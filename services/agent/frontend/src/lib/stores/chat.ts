/**
 * Chat store — manages WebSocket connection and message history.
 */
import { writable, get } from 'svelte/store';
import type { ChatEvent } from '$lib/api';

export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	events: ChatEvent[];
	timestamp: number;
}

export const messages = writable<ChatMessage[]>([]);
export const isConnected = writable(false);
export const isProcessing = writable(false);

let ws: WebSocket | null = null;
let currentEvents: ChatEvent[] = [];
let currentContent = '';

function getWsUrl(): string {
	const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	return `${protocol}//${window.location.host}/ws/chat`;
}

export function connect() {
	if (ws && ws.readyState === WebSocket.OPEN) return;

	ws = new WebSocket(getWsUrl());

	ws.onopen = () => {
		isConnected.set(true);
	};

	ws.onclose = () => {
		isConnected.set(false);
		// Auto-reconnect after 3s
		setTimeout(connect, 3000);
	};

	ws.onerror = () => {
		isConnected.set(false);
	};

	ws.onmessage = (event) => {
		const data: ChatEvent = JSON.parse(event.data);
		currentEvents.push(data);

		if (data.type === 'done') {
			currentContent = data.content || '';
			messages.update((msgs) => {
				// Replace the last assistant message placeholder
				const last = msgs[msgs.length - 1];
				if (last && last.role === 'assistant') {
					last.content = currentContent;
					last.events = [...currentEvents];
				}
				return [...msgs];
			});
			isProcessing.set(false);
			currentEvents = [];
			currentContent = '';
		} else if (data.type === 'error') {
			messages.update((msgs) => {
				const last = msgs[msgs.length - 1];
				if (last && last.role === 'assistant') {
					last.content = data.error || 'An error occurred';
					last.events = [...currentEvents];
				}
				return [...msgs];
			});
			isProcessing.set(false);
			currentEvents = [];
		} else {
			// Update the placeholder with live events
			messages.update((msgs) => {
				const last = msgs[msgs.length - 1];
				if (last && last.role === 'assistant') {
					last.events = [...currentEvents];
				}
				return [...msgs];
			});
		}
	};
}

export function sendMessage(text: string) {
	if (!ws || ws.readyState !== WebSocket.OPEN || !text.trim()) return;

	// Add user message
	messages.update((msgs) => [
		...msgs,
		{ role: 'user', content: text, events: [], timestamp: Date.now() }
	]);

	// Add assistant placeholder
	messages.update((msgs) => [
		...msgs,
		{ role: 'assistant', content: '', events: [], timestamp: Date.now() }
	]);

	currentEvents = [];
	isProcessing.set(true);

	ws.send(JSON.stringify({ message: text }));
}

export function clearMessages() {
	messages.set([]);
}

export function disconnect() {
	if (ws) {
		ws.close();
		ws = null;
	}
}
