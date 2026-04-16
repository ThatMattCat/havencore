/**
 * Chat store — manages WebSocket connection and message history.
 */
import { writable, get } from 'svelte/store';
import type { ChatEvent, TurnMetric } from '$lib/api';

export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	events: ChatEvent[];
	timestamp: number;
	metric?: TurnMetric;
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export const messages = writable<ChatMessage[]>([]);
export const isConnected = writable(false);
export const isProcessing = writable(false);
export const connectionState = writable<ConnectionState>('connecting');

let ws: WebSocket | null = null;
let currentEvents: ChatEvent[] = [];
let currentContent = '';
let currentMetric: TurnMetric | undefined;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
const RECONNECT_DELAY_MS = 3000;

function getWsUrl(): string {
	const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	return `${protocol}//${window.location.host}/ws/chat`;
}

function scheduleReconnect() {
	if (reconnectTimer) return;
	reconnectTimer = setTimeout(() => {
		reconnectTimer = null;
		connect();
	}, RECONNECT_DELAY_MS);
}

export function connect() {
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	connectionState.update((s) => (s === 'disconnected' || s === 'reconnecting' ? 'reconnecting' : 'connecting'));

	ws = new WebSocket(getWsUrl());

	ws.onopen = () => {
		isConnected.set(true);
		connectionState.set('connected');
	};

	ws.onclose = () => {
		isConnected.set(false);
		connectionState.set('reconnecting');
		isProcessing.set(false);
		scheduleReconnect();
	};

	ws.onerror = () => {
		isConnected.set(false);
	};

	ws.onmessage = (event) => {
		const data: ChatEvent = JSON.parse(event.data);
		currentEvents.push(data);

		if (data.type === 'metric') {
			const { type: _t, ...payload } = data;
			currentMetric = payload as TurnMetric;
			messages.update((msgs) => {
				const last = msgs[msgs.length - 1];
				if (last && last.role === 'assistant') {
					last.metric = currentMetric;
					last.events = [...currentEvents];
				}
				return [...msgs];
			});
			return;
		}

		if (data.type === 'done') {
			currentContent = data.content || '';
			messages.update((msgs) => {
				// Replace the last assistant message placeholder
				const last = msgs[msgs.length - 1];
				if (last && last.role === 'assistant') {
					last.content = currentContent;
					last.events = [...currentEvents];
					if (currentMetric) last.metric = currentMetric;
				}
				return [...msgs];
			});
			isProcessing.set(false);
			currentEvents = [];
			currentContent = '';
			currentMetric = undefined;
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
			currentMetric = undefined;
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
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	if (ws) {
		ws.onclose = null;
		ws.close();
		ws = null;
	}
	connectionState.set('disconnected');
}

export function retryNow() {
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	connect();
}
