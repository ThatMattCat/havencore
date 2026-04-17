/**
 * Chat store — manages WebSocket connection, message history, and session identity.
 *
 * Session identity is client-owned: the store persists `currentSessionId` in
 * sessionStorage and sends it as the first WS frame. On first connect (or
 * after `startNewChat()`), the server mints a session and echoes the id back
 * via a `{"type": "session"}` event.
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
export const currentSessionId = writable<string | null>(null);

const SESSION_STORAGE_KEY = 'haven.chat.session_id';

function readPersistedSessionId(): string | null {
	if (typeof window === 'undefined') return null;
	try {
		return window.sessionStorage.getItem(SESSION_STORAGE_KEY);
	} catch {
		return null;
	}
}

function persistSessionId(sid: string | null) {
	if (typeof window === 'undefined') return;
	try {
		if (sid) {
			window.sessionStorage.setItem(SESSION_STORAGE_KEY, sid);
		} else {
			window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
		}
	} catch {}
}

// Hydrate store from sessionStorage on module load (browser only).
if (typeof window !== 'undefined') {
	const persisted = readPersistedSessionId();
	if (persisted) currentSessionId.set(persisted);
}

export function setSessionId(sid: string | null) {
	currentSessionId.set(sid);
	persistSessionId(sid);
}

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
		// Send our session preference (if any) as the first frame. Server
		// will either honor it, cold-resume from DB, or mint a new one.
		const sid = get(currentSessionId);
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type: 'session', session_id: sid }));
		}
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

		// Session assignment (server → client). Always first non-turn frame.
		if (data.type === 'session') {
			const sid = (data as any).session_id as string | undefined;
			if (sid) setSessionId(sid);
			return;
		}

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

/**
 * Start a brand-new chat session: clear visible messages, drop the persisted
 * session_id, and reconnect the WS so the server mints a fresh session.
 */
export function startNewChat() {
	messages.set([]);
	isProcessing.set(false);
	currentEvents = [];
	currentContent = '';
	currentMetric = undefined;
	setSessionId(null);
	if (ws) {
		ws.onclose = null;
		try { ws.close(); } catch {}
		ws = null;
	}
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	connect();
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
