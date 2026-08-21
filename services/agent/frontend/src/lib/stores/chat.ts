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
	/**
	 * Stable per-entry id, used to follow the in-flight assistant placeholder
	 * even when another entry (e.g. a `summary_reset` card) is spliced into the
	 * array mid-turn. Optional because resumed/history-hydrated entries are
	 * never turn targets.
	 */
	id?: number;
	role: 'user' | 'assistant' | 'summary';
	content: string;
	events: ChatEvent[];
	timestamp: number;
	metric?: TurnMetric;
	reason?: string;
}

let messageIdSeq = 0;
function nextMessageId(): number {
	return ++messageIdSeq;
}

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

export const messages = writable<ChatMessage[]>([]);
export const isConnected = writable(false);
export const isProcessing = writable(false);
export const connectionState = writable<ConnectionState>('connecting');
export const currentSessionId = writable<string | null>(null);
export const currentDeviceName = writable<string | null>(null);

const SESSION_STORAGE_KEY = 'haven.chat.session_id';
const DEVICE_NAME_STORAGE_KEY = 'haven.device_name';

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

// Hydrate device name from localStorage (per-browser, persists across tabs).
if (typeof window !== 'undefined') {
	try {
		const v = window.localStorage.getItem(DEVICE_NAME_STORAGE_KEY);
		if (v && v.trim()) currentDeviceName.set(v);
	} catch {}
}

export function setSessionId(sid: string | null) {
	currentSessionId.set(sid);
	persistSessionId(sid);
}

export function setDeviceName(name: string | null) {
	const cleaned = name?.trim() || null;
	currentDeviceName.set(cleaned);
	if (typeof window !== 'undefined') {
		try {
			if (cleaned) window.localStorage.setItem(DEVICE_NAME_STORAGE_KEY, cleaned);
			else window.localStorage.removeItem(DEVICE_NAME_STORAGE_KEY);
		} catch {}
	}
	// Push the change to an open WS so the active session adopts the new label.
	// Backend treats empty/whitespace as a no-op, so unsetting locally won't
	// clear the server-side label until a new WS opens.
	if (ws && ws.readyState === WebSocket.OPEN && cleaned) {
		ws.send(JSON.stringify({ type: 'session', device_name: cleaned }));
	}
}

let ws: WebSocket | null = null;
let currentEvents: ChatEvent[] = [];
let currentContent = '';
let currentMetric: TurnMetric | undefined;
/**
 * Id of the empty assistant bubble the current turn is streaming into, or
 * null between turns. Tracked explicitly because the placeholder is NOT
 * reliably the last array element: `orchestrator.run()` yields SUMMARY_RESET
 * as the very first event of a turn, and its card is a pane-level entry of
 * its own. Targeting `msgs[msgs.length - 1]` instead used to make every later
 * event of that turn (thinking / tool_call / reasoning / metric / done) hit a
 * non-assistant entry and get dropped, leaving the bubble permanently empty.
 */
let pendingAssistantId: number | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
const RECONNECT_DELAY_MS = 3000;

/**
 * Apply `mutate` to the message this turn's events belong to.
 *
 * Resolution order:
 *  - a tracked in-flight placeholder, located by id wherever it sits;
 *  - if it's tracked but no longer present (the pane was cleared or replaced
 *    by a history resume mid-turn) the event is dropped rather than written
 *    onto an unrelated message;
 *  - with nothing in flight, fall back to a trailing assistant entry so
 *    out-of-band frames behave as they always did.
 */
function updateTurnMessage(mutate: (msg: ChatMessage) => void) {
	messages.update((msgs) => {
		let target: ChatMessage | undefined;
		if (pendingAssistantId !== null) {
			target = msgs.find((m) => m.id === pendingAssistantId);
		} else {
			const last = msgs[msgs.length - 1];
			if (last && last.role === 'assistant') target = last;
		}
		if (target) mutate(target);
		return [...msgs];
	});
}

function endTurn() {
	isProcessing.set(false);
	pendingAssistantId = null;
	currentEvents = [];
	currentContent = '';
	currentMetric = undefined;
}

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
		const dname = get(currentDeviceName);
		const frame: Record<string, unknown> = {
			type: 'session',
			session_id: sid,
			idle_timeout: -1,
		};
		if (dname) frame.device_name = dname;
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify(frame));
		}
	};

	ws.onclose = () => {
		isConnected.set(false);
		connectionState.set('reconnecting');
		// The turn is abandoned with the socket; stop tracking its placeholder
		// so a late frame can't land on a stale bubble after reconnect.
		endTurn();
		scheduleReconnect();
	};

	ws.onerror = () => {
		isConnected.set(false);
	};

	ws.onmessage = (event) => {
		handleChatEvent(JSON.parse(event.data) as ChatEvent);
	};
}

/**
 * Reduce a single server frame into the store. Exported (rather than living
 * inside the WS closure) so the turn-routing rules are testable in isolation.
 */
export function handleChatEvent(data: ChatEvent) {
	// Session assignment (server → client). Always first non-turn frame.
	if (data.type === 'session') {
		const sid = (data as any).session_id as string | undefined;
		if (sid) setSessionId(sid);
		return;
	}

	// Inline marker for server-side session summarization. Rendered as
	// its own pane-level entry (not buffered into the current turn),
	// and may arrive mid-turn or between turns via the pool's pub/sub.
	if (data.type === 'summary_reset') {
		const summary = (data as any).summary as string | undefined;
		const reason = (data as any).reason as string | undefined;
		messages.update((msgs) => {
			const entry: ChatMessage = {
				id: nextMessageId(),
				role: 'summary',
				content: summary || '',
				events: [],
				timestamp: Date.now(),
				reason,
			};
			const idx =
				pendingAssistantId === null
					? -1
					: msgs.findIndex((m) => m.id === pendingAssistantId);
			// Mid-turn: the server summarizes before it streams the reply, so
			// the card belongs directly above the in-flight placeholder —
			// after the user's message, before the assistant's answer.
			if (idx >= 0) return [...msgs.slice(0, idx), entry, ...msgs.slice(idx)];
			return [...msgs, entry];
		});
		return;
	}

	// Live2D facial-expression cue (Phase C). The dashboard has no avatar
	// rig yet, so this frame is logged and dropped — the companion app's
	// Live2D overlay is the real consumer. Logged rather than silently
	// swallowed so the channel is visible when debugging the WS stream.
	if (data.type === 'avatar_state') {
		console.debug('[chat] avatar_state', data.expression);
		return;
	}

	currentEvents.push(data);

	if (data.type === 'metric') {
		const { type: _t, ...payload } = data;
		currentMetric = payload as TurnMetric;
		updateTurnMessage((msg) => {
			msg.metric = currentMetric;
			msg.events = [...currentEvents];
		});
		return;
	}

	if (data.type === 'done') {
		currentContent = data.content || '';
		// Fill in the placeholder created by sendMessage().
		updateTurnMessage((msg) => {
			msg.content = currentContent;
			msg.events = [...currentEvents];
			if (currentMetric) msg.metric = currentMetric;
		});
		endTurn();
	} else if (data.type === 'error') {
		updateTurnMessage((msg) => {
			msg.content = data.error || 'An error occurred';
			msg.events = [...currentEvents];
		});
		endTurn();
	} else {
		// thinking / tool_call / tool_result / reasoning / device_action and any
		// future frame: stream the buffered events onto the placeholder.
		updateTurnMessage((msg) => {
			msg.events = [...currentEvents];
		});
	}
}

export function sendMessage(text: string) {
	if (!ws || ws.readyState !== WebSocket.OPEN || !text.trim()) return;

	// Add the user message, then the assistant placeholder this turn's events
	// stream into. The placeholder is followed by id from here on — a
	// `summary_reset` card can land between it and the end of the array.
	const placeholderId = nextMessageId();
	messages.update((msgs) => [
		...msgs,
		{ id: nextMessageId(), role: 'user', content: text, events: [], timestamp: Date.now() },
		{ id: placeholderId, role: 'assistant', content: '', events: [], timestamp: Date.now() }
	]);
	pendingAssistantId = placeholderId;

	currentEvents = [];
	isProcessing.set(true);

	ws.send(JSON.stringify({ message: text }));
}

export function clearMessages() {
	messages.set([]);
	// `pendingAssistantId` is deliberately left set: updateTurnMessage() can no
	// longer find that id, so any frame still arriving for the cleared turn is
	// discarded. Nulling it here would instead re-arm the trailing-assistant
	// fallback and let a late reply overwrite whatever is put in the pane next
	// (history "Resume" does clearMessages() + messages.set(...) back to back).
	// The next sendMessage(), endTurn() or startNewChat() retires the id.
}

/**
 * Start a brand-new chat session: clear visible messages, drop the persisted
 * session_id, and reconnect the WS so the server mints a fresh session.
 */
export function startNewChat() {
	messages.set([]);
	endTurn();
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
