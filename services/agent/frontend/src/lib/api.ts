/**
 * API client for HavenCore backend endpoints.
 * All paths are relative — Vite proxy (dev) or same-origin (prod) handles routing.
 */

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, init);
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${detail}`);
	}
	return res.json();
}

// --- Status ---

export interface SystemStatus {
	agent: { name: string; healthy: boolean };
	mcp: {
		configured_servers: string[];
		connected_servers: string[];
		failed_servers: Record<string, string>;
		total_mcp_tools: number;
		tools_by_server: Record<string, number>;
	};
	database: { connected: boolean };
	llm: { healthy: boolean; models?: any; error?: string };
}

export function getStatus(): Promise<SystemStatus> {
	return fetchJSON('/api/status');
}

// --- Tools ---

export interface ToolInfo {
	name: string;
	description: string;
	parameters: any;
}

export interface ToolsResponse {
	tools_by_server: Record<string, ToolInfo[]>;
	total: number;
}

export function getTools(): Promise<ToolsResponse> {
	return fetchJSON('/api/tools');
}

// --- Conversations ---

export interface ConversationSummary {
	session_id: string;
	created_at: string;
	message_count: number;
	agent_name: string;
	metadata: Record<string, any>;
}

export interface ConversationDetail {
	messages: any[];
	created_at: string;
	metadata: Record<string, any>;
}

export function listConversations(limit = 20, offset = 0): Promise<{ conversations: ConversationSummary[]; limit: number; offset: number }> {
	return fetchJSON(`/api/conversations?limit=${limit}&offset=${offset}`);
}

export function getConversation(sessionId: string): Promise<{ conversation: ConversationDetail[] }> {
	return fetchJSON(`/api/conversations/${sessionId}`);
}

// --- Chat ---

export interface ChatResponse {
	response: string;
	events: ChatEvent[];
}

export interface ChatEvent {
	type: 'thinking' | 'tool_call' | 'tool_result' | 'done' | 'error';
	[key: string]: any;
}

export function chatSync(message: string): Promise<ChatResponse> {
	return fetchJSON('/api/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ message }),
	});
}

// --- Home Assistant ---

export interface HAEntity {
	entity_id: string;
	state: string;
	friendly_name: string;
	domain: string;
	last_changed: string;
}

export interface HADomainSummary {
	domain: string;
	total: number;
	active: number;
}

export function getEntities(domain?: string): Promise<{ entities: HAEntity[]; count: number }> {
	const q = domain ? `?domain=${domain}` : '';
	return fetchJSON(`/api/ha/entities${q}`);
}

export function getEntitySummary(): Promise<{ domains: HADomainSummary[] }> {
	return fetchJSON('/api/ha/entities/summary');
}

export function getAutomations(): Promise<{ automations: any[]; count: number }> {
	return fetchJSON('/api/ha/automations');
}

export function getScenes(): Promise<{ scenes: any[]; count: number }> {
	return fetchJSON('/api/ha/scenes');
}
