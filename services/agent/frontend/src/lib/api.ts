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

// --- LLM provider toggle ---

export interface LLMProviderInfo {
	provider: string;
	model: string | null;
	valid: string[];
	since: string | null;
}

export function getLLMProvider(): Promise<LLMProviderInfo> {
	return fetchJSON('/api/system/llm-provider');
}

export function setLLMProvider(provider: string): Promise<LLMProviderInfo> {
	return fetchJSON('/api/system/llm-provider', {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({ provider }),
	});
}

// --- Conversations ---

export interface ConversationSummary {
	id: number;
	session_id: string;
	created_at: string;
	message_count: number;
	agent_name: string;
	metadata: Record<string, any>;
}

export interface ConversationDetail {
	id: number;
	messages: any[];
	created_at: string;
	metadata: Record<string, any>;
}

export function getConversationDeviceName(c: { metadata?: Record<string, any> }): string | null {
	const v = c.metadata?.device_name;
	return typeof v === 'string' && v.trim() ? v : null;
}

export function listConversations(limit = 20, offset = 0): Promise<{ conversations: ConversationSummary[]; limit: number; offset: number }> {
	return fetchJSON(`/api/conversations?limit=${limit}&offset=${offset}`);
}

export function getConversation(
	sessionId: string,
	flushId?: number,
): Promise<{ conversation: ConversationDetail[] }> {
	const qs = flushId != null ? `?id=${flushId}` : '';
	return fetchJSON(`/api/conversations/${sessionId}${qs}`);
}

export interface ResumeResponse {
	session_id: string;
	resumed: boolean;
	message_count: number;
}

export function resumeConversation(sessionId: string): Promise<ResumeResponse> {
	return fetchJSON(`/api/conversations/${sessionId}/resume`, {
		method: 'POST',
	});
}

// --- Chat ---

export interface ChatResponse {
	response: string;
	events: ChatEvent[];
}

export interface ChatEvent {
	type: 'thinking' | 'tool_call' | 'tool_result' | 'metric' | 'done' | 'error';
	[key: string]: any;
}

export interface TurnMetric {
	llm_ms: number;
	tool_ms_total: number;
	total_ms: number;
	iterations: number;
	tool_calls: { name: string; ms: number }[];
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

// --- TTS ---

export interface TtsVoicesResponse {
	voices: { id: string; label: string }[];
	formats: string[];
}

export function getTtsVoices(): Promise<TtsVoicesResponse> {
	return fetchJSON('/api/tts/voices');
}

export async function ttsSpeak(body: {
	text: string;
	voice?: string;
	format?: string;
	speed?: number;
}): Promise<Blob> {
	const res = await fetch('/api/tts/speak', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${detail}`);
	}
	return res.blob();
}

export function getTtsHealth(): Promise<{ status: string }> {
	return fetchJSON('/api/tts/health');
}

// --- STT ---

export async function sttTranscribe(
	file: File,
	opts?: { language?: string; response_format?: string },
): Promise<{ text: string }> {
	const form = new FormData();
	form.append('file', file);
	if (opts?.language) form.append('language', opts.language);
	form.append('response_format', opts?.response_format || 'json');
	const res = await fetch('/api/stt/transcribe', { method: 'POST', body: form });
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${detail}`);
	}
	return res.json();
}

export function getSttHealth(): Promise<{ status: string }> {
	return fetchJSON('/api/stt/health');
}

// --- Vision ---

export async function visionAsk(
	image: File,
	prompt: string,
	opts?: { max_tokens?: number; temperature?: number },
): Promise<{ response: string; latency_ms: number; usage?: any }> {
	const form = new FormData();
	form.append('image', image);
	form.append('prompt', prompt);
	if (opts?.max_tokens !== undefined) form.append('max_tokens', String(opts.max_tokens));
	if (opts?.temperature !== undefined) form.append('temperature', String(opts.temperature));
	const res = await fetch('/api/vision/ask', { method: 'POST', body: form });
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${detail}`);
	}
	return res.json();
}

export function getVisionHealth(): Promise<{ status: string; models?: any }> {
	return fetchJSON('/api/vision/health');
}

// --- ComfyUI ---

export interface ComfyJobStatus {
	job_id: string;
	status: 'pending' | 'done' | 'error';
	elapsed_ms: number;
	images: { filename: string; url: string }[];
	error?: string;
}

export function comfyGenerate(body: {
	prompt: string;
	negative_prompt?: string;
	seed?: number;
	steps?: number;
	workflow?: string;
}): Promise<{ job_id: string; status: string }> {
	return fetchJSON('/api/comfy/generate', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
}

export function comfyStatus(jobId: string): Promise<ComfyJobStatus> {
	return fetchJSON(`/api/comfy/status/${jobId}`);
}

export function getComfyHealth(): Promise<{ status: string }> {
	return fetchJSON('/api/comfy/health');
}

// --- Metrics ---

export interface TurnRow {
	id: number;
	session_id: string | null;
	device_name: string | null;
	created_at: string;
	llm_ms: number;
	tool_ms_total: number;
	total_ms: number;
	iterations: number;
	tool_calls: { name: string; ms: number }[];
}

export interface MetricsSummary {
	turns: number;
	avg_llm_ms: number;
	avg_total_ms: number;
	p95_total_ms: number;
	turns_today: number;
	per_day: { day: string; turns: number }[];
}

export interface TopTool {
	name: string;
	count: number;
	avg_ms: number;
}

export function getMetricTurns(limit = 50): Promise<{ turns: TurnRow[]; limit: number }> {
	return fetchJSON(`/api/metrics/turns?limit=${limit}`);
}

export function getMetricSummary(days = 7): Promise<MetricsSummary> {
	return fetchJSON(`/api/metrics/summary?days=${days}`);
}

export function getTopTools(days = 7, limit = 10): Promise<{ tools: TopTool[]; days: number }> {
	return fetchJSON(`/api/metrics/top-tools?days=${days}&limit=${limit}`);
}
