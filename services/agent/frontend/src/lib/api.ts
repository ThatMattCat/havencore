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
	messages: any[];
}

export function resumeConversation(sessionId: string): Promise<ResumeResponse> {
	return fetchJSON(`/api/conversations/${sessionId}/resume`, {
		method: 'POST',
	});
}

export function deleteConversation(sessionId: string, flushId: number): Promise<{ deleted: number }> {
	return fetchJSON(`/api/conversations/${sessionId}?id=${flushId}`, {
		method: 'DELETE',
	});
}

// --- Chat ---

export interface ChatResponse {
	response: string;
	events: ChatEvent[];
}

export interface ChatEvent {
	type: 'thinking' | 'tool_call' | 'tool_result' | 'reasoning' | 'metric' | 'done' | 'error' | 'summary_reset' | 'session';
	[key: string]: any;
}

export interface TurnMetric {
	llm_ms: number;
	tool_ms_total: number;
	total_ms: number;
	iterations: number;
	tool_calls: { name: string; ms: number }[];
	cache_read_tokens?: number;
	cache_creation_tokens?: number;
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
	cache_read_tokens?: number;
	cache_creation_tokens?: number;
}

export interface MetricsSummary {
	turns: number;
	avg_llm_ms: number;
	avg_total_ms: number;
	p95_total_ms: number;
	turns_today: number;
	per_day: { day: string; turns: number }[];
	cache_read_total?: number;
	cache_create_total?: number;
	cache_hit_rate?: number | null;
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

// --- Face recognition ---

export type AccessLevel = 'unknown' | 'resident' | 'guest' | 'blocked';
export const ACCESS_LEVELS: AccessLevel[] = ['unknown', 'resident', 'guest', 'blocked'];

export interface FaceImage {
	id: string;
	path: string;
	is_primary: boolean;
	source: string;
	quality_score: number | null;
	created_at: string;
}

export interface Person {
	id: string;
	name: string;
	access_level: AccessLevel;
	notes: string | null;
	image_count: number;
	created_at: string;
	updated_at: string;
}

export interface PersonDetail extends Person {
	images: FaceImage[];
}

export interface FaceDetection {
	id: string;
	event_id: string;
	camera: string;
	captured_at: string;
	person_id: string | null;
	person_name: string | null;
	confidence: number | null;
	quality_score: number | null;
	snapshot_path: string;
	review_state: 'auto' | 'confirmed' | 'rejected' | 'pending';
	embedding_contributed: boolean;
	// InsightFace genderage outputs — informational, not used in matching.
	age: number | null;
	sex: 'M' | 'F' | null;
}

export interface FaceCamera {
	sensor_entity: string;
	camera_entity: string;
	camera_exists: boolean;
	current_state: string;
}

// URL helpers — return the agent-proxy stream URLs so callers can drop
// them straight into <img src=...>.
export function faceImageUrl(faceImageId: string): string {
	return `/api/face/face_images/${faceImageId}/bytes`;
}

export function detectionSnapshotUrl(detectionId: string): string {
	return `/api/face/detections/${detectionId}/snapshot`;
}

export function listPeople(): Promise<Person[]> {
	return fetchJSON('/api/face/people');
}

export function getPerson(personId: string): Promise<PersonDetail> {
	return fetchJSON(`/api/face/people/${personId}`);
}

export function createPerson(body: {
	name: string;
	access_level?: AccessLevel;
	notes?: string;
}): Promise<Person> {
	return fetchJSON('/api/face/people', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
}

export function updatePerson(
	personId: string,
	body: { access_level?: AccessLevel; notes?: string },
): Promise<Person> {
	return fetchJSON(`/api/face/people/${personId}`, {
		method: 'PATCH',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
}

export function deletePerson(
	personId: string,
): Promise<{ id: string; images_removed: number; qdrant_points_removed: number }> {
	return fetchJSON(`/api/face/people/${personId}`, { method: 'DELETE' });
}

export async function enrollImage(
	personId: string,
	file: File,
	opts?: { isPrimary?: boolean; source?: string },
): Promise<{ id: string; quality_score: number; faces_detected: number }> {
	const form = new FormData();
	form.append('file', file);
	form.append('is_primary', opts?.isPrimary ? 'true' : 'false');
	form.append('source', opts?.source ?? 'upload');
	const res = await fetch(`/api/face/people/${personId}/images`, {
		method: 'POST',
		body: form,
	});
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new Error(`${res.status}: ${detail}`);
	}
	return res.json();
}

export function setPrimaryImage(personId: string, faceImageId: string): Promise<FaceImage> {
	return fetchJSON(`/api/face/people/${personId}/images/${faceImageId}/set-primary`, {
		method: 'POST',
	});
}

export function deleteFaceImage(
	personId: string,
	faceImageId: string,
): Promise<{ id: string; deleted: boolean }> {
	return fetchJSON(`/api/face/people/${personId}/images/${faceImageId}`, {
		method: 'DELETE',
	});
}

export interface DetectionsQuery {
	camera?: string;
	person_id?: string;
	since_seconds_ago?: number;
	review_state?: 'auto' | 'confirmed' | 'rejected' | 'pending';
	unknowns_only?: boolean;
	limit?: number;
}

export function listDetections(q: DetectionsQuery = {}): Promise<FaceDetection[]> {
	const params = new URLSearchParams();
	if (q.camera) params.set('camera', q.camera);
	if (q.person_id) params.set('person_id', q.person_id);
	if (q.since_seconds_ago != null) params.set('since_seconds_ago', String(q.since_seconds_ago));
	if (q.review_state) params.set('review_state', q.review_state);
	if (q.unknowns_only) params.set('unknowns_only', 'true');
	if (q.limit != null) params.set('limit', String(q.limit));
	const qs = params.toString();
	return fetchJSON(`/api/face/detections${qs ? `?${qs}` : ''}`);
}

export interface ConfirmResult {
	detection_id: string;
	person_id: string;
	person_name: string;
	embedding_contributed: boolean;
	quality_score: number | null;
	faces_detected: number;
}

export function confirmDetection(
	detectionId: string,
	body: { person_id?: string; name?: string },
): Promise<ConfirmResult> {
	return fetchJSON(`/api/face/detections/${detectionId}/confirm`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
}

export function rejectDetection(detectionId: string): Promise<FaceDetection> {
	return fetchJSON(`/api/face/detections/${detectionId}/reject`, { method: 'POST' });
}

export function listFaceCameras(): Promise<FaceCamera[]> {
	return fetchJSON('/api/face/cameras');
}

export interface FaceJob {
	job_id: string;
	type: 'rescan_unknowns' | 'rebuild_embeddings';
	status: 'running' | 'done' | 'error';
	phase: string;
	started_at: number;
	finished_at: number | null;
	elapsed_ms: number;
	totals: Record<string, number>;
	errors: { detection_id?: string | null; face_image_id?: string | null; reason: string; detail?: string }[];
}

export function startRescanUnknowns(): Promise<{ job_id: string; status: string }> {
	return fetchJSON('/api/face/admin/rescan-unknowns', { method: 'POST' });
}

export function getFaceJob(jobId: string): Promise<FaceJob> {
	return fetchJSON(`/api/face/admin/jobs/${jobId}`);
}

export function bulkDeleteDetections(
	scope: 'rejected' | 'all_unknowns',
): Promise<{ rows_deleted: number; files_unlinked: number; scope: string }> {
	return fetchJSON('/api/face/detections/bulk-delete', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ scope }),
	});
}

// --- Camera ↔ zone mapping ---

export interface CameraZoneRow {
	camera_entity: string;
	sensor_entity: string | null;
	camera_exists: boolean;
	current_state: string | null;
	zone: string | null;
	zone_label: string | null;
	notes: string | null;
	updated_at: string | null;
}

export interface CameraZonesResponse {
	cameras: CameraZoneRow[];
	zones: string[];
}

export function listCameraZones(): Promise<CameraZonesResponse> {
	return fetchJSON('/api/cameras');
}

export function setCameraZone(
	cameraEntity: string,
	body: { zone: string; zone_label?: string | null; notes?: string | null },
): Promise<{ camera_entity: string; zone: string; zone_label: string | null; notes: string | null }> {
	return fetchJSON(`/api/cameras/${encodeURIComponent(cameraEntity)}/zone`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
}

export function clearCameraZone(
	cameraEntity: string,
): Promise<{ camera_entity: string; deleted: boolean }> {
	return fetchJSON(`/api/cameras/${encodeURIComponent(cameraEntity)}/zone`, {
		method: 'DELETE',
	});
}
