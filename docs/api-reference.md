# API Reference

HavenCore exposes two surfaces:

1. **OpenAI-compatible APIs** — `/v1/chat/completions`, `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/models`. These ride through the Nginx gateway at `http://localhost` and are used by the voice pipeline (edge devices, external integrations).
2. **Agent dashboard APIs** — `/api/*` and `/ws/*`, served by the agent on port 6002 (also available through Nginx). Used by the SvelteKit dashboard at `http://localhost:6002/` for chat, metrics, service playgrounds, Home Assistant state, and live logs.

## Authentication

The OpenAI-compatible endpoints forwarded to vLLM use the `LLM_API_KEY` configured in `.env`:

```bash
LLM_API_KEY="your_secret_key"
curl -H "Authorization: Bearer your_secret_key" http://localhost/v1/chat/completions ...
```

The `/api/*` dashboard endpoints are unauthenticated — the dashboard is intended for a private/home network. Do not expose port 6002 to the public internet without adding your own auth in front of it.

## Chat Completions API

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint for conversational AI. **Stateless**: each request builds an ephemeral orchestrator for the one call, runs the agent loop, and discards it. The endpoint never touches the session pool, the conversation-history DB, or the `turn_metrics` table. Callers must supply their full message history in the request body — the server remembers nothing between calls. For a pool-backed, history-tracked chat surface, use `/api/chat` or `/ws/chat` instead.

**Endpoint**: `POST http://localhost/v1/chat/completions`

#### Request Headers
```
Content-Type: application/json
Authorization: Bearer your_api_key
```

#### Request Body
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant."
    },
    {
      "role": "user", 
      "content": "Hello! How are you today?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model identifier (can be any value) |
| `messages` | array | Yes | - | Array of message objects |
| `temperature` | number | No | 0.7 | Randomness (0.0-2.0) |
| `max_tokens` | number | No | 1024 | Maximum response length |
| `stream` | boolean | No | false | Enable streaming responses |

#### Message Object
```json
{
  "role": "user|assistant|system",
  "content": "Message content"
}
```

#### Response
```json
{
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 31,
    "total_tokens": 87
  }
}
```

#### Tool Calling

HavenCore supports tool calling for external integrations:

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather like in San Francisco?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather_forecast",
        "description": "Get weather forecast for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name or coordinates"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

#### Available Tools

Tools are grouped into MCP servers. Each server has its own reference
doc with the full tool list, arguments, config, and troubleshooting.

- **Home Assistant** — 20 tools (domain state / services, light & climate
  helpers, scenes, scripts, automations, notifications, areas, presence,
  timers, Jinja templates, history, calendar (read + create), media
  transport). See
  [MCP Home Assistant](services/agent/tools/home-assistant.md).
- **Plex** — library search + cloud-relay playback
  (`plex_search`, `plex_list_recent`, `plex_list_on_deck`,
  `plex_list_clients`, `plex_play`). See
  [MCP Plex](services/agent/tools/plex.md) and [Media Control](integrations/media-control.md).
- **Music Assistant** — audio-only playback router for speakers,
  Chromecasts, and Google Homes. See
  [MCP Music Assistant](services/agent/tools/music-assistant.md).
- **General Tools** — `get_weather_forecast`, `brave_search`,
  `search_wikipedia`, `wolfram_alpha`, `generate_image`, `send_signal_message`,
  `query_multimodal_api`. See [MCP General](services/agent/tools/general.md).
- **Qdrant** — semantic memory (`create_memory`, `search_memories`). See
  [MCP Qdrant](services/agent/tools/qdrant.md).
- **MQTT / Cameras** — `get_camera_snapshots`. See [MCP MQTT](services/agent/tools/mqtt.md).

#### Example Request
```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Turn on the living room lights"}
    ]
  }'
```

## Audio APIs

### Speech Synthesis (Text-to-Speech)

#### POST /v1/audio/speech

Convert text to spoken audio using Kokoro TTS.

**Endpoint**: `POST http://localhost/v1/audio/speech`

#### Request Body
```json
{
  "input": "Hello, this is a test of the text to speech system.",
  "model": "tts-1",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | - | Text to convert to speech |
| `model` | string | No | "tts-1" | TTS model (any value accepted) |
| `voice` | string | No | "alloy" | Native Kokoro voice id (e.g. `af_heart`, `af_bella`, `am_michael`) or an OpenAI alias (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| `response_format` | string | No | "mp3" | Audio format (mp3, wav, opus, aac, flac, pcm) |
| `speed` | number | No | 1.0 | Playback speed (0.25-4.0) |

**Note**: Native Kokoro voice ids pass through to the pipeline (filtered to the configured `TTS_LANGUAGE`). OpenAI aliases resolve to the configured default voice (`TTS_VOICE`, fallback `af_heart`). Unknown names fall back to the default. `GET /v1/voices` returns the accepted catalog. Output is always WAV format regardless of `response_format`.

#### Response
Returns raw audio binary data with appropriate Content-Type header.

#### Example Request
```bash
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, HavenCore is working perfectly!",
    "model": "tts-1",
    "voice": "alloy"
  }' \
  --output speech.wav
```

### Speech Recognition (Speech-to-Text)

#### POST /v1/audio/transcriptions

Transcribe audio to text using Whisper models.

**Endpoint**: `POST http://localhost/v1/audio/transcriptions`

#### Request Format
Multipart form data with audio file:

```bash
curl -X POST http://localhost/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "language=en"
```

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | Yes | - | Audio file to transcribe |
| `model` | string | No | "whisper-1" | Model identifier |
| `language` | string | No | auto | Language code (en, es, fr, etc.) |
| `prompt` | string | No | - | Optional prompt to guide transcription |
| `response_format` | string | No | "json" | Response format (json, text, srt, vtt) |
| `temperature` | number | No | 0 | Sampling temperature |

#### Supported Audio Formats
- WAV
- MP3
- MP4
- MPEG
- MPGA
- M4A
- WEBM

#### Response
```json
{
  "text": "Hello, this is a transcription of the audio file."
}
```

## Model Management API

### GET /v1/models

List available models in the system.

**Endpoint**: `GET http://localhost/v1/models`

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "created": 1677610602,
      "owned_by": "openai"
    },
    {
      "id": "whisper-1",
      "object": "model", 
      "created": 1677610602,
      "owned_by": "openai"
    }
  ]
}
```

## System Management APIs

### Health Check Endpoints

#### GET /health
System-wide health check through the gateway.

```bash
curl http://localhost/health
```

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "agent": "healthy",
    "tts": "healthy", 
    "stt": "healthy",
    "llm": "healthy"
  }
}
```

#### Individual Service Health Checks

| Service | Endpoint | Description |
|---------|----------|-------------|
| Agent | `GET http://localhost:6002/health` | Agent service status |
| TTS | `GET http://localhost:6005/health` | Text-to-speech service |
| STT | `GET http://localhost:6001/health` | Speech-to-text service |
| LLM | `GET http://localhost:8000/health` | LLM backend status |

### Satellite OTA firmware

Static blobs served straight from disk by nginx (not the agent), on the
same gateway port as the rest of the API. Plain HTTP, no auth, LAN-only —
the satellite firmware's "Pull" button GETs `satellite.bin` and streams
it into `esp_https_ota`.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/firmware/satellite.bin` | Whole-file firmware image (`application/octet-stream`). Whole-file GETs only — no range support needed. |
| `GET` | `/firmware/satellite.json` | Optional version sidecar (`application/json`): `{"version", "size", "sha256"}`. Satellite ignores this today; reserved for future version-skip logic. |

Bare `/firmware/` returns 403 (autoindex disabled). Upload path and the
build-host scp/rsync recommendation are documented in
[`services/nginx/README.md`](services/nginx/README.md#satellite-ota-firmware-firmware).

### MCP Management APIs

#### GET /mcp/status
Get status of MCP (Model Context Protocol) connections.

**Endpoint**: `GET http://localhost:6002/mcp/status`

#### Response
```json
{
  "mcp_enabled": true,
  "active_servers": 2,
  "servers": [
    {
      "name": "filesystem",
      "status": "connected",
      "tools": 5,
      "last_ping": "2024-01-15T10:30:00Z"
    }
  ]
}
```

## Agent Dashboard APIs

The agent service at `http://localhost:6002` serves both the SvelteKit dashboard SPA and a set of JSON/WebSocket endpoints under `/api/*` and `/ws/*`. Full endpoint detail lives in [`services/agent/README.md`](services/agent/README.md); this section is a quick reference.

### REST (`/api/*`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/chat` | One-shot message with tool event log. Accepts `X-Session-Id` request header (optional); echoes the active `X-Session-Id` response header. Accepts `X-Idle-Timeout: <seconds>` (optional) to set the per-session idle window for summarize-and-reset; value is clamped to `[CONVERSATION_TIMEOUT_MIN, CONVERSATION_TIMEOUT_MAX]` and bad values are log-and-ignored. The sentinel `-1` means "never auto-summarize" and bypasses the clamp — used by the dashboard on every WS open. Accepts `X-Device-Name` (optional, e.g. `Kitchen Speaker`) — a human-readable label for the satellite/client driving the session; persists with every flush and rides each `turn_metrics` row. Trimmed, ASCII control chars stripped, capped at 64 chars; empty/whitespace values are no-ops (won't clobber a previously set name). |
| `GET`  | `/api/status` | Agent, MCP, DB, vLLM health. Includes `agent.sessions` = `{active_sessions, max_size, sweep_running}` for the session pool. |
| `GET`  | `/api/tools` | Registered tools grouped by MCP server |
| `GET`  | `/api/conversations` | Paginated conversation history. Each row carries its primary-key `id` (addresses a specific stored flush) alongside `session_id`, `created_at`, `message_count`, and `metadata` (which includes a `device_name` key — the label of the device that wrote that flush, captured per-flush so renames are preserved historically). |
| `GET`  | `/api/conversations/{session_id}` | Stored conversation snapshots for a session. Without query args, returns every flush for that `session_id` newest-first. With `?id=<flush_id>` (the `id` from a list row), returns only that single flush; mismatched `session_id`/`id` pairs 404. Each returned record includes its `id`, `messages`, `created_at`, and `metadata`. |
| `POST` | `/api/conversations/{session_id}/resume` | Hydrate a stored session into the live pool. Returns `{session_id, resumed, message_count, messages}` where `messages` is the post-hydrate orchestrator buffer with the leading base system prompt filtered out (any `[Prior conversation summary]` system message is preserved). The dashboard's "Resume" button on `/history` consumes this payload directly to pre-populate the chat transcript before navigating to `/chat`, so the user lands on what the model will actually see on the next turn. |
| `DELETE` | `/api/conversations/{session_id}?id=<flush_id>` | Delete a single stored flush row. The `id` query parameter is required and the `(session_id, id)` pair must match — a mismatched pair 404s. Returns `{"deleted": 1}` on success. Granularity is per-flush because `/history` lists one row per flush; bulk session deletion is not exposed. If the session is currently live in the pool, the in-memory orchestrator is untouched — the next flush will create a new row. The dashboard's `/history` page exposes this via a per-row delete button (with a browser-native confirm prompt). |
| `DELETE` | `/api/conversations` | Delete every stored conversation flush. Irreversible. Returns `{"deleted": N}`; a no-op against an already-empty table is a 200 with `deleted: 0`. Live pool sessions are untouched — their next flush creates fresh rows. The dashboard exposes this through a "Delete all" button at the top of `/history`, gated by an explicit confirmation prompt. |
| `GET`  | `/api/ha/entities` | HA entities (optionally `?domain=light`) |
| `GET`  | `/api/ha/entities/summary` | Entity counts per domain |
| `GET`  | `/api/ha/automations` | HA automations |
| `GET`  | `/api/ha/scenes` | HA scenes |
| `GET`  | `/api/metrics/turns` | Recent per-turn timings. Each turn row carries `device_name` (string or `null`) — denormalized from the orchestrator at write time so the dashboard can label rows by room/device without joining `conversation_histories`. Rows also carry `cache_read_tokens` and `cache_creation_tokens` (Anthropic prompt-cache counters summed across the turn's LLM calls; `0` for vLLM turns and legacy rows). |
| `GET`  | `/api/metrics/summary` | Daily aggregates, p95. Also exposes `cache_read_total` / `cache_create_total` (sums of the per-turn cache counters over the window) and a derived `cache_hit_rate = read / (read + create)`, guarded against zero. |
| `GET`  | `/api/metrics/top-tools` | Tool invocation counts + avg latency |
| `POST` | `/api/tts/speak` | Synthesize speech (returns audio binary) |
| `GET`  | `/api/tts/voices` | Voice alias list |
| `POST` | `/api/stt/transcribe` | Multipart transcription proxy |
| `POST` | `/api/vision/ask` | Multipart `file` (image OR short video) + `prompt` to the vision LLM (`vllm-vision`). MIME branches the upload to an `image_url` vs `video_url` chat-completion content part; unknown MIME → 415. The legacy `image` field name is still accepted. Returns `{response, latency_ms, usage, model, media_type}`. Used by the dashboard playground. nginx cap on `/api/` is 100 MB. |
| `POST` | `/api/vision/ask_url` | JSON `{text, image_url, max_tokens?, temperature?}` for image-URL inputs (the vision service fetches the URL itself). Used by the `query_multimodal_api` MCP tool — single chokepoint for logging/metrics. Returns `{response, latency_ms, usage, model}`. Image-only by design; for video, the multipart `/api/vision/ask` endpoint or the `mcp_vision_tools` server are the right entry points. |
| `POST` | `/api/comfy/generate` | Queue ComfyUI workflow |
| `GET`  | `/api/comfy/status/{prompt_id}` | Poll generation status |
| `GET`  | `/api/comfy/view` | Stream a generated image |
| `GET`  | `/api/{tts,stt,vision,comfy}/health` | Per-service health proxies |
| `GET`  | `/api/autonomy/status` | Autonomy engine state (running/paused, last dispatch, next-due) |
| `POST` | `/api/autonomy/pause` | Runtime kill switch — stop dispatch without restart |
| `POST` | `/api/autonomy/resume` | Clear runtime kill switch |
| `GET`  | `/api/autonomy/items` | List agenda items (scheduled autonomous behaviors) |
| `GET`  | `/api/autonomy/runs` | Recent run history; `?include_messages=1` for full traces |
| `POST` | `/api/autonomy/trigger/{id}` | Fire an agenda item immediately (bypasses schedule + rate limit) |
| `GET`  | `/api/memory/stats` | Per-tier counts (L2/L3/L4), pending proposals, estimated L4 prompt tokens |
| `GET`  | `/api/memory/l4` | List approved L4 entries (persistent context injected into every prompt) |
| `POST` | `/api/memory/l4` | Create an L4 entry directly (`{text, importance, tags}`) |
| `PATCH`| `/api/memory/l4/{id}` | Update an L4 entry's text, importance, or tags |
| `DELETE`| `/api/memory/l4/{id}` | Demote an L4 entry to L3 (the underlying memory is preserved) |
| `GET`  | `/api/memory/l4/proposals` | L3 entries flagged `pending_l4_approval` by the nightly job |
| `POST` | `/api/memory/l4/proposals/{id}/approve` | Promote a proposal to L4 |
| `POST` | `/api/memory/l4/proposals/{id}/reject` | Clear the pending flag, leave entry at L3 |
| `GET`  | `/api/memory/l2?limit=&offset=` | Browse L2 episodic entries with paging |
| `DELETE`| `/api/memory/l2/{id}` | Hard-delete an L2 entry |
| `GET`  | `/api/memory/l3?limit=&offset=` | Browse consolidated L3 summaries with paging |
| `GET`  | `/api/memory/l3/{id}/sources` | Resolve an L3 entry's sources — returns `source_texts` from the payload when present (post-absorption), falls back to `retrieve(source_ids)` for legacy L3s |
| `DELETE`| `/api/memory/l3/{id}` | Hard-delete an L3 entry |
| `POST` | `/api/memory/search` | Semantic search across selected tiers (`{q, tiers:[L2,L3,L4], limit}`) |
| `POST` | `/api/memory/admin/purge` | Hygiene delete: body `{tier: "L2"\|"L3"\|"all", source: "<tag>"}` or `{ids: [...]}`. Requires `source` or `ids` to avoid wildcard wipes. Returns `{deleted_ids, count}` |
| `GET`  | `/api/memory/runs?limit=` | History of `memory_review` consolidation runs |
| `POST` | `/api/memory/runs/trigger` | Run the consolidation pipeline now (delegates to autonomy engine) |
| `GET`  | `/api/agent/phase` | Current operational phase: `{phase: "learning"\|"operating", since: "<iso>"}` |
| `POST` | `/api/agent/phase` | Set the phase (body `{phase}`); refreshes active sessions' system prompts on return |
| `GET`  | `/api/system/llm-provider` | Active LLM provider: `{provider: "vllm"\|"anthropic"\|"openai", model, valid: [...], since}` |
| `POST` | `/api/system/llm-provider` | Switch provider (body `{provider}`); hot-swaps `app.state.provider`. `/v1/chat/completions` stays pinned to vLLM regardless |

### Face recognition (agent proxy)

The agent's `/api/face/*` is a thin pass-through to the
[face-recognition service](services/face-recognition/README.md) at
`FACE_REC_API_BASE` (default `http://face-recognition:6006`). Same
routes, same shapes, same status codes — only the path prefix changes
(agent `/api/face/*` ↔ face-recognition `/api/*`). The proxy exists so
the dashboard stays single-port and nginx config doesn't need to change;
it's also the only surface the SvelteKit `/people` UI calls.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/api/face/people` | List enrolled people with `image_count` |
| `POST` | `/api/face/people` | Create — `{name, access_level?, notes?}` |
| `GET`  | `/api/face/people/{id}` | Detail with the image gallery |
| `PATCH`| `/api/face/people/{id}` | Partial update of `access_level` and/or `notes` |
| `DELETE`| `/api/face/people/{id}` | Cascade delete (rows + files + Qdrant points) |
| `POST` | `/api/face/people/{id}/images` | Multipart upload — picks the highest-`det_score` face, persists |
| `POST` | `/api/face/people/{id}/enroll-from-camera` | Burst-capture from HA, picks the highest-quality face — `{camera, is_primary?}` |
| `DELETE`| `/api/face/people/{id}/images/{img_id}` | Delete a non-primary image |
| `POST` | `/api/face/people/{id}/images/{img_id}/set-primary` | Atomic primary swap |
| `GET`  | `/api/face/face_images/{id}/bytes` | Stream a face_image JPEG |
| `GET`  | `/api/face/detections` | Filters: `camera`, `person_id`, `since_seconds_ago`, `review_state`, `unknowns_only`, `limit` |
| `GET`  | `/api/face/detections/{id}/snapshot` | Stream a detection snapshot JPEG |
| `POST` | `/api/face/detections/{id}/confirm` | Body `{person_id}` or `{name}` (exactly one) — re-embeds + persists + marks `confirmed` |
| `POST` | `/api/face/detections/{id}/reject` | Marks `rejected` so it stops appearing in the unknowns queue |
| `POST` | `/api/face/detections/bulk-delete` | Body `{scope: "rejected"\|"all_unknowns"}` — mass-deletes unknown rows + snapshot files; returns `{rows_deleted, files_unlinked, scope}` |
| `POST` | `/api/face/admin/rescan-unknowns` | Kicks the rescan-unknowns admin job (re-match every unknown against the current index, contribute high-quality matches to the gallery). Returns `{job_id, status: "running"}` immediately; poll `/api/face/admin/jobs/{id}` |
| `GET`  | `/api/face/admin/jobs/{id}` | Poll a face-recognition admin job (rescan or rebuild). Status, phase, totals, errors |
| `GET`  | `/api/face/cameras` | Discovery (HA person-sensor → camera entity mapping) |
| `GET`  | `/api/face/health` | Upstream `/health` (model providers, db, qdrant, mqtt, retention) |

Other operator endpoints — `POST /api/admin/retention/sweep`, `POST
/api/admin/rebuild-embeddings`, `GET /api/admin/jobs?limit=` — are
intentionally **not** proxied (not part of the dashboard surface). Hit
them directly on port 6006. See
[Face Recognition Service](services/face-recognition/README.md#admin--operator)
for shapes.

### Cameras (zone mapping)

`/api/cameras/*` manages the camera-to-zone map the autonomy engine reads
when triaging camera/sensor events. Zones are free-text slugs
(`front_door`, `backyard`, `driveway`, `side_yard`, …) — the LLM reasons
about zones, not raw HA camera entity_ids, so the same notification
logic generalizes across deployments. Backed by the `camera_zones`
Postgres table; updates fan out via `LISTEN/NOTIFY` so the autonomy
engine's in-memory zone cache refreshes without a restart. Backs the
`/cameras` SvelteKit page. See
[autonomy/cameras.md](services/agent/autonomy/cameras.md) for the
end-to-end flow.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`    | `/api/cameras` | Returns `{cameras: [...], zones: [...]}`. `cameras[]` is the discovered HA camera list (proxied from face-rec's `/api/cameras`) left-joined with `camera_zones` — each row is `{camera_entity, sensor_entity, camera_exists, current_state, zone, zone_label, notes, updated_at}`. Orphan rows whose entity_id is no longer reported by face-rec are still included so they can be cleaned up. `zones[]` is the distinct list of in-use zone slugs for autocomplete |
| `PUT`    | `/api/cameras/{entity}/zone` | Upsert. Body: `{zone: string, zone_label?: string, notes?: string}`. `entity` is the HA camera entity_id (e.g. `camera.front_duo_3_clear`); the path uses FastAPI's `:path` converter so `.` is preserved |
| `DELETE` | `/api/cameras/{entity}/zone` | Clear the assignment. Returns `{camera_entity, deleted: bool}` |

### Push registration (companion-app)

`/api/push/register` is the agent's inbound registration surface for
[UnifiedPush](https://unifiedpush.org/) endpoints — used by the
[`havencore-companion-app`](https://github.com/ThatMattCat/havencore-companion-app)
to deliver autonomy briefings, anomaly alerts, and ad-hoc agent
messages to a phone without keeping a WebSocket open. Endpoints are
self-contained URLs produced by the user's distributor (typically the
ntfy Android app), and the agent stores them verbatim — there is no
agent-side ntfy server URL config. See the
[companion-app integration](integrations/companion-app.md) walkthrough
for the end-to-end flow and setup checklist.

LAN-only stack — these routes are unauthenticated like the rest of
`/api/*`. Backed by the `push_devices` Postgres table.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/push/register` | Upsert a device. Body: `{device_id: <uuid>, device_label: string, endpoint: string, platform?: "android"}`. `endpoint` must be `http(s)://...`. Always upserts on `device_id` (no 409 path in v1) — re-registering with a rotated endpoint replaces the prior row. Returns `{"ok": true}`. 400s on malformed `device_id` (must be UUID) or non-`http(s)` endpoint |
| `DELETE` | `/api/push/register/{device_id}` | Remove a registered device. Returns `{"ok": true}` on success or 404 `{"detail": "device not registered"}` if the row doesn't exist (companion app treats 404 as success) |
| `GET`  | `/api/push/register` | List all registered devices: `{devices: [{device_id, device_label, endpoint, platform, registered_at, last_seen_at}, ...]}`. Used for debugging — there is no dashboard UI for it in v1 |

When the autonomy engine fires with `_notify_channel = "ntfy"`, the
`NtfyFanoutNotifier` reads this table at send-time and POSTs the
[wire-format envelope](integrations/companion-app.md#wire-format) to
every registered endpoint. Devices added between dispatches are picked
up on the next fire without an engine restart.

### WebSockets

| URL | Direction | Purpose |
|-----|-----------|---------|
| `/ws/chat` | bidirectional | Streaming chat with `thinking`, `tool_call`, `tool_result`, `reasoning`, `metric`, `done`, `error`, and `summary_reset` events. `reasoning` frames (`{"type":"reasoning","content":"...","iteration":N}`) carry chain-of-thought from reasoning-capable models (e.g. GLM-4.5-Air via vLLM's `--reasoning-parser glm45`) — they arrive before the `done` of the iteration that produced them and are **dashboard-only on the wire**: `/api/chat` filters them out of its returned `events[]`, `/v1/chat/completions` naturally drops them, and they're not recorded as a separate event stream. The same CoT is, however, normalized onto the corresponding assistant message as `reasoning_content` and persisted with the rest of the conversation, since GLM-4.5-Air's chat template reads that field to render `<think>…</think>` for assistant messages within the in-progress agentic tool-call loop. `turn_metrics` does not record reasoning content. Session handshake: clients MAY send `{"type":"session","session_id":"...","idle_timeout":90,"device_name":"Kitchen Speaker"}` as the first frame to resume/bind a session, set a per-session idle window, and/or label the device; all fields optional. A later `{"type":"session", ...}` mid-stream may update `idle_timeout` or `device_name` on the active session (omitted fields are left untouched; `session_id` is honored only on the first frame). The server responds once with `{"type":"session","session_id":"..."}` before any turn events. `summary_reset` frames (`{"type":"summary_reset","reason":"idle_timeout_summarize","summary":"..."}`) may arrive in two flavors: inline at turn start when `run()` detects a pre-turn compaction, or pushed between turns when the background pool sweep compacts a session (per-session pub/sub on the pool) — both carry the same shape and the Chat pane renders them as a "Conversation summarized" marker. The `reason` field is `idle_timeout_summarize` when the session crossed its idle window, or `context_size_summarize` when its serialized message bytes crossed the provider-derived token budget (see `CONVERSATION_CONTEXT_LIMIT_FRACTION` / `CONVERSATION_CONTEXT_LIMIT_TOKENS` in [`configuration.md`](configuration.md#agent-runtime-tuning)). Idle and size are independent axes — a session pinned to `idle_timeout=-1` will still receive a `context_size_summarize` frame if it grows past the size threshold. |
| `/ws/logs` | server → client | Live tail of the agent's in-process log ring buffer |

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "error": {
    "message": "Missing required parameter 'input'",
    "type": "invalid_request_error",
    "param": "input",
    "code": "missing_parameter"
  }
}
```

#### 401 Unauthorized
```json
{
  "error": {
    "message": "Invalid API key provided",
    "type": "authentication_error"
  }
}
```

#### 500 Internal Server Error
```json
{
  "error": {
    "message": "Internal server error",
    "type": "internal_error"
  }
}
```

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `missing_parameter` | Required parameter not provided | Check request body |
| `invalid_parameter` | Parameter value is invalid | Verify parameter format |
| `authentication_error` | API key invalid or missing | Check Authorization header |
| `internal_error` | Server-side error | Check service logs |

## SDK Examples

### Python
```python
import requests

# Chat completion
response = requests.post(
    "http://localhost/v1/chat/completions",
    headers={
        "Authorization": "Bearer your_api_key",
        "Content-Type": "application/json"
    },
    json={
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)

print(response.json())
```

### JavaScript
```javascript
// Using fetch API
const response = await fetch('http://localhost/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_api_key',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: 'gpt-3.5-turbo',
    messages: [
      { role: 'user', content: 'Hello!' }
    ]
  })
});

const data = await response.json();
console.log(data);
```

### cURL
```bash
# Chat completion
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Text-to-speech
curl -X POST http://localhost/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "alloy"}' \
  --output audio.wav

# Speech-to-text
curl -X POST http://localhost/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-1"
```

## WebSocket APIs

The agent dashboard already uses WebSockets for real-time features:

- **`/ws/chat`** — streaming conversation with inline tool-call visibility, per-turn timing metrics, and thinking indicators. See the Agent Dashboard APIs section above and `services/agent/README.md` for the event schema.
- **`/ws/logs`** — one-way stream of structured log records from the agent's in-process ring buffer (the last 500 records are replayed on connect).

---

**Next Steps**:
- [Tool Development](services/agent/tools/development.md) - Creating custom tools
- [Integration Guides](integrations/home-assistant.md) - External service setup
- [Troubleshooting](troubleshooting.md) - API debugging and issues