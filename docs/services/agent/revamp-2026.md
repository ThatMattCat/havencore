# Agent Service Revamp (April 2025)

This document details the comprehensive revamp of the Agent service across three phases: backend refactoring, new API surface, and a custom SvelteKit dashboard replacing the old Gradio UI.

## Motivation

The agent service had accumulated technical debt through multiple prior revamps:

- **Gradio limitation**: The built-in Gradio UI was a basic text-in/text-out interface with no visibility into tool calls, conversation history, or home state.
- **Async/sync bridge**: An `AsyncToolExecutor` class existed solely to bridge Gradio's synchronous calls to the async MCP tool execution, adding unnecessary threading complexity.
- **Dual-port architecture**: Gradio ran on port 6002 and a separate FastAPI app on port 6006, complicating networking and nginx config.
- **Dead code**: Commented-out LogScale handlers, unused imports, hardcoded config values, and stale log messages.
- **No conversation retrieval**: Conversations were stored in PostgreSQL but the code to list/browse them was never implemented.
- **No streaming**: Responses were collected in full before returning, with no real-time feedback during tool execution.

---

## Phase 1: Backend Refactoring

### 1.1 Removed Gradio, consolidated to async FastAPI

**What changed:**
- Deleted the `AsyncToolExecutor` class (persistent event loop in a background thread)
- Deleted the `SeleneRunner` and `SeleneAgent` classes
- Removed Gradio imports, interface setup, and the threading model
- Switched from `OpenAI` (sync) to `AsyncOpenAI` client
- Consolidated from two ports (6002 Gradio + 6006 FastAPI) to a single port 6002
- Replaced class-based architecture with FastAPI lifespan pattern and `app.state` for shared state

**Files modified:**
- `selene_agent/selene_agent.py` — Reduced from ~640 lines to ~380. Now a clean FastAPI app with lifespan, router registration, and OpenAI-compatible endpoints.
- `pyproject.toml` — Removed `gradio` dependency, added `websockets`
- `compose.yaml` — Updated health check to `/health`, removed port 6006 mapping

### 1.2 Cleaned up dead code and fixed config

**What changed:**
- Deleted ~87 lines of commented-out LogScale handler code from `logger.py`
- Removed duplicate imports and commented-out code blocks from `selene_agent.py`
- Made previously hardcoded values configurable via environment variables:
  - `QDRANT_HOST`, `QDRANT_PORT`, `EMBEDDINGS_URL`, `EMBEDDING_DIM`
  - New: `CONVERSATION_TIMEOUT` (default 180s), `TOOL_RESULT_MAX_CHARS` (default 8000)

**Files modified:**
- `selene_agent/utils/logger.py` — Reduced from ~207 to ~116 lines
- `selene_agent/utils/config.py` — Environment-driven config for all values

### 1.3 Centralized logging in MCP servers

**What changed:**
- All MCP server modules were using `logging.basicConfig()` / `logging.getLogger(__name__)`, so their logs never reached Grafana Loki
- Replaced with `from selene_agent.utils.logger import get_logger; logger = get_logger('loki')` in every module

**Files modified:**
- `modules/mcp_general_tools/mcp_server.py`
- `modules/mcp_homeassistant_tools/mcp_server.py`
- `modules/mcp_homeassistant_tools/ha_media_controller.py`
- `modules/mcp_mqtt_tools/mcp_server.py`
- `modules/mcp_qdrant_tools/qdrant_mcp_server.py`
- `selene_agent/utils/mcp_client_manager.py`
- `selene_agent/utils/conversation_db.py`

### 1.4 Extracted orchestrator from query()

**What changed:**
- The old `query()` method (~130 lines) mixed conversation management, LLM calls, tool extraction, tool execution, and error handling in one function
- Extracted into a dedicated `AgentOrchestrator` class with an event-based design

**New file: `selene_agent/orchestrator.py` (~294 lines)**

Key design:
```
EventType: THINKING | TOOL_CALL | TOOL_RESULT | DONE | ERROR

AgentOrchestrator.run(user_message) -> AsyncGenerator[AgentEvent]
  - Yields typed events as the agent loop progresses
  - Enables both streaming (SSE/WebSocket) and non-streaming consumption
  - Includes tool result truncation (max 8000 chars)
  - Session timeout detection (configurable via `CONVERSATION_TIMEOUT`; since extended with a per-session override and a summarize-and-continue reset — see [`conversation-history.md`](conversation-history.md))
  - Max 8 tool iterations per request to prevent runaway loops
```

Helper function `collect_response()` collects all events and returns the final text for non-streaming endpoints.

### 1.5 Conversation retrieval

**What changed:**
- Added `list_conversations(limit, offset)` method to `conversation_db.py`
- Returns paginated conversation list with session ID, timestamps, message counts, and metadata
- Previously stored conversations can now be browsed through the API

### 1.6 Streaming support

**What changed:**
- The orchestrator's event-based `run()` method feeds two streaming interfaces:
  - **SSE**: `POST /v1/chat/completions` with `stream: true` returns Server-Sent Events in OpenAI streaming format (`data: {"choices": [{"delta": {...}}]}`)
  - **WebSocket**: `WS /ws/chat` sends JSON events with tool visibility (see Phase 2)

### 1.7 MCP failure tracking

**What changed:**
- `MCPClientManager` now tracks `failed_servers: Dict[str, str]` during initialization
- Failed server names are injected as a system message so the LLM knows which capabilities are unavailable
- Server status endpoint includes failed server details with error messages

**Files modified:**
- `selene_agent/utils/mcp_client_manager.py` — Added `failed_servers` tracking, updated `get_server_status()`

---

## Phase 2: API Surface

New FastAPI routers under `selene_agent/api/` provide the data layer for the dashboard.

### 2.1 Chat API (`api/chat.py`)

| Endpoint | Description |
|----------|-------------|
| `POST /api/chat` | Non-streaming chat, returns `{response, events}` with full tool event log |
| `WS /ws/chat` | WebSocket streaming with real-time tool visibility |

WebSocket message format:
```json
{"type": "tool_call", "tool": "get_weather", "args": {...}}
{"type": "tool_result", "tool": "get_weather", "result": "..."}
{"type": "done", "content": "full response text"}
{"type": "error", "error": "error message"}
```

### 2.2 Conversations API (`api/conversations.py`)

| Endpoint | Description |
|----------|-------------|
| `GET /api/conversations?limit=20&offset=0` | Paginated conversation list |
| `GET /api/conversations/{session_id}` | Full conversation detail with all messages |

### 2.3 Status API (`api/status.py`)

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | System health: agent, MCP servers, database, vLLM backend |
| `GET /api/tools` | All registered tools grouped by MCP server with descriptions |
| `GET /api/mcp/status` | MCP connection details (migrated legacy endpoint) |

### 2.4 Home Assistant API (`api/homeassistant.py`)

| Endpoint | Description |
|----------|-------------|
| `GET /api/ha/entities?domain=light` | Entity states, optionally filtered by domain |
| `GET /api/ha/entities/summary` | Domain counts with active entity counts |
| `GET /api/ha/automations` | Automation list with last-triggered times |
| `GET /api/ha/scenes` | Scene list |

All HA endpoints proxy to the Home Assistant REST API using `HAOS_URL` and `HAOS_TOKEN` from config.

### Router registration

All routers are registered in `selene_agent.py`:
```python
app.include_router(chat_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(status_router, prefix="/api")
app.include_router(ha_router, prefix="/api")
app.include_router(chat_ws_router, prefix="/ws")
```

Shared state is accessed via `request.app.state.session_pool` and `request.app.state.mcp_manager`, set during the FastAPI lifespan. (At the time of the rewrite this was a singleton `request.app.state.orchestrator`; it was superseded by the per-session pool later — see [services/agent/README.md](../../../services/agent/README.md#session-pool).)

---

## Phase 3: SvelteKit Frontend

A custom dark-themed dashboard built with SvelteKit 5, replacing Gradio entirely.

### 3.1 Project setup

**Location:** `services/agent/frontend/`

- SvelteKit 5 with `@sveltejs/adapter-static` (SPA mode, `fallback: 'index.html'`)
- Vite dev proxy forwards `/api`, `/ws`, `/v1` to `localhost:6002` for development
- Production: built to static files, served by FastAPI's `StaticFiles` mount

**Key files:**
- `package.json` — Dependencies: svelte 5, sveltekit 2, vite 6, marked (markdown rendering)
- `svelte.config.js` — Static adapter with SPA fallback
- `vite.config.js` — Dev proxy configuration
- `src/lib/api.ts` — Typed fetch wrappers for all API endpoints
- `src/lib/stores/chat.ts` — WebSocket connection store with auto-reconnect, message/connection/processing state

### 3.2 Layout and navigation

**File:** `src/routes/+layout.svelte`

- Fixed sidebar with navigation: Dashboard, Chat, Devices, History, System
- Dark theme (`#0f1117` background, `#161822` surfaces, `#2d3148` borders)
- Purple accent (`#6366f1` / `#8b5cf6`) matching Selene branding
- SVG icons for each nav item
- Active route highlighting

### 3.3 Dashboard (`src/routes/+page.svelte`)

Card-grid layout showing:
- **System Status** — Agent, LLM, database, and MCP server health with green/red indicators
- **Quick Chat** — Link to the full chat page
- **Available Tools** — Total count and per-server breakdown
- **Recent Conversations** — Last 5 with timestamps and message counts
- **Home Summary** — Entity counts by domain with active counts (gracefully handles HA being unavailable)

### 3.4 Chat (`src/routes/chat/+page.svelte`)

Full chat interface with:
- WebSocket connection to `/ws/chat` with connection status indicator
- Streaming text display (response chunks append in real-time)
- **Tool call cards** inline in the conversation: collapsible cards showing tool name, arguments, and results
- Markdown rendering via `marked` (GFM, line breaks enabled)
- Animated thinking indicator (three pulsing dots) during processing
- User/assistant avatars with distinct styling
- **Push-to-talk mic** in the input bar — records from the browser's `MediaRecorder`, uploads the clip to `/api/stt/transcribe` on stop, and auto-sends the transcript as the user message. Disabled while connecting, processing a turn, or transcribing.
- **Auto-speak toggle** in the header — when enabled, each completed assistant turn is synthesized via `/api/tts/speak` (default voice `af_heart`, mp3) and played inline. A small green dot on the icon indicates active playback; a new send or toggling off interrupts playback. Preference persists in `localStorage` under `chat.autoSpeak`. On mount the page snapshots the current message count so navigating back to `/chat` never replays prior turns.
- Clear button to reset conversation

**Component:** `src/lib/components/ToolCallCard.svelte` — Expandable card with tool icon (wrench for calls, checkmark for results), tool name, and collapsible JSON body.

### 3.5 History (`src/routes/history/+page.svelte`)

Two-panel layout:
- **Left panel** — Paginated conversation list with timestamps and message counts
- **Right panel** — Full conversation detail showing all messages (user, assistant, tool) with role-colored labels
- System messages are filtered out
- Pagination controls (previous/next)

### 3.6 Devices (`src/routes/devices/+page.svelte`)

- **Domain grid** — Clickable cards for each HA domain with emoji icons and active/total counts
- **Entity detail** — Click a domain to see all entities with friendly names, state badges, entity IDs, and last-changed times
- **Automations list** — All automations with on/off status and last-triggered timestamps
- **Scenes list** — All available scenes

### 3.7 System (`src/routes/system/+page.svelte`)

Card-grid layout showing:
- **Agent** — Name and health status
- **LLM Backend** — Online/offline status and loaded model name
- **Database** — PostgreSQL connection status
- **MCP Servers** — Connected/failed servers with tool counts and error details
- **Tool listings** — One card per MCP server showing all registered tools with descriptions

### Build pipeline

**Dockerfile** (multi-stage):
```
Stage 1: node:18-alpine
  - npm ci && npm run build (SvelteKit static output)

Stage 2: pytorch/pytorch (runtime)
  - COPY --from=frontend-build /frontend/build /srv/agent-static
  - pip install agent package
```

The build output lands in `/srv/agent-static` — outside `/app` — so the development bind mount (`./services/agent/:/app`) doesn't shadow it.

**FastAPI static mount** (in `selene_agent.py`, after all route definitions):
```python
app.mount("/", StaticFiles(directory="/srv/agent-static", html=True), name="frontend")
```

**Nginx** (`nginx.conf`):
- Root `/` proxies to agent service (serves the SPA)
- `/api/` proxies to agent with 300s timeout
- `/ws/` proxies with WebSocket upgrade headers
- `/agent/` redirects to `/` (legacy path)
- All `/v1/*` and `/v2/*` API routes unchanged

---

## Port and networking changes

| Before | After |
|--------|-------|
| Port 6002: Gradio UI | Port 6002: FastAPI (all endpoints + SPA) |
| Port 6006: FastAPI OpenAI-compat API | Removed |
| Nginx `/`: JSON gateway message | Nginx `/`: SvelteKit dashboard |
| Nginx `/agent/`: Gradio proxy | Nginx `/agent/`: Redirect to `/` |

---

## Backward compatibility

The following endpoints are preserved for the voice pipeline and external integrations:

- `POST /v1/chat/completions` — OpenAI-compatible chat (now supports `stream: true`)
- `GET /v1/models` — Model listing
- `GET /health` — Health check
- `GET /mcp/status` — Legacy MCP status (also available at `/api/mcp/status`)

---

## Development workflow

**Frontend development** (hot reload):
```bash
cd services/agent/frontend
npm install
npm run dev
# Opens on localhost:5173, proxies API calls to localhost:6002
```

**Full stack** (Docker):
```bash
docker compose build agent
docker compose up -d agent
# Dashboard available at http://localhost/ via nginx
# Or directly at http://localhost:6002/
```

Note: The built SPA lives at `/srv/agent-static` inside the image, outside the `./services/agent/:/app` bind mount, so frontend changes require rebuilding the agent image (`docker compose build agent`). For a faster dev loop, use `npm run dev` with the Vite proxy; the runtime also falls back to `frontend/build/` on the host if `/srv/agent-static` is missing.

---

## Follow-up: Service Playgrounds + Metrics (2026-04)

A second pass added surfaces for the sibling services and lightweight observability. No new infrastructure services — everything rides on the existing FastAPI + Postgres stack.

### Per-turn metrics

- `orchestrator.py` instruments each turn: LLM call duration, per-tool-call duration, total duration, iteration count.
- A new `METRIC` event is yielded right before `DONE`, forwarded over `/ws/chat`, and persisted to a new `turn_metrics` Postgres table via `utils/metrics_db.py` (reuses the existing asyncpg pool).
- `GET /api/metrics/turns`, `/api/metrics/summary`, `/api/metrics/top-tools` drive the new **Metrics** dashboard page.
- The **Chat** page renders a compact badge row (`LLM 812ms · Tools 240ms · Total 1.09s · 2 iter`) under each assistant reply.

### Live log stream

- `utils/logger.py` gained an in-process ring-buffer handler (500 records, fan-out via `asyncio.Queue`).
- `api/logs.py` exposes `WS /ws/logs`: flushes the ring on connect, streams new records thereafter.
- `frontend/src/lib/components/LogStream.svelte` renders this on the **System** page with pause, level filter, and clear.

### Service playgrounds

New `/playgrounds/*` routes in the dashboard, backed by new agent API proxies. Everything is same-origin — no CORS, no nginx changes.

| Playground | Dashboard route | Agent endpoint(s) | Proxies to |
|-----------|-----------------|-------------------|------------|
| TTS | `/playgrounds/tts` | `POST /api/tts/speak`, `GET /api/tts/voices`, `GET /api/tts/health` | `text-to-speech:6005` |
| STT | `/playgrounds/stt` | `POST /api/stt/transcribe`, `GET /api/stt/health` | `speech-to-text:6001` |
| Vision | `/playgrounds/vision` | `POST /api/vision/ask`, `POST /api/vision/ask_url`, `GET /api/vision/health` | `vllm-vision:8000` (originally `iav-to-text:8100`; repointed when `vllm-vision` replaced it) |
| ComfyUI | `/playgrounds/comfy` | `POST /api/comfy/generate`, `GET /api/comfy/status/{id}`, `GET /api/comfy/view`, `GET /api/comfy/health` | `text-to-image:8188` |

Implementation notes:

- **STT playground**: uses the browser `MediaRecorder` API (WebM/Opus) and uploads the finished blob to `/api/stt/transcribe`. Whisper's file loader handles WebM natively. A WS streaming mode was prototyped but shelved because the STT service's legacy WebSocket expects raw int16 PCM — a transcode-in-browser step we don't need for a "record then transcribe" playground.
- **Vision playground**: encodes the image as a base64 data URL and forwards it as an OpenAI `image_url` chat-completion to the vision backend (originally `iav-to-text`; now `vllm-vision`, configured via `VISION_API_BASE` / `VISION_SERVED_NAME`). Avoided a shared-volume mount between the agent and vision containers. The `query_multimodal_api` MCP tool reuses the same proxy via a sibling `/api/vision/ask_url` JSON endpoint, keeping a single chokepoint for vision calls.
- **ComfyUI playground**: reuses `modules/mcp_general_tools/comfyui_tools.py:SimpleComfyUI` for workflow loading and queue polling — no duplicated queueing logic.
- **Dependencies**: added `python-multipart` to `pyproject.toml` (needed by FastAPI `File`/`Form` params for the new multipart endpoints).

### Speech-to-text cleanup

In parallel, `services/speech-to-text/app/main.py` dropped its `gradio_client` dependency and no longer round-trips final transcripts through the agent's (now non-existent) Gradio server. The port-6000 WebSocket now emits a direct `{"text": "...", "final": true, "trace_id": "..."}` message on completion. The voice pipeline (edge devices) was unaffected — ESP32s use the HTTP endpoint on port 6001, not the WebSocket.
