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
  - Session timeout detection (configurable via CONVERSATION_TIMEOUT)
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

Shared state is accessed via `request.app.state.orchestrator` and `request.app.state.mcp_manager`, set during the FastAPI lifespan.

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
  - COPY --from=frontend-build /frontend/build /app/static
  - pip install agent package
```

**FastAPI static mount** (in `selene_agent.py`, after all route definitions):
```python
app.mount("/", StaticFiles(directory="/app/static", html=True), name="frontend")
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

Note: The Docker volume mount (`./services/agent/:/app`) overrides `/app/static` in development. Use `npm run dev` with the Vite proxy for frontend development, and the Docker build for production deployment.
