# Selene Agent

The core AI agent service for HavenCore. Selene receives natural language input, orchestrates tool calls via MCP servers, queries the LLM (vLLM), and returns responses. She serves as the central intelligence layer between users and all connected services (Home Assistant, web search, memory, MQTT devices, etc.).

## Architecture

```
                         Port 6002
                            |
                   ┌────────┴────────┐
                   │   FastAPI App   │
                   │   (uvicorn)     │
                   ├─────────────────┤
                   │  SvelteKit SPA  │  ← Static files (/, /chat, /devices, etc.)
                   │  REST API       │  ← /api/*
                   │  WebSocket      │  ← /ws/chat
                   │  OpenAI-compat  │  ← /v1/chat/completions
                   └────────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌────┴────┐ ┌──────┴──────┐
        │ Orchestrator│ │  MCP   │ │ Conversation│
        │ (agent loop)│ │Manager │ │     DB      │
        └─────┬─────┘ └────┬────┘ │ (PostgreSQL)│
              │             │      └─────────────┘
              │    ┌────────┼────────┬──────────┐
              │    │        │        │          │
              ▼    ▼        ▼        ▼          ▼
           vLLM  general  home     qdrant     mqtt
          (8000) _tools  assistant _tools    _tools
                          _tools
```

Everything runs on a single port (6002). There is no Gradio — the UI is a custom SvelteKit dashboard built into the Docker image and served as static files by FastAPI.

## Endpoints

### Browser-Accessible (SvelteKit Dashboard)

These all serve the SPA and are meant to be opened in a browser:

| URL | Page | Description |
|-----|------|-------------|
| `http://HOST:6002/` | Dashboard | System status, recent conversations, tool summary, HA device overview |
| `http://HOST:6002/chat` | Chat | Real-time chat with Selene via WebSocket. Shows tool calls inline, markdown rendering, thinking indicators, per-turn timing badges |
| `http://HOST:6002/devices` | Devices | Home Assistant entities grouped by domain, automations, scenes |
| `http://HOST:6002/history` | History | Browse stored conversations with full message detail |
| `http://HOST:6002/playgrounds` | Playgrounds | Index of per-service playgrounds (TTS, STT, Vision, ComfyUI) with health badges |
| `http://HOST:6002/playgrounds/tts` | TTS Playground | Synthesize speech from text with voice + format selection; plays the result inline |
| `http://HOST:6002/playgrounds/stt` | STT Playground | Transcribe an uploaded audio file or a clip recorded from the browser microphone |
| `http://HOST:6002/playgrounds/vision` | Vision Playground | Send an image + prompt to the vision LLM (iav-to-text) and render the response |
| `http://HOST:6002/playgrounds/comfy` | ComfyUI Playground | Queue an image-generation prompt and view the rendered output |
| `http://HOST:6002/metrics` | Metrics | Per-turn LLM/tool/total latencies, daily activity, top tools, p95 stats |
| `http://HOST:6002/system` | System | MCP server status, loaded LLM model, DB connection, per-server tool listings, live log stream |

### REST API (`/api/*`)

JSON endpoints consumed by the dashboard frontend. Can also be called directly.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message, get full response + tool event log. Body: `{"message": "..."}`. Returns: `{"response": "...", "events": [...]}` |
| `GET` | `/api/status` | System health — agent, MCP servers (connected/failed), database, vLLM model info |
| `GET` | `/api/tools` | All registered tools grouped by MCP server, with descriptions and parameter schemas |
| `GET` | `/api/conversations?limit=20&offset=0` | Paginated list of stored conversations |
| `GET` | `/api/conversations/{session_id}` | Full message history for a specific conversation |
| `GET` | `/api/mcp/status` | MCP connection details (configured, connected, failed servers) |
| `GET` | `/api/ha/entities?domain=light` | Home Assistant entity states, optionally filtered by domain |
| `GET` | `/api/ha/entities/summary` | Entity counts per domain with active counts |
| `GET` | `/api/ha/automations` | HA automations with state and last-triggered time |
| `GET` | `/api/ha/scenes` | HA scenes |
| `GET` | `/api/metrics/turns?limit=50` | Recent per-turn timings from the `turn_metrics` table |
| `GET` | `/api/metrics/summary?days=7` | Aggregates: turns/day, avg llm/total ms, p95 total ms |
| `GET` | `/api/metrics/top-tools?days=7&limit=10` | Tool call counts and average latency |
| `POST` | `/api/tts/speak` | Proxy to text-to-speech. Body: `{"text", "voice?", "format?", "speed?"}`. Streams binary audio back |
| `GET` | `/api/tts/voices` | Static voice list (OpenAI aliases all mapped to `af_heart`) |
| `GET` | `/api/tts/health` | TTS service health proxy |
| `POST` | `/api/stt/transcribe` | Multipart proxy to `/v1/audio/transcriptions`. Fields: `file`, `language?`, `response_format?` |
| `GET` | `/api/stt/health` | STT service health proxy |
| `POST` | `/api/vision/ask` | Multipart: `image` + `prompt`. Encodes image as data URL and forwards to iav-to-text. Returns `{response, latency_ms}` |
| `GET` | `/api/vision/health` | Vision service health proxy |
| `POST` | `/api/comfy/generate` | Body: `{prompt, negative_prompt?, seed?, steps?}`. Queues workflow, returns `{prompt_id}` |
| `GET` | `/api/comfy/status/{prompt_id}` | Returns `{status: "pending"\|"done", images: [...]}`  |
| `GET` | `/api/comfy/view?filename=...&subfolder=...` | Streams a generated image from ComfyUI |
| `GET` | `/api/comfy/health` | ComfyUI service health proxy |

### WebSocket Endpoints

| URL | Direction | Description |
|-----|-----------|-------------|
| `/ws/chat` | bidirectional | Streaming chat with real-time tool visibility |
| `/ws/logs` | server → client | Live tail of the in-process log ring buffer (500 records) |

#### `/ws/chat`

Used by the chat page.

```
Connect:  ws://HOST:6002/ws/chat

Send:     {"message": "What's the weather like?"}

Receive (in order):
  {"type": "thinking", "iteration": 1}
  {"type": "tool_call", "tool": "get_weather_forecast", "args": {"location": "..."}}
  {"type": "tool_result", "tool": "get_weather_forecast", "result": "..."}
  {"type": "thinking", "iteration": 2}
  {"type": "metric", "payload": {"llm_ms": 812, "tool_ms_total": 240, "tool_calls": [...], "total_ms": 1089, "iterations": 2}}
  {"type": "done", "content": "It's 72F and sunny..."}

Error:
  {"type": "error", "error": "description of what went wrong"}
```

The connection stays open for multiple messages. Each message triggers a full agent loop (LLM call, optional tool calls, final response). The `metric` event is emitted once per turn, just before `done`, and is persisted to the `turn_metrics` table.

#### `/ws/logs`

On connect the server flushes the current ring buffer, then streams new records as they arrive. Each message is JSON: `{"type": "log", "level", "message", "timestamp", "trace_id"}`. Periodic `{"type": "ping"}` frames keep the connection alive.

### OpenAI-Compatible API (`/v1/*`)

These endpoints maintain backward compatibility with the voice pipeline and any external integrations that speak the OpenAI protocol.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions. Supports `stream: true` for SSE streaming. Only the last user message is processed (agent manages its own conversation history). |
| `GET` | `/v1/models` | Lists the agent as an available model |

**Non-streaming example:**
```bash
curl -X POST http://HOST:6002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Turn on the living room lights"}]}'
```

**Streaming example:**
```bash
curl -X POST http://HOST:6002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

SSE streaming follows the OpenAI format: `data: {"choices": [{"delta": {"content": "..."}}]}`

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{"status": "healthy", "agent": "selene"}` |
| `GET` | `/mcp/status` | Legacy MCP status endpoint (same as `/api/mcp/status`) |

## Agent Orchestrator

The core agent loop lives in `selene_agent/orchestrator.py`. It:

1. Receives a user message
2. Checks for session timeout (configurable, default 180s) — if expired, stores the conversation to PostgreSQL and resets
3. Appends the user message with system context (timestamp, location)
4. Calls the LLM with the conversation history and available tools
5. If the LLM requests tool calls, executes them via MCP and loops back to step 4
6. Returns the final text response

Each step yields typed events (`THINKING`, `TOOL_CALL`, `TOOL_RESULT`, `METRIC`, `DONE`, `ERROR`) that enable streaming and tool visibility in the UI. The `METRIC` event carries per-turn timings (LLM latency, per-tool latencies, total, iteration count) and is both forwarded over the chat WebSocket and persisted to the `turn_metrics` Postgres table for the Metrics page.

**Safety limits:**
- Max 8 tool iterations per request (prevents runaway loops)
- Tool results truncated to 8000 chars (configurable via `TOOL_RESULT_MAX_CHARS`)
- Session timeout stores conversation and resets context (configurable via `CONVERSATION_TIMEOUT`)

## MCP Tool Servers

Tools are provided by MCP (Model Context Protocol) servers, each running as a subprocess managed by `MCPClientManager`. The LLM sees all tools as OpenAI function-calling format and decides which to invoke.

### `general_tools`
| Tool | Description |
|------|-------------|
| `generate_image` | Generate images via ComfyUI |
| `send_signal_message` | Send Signal message (text + images/video) via signal-cli-rest-api |
| `query_multimodal_api` | Send images/audio to the vision LLM for analysis |
| `wolfram_alpha` | Query Wolfram Alpha for math, science, facts |
| `get_weather_forecast` | Weather data from WeatherAPI |
| `brave_search` | Web search via Brave Search API |
| `search_wikipedia` | Search and retrieve Wikipedia articles |

### `homeassistant`
| Tool | Description |
|------|-------------|
| `ha_get_domain_entity_states` | List entities and their states for a given HA domain |
| `ha_get_domain_services` | List available services for an HA domain |
| `ha_execute_service` | Execute any HA service (turn on lights, lock doors, etc.) |
| `ha_control_media_player` | Play, pause, skip, volume, etc. on media players |
| `ha_stream_media` | Stream media URLs to HA media players |
| `ha_find_media_items` | Search for media content |

### `mcp_server_qdrant` (Semantic Memory)
| Tool | Description |
|------|-------------|
| `create_memory` | Store a memory with embeddings in Qdrant |
| `search_memories` | Semantic search across stored memories |

### `mqtt`
| Tool | Description |
|------|-------------|
| `get_camera_snapshots` | Capture snapshots from MQTT-connected cameras |

### `mcp_server_fetch` (built-in)
| Tool | Description |
|------|-------------|
| `fetch` | Fetch a URL and extract content as markdown |

## Configuration

All configuration is via environment variables (loaded in `selene_agent/utils/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | — | vLLM endpoint URL (e.g. `http://vllm:8000/v1`) |
| `LLM_API_KEY` | — | API key for the LLM backend |
| `AGENT_NAME` | `""` | Name of the assistant persona |
| `MCP_SERVERS` | `{}` | JSON array of MCP server configs |
| `CONVERSATION_TIMEOUT` | `180` | Seconds of inactivity before conversation resets |
| `TOOL_RESULT_MAX_CHARS` | `8000` | Max characters per tool result before truncation |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | — | PostgreSQL connection for conversation storage |
| `QDRANT_HOST` | `qdrant` | Qdrant vector DB host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDINGS_URL` | `http://embeddings:3000` | Text embeddings service URL |
| `HAOS_URL` | — | Home Assistant URL |
| `HAOS_TOKEN` | — | Home Assistant long-lived access token |
| `BRAVE_SEARCH_API_KEY` | — | Brave Search API key |
| `WOLFRAM_ALPHA_API_KEY` | — | Wolfram Alpha API key |
| `WEATHER_API_KEY` | — | WeatherAPI key |
| `CURRENT_LOCATION` | `New York, NY` | Location context for weather/local queries |
| `CURRENT_TIMEZONE` | — | Timezone for timestamp formatting |
| `LOKI_URL` | — | Grafana Loki push URL for centralized logging |
| `DEBUG_LOGGING` | `0` | Set to `1` for debug-level logs |

## Project Structure

```
services/agent/
├── Dockerfile               # Multi-stage: Node frontend build + Python runtime
├── pyproject.toml            # Python package definition, entry point: agent
├── frontend/                 # SvelteKit dashboard (built to static files)
│   ├── package.json
│   ├── svelte.config.js      # Static adapter with SPA fallback
│   ├── vite.config.js        # Dev proxy to FastAPI on :6002
│   └── src/
│       ├── lib/
│       │   ├── api.ts        # Typed API client (chat, HA, metrics, TTS, STT, vision, comfy)
│       │   ├── stores/
│       │   │   └── chat.ts   # WebSocket chat store (handles metric events)
│       │   └── components/   # Card, StatusBadge, ToolCallCard, LogStream
│       └── routes/
│           ├── +layout.svelte
│           ├── +page.svelte             # Dashboard
│           ├── chat/+page.svelte        # Chat (with timing badges)
│           ├── devices/+page.svelte     # HA Devices
│           ├── history/+page.svelte     # Conversation History
│           ├── metrics/+page.svelte     # Per-turn metrics + top tools
│           ├── playgrounds/+page.svelte # Playground index
│           ├── playgrounds/tts/+page.svelte
│           ├── playgrounds/stt/+page.svelte
│           ├── playgrounds/vision/+page.svelte
│           ├── playgrounds/comfy/+page.svelte
│           └── system/+page.svelte      # System Status + live log stream
└── selene_agent/
    ├── selene_agent.py       # FastAPI app, lifespan, OpenAI-compat endpoints, static mount
    ├── orchestrator.py       # Agent loop with event-based streaming + per-turn metrics
    ├── api/
    │   ├── chat.py           # POST /api/chat, WS /ws/chat (persists metrics)
    │   ├── conversations.py  # GET /api/conversations
    │   ├── status.py         # GET /api/status, /api/tools
    │   ├── homeassistant.py  # GET /api/ha/*
    │   ├── metrics.py        # GET /api/metrics/{turns,summary,top-tools}
    │   ├── logs.py           # WS /ws/logs
    │   ├── tts.py            # POST /api/tts/speak, voices/health proxies
    │   ├── stt.py            # POST /api/stt/transcribe + health proxy
    │   ├── vision.py         # POST /api/vision/ask + health proxy
    │   └── comfy.py          # POST /api/comfy/generate, status/view/health
    ├── utils/
    │   ├── config.py         # Environment-driven configuration
    │   ├── logger.py         # Loki + ring-buffer log handlers
    │   ├── conversation_db.py # PostgreSQL conversation storage/retrieval
    │   ├── metrics_db.py     # PostgreSQL turn_metrics read/write (shared pool)
    │   └── mcp_client_manager.py # MCP server lifecycle, tool discovery
    └── modules/              # MCP tool servers (each runs as a subprocess)
        ├── mcp_general_tools/
        ├── mcp_homeassistant_tools/
        ├── mcp_mqtt_tools/
        └── mcp_qdrant_tools/
```

## Development

**Frontend hot-reload (recommended for UI work):**
```bash
cd services/agent/frontend
npm install
npm run dev
# Opens on localhost:5173, proxies API calls to localhost:6002
```

**Restart agent after Python changes:**
```bash
# Code is volume-mounted, so just restart the container
docker compose restart agent
```

**Build and deploy:**
```bash
docker compose build agent
docker compose up -d agent
```

The Dockerfile uses a multi-stage build: stage 1 builds the SvelteKit frontend with Node 18, stage 2 installs the Python package and copies the built static files into `/srv/agent-static` (outside `/app` so the Docker volume mount doesn't shadow them). At runtime, FastAPI serves from `/srv/agent-static` if present, otherwise falls back to `frontend/build/` for local dev (run `npm run build` in `frontend/` to refresh).

## Port Summary

| Port | Protocol | Served By | Purpose |
|------|----------|-----------|---------|
| 6002 | HTTP | FastAPI/uvicorn | All endpoints: SPA, REST API, OpenAI-compat, health |
| 6002 | WebSocket | FastAPI/uvicorn | `/ws/chat` streaming chat |

Via nginx (port 80), all paths are proxied to the agent on 6002:
- `/` → SvelteKit dashboard
- `/api/*` → REST API (300s timeout)
- `/ws/*` → WebSocket (upgrade headers)
- `/v1/*` → OpenAI-compatible endpoints (300s timeout)
