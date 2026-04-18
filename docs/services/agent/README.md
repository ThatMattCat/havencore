# Agent Service

The core AI agent — Python + FastAPI + a built-in SvelteKit dashboard, all served from a single port (6002). Handles conversation, tool calling, conversation history, per-turn metrics, OpenAI-compatible endpoints for the voice pipeline, and proxies to sibling services for the dashboard playgrounds.

## Subtopics

- [Tools (MCP servers)](tools/README.md) — the agent's tool inventory lives here: HA, Plex, general, Qdrant, MQTT, plus a tool-development guide.
- [Conversation history](conversation-history.md) — how timed-out conversations get persisted to Postgres.
- [Autonomy engine (v1)](autonomy/README.md) — proactive background behaviors (morning briefing, ambient anomaly sweep) that wake on a schedule, run a tier-filtered autonomous turn, and notify via Signal message or HA push.
- [Revamp 2026](revamp-2026.md) — architectural notes on the April 2026 rewrite (Gradio removal, async FastAPI, dashboard, streaming).

## Responsibilities

- Core AI conversation engine
- Tool calling and function execution
- Session and conversation management
- Integration orchestration with sibling services

## Architecture

```
Request → FastAPI → Selene Agent → Tool Registry → External APIs
    ↓         ↓           ↓             ↓              ↓
Response ← JSON ← Agent Logic ← Tool Results ← API Responses
```

## Key components

- `selene_agent/selene_agent.py` — FastAPI app with lifespan, serves the SvelteKit SPA + `/api/*` + `/ws/*` + OpenAI-compat endpoints; mounts routers from `selene_agent/api/`.
- `selene_agent/orchestrator.py` — event-based agent loop (THINKING / TOOL_CALL / TOOL_RESULT / METRIC / DONE / ERROR) with per-turn LLM and tool-call timing instrumentation. Each conversation session owns its own `AgentOrchestrator` instance (messages, `session_id`, `last_query_time`); singletons (OpenAI client, MCP manager, model, tools) are shared across all sessions.
- `selene_agent/utils/session_pool.py` — `SessionOrchestratorPool` keyed by `session_id`, with per-session `asyncio.Lock`, LRU cap (64), 30s background idle sweep, cold-resume from `conversation_db`, and shutdown flush. `/api/chat` and `/ws/chat` route through the pool; `/v1/chat/completions` builds an ephemeral orchestrator per request and never touches the pool. The autonomy engine also bypasses the pool (it already builds its own ephemeral orchestrators per task in `autonomy/turn.py`).
- `selene_agent/utils/mcp_client_manager.py` — MCP client that discovers and executes tools from MCP servers, with a `UnifiedTool` abstraction.
- `selene_agent/utils/conversation_db.py` — PostgreSQL conversation persistence. See [Conversation history](conversation-history.md).

### Session identity on the wire

| Surface | Client → server | Server → client |
|---|---|---|
| `POST /api/chat` | `X-Session-Id` request header (optional); `X-Idle-Timeout: <seconds>` (optional, per-session); `X-Device-Name: <label>` (optional, satellite/client label) | `X-Session-Id` response header |
| `WS /ws/chat` | First frame `{"type":"session","session_id":"...","idle_timeout":N,"device_name":"..."}` (all fields optional; `idle_timeout` and `device_name` may also be sent on later session frames mid-stream) | First frame `{"type":"session","session_id":"..."}` |
| `POST /v1/chat/completions` | — (ignored; stateless, no device-name attribution) | — |

Missing/unknown `session_id` → the pool mints a new UUID and returns it. Known `session_id` that isn't in memory → the pool cold-resumes from `conversation_db` and rehydrates the orchestrator (calls `prepare()` to prepend the L4 block without clobbering restored messages). `device_name` rides with every flushed history row and is denormalized onto every `turn_metrics` row so the dashboard can render human-readable labels instead of opaque session ids — see [Conversation history → Device attribution](conversation-history.md#device-attribution).

## API endpoints

All endpoints live on a single port (6002). The SvelteKit dashboard is built into the image and served as static files by FastAPI — there is no Gradio.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | SvelteKit dashboard SPA (Dashboard, Chat, Devices, Memory, History, Playgrounds, Metrics, System) |
| `/api/chat` | POST | Send a message, get full response + tool event log |
| `/api/status` | GET | Health: agent, MCP servers, DB, vLLM |
| `/api/tools` | GET | Registered tools grouped by MCP server |
| `/api/conversations` | GET | Paginated conversation history |
| `/api/conversations/{session_id}/resume` | POST | Hydrate a stored session into the live pool so `/chat` can continue it |
| `/api/ha/*` | GET | Home Assistant entities, automations, scenes |
| `/api/metrics/*` | GET | Per-turn timings, daily aggregates, top tools |
| `/api/tts/*` | POST/GET | Proxies to text-to-speech |
| `/api/stt/*` | POST/GET | Proxies to speech-to-text |
| `/api/vision/*` | POST/GET | Proxies to iav-to-text |
| `/api/comfy/*` | POST/GET | Proxies to text-to-image (ComfyUI) |
| `/api/autonomy/status` | GET | Autonomy engine state (running/paused, last dispatch, next-due) |
| `/api/autonomy/pause` | POST | Runtime kill switch — stop dispatch without restart |
| `/api/autonomy/resume` | POST | Clear runtime kill switch |
| `/api/autonomy/items` | GET | List agenda items (scheduled autonomous behaviors) |
| `/api/autonomy/runs` | GET | Recent run history; `include_messages=1` for full traces |
| `/api/autonomy/trigger/{id}` | POST | Fire an agenda item immediately, bypassing schedule + rate limit |
| `/api/memory/*` | GET/POST/PATCH/DELETE | Tiered memory (L2/L3/L4): stats, L4 CRUD, proposal approve/reject, L3 browse + source drill-down, semantic search, run history + manual trigger. See [autonomy/memory/README.md](autonomy/memory/README.md). |
| `/ws/chat` | WS | Streaming chat with tool visibility + metric events |
| `/ws/logs` | WS | Live server log tail |
| `/v1/chat/completions` | POST | OpenAI-compatible chat — **stateless**: each request builds an ephemeral orchestrator; no pool, no history persistence, no metrics. The caller owns its own history. |
| `/v1/models` | GET | Lists the agent as an available model |
| `/health` | GET | Service health check |
| `/mcp/status` | GET | MCP connection status |

See [API Reference](../../api-reference.md) for full request/response schemas.

## Configuration

Agent-specific environment variables (full list in [Configuration](../../configuration.md)):

```bash
AGENT_NAME="Selene"
LLM_API_KEY="your_api_key"
CURRENT_LOCATION="San Francisco, CA"
CURRENT_TIMEZONE="America/Los_Angeles"
```

## Available tools

Tools are grouped into MCP servers. Each server has its own reference doc under [tools/](tools/README.md):

| MCP server | Tools | Doc |
|------------|-------|-----|
| Home Assistant | 19 — domain state / service calls, opinionated light & climate control, scenes, scripts, automations, notifications, areas, presence, timers, Jinja templates, history, calendar, media transport | [tools/home-assistant.md](tools/home-assistant.md) |
| Plex | 5 — `plex_search`, `plex_list_recent`, `plex_list_on_deck`, `plex_list_clients`, `plex_play` | [tools/plex.md](tools/plex.md) |
| Music Assistant | Audio-only playback router for speakers, Chromecasts, and Google Homes (queue / play / pause / transport). | [tools/music-assistant.md](tools/music-assistant.md) |
| General Tools | Up to 7 (credential-gated) — `generate_image`, `send_signal_message`, `query_multimodal_api`, `wolfram_alpha`, `get_weather_forecast`, `brave_search`, `search_wikipedia` | [tools/general.md](tools/general.md) |
| Qdrant | 2 — `create_memory`, `search_memories` | [tools/qdrant.md](tools/qdrant.md) |
| MQTT / Cameras | 1 (when MQTT is connected) — `get_camera_snapshots` | [tools/mqtt.md](tools/mqtt.md) |

See [Media Control](../../integrations/media-control.md) for the split between Plex and Home Assistant on TV playback, required TV setup, and the optional wake/launch mapping.

## Development and debugging

```bash
# View agent logs
docker compose logs -f agent

# Access Python console
docker compose exec agent python

# Test tool registry
curl http://localhost:6002/api/tools

# Check MCP status
curl http://localhost:6002/mcp/status
```

## Performance tuning

- **Memory management**: conversation history cleanup
- **Tool optimization**: parallel tool execution
- **Response caching**: repeated query optimization
- **GPU utilization**: efficient model inference
