# Agent Service

The core AI agent — Python + FastAPI + a built-in SvelteKit dashboard, all served from a single port (6002). Handles conversation, tool calling, conversation history, per-turn metrics, OpenAI-compatible endpoints for the voice pipeline, and proxies to sibling services for the dashboard playgrounds.

## Subtopics

- [Tools (MCP servers)](tools/README.md) — the agent's tool inventory lives here: HA, Plex, Music Assistant, general, Qdrant, MQTT, GitHub, plus a tool-development guide.
- [Conversation history](conversation-history.md) — how timed-out conversations get persisted to Postgres.
- [Autonomy engine](autonomy/README.md) — proactive background behaviors that wake on a schedule or in response to live MQTT/webhook events, run a tier-filtered autonomous turn, and notify via Signal, HA push, or speaker. Includes scheduled briefing + anomaly sweep, user-programmable reminders / watches / routines, the supervised act tier, and the [camera/sensor event pipeline](autonomy/cameras.md) that wires face-recognition (and future vehicle/motion/doorbell sources) through `watch_llm` for proactive notifications.
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
- `selene_agent/orchestrator.py` — event-based agent loop (THINKING / TOOL_CALL / TOOL_RESULT / REASONING / METRIC / SUMMARY_RESET / DONE / ERROR). `REASONING` is chain-of-thought surfaced from reasoning-capable models (e.g. GLM-4.5-Air via vLLM's `--reasoning-parser glm45`); the WS-frame form is dashboard-only (yielded on `/ws/chat`, filtered out of `/api/chat`'s `events[]`, naturally absent from `/v1/chat/completions`). The same CoT is also normalized onto the assistant message as `reasoning_content` and appended to `self.messages` — GLM-4.5-Air's `chat_template.jinja` reads that field and renders `<think>…</think>` only for assistant messages newer than the most recent user message (i.e. the in-progress agentic tool-call loop, where the model expects to see its own prior reasoning before the next call). For completed prior turns the template auto-emits empty `<think></think>`, so retaining the field across turns is harmless. with per-turn LLM and tool-call timing instrumentation. Each conversation session owns its own `AgentOrchestrator` instance (messages, `session_id`, `last_query_time`); singletons (OpenAI client, MCP manager, model, tools) are shared across all sessions. `SUMMARY_RESET` is yielded at turn start when `_check_session_timeout` compacted the session just before the incoming turn — the client renders an inline "Conversation summarized" marker above the next assistant reply.
- `selene_agent/utils/session_pool.py` — `SessionOrchestratorPool` keyed by `session_id`, with per-session `asyncio.Lock`, LRU cap (64), 30s background idle sweep, cold-resume from `conversation_db`, and shutdown flush. `/api/chat` and `/ws/chat` route through the pool; `/v1/chat/completions` builds an ephemeral orchestrator per request and never touches the pool. The autonomy engine also bypasses the pool (it already builds its own ephemeral orchestrators per task in `autonomy/turn.py`). The pool also exposes a per-session pub/sub surface (`subscribe(sid) → asyncio.Queue`, `unsubscribe`, `publish`) so background tasks can reach connected WS clients out-of-band — the idle sweep uses it to push `summary_reset` frames when a session is compacted between turns.
- `selene_agent/providers/` — pluggable LLM provider seam (`vllm`, `anthropic`, `openai` stub). Every agent call goes through `provider_getter() -> LLMProvider`, a closure over `app.state.provider` that resolves live on each turn so the dashboard System-page toggle takes effect without a session rebuild. `/v1/chat/completions` is pinned to vLLM regardless of the toggle. See [LLM provider toggle](#llm-provider-toggle) below.
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
| `/` | GET | SvelteKit dashboard SPA (Dashboard, Chat, Playgrounds, Metrics, Devices, People, Cameras, Memory, Autonomy, History, System) |
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
| `/api/memory/*` | GET/POST/PATCH/DELETE | Tiered memory (L2/L3/L4): stats, L2/L3/L4 browse, L4 CRUD, proposal approve/reject, L3 source drill-down, semantic search, run history + manual trigger, `admin/purge` hygiene endpoint. See [autonomy/memory/README.md](autonomy/memory/README.md). |
| `/api/face/*` | GET/POST/PATCH/DELETE | Pass-through proxy to the [face-recognition service](../face-recognition/README.md) (port 6006). Mirrors the upstream `/api/*` surface 1:1 — only the path prefix changes. Backs the SvelteKit `/people` routes (people grid + detail, detections timeline, unknowns review queue) and streams snapshot/face-image bytes so the dashboard never sees on-disk paths. Reads `FACE_REC_API_BASE`. See [API Reference → Face recognition (agent proxy)](../../api-reference.md#face-recognition-agent-proxy). |
| `/api/cameras` | GET | Discovered HA cameras (proxied from face-rec) left-joined with the autonomy `camera_zones` table. Backs the `/cameras` dashboard page where the operator maps each camera to a zone slug (`front_door`, `backyard`, `driveway`, …). Zones are what the autonomy LLM reasons about, not raw entity_ids. |
| `/api/cameras/{entity}/zone` | PUT/DELETE | Upsert or clear a zone assignment. Updates fan out via Postgres `LISTEN/NOTIFY` so the autonomy engine's in-memory zone cache refreshes without a restart. See [autonomy/cameras.md](autonomy/cameras.md). |
| `/api/agent/phase` | GET/POST | Read or set the agent's operational phase (`learning` \| `operating`). Writes to the `agent_state` Postgres table and refreshes active sessions' system prompts. |
| `/api/system/llm-provider` | GET/POST | Read or switch the active LLM provider (`vllm` \| `anthropic`; `openai` is stubbed). Writes to the `agent_state` table and hot-swaps `app.state.provider`. See [LLM provider toggle](#llm-provider-toggle). |
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
| Home Assistant | 20 — domain state / service calls, opinionated light & climate control, scenes, scripts, automations, notifications, areas, presence, timers, Jinja templates, history, calendar (read + create), media transport | [tools/home-assistant.md](tools/home-assistant.md) |
| Plex | 5 — `plex_search`, `plex_list_recent`, `plex_list_on_deck`, `plex_list_clients`, `plex_play` | [tools/plex.md](tools/plex.md) |
| Music Assistant | 7 — audio-only playback router for speakers, Chromecasts, and Google Homes (search / players / queue / play / announcement / transport). | [tools/music-assistant.md](tools/music-assistant.md) |
| General Tools | Up to 7 (credential-gated) — `generate_image`, `send_signal_message`, `query_multimodal_api`, `wolfram_alpha`, `get_weather_forecast`, `brave_search`, `search_wikipedia` | [tools/general.md](tools/general.md) |
| Qdrant | 3 — `create_memory`, `search_memories`, `delete_memory` | [tools/qdrant.md](tools/qdrant.md) |
| MQTT / Cameras | 1 (when MQTT is connected) — `get_camera_snapshots` | [tools/mqtt.md](tools/mqtt.md) |
| GitHub | 7 — repo code search / read / list / pull-latest + list/get/create GitHub Issues (untrusted issue text is wrapped in `<UNTRUSTED_USER_TEXT>` blocks) | [tools/github.md](tools/github.md) |

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

## LLM provider toggle

The agent's text-to-text LLM calls are routed through a pluggable provider
abstraction in `selene_agent/providers/`. STT/TTS always stay local.

| Provider | Default model | Notes |
|---|---|---|
| `vllm` | `gpt-3.5-turbo` (served name for the local GLM-4.5-Air-AWQ-FP16Mix) | Default. Direct `AsyncOpenAI` wrapper — zero translation overhead. Captures `reasoning` / `reasoning_content` extras from vLLM's reasoning-parser output, surfaces them as `REASONING` events (dashboard-only on the wire), and normalizes them onto the assistant message as `reasoning_content` so GLM-4.5-Air's chat template can render `<think>…</think>` on the next in-turn iteration. The legacy `reasoning` alias (vLLM glm45 parser's name) is dropped on append. Defensively pulls `<think>…</think>` blocks out of raw content too, folding them into the same `reasoning_content` field. |
| `anthropic` | `claude-opus-4-7` | Uses the official `AsyncAnthropic` SDK. Translates OpenAI-shaped messages ↔ Anthropic `tool_use`/`tool_result` blocks; strips `temperature`/`top_p` for Opus 4.7 (rejected by the API), forwards them for older Anthropic models. |
| `openai` | — | Stubbed; falls back to vLLM with a warning. |

**Switching**: Dashboard → **System** → **Agent LLM Provider**. The
change is live on the next turn; no restart, no session rebuild. The
selection is persisted to the `agent_state` table (`llm_provider` key)
and survives container restarts. Seeded on first boot from `LLM_PROVIDER`
env var; after that the DB wins.

**What is not affected**: `/v1/chat/completions` (OpenAI-compat external
surface for the voice pipeline) is pinned to vLLM regardless of the
toggle — voice satellites keep talking to the local model even when the
dashboard is pointed at Anthropic.

**Prompt caching (Anthropic only)**: when the Anthropic provider is
active, three ephemeral `cache_control` breakpoints are attached to
every request — the last tool entry (caches the tools array), the system
block (caches tools + system), and the last conversation message (caches
the accumulated history). 5-minute TTL, refreshed on each hit. Cache
hits and writes are logged at INFO, e.g.
`[anthropic] cache read=12963 create=79 input=5 output=76`.

The same counters are also plumbed through to `turn_metrics`: the
`LLMProvider` protocol exposes `pop_last_cache_stats() -> {"read", "create"}`
(stash-and-reset semantics — the Anthropic provider captures the usage
fields on each call, vLLM and the OpenAI stub return zeros). The
orchestrator sums these across a turn's LLM iterations and writes them
to the new `cache_read_tokens` / `cache_creation_tokens` columns, which
the Metrics dashboard renders as a "Cache hit rate (7d)" tile
(`read / (read + create)`).

## Performance tuning

- **Memory management**: conversation history cleanup
- **Tool optimization**: parallel tool execution
- **Response caching**: repeated query optimization
- **GPU utilization**: efficient model inference
