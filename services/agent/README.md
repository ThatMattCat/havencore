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
            ┌───────────────┼───────────────┐
            │               │               │
     ┌──────┴───────┐ ┌─────┴─────┐ ┌──────┴──────┐
     │Session Pool  │ │   MCP     │ │ Conversation│
     │(per-session  │ │ Manager   │ │     DB      │
     │orchestrators)│ │           │ │ (PostgreSQL)│
     └──────┬───────┘ └─────┬─────┘ └─────────────┘
            │               │
            │   ┌───────────┼──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
            │   │           │          │          │          │          │          │          │          │          │          │
            ▼   ▼           ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼          ▼
         vLLM general    home        face      vision    device      plex     music      qdrant   reminder    mqtt      github
        (8000) _tools  assistant    _tools     _tools   _action     _tools  assistant    _tools    _tools    _tools     _tools
                       _tools                            _tools              _tools
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
| `http://HOST:6002/playgrounds/vision` | Vision Playground | Send an image + prompt to the vision LLM (vllm-vision) and render the response |
| `http://HOST:6002/playgrounds/comfy` | ComfyUI Playground | Queue an image-generation prompt and view the rendered output |
| `http://HOST:6002/metrics` | Metrics | Per-turn LLM/tool/total latencies, daily activity, top tools, p95 stats |
| `http://HOST:6002/system` | System | MCP server status, loaded LLM model, DB connection, per-server tool listings, live log stream |

### REST API (`/api/*`)

JSON endpoints consumed by the dashboard frontend. Can also be called directly.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message, get full response + tool event log. Body: `{"message": "..."}`. Optional `X-Session-Id` header binds to an existing session; response echoes the active `X-Session-Id`. Returns: `{"response": "...", "session_id": "...", "events": [...]}` |
| `GET` | `/api/status` | System health — agent (incl. `sessions: {active_sessions, max_size, sweep_running}`), MCP servers (connected/failed), database, vLLM model info |
| `GET` | `/api/tools` | All registered tools grouped by MCP server, with descriptions and parameter schemas |
| `GET` | `/api/conversations?limit=20&offset=0` | Paginated list of stored conversations |
| `GET` | `/api/conversations/{session_id}` | Full message history for a specific conversation |
| `POST` | `/api/conversations/{session_id}/resume` | Hydrate a stored session into the live pool so `/chat` can continue it. Returns `{session_id, resumed, message_count}` |
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
| `POST` | `/api/vision/ask` | Multipart: `image` + `prompt`. Encodes image as data URL and forwards to vllm-vision. Returns `{response, latency_ms}` |
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

Optional first frame (client → server):
  {"type": "session", "session_id": "existing-uuid-or-device-id"}

Server always announces the active session before the first turn:
  {"type": "session", "session_id": "..."}

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

The connection stays open for multiple messages. Each message triggers a full agent loop (LLM call, optional tool calls, final response), serialized per-session via the pool's per-session `asyncio.Lock` so in-flight turns never overlap on the same `session_id`. The `metric` event is emitted once per turn, just before `done`, and is persisted to the `turn_metrics` table.

#### `/ws/logs`

On connect the server flushes the current ring buffer, then streams new records as they arrive. Each message is JSON: `{"type": "log", "level", "message", "timestamp", "trace_id"}`. Periodic `{"type": "ping"}` frames keep the connection alive.

### OpenAI-Compatible API (`/v1/*`)

These endpoints maintain backward compatibility with the voice pipeline and any external integrations that speak the OpenAI protocol.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions. Supports `stream: true` for SSE streaming. **Stateless** — each request builds an ephemeral orchestrator, runs the agent loop, and discards it. No pool, no history persistence, no `turn_metrics` writes. The caller owns its full message history in the request body. For pool-backed, history-tracked chat, use `/api/chat` or `/ws/chat` instead. |
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
2. Appends the user message with system context (timestamp, location)
3. Calls the LLM with the conversation history and available tools
4. If the LLM requests tool calls, executes them via MCP and loops back to step 3
5. Returns the final text response

Each step yields typed events (`THINKING`, `TOOL_CALL`, `TOOL_RESULT`, `METRIC`, `DONE`, `ERROR`) that enable streaming and tool visibility in the UI. The `METRIC` event carries per-turn timings (LLM latency, per-tool latencies, total, iteration count) and is both forwarded over the chat WebSocket and persisted to the `turn_metrics` Postgres table for the Metrics page.

### Session pool

Each conversation owns its own `AgentOrchestrator` (messages, `session_id`, `last_query_time`); the expensive singletons (OpenAI client, MCP manager, model, tool list) are shared. The `SessionOrchestratorPool` in `selene_agent/utils/session_pool.py` manages the lifecycle:

- **Per-session `asyncio.Lock`** — turns on the same `session_id` serialize, turns across sessions run concurrently
- **LRU cap (default 64)** — when the pool fills, the least-recently-used session is flushed to Postgres and evicted
- **30-second idle sweep (summarize-and-continue)** — background task persists sessions past their effective idle window (per-session override, else `CONVERSATION_TIMEOUT`, default 90s), runs a one-shot LLM summary, and resets `messages` to `[system + L4, summary, last 2 exchanges]` with the same `session_id`. On summary LLM timeout/failure, falls back to keep-tail-only. Busy sessions are skipped, not blocked
- **Shutdown flush** — on restart/stop/SIGTERM, every non-empty session is persisted before exit
- **Cold resume** — an unknown `session_id` that exists in `conversation_histories` is rehydrated; `prepare()` re-prepends the L4 block without clobbering the restored messages

`/api/chat` and `/ws/chat` route through the pool. `/v1/chat/completions` bypasses it entirely (stateless, ephemeral orchestrator per request). The autonomy engine also bypasses the pool — it builds its own ephemeral orchestrators per task in `autonomy/turn.py`.

**Safety limits:**
- Max 8 tool iterations per request (prevents runaway loops)
- Tool results truncated to 8000 chars (configurable via `TOOL_RESULT_MAX_CHARS`)
- Session timeout persists the full conversation and resets to `[system, summary, last N exchanges]` (configurable via `CONVERSATION_TIMEOUT`; per-session override via `X-Idle-Timeout` header or WS `idle_timeout` field)

## MCP Tool Servers

Tools are provided by MCP (Model Context Protocol) servers, each running as a subprocess managed by `MCPClientManager`. The LLM sees all tools as OpenAI function-calling format and decides which to invoke.

### `general_tools`
| Tool | Description |
|------|-------------|
| `generate_image` | Generate images via ComfyUI |
| `send_signal_message` | Send Signal message (text + images/video) via signal-cli-rest-api |
| `query_multimodal_api` | Send images/audio to the vision LLM (vllm-vision) for analysis |
| `wolfram_alpha` | Query Wolfram Alpha for math, science, facts |
| `get_weather_forecast` | Weather data from WeatherAPI |
| `brave_search` | Web search via Brave Search API |
| `search_wikipedia` | Search and retrieve Wikipedia articles |

### `homeassistant`
| Tool | Description |
|------|-------------|
| `ha_list_entities` | List HA entities filtered by domain and/or area. Service-only domains (notify, tts, script) return a hint pointing at `ha_list_services`. |
| `ha_list_services` | List callable services for an HA domain (use for notify/tts/script). |
| `ha_execute_service` | Execute any HA service (turn on lights, lock doors, etc.) |
| `ha_control_light` | Opinionated light control (on/off, brightness, color) |
| `ha_control_climate` | Opinionated climate / thermostat control |
| `ha_activate_scene` | Activate a Home Assistant scene |
| `ha_trigger_script` | Run an HA script |
| `ha_trigger_automation` | Trigger an HA automation by id |
| `ha_toggle_automation` | Enable / disable an HA automation |
| `ha_send_notification` | Send a Home Assistant notification |
| `ha_list_areas` | List defined HA areas |
| `ha_get_presence` | Read presence (`person`/`device_tracker`) state |
| `ha_set_timer` | Create / arm a Home Assistant timer |
| `ha_cancel_timer` | Cancel a running HA timer |
| `ha_evaluate_template` | Render a Jinja2 template against HA state |
| `ha_get_entity_history` | Retrieve recent state history for an entity |
| `ha_get_calendar_events` | Pull events from an HA calendar |
| `ha_create_calendar_event` | Create an event on an HA calendar |
| `ha_control_media_player` | Play, pause, skip, volume, etc. on media players |

### `face`
| Tool | Description |
|------|-------------|
| `face_who_is_at` | Run face recognition against a camera snapshot and return identified people |
| `face_recent_visitors` | List recently identified people across the face-recognition log |
| `face_list_known_people` | Enumerate enrolled identities |
| `face_enroll_person` | Enroll a new face identity from a snapshot |
| `face_set_access_level` | Update an enrolled person's access level |

### `vision`
| Tool | Description |
|------|-------------|
| `describe_image` | Describe an arbitrary image via the vision LLM |
| `describe_camera_snapshot` | Pull a snapshot from a named camera and describe it |
| `compare_snapshots` | Compare two images and summarize what changed |
| `identify_object` | Identify the primary object / subject in an image |
| `read_text_in_image` | OCR the text content of an image |

### `device_action`
| Tool | Description |
|------|-------------|
| `set_alarm` | Wire a "set alarm" event to a connected companion-app device |
| `take_photo` | Ask the companion app to capture a photo from a phone camera |
| `identify_object_in_photo` | Take a photo via the companion app and identify its subject |
| `read_text_from_image` | Take a photo via the companion app and OCR it |
| `who_is_in_view` | Take a photo via the companion app and run face recognition on it |

### `plex`
| Tool | Description |
|------|-------------|
| `plex_search` | Search the Plex library for movies, shows, music, etc. |
| `plex_list_recent` | Recently added items |
| `plex_list_on_deck` | "On Deck" / continue-watching list |
| `plex_list_clients` | Reachable Plex clients |
| `plex_play` | Play media on a Plex client (cloud-relay path) |

### `music_assistant`
| Tool | Description |
|------|-------------|
| `mass_search` | Search Music Assistant providers (audio only) |
| `mass_list_players` | Enumerate Music Assistant players (speakers / Chromecasts / Google Homes) |
| `mass_play_media` | Queue and play a media item on a player |
| `mass_get_queue` | Inspect a player's current queue |
| `mass_queue_clear` | Clear the queue on a player |
| `mass_play_announcement` | Play a TTS announcement on one or more players |
| `mass_playback_control` | Transport: play / pause / stop / next / previous / seek / volume |

### `qdrant` (Semantic Memory)
| Tool | Description |
|------|-------------|
| `create_memory` | Store a memory with embeddings in Qdrant |
| `search_memories` | Semantic search across stored memories |
| `delete_memory` | Delete a stored memory by id |

### `reminder`
| Tool | Description |
|------|-------------|
| `schedule_reminder` | Schedule a one-shot reminder via the autonomy engine |
| `list_reminders` | List currently scheduled reminders |
| `cancel_reminder` | Cancel a scheduled reminder by id |

### `mqtt`
| Tool | Description |
|------|-------------|
| `get_camera_snapshots` | Capture snapshots from MQTT-connected cameras |

### `github` (Self-inspection)
| Tool | Description |
|------|-------------|
| `github_search_code` | Search the local HavenCore source clone |
| `github_read_file` | Read a file from the local HavenCore source clone |
| `github_list_dir` | List a directory in the local HavenCore source clone |
| `github_pull_latest` | Refresh the container-managed local clone from `origin` |
| `github_list_issues` | List GitHub Issues on the HavenCore repo |
| `github_get_issue` | Fetch a single issue (untrusted text wrapped in `<UNTRUSTED_USER_TEXT>`) |
| `github_create_issue` | File a new issue on the HavenCore repo |

## Configuration

All configuration is via environment variables (loaded in `selene_agent/utils/config.py`):

**LLM / providers**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE` | — | LLM endpoint URL (e.g. `http://vllm:8000/v1`) |
| `LLM_API_KEY` | — | API key for the LLM backend |
| `LLM_PROVIDER` | `vllm` | Seed value for the agent-LLM provider (`vllm`, `anthropic`, `openai`). Persisted in `agent_state`; this env var is only the first-boot fallback. The OpenAI-compat `/v1/chat/completions` endpoint stays pinned to vLLM regardless. |
| `ANTHROPIC_API_KEY` | — | API key when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Model id for the Anthropic provider |
| `VISION_API_BASE` | — | OpenAI-compat endpoint for the vision vLLM (e.g. `http://vllm-vision:8000/v1`) |
| `VISION_API_KEY` | — | API key for the vision LLM backend |
| `VISION_SERVED_NAME` | `gpt-4-vision` | Model name the vision vLLM is served as |
| `AGENT_NAME` | `""` | Name of the assistant persona |
| `MCP_SERVERS` | `{}` | JSON array of MCP server configs |

**Sessions / context**

| Variable | Default | Description |
|----------|---------|-------------|
| `CONVERSATION_TIMEOUT` | `90` | Seconds of inactivity before summarize-and-reset fires (per-session override via `X-Idle-Timeout` header or WS `idle_timeout` field) |
| `CONVERSATION_TIMEOUT_MIN` | `10` | Lower clamp for per-session overrides |
| `CONVERSATION_TIMEOUT_MAX` | `3600` | Upper clamp for per-session overrides |
| `SESSION_SUMMARY_MAX_TOKENS` | `400` | Cap on the summarize-on-timeout LLM recap length |
| `SESSION_SUMMARY_TAIL_EXCHANGES` | `2` | Raw user/assistant pairs preserved after summarize-and-reset |
| `SESSION_SUMMARY_LLM_TIMEOUT_SEC` | `15` | Summary LLM call wall-clock timeout; falls back to keep-tail-only |
| `CONVERSATION_CONTEXT_LIMIT_FRACTION` | `0.75` | Fraction of the active provider's `max_model_len` at which to summarize before overflow |
| `CONVERSATION_CONTEXT_LIMIT_TOKENS` | `0` | Absolute token ceiling override; `0` means use the fraction instead |
| `TOOL_RESULT_MAX_CHARS` | `8000` | Max characters per tool result before truncation |
| `MCP_TOOL_TIMEOUT_SECONDS` | `120` | Per-tool-call wall-clock timeout |

**Storage / infra**

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | — | PostgreSQL connection for conversation storage |
| `QDRANT_HOST` | `qdrant` | Qdrant vector DB host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `EMBEDDINGS_URL` | `http://embeddings:3000` | Text embeddings service URL |
| `EMBEDDING_DIM` | `1024` | Embedding vector dimensionality (matches the embeddings model) |

**External APIs**

| Variable | Default | Description |
|----------|---------|-------------|
| `HAOS_URL` | — | Home Assistant URL |
| `HAOS_TOKEN` | — | Home Assistant long-lived access token |
| `HAOS_USE_SSL` | — | Set truthy to use `https`/`wss` to Home Assistant |
| `PLEX_URL` | — | Plex base URL |
| `PLEX_TOKEN` | — | Plex auth token |
| `PLEX_CLIENT_HA_MAP` | — | JSON mapping Plex client names to HA media-player entities for the wake/launch flow |
| `MASS_URL` | — | Music Assistant URL |
| `MASS_TOKEN` | — | Music Assistant API token |
| `BRAVE_SEARCH_API_KEY` | — | Brave Search API key |
| `WOLFRAM_ALPHA_API_KEY` | — | Wolfram Alpha API key |
| `WEATHER_API_KEY` | — | WeatherAPI key |

**Companion-app camera tools**

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPANION_PHOTO_UPLOAD_TIMEOUT_SEC` | `25` | How long `take_photo` / chained vision tools wait for the phone before erroring |
| `COMPANION_BLOB_TTL_SEC` | `600` | TTL for uploaded captures held in the in-memory BlobStore |
| `COMPANION_BLOB_MAX_BYTES` | `10485760` | Per-blob size cap (10 MB) |

**Autonomy engine**

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTONOMY_ENABLED` | `true` | Master switch for the background autonomy engine |
| `AUTONOMY_DISPATCH_INTERVAL_SECONDS` | `30` | Tick interval for the dispatch loop |
| `AUTONOMY_BRIEFING_CRON` | `0 8 * * *` | Cron schedule for the daily briefing |
| `AUTONOMY_ANOMALY_CRON` | `*/15 * * * *` | Cron schedule for anomaly scans |
| `AUTONOMY_ANOMALY_COOLDOWN_MIN` | `30` | Per-entity cooldown between anomaly notifications |
| `AUTONOMY_MAX_RUNS_PER_HOUR` | `20` | Global rate cap on autonomy runs |
| `AUTONOMY_TURN_TIMEOUT_SEC` | `60` | Wall-clock timeout per autonomy turn |
| `AUTONOMY_BRIEFING_NOTIFY_TO` | — | Notification target for the daily briefing (falls back to `AUTONOMY_BRIEFING_EMAIL_TO`) |
| `AUTONOMY_HA_NOTIFY_TARGET` | — | Default HA `notify` target for autonomy notifications |
| `NTFY_PUBLISH_TOKEN` | — | Bearer token for publishing to the self-hosted ntfy server |
| `AUTONOMY_BRIEFING_CAMERA_ENTITIES` | — | Comma-separated HA camera entities included in the briefing |
| `AUTONOMY_ANOMALY_WATCH_DOMAINS` | `binary_sensor,lock,cover` | Comma-separated HA domains the anomaly scan watches |
| `AUTONOMY_WEBHOOK_ENABLED` | `false` | Enable the inbound webhook handler for reactive autonomy |
| `AUTONOMY_MQTT_ENABLED` | `false` | Enable the MQTT listener for reactive autonomy |
| `AUTONOMY_MQTT_CLIENT_ID` | `selene-autonomy` | MQTT client id for the autonomy listener |
| `AUTONOMY_MQTT_RECONNECT_MAX_SEC` | `60` | Max backoff for MQTT reconnect attempts |
| `AUTONOMY_DEFAULT_QUIET_START` | — | Default quiet-hours start time (HH:MM) |
| `AUTONOMY_DEFAULT_QUIET_END` | — | Default quiet-hours end time (HH:MM) |
| `AUTONOMY_DEFAULT_QUIET_POLICY` | `defer` | Quiet-hours policy: `defer` or `drop` |
| `AUTONOMY_DEFAULT_EVENT_RATE_LIMIT` | `10/min` | Default per-trigger event-rate limit |
| `AUTONOMY_SPEAKER_DEFAULT_DEVICE` | — | Default speaker target for autonomy TTS announcements |
| `AUTONOMY_SPEAKER_DEFAULT_VOICE` | `af_heart` | Default Kokoro voice for autonomy TTS |
| `AUTONOMY_SPEAKER_DEFAULT_VOLUME` | `0.5` | Default volume for autonomy announcements |
| `AUTONOMY_TTS_AUDIO_TTL_SEC` | `600` | TTL for cached autonomy TTS audio blobs |
| `AUTONOMY_ACT_ENABLED` | `false` | Allow autonomy turns to actuate devices (vs. notify-only) |
| `AUTONOMY_ACT_DEFAULT_CONFIRMATION_TIMEOUT_SEC` | `300` | Default confirmation window for actuating autonomy actions |
| `AGENT_BASE_URL` | — | Public base URL the agent advertises in notifications (deep links etc.) |
| `AGENT_INTERNAL_BASE_URL` | `http://agent:6002` | Agent's own HTTP base inside the Docker network (used for audio URLs handed to Music Assistant) |
| `AUTONOMY_MEMORY_REVIEW_CRON` | `0 3 * * *` | Cron schedule for the nightly memory consolidation pass |
| `AUTONOMY_MEMORY_MAX_SCAN` | `5000` | Max memories scanned per consolidation run |
| `AUTONOMY_MEMORY_LLM_CALL_CAP` | `20` | Max LLM calls the consolidation pass may make |

**Memory tiers (L1–L4)**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_HALF_LIFE_DAYS` | `60` | Decay half-life applied to importance scoring |
| `MEMORY_ACCESS_COEF` | `0.5` | Weight of access-count in importance scoring |
| `MEMORY_HDBSCAN_MIN_CLUSTER_SIZE` | `5` | HDBSCAN min cluster size for consolidation clustering |
| `MEMORY_HDBSCAN_MIN_SAMPLES` | `3` | HDBSCAN min samples for consolidation clustering |
| `MEMORY_L4_MIN_IMPORTANCE` | `4` | Minimum importance for an entry to be promoted to L4 |
| `MEMORY_L4_MIN_AGE_DAYS` | `14` | Minimum age before an entry is eligible for L4 |
| `MEMORY_L4_MIN_ACCESS_COUNT` | `3` | Minimum access count before L4 promotion |
| `MEMORY_L4_MAX_ENTRIES` | `20` | Hard cap on the L4 system-prompt block |
| `MEMORY_L4_WARN_TOKENS` | `1500` | Token-budget warning threshold for the L4 block |
| `MEMORY_L2_PRUNE_AGE_DAYS` | `180` | Age at which low-importance L2 entries are pruned |
| `MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD` | `0.5` | Importance floor below which aged L2 entries are pruned |
| `MEMORY_L3_RANK_BOOST` | `1.2` | Re-rank multiplier applied to L3 cluster summaries |
| `MEMORY_RETRIEVAL_ENABLED` | `true` | Enable per-turn retrieval injection (embed user message, pull top-K L2/L3 into the prompt) |
| `MEMORY_RETRIEVAL_TOPK_LEARNING` | `5` | Top-K retrieval depth in the LEARNING phase |
| `MEMORY_RETRIEVAL_TOPK_OPERATING` | `3` | Top-K retrieval depth in the OPERATING phase |
| `MEMORY_RETRIEVAL_MIN_SCORE` | `0.3` | Minimum similarity score for a retrieval hit to be injected |
| `AGENT_PHASE_DEFAULT` | `learning` | Seed value for the agent operational phase (`learning` / `operating`); persisted in `agent_state` after first read |

**Misc**

| Variable | Default | Description |
|----------|---------|-------------|
| `CURRENT_LOCATION` | `New York, NY` | Location context for weather/local queries |
| `CURRENT_ZIPCODE` | `10001` | Zip code context for weather/local queries |
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
    │   ├── agent.py          # GET/POST /api/agent/* (provider, phase, agent_state)
    │   ├── autonomy.py       # GET/POST /api/autonomy/* (runs, schedules, triggers)
    │   ├── cameras.py        # GET /api/cameras/* (snapshot proxy)
    │   ├── chat.py           # POST /api/chat, WS /ws/chat (persists metrics)
    │   ├── comfy.py          # POST /api/comfy/generate, status/view/health
    │   ├── companion.py      # Companion-app device link, photo upload, push device registration
    │   ├── conversations.py  # GET /api/conversations
    │   ├── face.py           # GET/POST /api/face/* (people, snapshots, identify)
    │   ├── homeassistant.py  # GET /api/ha/*
    │   ├── logs.py           # WS /ws/logs
    │   ├── memory.py         # GET/POST /api/memory/* (browse + tier inspection)
    │   ├── metrics.py        # GET /api/metrics/{turns,summary,top-tools}
    │   ├── push.py           # Push-device registration for ntfy/UnifiedPush
    │   ├── status.py         # GET /api/status, /api/tools
    │   ├── stt.py            # POST /api/stt/transcribe + health proxy
    │   ├── tts.py            # POST /api/tts/speak, voices/health proxies
    │   ├── tts_audio.py      # Cached TTS audio blob serving for autonomy announcements
    │   └── vision.py         # POST /api/vision/ask + health proxy
    ├── autonomy/             # Background engine
    │   ├── engine.py         # Dispatch loop, scheduler integration
    │   ├── turn.py           # Per-task ephemeral orchestrator + tool gating
    │   ├── schedule.py       # Cron + one-shot reminder scheduling
    │   ├── tool_gating.py    # Per-trigger tool allow/deny lists
    │   ├── trigger_match.py  # Webhook + MQTT trigger matching
    │   ├── quiet_hours.py    # Quiet-hours policy
    │   ├── event_rate_limit.py # Per-trigger rate limiting
    │   ├── notifiers.py      # ntfy / HA / email notification fan-out
    │   ├── memory_clustering.py / memory_math.py # Nightly L2→L3/L4 consolidation
    │   ├── reminder_personalize.py # Reminder phrasing
    │   ├── mqtt_listener.py / sensor_events.py # Reactive autonomy ingest
    │   ├── db.py             # autonomy_runs / agenda_items DAL
    │   ├── handlers/         # Per-trigger handlers (briefing, anomaly, reminder, ...)
    │   └── seeds/            # Built-in trigger / agenda seed data
    ├── providers/            # Pluggable LLM providers (vllm, anthropic, openai)
    │   ├── base.py
    │   ├── factory.py
    │   ├── vllm.py
    │   ├── anthropic.py
    │   └── openai.py
    ├── services/             # Shared service clients
    │   ├── tts_client.py     # Kokoro TTS HTTP client
    │   └── audio_store.py    # In-memory TTL audio blob store
    ├── utils/
    │   ├── agent_state.py    # Postgres-backed agent_state (provider, phase) read/write
    │   ├── config.py         # Environment-driven configuration
    │   ├── conversation_db.py # PostgreSQL conversation storage/retrieval
    │   ├── l4_context.py     # L4 system-prompt block builder
    │   ├── log_stream.py     # In-process log ring buffer for /ws/logs
    │   ├── logger.py         # Loki + ring-buffer log handlers
    │   ├── mcp_client_manager.py # MCP server lifecycle, tool discovery
    │   ├── metrics_db.py     # PostgreSQL turn_metrics read/write (shared pool)
    │   ├── push_db.py        # push_devices table DAL
    │   ├── retrieval.py      # Per-turn L2/L3 retrieval injection
    │   ├── session_pool.py   # SessionOrchestratorPool (LRU + idle sweep + flush + cold-resume)
    │   └── tokens.py         # Token-counting helpers
    └── modules/              # MCP tool servers (each runs as a subprocess)
        ├── mcp_device_action_tools/
        ├── mcp_face_tools/
        ├── mcp_general_tools/
        ├── mcp_github_tools/
        ├── mcp_homeassistant_tools/
        ├── mcp_mqtt_tools/
        ├── mcp_music_assistant_tools/
        ├── mcp_plex_tools/
        ├── mcp_qdrant_tools/
        ├── mcp_reminder_tools/
        └── mcp_vision_tools/
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
