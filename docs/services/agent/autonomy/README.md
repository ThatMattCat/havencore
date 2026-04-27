# Autonomy Engine

The autonomy engine makes Selene proactive. Instead of only reacting to voice
or chat input, the agent wakes on its own schedule — or in response to live
events — to perform checks, produce summaries, surface anomalies, or run
user-programmed reminders / watches / routines, then exits until the next
event. It runs as an asyncio background task inside the existing `agent`
FastAPI process; no extra container.

- **v1** (merged): scheduled `briefing` + `anomaly_sweep` on a `notify`-tier
  tool allow-list with global rate limiting, per-signature cooldown, and
  audit logging.
- **v2** (merged): L1–L4 memory tiers + a nightly `memory_review`
  consolidation agenda item. See [memory/README.md](memory/README.md).
- **v3** (implemented, flagged off): user-programmable `reminder`, `watch`,
  `routine` kinds; reactive HA webhook + live MQTT triggers; quiet-hours
  defer/drop; per-item event rate limits; live run WebSocket feed;
  `/autonomy` dashboard page. Full detail in [v3.md](v3.md).
- **v4** (implemented; `act` tier flagged off): `speak` tier with a
  `SpeakerNotifier` that renders Kokoro TTS and plays through Music
  Assistant, LLM-judged `watch_llm` kind, and a supervised `act` tier with
  per-item `action_allow_list` + optional confirmation gate. Full detail in
  [v4.md](v4.md).
- **Camera/sensor events**: generic proactive-notification pipeline on top
  of `watch_llm` — face-recognition is the first source, vehicles/motion/
  doorbell plug in via the same `haven/<domain>/<kind>` MQTT topic schema.
  Camera-to-zone mapping lives in Postgres + a `/cameras` dashboard page so
  the LLM reasons about zones, not raw camera entity_ids. See
  [cameras.md](cameras.md).

## What v1 does

- **Morning briefing** (`kind='briefing'`, default cron `0 8 * * *`) — gathers
  calendar, weather, and optional overnight history into a single LLM pass,
  then sends the result via the `send_signal_message` MCP tool.
- **Ambient anomaly sweep** (`kind='anomaly_sweep'`, default cron
  `*/15 * * * *`) — snapshots presence + watched-domain entity states, queries
  memory for household-routine context, asks the LLM for a strict JSON
  judgment, and — if unusual — pushes an HA notification. Per-signature
  cooldown prevents spam.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ agent service (FastAPI, single process)                          │
│                                                                  │
│  lifespan ──► existing AgentOrchestrator (user-facing chat)      │
│           └─► AutonomyEngine (asyncio background task)           │
│                ├─ Dispatcher loop (30s tick, configurable)       │
│                │   ├─ reads agenda_items (enabled AND due?)      │
│                │   ├─ global rate limit + kill switch gates      │
│                │   └─ fires handler per item.kind                │
│                │                                                 │
│                ├─ AutonomousTurn                                 │
│                │   (fresh AgentOrchestrator, isolated session,   │
│                │    tier-filtered tool set, captured messages +  │
│                │    metrics, hard timeout)                       │
│                │                                                 │
│                └─ Notifier (protocol)                            │
│                     ├─ SignalNotifier  → send_signal_message     │
│                     ├─ HAPushNotifier  → ha_send_notification    │
│                     └─ NullNotifier                              │
│                                                                  │
│  new REST: /api/autonomy/{status,pause,resume,items,runs,trigger}│
└──────────────────────────────────────────────────────────────────┘
```

**Invariant:** autonomous turns never touch any user session in
`app.state.session_pool`. Each run constructs a fresh `AgentOrchestrator`
directly (bypassing the pool entirely), drives its event stream to
completion, captures the trace into `autonomy_runs.messages`, and is
discarded. User conversation state stays clean.

## Components

Code lives in `services/agent/selene_agent/autonomy/`:

| File | Role |
|------|------|
| `engine.py` | Dispatcher loop, lifecycle, rate limiting, per-signature cooldown, run persistence |
| `turn.py` | `AutonomousTurn` — single-use orchestrator with custom prompt + filtered tools + timeout |
| `schedule.py` | `croniter`-based `next_fire_at()` computation in `CURRENT_TIMEZONE`, stores UTC |
| `tool_gating.py` | Per-tier allow-lists (`observe`, `notify`) and explicit `V1_DENY` set |
| `notifiers.py` | `Notifier` protocol + `SignalNotifier`, `HAPushNotifier`, `NullNotifier` |
| `db.py` | `agenda_items` + `autonomy_runs` access layer (reuses the `conversation_db` pool) |
| `handlers/briefing.py` | Deterministic gather → LLM summarize → Signal message |
| `handlers/anomaly.py` | State snapshot + memory context → LLM JSON judgment → push + cooldown |

REST router: `services/agent/selene_agent/api/autonomy.py`. Wired into the
FastAPI lifespan at `services/agent/selene_agent/selene_agent.py`.

## Data model

Two Postgres tables, created on agent startup via
`autonomy.db.ensure_schema()` in the same pattern as `turn_metrics`.

### `agenda_items`

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid primary key | |
| `kind` | text | `'briefing'` or `'anomaly_sweep'` in v1 |
| `schedule_cron` | text | standard 5-field cron |
| `next_fire_at` | timestamptz | authoritative for dispatch; stored UTC |
| `last_fired_at` | timestamptz | nullable |
| `config` | jsonb | kind-specific config (email target, watched domains, …) |
| `autonomy_level` | text | `'observe'` or `'notify'` in v1 |
| `enabled` | boolean | flip to pause a single row |
| `created_by` | text | `'system'` for seeded defaults, `'user'` / `'llm'` otherwise |
| `created_at` | timestamptz | |

Two default rows are seeded on startup (idempotent upsert keyed on
`(kind, created_by='system')`): one `briefing`, one `anomaly_sweep`. The
`schedule_cron` and `config` fields are refreshed from env each startup so
operators can tune via `.env` without touching the DB. `enabled` is preserved
across restarts so a manual pause survives.

### `autonomy_runs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid primary key | |
| `agenda_item_id` | uuid → `agenda_items(id)` | set null if the item is deleted |
| `kind` | text | denormalized for query convenience |
| `triggered_at` / `completed_at` | timestamptz | |
| `status` | text | `ok` / `error` / `skipped_cooldown` / `skipped_killswitch` / `rate_limited` |
| `summary` | text | one-line human summary (e.g. `nominal`, `garage open >10min`) |
| `severity` | text | `none` / `low` / `med` / `high` (anomaly-only) |
| `signature_hash` | text | stable sha1(first 16) of the LLM's signature slug — drives cooldown dedup |
| `notified_via` | text | `email` / `ha_push` / null |
| `messages` | jsonb | full message trace from the turn |
| `metrics` | jsonb | `{llm_ms, tool_ms_total, total_ms, iterations, tool_calls, autonomy_level, tools_allowed}` |
| `error` | text | nullable |

All dispatch outcomes are persisted — including rate-limited and cooldown
skips — so the `autonomy_runs` table is a complete ledger of engine activity.

## Configuration

All autonomy env vars live together in `.env.tmpl` and are surfaced through
`shared/configs/shared_config.py` and
`services/agent/selene_agent/utils/config.py`.

```bash
AUTONOMY_ENABLED=true
AUTONOMY_DISPATCH_INTERVAL_SECONDS=30
AUTONOMY_BRIEFING_CRON="0 8 * * *"
AUTONOMY_ANOMALY_CRON="*/15 * * * *"
AUTONOMY_ANOMALY_COOLDOWN_MIN=30
AUTONOMY_MAX_RUNS_PER_HOUR=20
AUTONOMY_TURN_TIMEOUT_SEC=60
AUTONOMY_BRIEFING_NOTIFY_TO=""       # Signal recipient for the morning briefing
AUTONOMY_HA_NOTIFY_TARGET=""         # e.g. notify.mobile_app_pixel_8
AUTONOMY_BRIEFING_CAMERA_ENTITIES="" # comma-separated camera entity_ids
AUTONOMY_ANOMALY_WATCH_DOMAINS="binary_sensor,lock,cover"
```

Notes:

- Cron strings are interpreted in `CURRENT_TIMEZONE` before being converted
  to UTC for storage, matching the convention used by the user-facing
  orchestrator.
- `send_signal_message` sends via the `signal-api` container. Recipient
  precedence: per-notification `to` → `AUTONOMY_BRIEFING_NOTIFY_TO` →
  `SIGNAL_DEFAULT_RECIPIENT` → `SIGNAL_PHONE_NUMBER` (Note to Self). See
  `docs/services/agent/tools/general.md` for the one-time QR-link setup.
- `AUTONOMY_HA_NOTIFY_TARGET` may be written as `notify.mobile_app_<device>`
  or `mobile_app_<device>`; the leading `notify.` is stripped.

## REST API

All endpoints live under `/api/autonomy/*` on port 6002. No auth (same
convention as the rest of `/api/*`).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/api/autonomy/status` | Engine running/paused, last dispatch, runs in the last hour, next-due preview |
| `POST` | `/api/autonomy/pause` | Flip runtime kill switch (does not persist across restarts) |
| `POST` | `/api/autonomy/resume` | Clear runtime kill switch |
| `GET`  | `/api/autonomy/items` | List all `agenda_items` rows |
| `GET`  | `/api/autonomy/runs?limit=50&include_messages=0` | Recent runs; pass `include_messages=1` to include the full trace |
| `POST` | `/api/autonomy/trigger/{agenda_item_id}` | Fire a specific item now, bypassing the schedule and the rate limit |

The agent's `/health` endpoint gains an `autonomy` block reporting
`running`, `paused`, and `last_dispatch_at`.

## Tool gating

`AutonomousTurn` filters the base tool list before handing it to the
orchestrator. The allow-list is maintained in
`autonomy/tool_gating.py`:

- **`observe` tier** — read-only HA tools (`ha_get_*`), `search_memories`,
  general knowledge tools (`brave_search`, `wolfram_alpha`,
  `get_weather_forecast`, `search_wikipedia`, `query_multimodal_api`, `fetch`).
- **`notify` tier** — everything in `observe` plus the two notifier tools
  (`send_signal_message`, `ha_send_notification`).

An explicit `V1_DENY` set (HA actuators, scenes, scripts, automations,
`create_memory`, `delete_memory`, media playback) is enforced on top of the
allow-list for defense in depth. Notifiers themselves are invoked by
handlers directly — not via the LLM's tool-calling surface — so the LLM
cannot decide whether or where to notify.

## Guardrails

Non-negotiable v1 behavior:

1. **Kill switch** — `AUTONOMY_ENABLED=false` disables dispatch at startup.
   `POST /api/autonomy/pause` flips a runtime flag without restart.
2. **Global hourly rate limit** — `AUTONOMY_MAX_RUNS_PER_HOUR` (default 20).
   Scheduled dispatches above the cap are recorded as `status='rate_limited'`
   and not executed. Manual `/trigger` calls bypass this cap intentionally
   (operator override).
3. **Per-signature cooldown** — for `anomaly_sweep`, the LLM emits a stable
   `signature` slug; the engine hashes it and refuses to re-notify within
   `AUTONOMY_ANOMALY_COOLDOWN_MIN` minutes unless severity escalates
   (`low` → `med` → `high`). The cooldown skip still writes a run row.
4. **Hard tool allow-list** — enforced at turn construction; not trusted to
   LLM judgment.
5. **Session isolation** — each autonomous turn uses a fresh
   `AgentOrchestrator` with its own `session_id`; never appended to user
   conversation state.
6. **Per-turn timeout** — `AUTONOMY_TURN_TIMEOUT_SEC` (default 60). Overages
   are aborted and logged as `status='error'` with a timeout message.
7. **System prompt discipline** — autonomous prompts explicitly instruct:
   "You are running autonomously. Do not ask questions. Complete your
   assessment and exit. Output must match the required format."
8. **All outcomes persisted** — `ok`, `error`, `skipped_cooldown`,
   `rate_limited` all produce `autonomy_runs` rows for audit.

## Manual verification

With the agent running:

```bash
# Engine status
curl http://localhost:6002/api/autonomy/status

# List agenda items (grab an id)
curl http://localhost:6002/api/autonomy/items

# Force a run right now
curl -X POST http://localhost:6002/api/autonomy/trigger/<agenda_item_id>

# Recent run history (metadata only)
curl "http://localhost:6002/api/autonomy/runs?limit=20"

# Full message trace for the most recent runs
curl "http://localhost:6002/api/autonomy/runs?limit=5&include_messages=1"

# Pause / resume
curl -X POST http://localhost:6002/api/autonomy/pause
curl -X POST http://localhost:6002/api/autonomy/resume
```

Direct DB inspection:

```bash
docker compose exec postgres psql -U havencore -c \
  "select kind, status, severity, summary, triggered_at \
   from autonomy_runs order by triggered_at desc limit 20;"
```

## Memory integration

v1 gives the semantic-memory store a concrete job: the anomaly handler runs
`search_memories(query='household normal routine', limit=5)` as part of its
state bundle, so household-routine context informs the LLM's judgment.

Schema-only groundwork for v2 consolidation: newly created memories now carry
`tier: 'L2'` and `source_ids: []` in their Qdrant payload. Search results
back-fill `tier='L2'` for entries written before this change, so existing
data keeps working. v2 will populate `L3` / `L4` rows on top of this schema;
no behavior change in v1.

## Implemented across v1–v3

- **v1** — `briefing`, `anomaly_sweep`; global rate limiting; per-signature
  cooldown; `notify` tier; hard tool allow-list; session isolation; per-turn
  timeout; full audit log.
- **v2** — L1–L4 memory tiers; `memory_review` nightly consolidation with
  importance decay, L3 rank boost, and L4 promotion gate.
- **v3** — `reminder`, `watch`, `routine` kinds; HA webhook intake
  (`POST /api/autonomy/webhook/{name}`); live MQTT subscriber against the
  existing mosquitto broker; quiet-hours `defer` / `drop` policy with
  deferred-run sweep; per-item event rate limits; CRUD for agenda items
  (`POST` / `PATCH` / `DELETE /api/autonomy/items`); live run WebSocket feed
  (`/ws/autonomy/runs`) backed by Postgres `LISTEN`/`NOTIFY`; `/autonomy`
  dashboard page with agenda editor, filterable run history, and reactive
  source health.

## Deferred to v4

- `speak` autonomy tier with a `SpeakerNotifier` (TTS URL →
  `media_player.play_media` on a target).
- `act` tier with a per-action permission model — first novel action per
  signature requires explicit confirmation; confirmed actions join an
  allow-list.
- Smaller / cheaper gate model for quick triage sweeps that only escalate to
  the full chat LLM (GLM-4.5-Air-AWQ-FP16Mix) when interesting.
- LLM-judged watches (beyond the current deterministic template + optional
  HA state condition).
- Declarative routine DSL (routines today construct an `AutonomousTurn` with
  a freeform goal + tool override).
- Cross-item trigger chaining.
- Pattern learning that observes user-triggered routines and suggests
  matching agenda items.
