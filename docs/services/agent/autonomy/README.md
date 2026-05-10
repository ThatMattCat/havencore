# Autonomy Engine

The autonomy engine makes Selene proactive. Instead of only reacting to voice or chat input, the agent wakes on its own schedule — or in response to live events — to perform checks, produce summaries, surface anomalies, run user-programmed reminders / watches / routines, and (if you opt in) actuate devices through a supervised confirmation gate. It runs as an `asyncio` background task inside the existing `agent` FastAPI process; no extra container.

The engine is the same FastAPI process that serves the dashboard, so a single restart of the `agent` container reloads the dispatcher, MQTT subscriber, deferred-run sweep, and confirmation-timeout sweep together.

## What the engine does

- **Dispatch** — a 30-second tick (configurable) reads `agenda_items`, fires what's due, and writes a row to `autonomy_runs` for every outcome (success, error, cooldown skip, rate-limited, deferred).
- **Schedule + react** — items can be cron-driven (`schedule_cron`), reactive (`trigger_spec` against MQTT topics or HA webhooks), or both.
- **Isolate** — every fire constructs a fresh `AgentOrchestrator` with its own `session_id`. User chat sessions in `app.state.session_pool` are never touched.
- **Notify** — handlers select a delivery channel; the engine swaps in the right `Notifier`.
- **Audit** — every dispatch outcome is persisted, including skips and rate-limits, so `autonomy_runs` is a complete ledger.
- **Stream** — `WS /ws/autonomy/runs` pushes new runs to the dashboard via Postgres `LISTEN/NOTIFY`.

## Agenda kinds

| Kind | Purpose | Trigger |
|---|---|---|
| `briefing` | Morning briefing — pulls calendar + weather + optional overnight history, asks the LLM for a single summary, sends it through the chosen channel. | Cron (default `0 8 * * *`) |
| `anomaly_sweep` | Snapshot presence + watched-domain entity states, query memory for household routine context, ask the LLM for a strict JSON anomaly judgment, push if unusual. | Cron (default `*/15 * * * *`) |
| `reminder` | Scheduled notify with optional LLM body rewrite. `personalize: true` (default) re-renders the body in Selene's voice at fire time and may attach a generated image (Signal channel only). `one_shot: true` deletes the agenda row after the first successful fire. | Cron |
| `watch` | Reactive trigger-driven notify. Template-renders `body_template` against the trigger payload, optionally gates on an HA entity state condition, routes through the anomaly-style notification path so per-signature cooldown applies. | MQTT or HA webhook |
| `watch_llm` | Same trigger surface as `watch`, but the engine hands the event + a bounded state gather to the LLM and asks for a JSON judgment (`unusual`, `severity`, `summary`, `signature`, `evidence`). For "is this noteworthy?" calls that don't reduce to a boolean condition. Reuses anomaly cooldown. | MQTT or HA webhook |
| `routine` | Goal-oriented LLM turn — fresh `AutonomousTurn` with a per-item prompt, optional `tools_override` (must be a subset of the current tier's allow-list), result delivered via `deliver.channel`. | Cron |
| `act` | Two-phase, permission-gated actuation. Plan-only `observe`-tier turn produces a strict-JSON step list; engine validates each step against a per-item `action_allow_list` and either executes inline or parks `status='awaiting_confirmation'` with a deep-link to confirm. **Flagged off by default** — set `AUTONOMY_ACT_ENABLED=true` to enable. | Cron |
| `memory_review` | Nightly memory-tier consolidation pass. See [memory/README.md](memory/README.md). | Cron |
| `warmup` | 1-token vLLM ping with the live chat system prompt + tools schema. Refreshes the prefix cache's LRU position so user turns hit a warm cache between idle stretches. Bypasses `AUTONOMY_MAX_RUNS_PER_HOUR` (it's a maintenance ping, not a budgeted run) and is hidden from the default run-history view. | Cron (default `*/5 * * * *`) |

Camera/sensor events plug into the same surface via a generic `haven/<domain>/<kind>` MQTT topic schema — face-recognition is the first source, vehicles/motion/doorbell can drop in the same way. Camera-to-zone mapping lives in Postgres + a `/cameras` dashboard page so the LLM reasons about zones rather than raw camera entity_ids. See [cameras.md](cameras.md).

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ agent service (FastAPI, single process)                          │
│                                                                  │
│  lifespan ──► AgentOrchestrator (user-facing chat)               │
│           └─► AutonomyEngine (asyncio background task)           │
│                ├─ Dispatcher loop (30s tick, configurable)       │
│                │   ├─ reads agenda_items (enabled AND due?)      │
│                │   ├─ global rate limit + kill switch gates      │
│                │   ├─ deferred-run sweep (quiet-hours releases)  │
│                │   ├─ confirmation-timeout sweep (act items)     │
│                │   └─ fires handler per item.kind                │
│                │                                                 │
│                ├─ AutonomousTurn                                 │
│                │   (fresh AgentOrchestrator, isolated session,   │
│                │    tier-filtered tool set, captured messages +  │
│                │    metrics, hard timeout)                       │
│                │                                                 │
│                ├─ Reactive intake                                │
│                │   ├─ POST /api/autonomy/webhook/{name}          │
│                │   └─ paho-mqtt subscriber (diff-resubscribe on  │
│                │       agenda CRUD; reconnect with backoff)      │
│                │                                                 │
│                └─ Notifier (protocol)                            │
│                     ├─ SignalNotifier      → send_signal_message │
│                     ├─ HAPushNotifier      → ha_send_notification│
│                     ├─ SpeakerNotifier     → Kokoro TTS + MA     │
│                     ├─ NtfyFanoutNotifier  → companion-app push  │
│                     └─ NullNotifier                              │
│                                                                  │
│  REST: /api/autonomy/{status,pause,resume,items,runs,trigger,    │
│         webhook,events,runs/{id}/confirm,runs/awaiting}          │
│  WS:   /ws/autonomy/runs (Postgres LISTEN/NOTIFY)                │
└──────────────────────────────────────────────────────────────────┘
```

**Invariant:** autonomous turns never touch any user session in `app.state.session_pool`. Each run constructs a fresh `AgentOrchestrator` directly (bypassing the pool entirely), drives its event stream to completion, captures the trace into `autonomy_runs.messages`, and is discarded. User conversation state stays clean.

## Components

Code lives in `services/agent/selene_agent/autonomy/`:

| File | Role |
|------|------|
| `engine.py` | Dispatcher loop, lifecycle, rate limiting, per-signature cooldown, deferred-run sweep, confirmation-timeout sweep, run persistence |
| `turn.py` | `AutonomousTurn` — single-use orchestrator with custom prompt + filtered tools + timeout |
| `schedule.py` | `croniter`-based `next_fire_at()` computation in `CURRENT_TIMEZONE`, stored as UTC |
| `tool_gating.py` | Per-tier allow-lists (`observe`, `notify`, `speak`, `act`) + explicit deny set |
| `trigger_match.py` | MQTT `+` wildcard + payload-subset matcher; webhook-name matcher |
| `quiet_hours.py` | Cross-midnight, timezone-aware suppress-or-defer evaluation |
| `event_rate_limit.py` | Leaky-bucket per-item limiter parsed from `N/sec\|min\|hr` shorthand |
| `mqtt_listener.py` | Background `paho-mqtt` client; diff-subscribes on `asyncio.Event` from agenda CRUD |
| `notifiers.py` | `Notifier` protocol + `SignalNotifier`, `HAPushNotifier`, `SpeakerNotifier`, `NtfyNotifier` (single endpoint) / `NtfyFanoutNotifier` (DB-backed), `NullNotifier` |
| `db.py` | `agenda_items` + `autonomy_runs` access layer (reuses the `conversation_db` pool) |
| `handlers/briefing.py` | Deterministic gather → LLM summarize → notification |
| `handlers/anomaly.py` | State snapshot + memory context → LLM JSON judgment → notification + cooldown |
| `handlers/reminder.py` | Personalize body via LLM → notification; `one_shot` row delete |
| `handlers/watch.py` | Template render against trigger payload → notification with shared cooldown |
| `handlers/watch_llm.py` | Triage gather + LLM JSON judgment → notification; severity floor |
| `handlers/routine.py` | Generic `AutonomousTurn` with per-item tools_override and delivery channel |
| `handlers/act.py` | Plan / validate / execute three-phase actuator pipeline; awaiting-confirmation persistence |
| `handlers/warmup.py` | Direct vLLM 1-token ping with the live chat prefix; rebuilds the system prompt the same way `AgentOrchestrator.initialize()` does so the rendered prefix matches byte-for-byte |

REST router: `services/agent/selene_agent/api/autonomy.py`. Wired into the FastAPI lifespan at `services/agent/selene_agent/selene_agent.py`.

## Data model

Two Postgres tables, created on agent startup via `autonomy.db.ensure_schema()` in the same pattern as `turn_metrics`. Schema migration is idempotent — every column is added `IF NOT EXISTS` so re-runs against an upgraded cluster are safe.

### `agenda_items`

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid primary key | |
| `kind` | text | `briefing` / `anomaly_sweep` / `reminder` / `watch` / `watch_llm` / `routine` / `act` / `memory_review` / `warmup` |
| `name` | text | optional operator-facing label |
| `schedule_cron` | text | nullable for reactive items; standard 5-field cron in `CURRENT_TIMEZONE`, stored UTC |
| `trigger_spec` | jsonb | nullable; `{source: 'mqtt'\|'webhook', match: {...}}` |
| `next_fire_at` | timestamptz | authoritative for cron dispatch |
| `last_fired_at` | timestamptz | nullable |
| `config` | jsonb | kind-specific config (channel, recipient, watched domains, body template, action_allow_list, …) |
| `autonomy_level` | text | `observe` / `notify` / `speak` / `act` |
| `enabled` | boolean | flip to pause a single row |
| `created_by` | text | `system` for seeded defaults, `user` / `llm` otherwise |
| `created_at` | timestamptz | |

Table-level `CHECK`: `schedule_cron IS NOT NULL OR trigger_spec IS NOT NULL`.

Four default rows are seeded on startup (idempotent upsert keyed on `(kind, created_by='system')`): `briefing`, `anomaly_sweep`, `memory_review`, `warmup`. The `schedule_cron` and `config` fields are refreshed from env each startup so operators can tune via `.env` without touching the DB. `enabled` is preserved across restarts so a manual pause survives.

### `autonomy_runs`

| Column | Type | Notes |
|--------|------|-------|
| `id` | uuid primary key | |
| `agenda_item_id` | uuid → `agenda_items(id)` ON DELETE SET NULL | preserves audit trail when an item is deleted |
| `kind` | text | denormalized for query convenience |
| `triggered_at` / `completed_at` | timestamptz | |
| `scheduled_for` | timestamptz | populated when a run is deferred past quiet hours |
| `trigger_source` | text | `cron` / `mqtt` / `webhook` / `manual` / `deferred` |
| `trigger_event` | jsonb | raw event payload for reactive runs; `null` for cron |
| `status` | text | `ok` / `error` / `skipped_cooldown` / `skipped_killswitch` / `skipped_quiet_hours` / `skipped_rate_limit` / `skipped_trigger_mismatch` / `scheduled` / `rate_limited` / `awaiting_confirmation` / `confirmation_denied` / `confirmation_timeout` |
| `summary` | text | one-line human summary (e.g. `nominal`, `garage open >10min`) |
| `severity` | text | `none` / `low` / `med` / `high` (anomaly / watch_llm) |
| `signature_hash` | text | sha1(first 16) of the dedup signature — drives cooldown. For triggers carrying a normalized `sensor_event` (camera/face/etc.) the signature is `{domain}:{kind}:{zone}:{subject}`; otherwise it's the LLM-emitted slug. |
| `notified_via` | text | `signal` / `ha_push` / `speaker` / `ntfy` / null |
| `messages` | jsonb | full message trace from the turn |
| `metrics` | jsonb | `{llm_ms, tool_ms_total, total_ms, iterations, tool_calls, autonomy_level, tools_allowed}` |
| `error` | text | nullable |
| `confirmation_token` | text | random 24-byte token for `act` runs; never returned over list endpoints |
| `confirmation_response` | text | `approved` / `denied` / `timeout` |
| `action_audit` | jsonb | list of `{tool, args, rationale, outcome, result?, error?}` for `act` runs |

A trigger on `INSERT` notifies channel `autonomy_runs_ch` with the new row id; the `WS /ws/autonomy/runs` endpoint streams those frames. A partial index on `status='awaiting_confirmation'` keeps the timeout sweep cheap.

## Configuration

All env vars live together in `.env.example` and are surfaced through `shared/configs/shared_config.py` and `services/agent/selene_agent/utils/config.py`.

```bash
# Engine lifecycle
AUTONOMY_ENABLED=true
AUTONOMY_DISPATCH_INTERVAL_SECONDS=30
AUTONOMY_MAX_RUNS_PER_HOUR=20
AUTONOMY_TURN_TIMEOUT_SEC=60

# Built-in agenda items
AUTONOMY_BRIEFING_CRON="0 8 * * *"
AUTONOMY_ANOMALY_CRON="*/15 * * * *"
AUTONOMY_WARMUP_CRON="*/5 * * * *"        # vLLM prefix-cache warmup ping
AUTONOMY_ANOMALY_COOLDOWN_MIN=30
AUTONOMY_BRIEFING_NOTIFY_TO=""           # Signal recipient for the morning briefing
AUTONOMY_HA_NOTIFY_TARGET=""             # e.g. notify.mobile_app_pixel_8
AUTONOMY_BRIEFING_CAMERA_ENTITIES=""     # comma-separated camera entity_ids
AUTONOMY_ANOMALY_WATCH_DOMAINS="binary_sensor,lock,cover"

# Reactive intake (default: off — flip once you have items in place)
AUTONOMY_WEBHOOK_ENABLED=false
AUTONOMY_MQTT_ENABLED=false
AUTONOMY_MQTT_CLIENT_ID="selene-autonomy"
AUTONOMY_MQTT_RECONNECT_MAX_SEC=60

# Default quiet-hours window (applied when an item omits its own spec)
AUTONOMY_DEFAULT_QUIET_START=""          # "22:00"
AUTONOMY_DEFAULT_QUIET_END=""            # "07:00"
AUTONOMY_DEFAULT_QUIET_POLICY="defer"    # "defer" | "drop"
AUTONOMY_DEFAULT_EVENT_RATE_LIMIT="10/min"

# Speaker channel (Kokoro TTS → Music Assistant)
AUTONOMY_SPEAKER_DEFAULT_DEVICE=""       # MA player name
AUTONOMY_SPEAKER_DEFAULT_VOICE="af_heart"
AUTONOMY_SPEAKER_DEFAULT_VOLUME=0.5      # 0.0-1.0 (normalized to 0-100)
AUTONOMY_TTS_AUDIO_TTL_SEC=600           # AudioStore entry TTL

# act tier — flagged off by default
AUTONOMY_ACT_ENABLED=false
AUTONOMY_ACT_DEFAULT_CONFIRMATION_TIMEOUT_SEC=300

# Companion-app push channel
NTFY_PUBLISH_TOKEN=""                    # bearer for self-hosted ntfy with auth (LAN-only setups can leave empty)

# Public base URLs (act confirmation deep-links + MA-fetched announcement audio)
AGENT_BASE_URL=""
AGENT_INTERNAL_BASE_URL="http://agent:6002"
```

Notes:

- Cron strings are interpreted in `CURRENT_TIMEZONE` before being converted to UTC for storage, matching the convention used by the user-facing orchestrator.
- `send_signal_message` sends via the `signal-api` container. Recipient precedence: per-notification `to` → `AUTONOMY_BRIEFING_NOTIFY_TO` → `SIGNAL_DEFAULT_RECIPIENT` → `SIGNAL_PHONE_NUMBER` (Note to Self). See `docs/services/agent/tools/general.md` for the one-time QR-link setup.
- `AUTONOMY_HA_NOTIFY_TARGET` may be written as `notify.mobile_app_<device>` or `mobile_app_<device>`; the leading `notify.` is stripped.
- `NTFY_PUBLISH_TOKEN` is the bearer token the agent presents when POSTing to a registered companion-app endpoint. Empty (default) is correct for self-hosted ntfy with no auth — the most common LAN setup. The user's *distributor* (the ntfy Android app) handles its own server-side auth separately. End-to-end protocol + setup checklist in the [companion-app integration](../../../integrations/companion-app.md) walkthrough.
- `AGENT_INTERNAL_BASE_URL` matters for the speaker channel: the compose-network hostname `http://agent:6002` only works when Music Assistant shares the docker network. For deployments where MA runs on a separate host or VLAN, set it to the docker host's LAN-routable address (e.g. `http://10.0.0.134:6002`) so MA can fetch the staged TTS audio.

## REST + WS surface

All endpoints live under `/api/autonomy/*` on port 6002. No auth (same convention as the rest of `/api/*`).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET`  | `/api/autonomy/status` | Engine state, runs in last hour, MQTT health, deferred queue depth, next-due preview, active timezone |
| `POST` | `/api/autonomy/pause` / `/resume` | Runtime kill switch (does not persist across restarts) |
| `GET`  | `/api/autonomy/items` | List every agenda item |
| `POST` | `/api/autonomy/items` | Create a new agenda item |
| `PATCH`| `/api/autonomy/items/{id}` | Partial update; re-validates cron + trigger_spec; raises MQTT refresh |
| `DELETE` | `/api/autonomy/items/{id}` | Delete |
| `GET`  | `/api/autonomy/runs?limit=50&kind=&status=&trigger_source=&include_messages=0` | Filterable run history |
| `GET`  | `/api/autonomy/runs/{run_id}?include_messages=1` | Single run fetch (token stripped) |
| `GET`  | `/api/autonomy/runs/awaiting` | Unfiltered list of `awaiting_confirmation` runs (token stripped) |
| `POST` | `/api/autonomy/runs/{run_id}/confirm` | Body `{approved: bool, token: string}`; constant-time token compare |
| `POST` | `/api/autonomy/trigger/{id}?bypass_quiet=true` | Fire now, bypassing schedule + global rate limit; `bypass_quiet=true` ignores quiet hours too |
| `POST` | `/api/autonomy/webhook/{name}` | HA webhook intake; matches body against `trigger_spec.source='webhook'` items |
| `GET`  | `/api/autonomy/events/summary` | MQTT subscribed topics + webhook-item names + runs-by-source over last 24h + deferred queue depth |

`WS /ws/autonomy/runs` — primes with the last 25 runs, then streams one JSON frame `{type: 'run', run: {...}}` per newly-inserted row.

The agent's `/health` endpoint surfaces an `autonomy` block: `running`, `paused`, `last_dispatch_at`, `mqtt_connected`, `subscribed_topics`, `deferred_runs_pending`, `act_enabled`, `awaiting_confirmation`, `confirmation_timeouts_last_24h`, `speaker_default_device`.

## Tool gating tiers

`autonomy/tool_gating.py` publishes four tiers; each adds on top of the previous:

| Tier | Adds on top of previous |
|---|---|
| `observe` | Read-only HA state (`ha_get_*`, `ha_list_*`), `search_memories`, knowledge tools (`brave_search`, `wolfram_alpha`, `get_weather_forecast`, `search_wikipedia`, `query_multimodal_api`, `fetch`) |
| `notify` | `send_signal_message`, `ha_send_notification` |
| `speak` | (same surface as `notify`; delivery channel differs — handlers route through `SpeakerNotifier`) |
| `act`   | `ha_control_light`, `ha_control_switch`, `ha_control_climate`, `ha_control_media_player`, `ha_activate_scene`, `ha_trigger_script`, `ha_execute_service`, `mass_play_media`, `mass_playback_control` |

A hard deny set (`mcp_qdrant_tools.create_memory` / `delete_memory`, raw HA actuators outside the `act` allow set) is enforced on top of the allow-list for defense in depth. Notifiers themselves are invoked by handlers directly — not via the LLM's tool-calling surface — so the LLM cannot decide whether or where to notify.

`act`-tier items still enforce a per-item `action_allow_list`; the tier grant is *necessary*, the per-item allow-list is *sufficient*.

## Notification channels

Handlers select where a run delivers via the `_notify_channel` field on their result dict (anomaly / watch path) or `deliver.channel` on routine / act items. `engine._build_notifier(channel, to, cfg)` maps the string to a `Notifier`:

| Channel | Notifier | Wire path | Default for |
|---|---|---|---|
| `signal` (alias `email`) | `SignalNotifier` | `send_signal_message` MCP tool → `signal-api` container | briefing handler |
| `ha_push` | `HAPushNotifier` | `ha_send_notification` MCP tool → HA mobile-app integration | anomaly / watch (default) |
| `speaker` | `SpeakerNotifier` | Kokoro TTS render → `mass_play_announcement` (Music Assistant `Players.play_announcement`) | speak-tier items |
| `ntfy` | `NtfyFanoutNotifier` | direct HTTPS POST to every endpoint registered via `/api/push/register` | (opt-in via handler config) |
| _other / unset_ | `NullNotifier` | no-op (logged) | observe-tier and tests |

The `ntfy` channel reads the `push_devices` Postgres table at send-time (not at engine start), so devices registered between dispatches are picked up on the next fire. Payload shape is the [wire envelope](../../../integrations/companion-app.md#wire-format) (`v=1`, `type`, `title`, `body`, optional `session_id`, `severity`) — identical for every endpoint, capped defensively at 3000 chars in `body` to leave headroom under the 4 KB UnifiedPush byte cap. `NtfyFanoutNotifier.send()` returns True if at least one endpoint accepted; per-endpoint failures are logged but do not fail the run.

Optional `cfg` keys for the `ntfy` branch:
- `ntfy_session_id` — populates the wire envelope's `session_id` so a tap on the resulting Android notification deep-links to that chat session in the companion app.
- `ntfy_type` — one of `autonomy_brief` / `anomaly` / `reminder` / `act_confirm` / `ad_hoc` (default). Reserved for future per-type notification-channel splitting on the phone.

The speaker channel parks Kokoro TTS audio in an in-process `AudioStore` (random `secrets.token_urlsafe(16)` token, TTL default 10 min via `AUTONOMY_TTS_AUDIO_TTL_SEC`) and hands MA a URL of the form `{AGENT_BASE_URL}/api/tts/audio/{token}.mp3`. The dashboard's "Speak to device" card on `/playgrounds/tts` and `/autonomy` uses the same pipeline through `POST /api/tts/announce`. Entries are *not* evicted on first read — Music Assistant (and downstream Chromecast / Google Home players) typically fetch each announcement URL more than once (probe for codec/duration, then stream), so the store serves every request until the TTL passes.

## Reactive trigger matching

`autonomy/trigger_match.py` is the single source of truth. An incoming event matches an item's `trigger_spec` iff:

1. `trigger_spec.source == event.source` (strict equality).
2. **MQTT**: `trigger_spec.match.topic` matches the incoming topic, with `+` acting as a single-level wildcard (no `#`). Matching is literal otherwise.
3. **Webhook**: `trigger_spec.match.name == event.name` exactly. The `{name}` URL segment doubles as a shared secret (home-lab trust model, no auth).
4. `trigger_spec.match.payload` is a **recursive key subset** — every key listed must exist in the incoming payload with an equal value; payloads may contain additional keys without losing the match.

If the trigger_spec resolves to a non-match, the engine still inserts a `status='skipped_trigger_mismatch'` run row for auditability.

## Quiet hours

`config.quiet_hours = { start, end, policy }`, timezone-aware via `CURRENT_TIMEZONE`. Cross-midnight windows (`22:00` → `07:00`) work. When quiet hours suppress a run:

- `policy='defer'` — insert `status='scheduled'` + `scheduled_for = next_end_of_quiet`. The dispatcher sweeps every scheduled row that has become due on each tick and fires it with `trigger_source='deferred'`. Claiming uses `DELETE ... RETURNING` so there's no lingering `in_flight` state — the subsequent fire creates a fresh `autonomy_runs` row.
- `policy='drop'` — insert `status='skipped_quiet_hours'` and walk away.

`POST /api/autonomy/trigger/{id}?bypass_quiet=true` skips this gate.

If `config.quiet_hours` is omitted, the engine falls back to the `AUTONOMY_DEFAULT_QUIET_*` env. Malformed specs fail open (treated as no quiet hours) rather than silently blocking the item.

## Event rate limiting

`config.event_rate_limit = "N/sec" | "N/min" | "N/hr"`. Applied **only** to reactive triggers (`mqtt` / `webhook`). Implementation is an in-memory leaky bucket keyed on `item_id`; overflow is recorded as `status='skipped_rate_limit'`. The global hourly `AUTONOMY_MAX_RUNS_PER_HOUR` cap still applies on top.

## act tier — plan / validate / execute

When `AUTONOMY_ACT_ENABLED=true`, an `act` agenda item runs in three phases:

1. **Plan.** The handler spins an `AutonomousTurn` at the `observe` tier, so the LLM can read state but cannot actuate from inside the turn. The prompt carries the JSON schema of every allow-listed actuator tool (name, description, `parameters`) so the planner uses real parameter names rather than guessing. The system prompt asks for strict JSON:

   ```json
   {"steps": [{"tool": "ha_control_light", "args": {...}, "rationale": "..."}],
    "reasoning": "..."}
   ```

2. **Validate.** The engine iterates every proposed step and annotates an `action_audit` list: `pending` when the tool is in the item's `action_allow_list`, `skipped_not_allowed` otherwise, `malformed` for missing fields *or* args containing unresolved template placeholders (`<...>`, `{{...}}`, sentinel tokens like `"previous_step"` / `"placeholder"` / `"todo"`). Placeholders imply the LLM expected a follow-up resolution pass that does not exist, so the engine rejects those steps rather than handing raw templates to the MCP tool. The audit is **never dropped** — every proposed call is persisted for review.

3. **Execute.** If `require_confirmation=false`, the engine runs every `pending` step inline via the MCP manager and flips each to `executed` or `error`. If `require_confirmation=true` (default), the engine persists `status='awaiting_confirmation'` with a random `confirmation_token`, the full `action_audit`, and sends a notification containing a deep-link of the form `{AGENT_BASE_URL}/autonomy?confirm={run_id}&token={token}`.

The confirm endpoint (`POST /api/autonomy/runs/{run_id}/confirm` with body `{approved, token}`) validates the token (constant-time compare) and either cancels the run (`confirmation_denied`) or hands the stored audit back to `act.execute_approved`. The confirmation-timeout sweep runs on each dispatcher tick — runs past their `confirmation_timeout_sec` (default 300s) transition atomically to `confirmation_timeout`.

`awaiting_confirmation` runs **block the item's schedule** — the engine does not advance `next_fire_at` until the run is resolved. Cron-driven `act` items shouldn't fire a second plan while the first is parked.

Silent tool failures are promoted to errors: MCP tools that swallow HA errors and return `{success: false}` or an `error` key are detected by the executor and stamped `outcome='error'` rather than `executed`.

Example `act` config:

```json
{
  "kind": "act",
  "autonomy_level": "act",
  "config": {
    "prompt": "Turn on the reading lamp in the living room.",
    "action_allow_list": ["ha_control_light", "ha_activate_scene"],
    "require_confirmation": true,
    "confirmation_timeout_sec": 300,
    "deliver": {"channel": "signal", "to": "+15551234567"}
  }
}
```

## Config shapes

### `reminder`

```json
{
  "kind": "reminder",
  "name": "Laundry 17:00",
  "schedule_cron": "0 17 * * *",
  "autonomy_level": "notify",
  "config": {
    "title": "Laundry",
    "body": "Start a load — weather window opens in 2h.",
    "channel": "ha_push",
    "to": "mobile_app_pixel_8",
    "one_shot": false,
    "personalize": true,
    "quiet_hours": { "start": "22:00", "end": "07:00", "policy": "defer" }
  }
}
```

### `watch` (MQTT-triggered)

```json
{
  "kind": "watch",
  "name": "Front door after midnight",
  "trigger_spec": {
    "source": "mqtt",
    "match": { "topic": "home/door/front/state", "payload": { "state": "open" } }
  },
  "autonomy_level": "notify",
  "config": {
    "body_template": "Front door opened at {_ts}",
    "channel": "ha_push",
    "severity": "warn",
    "quiet_hours": { "start": "00:00", "end": "06:00", "policy": "defer" },
    "event_rate_limit": "3/min",
    "condition": { "entity_id": "binary_sensor.front_door", "state": "on", "min_duration_sec": 0 }
  }
}
```

### `routine`

```json
{
  "kind": "routine",
  "name": "Sunday weekly recap",
  "schedule_cron": "0 18 * * 0",
  "autonomy_level": "notify",
  "config": {
    "prompt": "Summarize this week's memories and suggest three focus areas for next week.",
    "tools_override": ["search_memories", "ha_send_notification"],
    "deliver": { "channel": "signal", "to": "+15551234567" }
  }
}
```

`tools_override` **must be a subset** of the current `autonomy_level` tier's allow-list; the engine raises on mismatch (surfaced as a failed run with a clear error).

## Guardrails

Non-negotiable engine behavior:

1. **Kill switch** — `AUTONOMY_ENABLED=false` disables dispatch at startup. `POST /api/autonomy/pause` flips a runtime flag without restart.
2. **Global hourly rate limit** — `AUTONOMY_MAX_RUNS_PER_HOUR` (default 20). Scheduled dispatches above the cap are recorded as `status='rate_limited'` and not executed. Manual `/trigger` calls bypass this cap intentionally (operator override).
3. **Per-signature cooldown** — for `anomaly_sweep`, `watch`, and `watch_llm`, the engine hashes a dedup signature and refuses to re-notify within `cooldown_min` (per-item, defaults to `AUTONOMY_ANOMALY_COOLDOWN_MIN`) unless severity escalates (`low` → `med` → `high`). For triggers carrying a normalized `sensor_event` (camera/face/etc.) the signature is derived deterministically from `{domain}:{kind}:{zone}:{subject}` so repeat detections hash to the same key regardless of how the LLM phrases its output. The cooldown skip still writes a run row.
4. **Hard tool allow-list** — enforced at turn construction; not trusted to LLM judgment.
5. **Session isolation** — each autonomous turn uses a fresh `AgentOrchestrator` with its own `session_id`; never appended to user conversation state.
6. **Per-turn timeout** — `AUTONOMY_TURN_TIMEOUT_SEC` (default 60). Overages are aborted and logged as `status='error'` with a timeout message.
7. **System prompt discipline** — autonomous prompts explicitly instruct: "You are running autonomously. Do not ask questions. Complete your assessment and exit. Output must match the required format."
8. **All outcomes persisted** — `ok`, `error`, `skipped_*`, `rate_limited`, `awaiting_confirmation`, `confirmation_*` all produce `autonomy_runs` rows for audit.
9. **Confirmation tokens never leak** — tokens never appear in `GET /api/autonomy/runs` responses or the WS feed. The notification channel is the only path that carries the token, so the confirmation channel must be one you trust to reach you.

## Dashboard

`/autonomy` (agent port 6002):

- **Header strip** — engine status, pause / resume, "New item" button.
- **Engine stats** — runs / hr with cap, deferred queue depth, MQTT connection + subscribed topic count, live WS status, awaiting-confirmation count.
- **Pending confirmations** — visible whenever one or more `act` runs are parked. Each row shows the proposed `action_audit` (tool + rationale per step) and offers approve/deny buttons. Deep-links of the form `/autonomy?confirm={run_id}&token={token}` (what the confirmation notification points at) scroll the card into view and use the URL token; query params are cleared after a response so a reload doesn't re-trigger the flow.
- **Agenda items** — table with per-row Run / Edit / Delete actions, an inline enabled toggle, and trigger description.
- **Live run feed** — WS-backed, last 50 runs.
- **Reactive sources** — currently subscribed MQTT topics + configured webhook paths + runs-by-source counts over the last 24h.
- **Run history** — filterable by kind / status / trigger_source.

The agenda editor (`AgendaForm.svelte`) renders kind-scoped fields and a live JSON preview of the request body.

## Memory integration

The anomaly handler runs `search_memories(query='household normal routine', limit=5)` as part of its state bundle, so household-routine context informs the LLM's judgment.

Memory tier mechanics (L1 in-session → L2 episodic → L3 consolidated → L4 promoted facts) and the nightly `memory_review` consolidation pass are documented in [memory/README.md](memory/README.md).

## Verification

### Unit (in-container)

```bash
docker compose exec agent pytest services/agent/tests/
```

Pure-logic suites live next to the modules they cover:

- `test_trigger_match.py` — MQTT `+` wildcard edges, payload subset, non-match auditing.
- `test_quiet_hours.py` — cross-midnight, tz conversion, malformed input fails open.
- `test_event_rate_limit.py` — burst + steady-state, `N/sec|min|hr` parsing.
- `test_reminder_handler.py`, `test_watch_handler.py`, `test_routine_handler.py`, `test_act_handler.py` — handler return contract + `one_shot` disable + `tools_override` subset enforcement + plan/validate/execute pipeline.
- `test_mqtt_listener.py` — subscription diffing, reconnect backoff.
- `test_autonomy_api.py` — CRUD round-trip, webhook fan-out, confirm endpoint.
- `test_deferred_runs.py` — quiet-hours `defer` → sweep → fire.

### Live (full stack up)

```bash
# Engine status
curl http://localhost:6002/api/autonomy/status

# List agenda items
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

Reactive-source hand checks once the flags are on:

1. Create a reminder via the dashboard; confirm `next_fire_at` in `/items`.
2. Create an MQTT watch on a topic you can publish to; `mosquitto_pub -h localhost -t <topic> -m '{...}'`; confirm a run row streams into the live feed.
3. Create a webhook watch; `curl -X POST -d '{"state":"open"}' http://localhost:6002/api/autonomy/webhook/<name>`; confirm matched + fired.
4. Build a quiet-hours window that covers now with `policy='defer'`; trigger the item; confirm `status='scheduled'`; walk the clock forward and confirm it fires.
5. Trigger repeatedly past the per-item rate limit and confirm `status='skipped_rate_limit'`.
6. Edit an MQTT item's topic via `PATCH`; confirm the listener diff-resubscribes (inspect `/events/summary`).
7. Disconnect the broker and confirm the listener reconnects with exponential backoff capped at `AUTONOMY_MQTT_RECONNECT_MAX_SEC`.

## Troubleshooting

- **WS `/ws/autonomy/runs` never connects** — check the Postgres trigger exists (`\d+ autonomy_runs` in psql). `ensure_schema()` creates it idempotently; if you nuked the trigger manually, restart the agent.
- **MQTT items created but `subscribed_topics` is empty** — confirm `AUTONOMY_MQTT_ENABLED=true`, the broker is reachable from the agent container (`MQTT_BROKER` / `MQTT_PORT`), and the item is `enabled=true`.
- **Deferred runs stuck pending** — the sweep runs on each dispatcher tick; if `AUTONOMY_ENABLED=false` or the engine is paused, scheduled rows stay put until the engine resumes.
- **`tools_override` blocks a routine** — the override must be a strict subset of the current tier's allow-list. Raise the autonomy_level or drop the over-reaching tool from the override.
- **Speaker channel plays the MA pre-announce chime then silence** — Music Assistant can't fetch the audio URL. Set `AGENT_INTERNAL_BASE_URL` to a host:port that MA can reach (typically the docker host's LAN IP, not `agent:6002`).
- **`act` item creates an error row immediately** — `AUTONOMY_ACT_ENABLED=false` short-circuits the handler. Flip to `true` and re-trigger.
- **`act` plan executes nothing** — every step probably hashed to `skipped_not_allowed`. Check the item's `action_allow_list` includes the tools the planner picked.
