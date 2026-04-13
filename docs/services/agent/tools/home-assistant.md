# MCP Server: Home Assistant (`mcp_homeassistant_tools`)

Reference doc for the Home Assistant MCP server bundled with the agent.
This is the server-side view — tool inventory, transport/dependencies,
configuration, and troubleshooting. End-user setup (token, URL, voice
examples) lives in [Home Assistant Integration](../../../integrations/home-assistant.md).
TV playback specifics live in [Media Control](../../../integrations/media-control.md).

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_homeassistant_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_homeassistant_tools` |
| Transport | MCP stdio (spawned by the agent's `MCPClientManager`) |
| Server name | `havencore-homeassistant` |
| HA REST client | `aiohttp`-based `HomeAssistantClient` (replaces the older blocking `homeassistant_api.Client`); also owns the short-lived WS client used for registry lookups |
| Media controller | REST-only `ha_media_controller.MediaController` (transport / volume / power on any `media_player` entity) |
| Tool count | 19 |

The module registers a single MCP stdio server. On startup it constructs a
`HomeAssistantClient` (REST + on-demand WS) and a REST `MediaController`.
If initialization fails the server still starts and every tool returns
`"Home Assistant unavailable: <reason>"` so the agent can surface the error
in chat instead of crashing.

When `HAOS_URL` or `HAOS_TOKEN` are unset, the server runs in **TEST
MODE**: tools return mock data so the agent can boot even without an HA
instance. Tests/mocks are intentional — don't treat them as real results.

## Tool inventory

Tools are grouped here by purpose; names match the MCP registrations.

### Generic REST tools

| Tool | Purpose |
|------|---------|
| `ha_get_domain_entity_states(domain)` | `GET /api/states`, filtered by domain prefix. For `media_player` this hands off to the media controller (which also reads `/api/states`) so you also get playback state. |
| `ha_get_domain_services(domain)` | `GET /api/services`, narrowed to one domain. Use this to discover which `notify.*` services exist on a given HA instance. |
| `ha_execute_service(entity_id, service, service_data?)` | Generic escape hatch: `POST /api/services/<domain>/<service>`. Domain is inferred from the entity ID. `service_data` is forwarded as a JSON object. |

### Device / automation control (Phase A)

| Tool | What it calls | Notes |
|------|---------------|-------|
| `ha_control_light(entity_id, state, brightness_pct?, color_name?, color_temp_kelvin?)` | `light.turn_on` / `turn_off` / `toggle` | Extras are only applied when `state=on`. |
| `ha_control_climate(entity_id, temperature?, hvac_mode?, fan_mode?)` | `climate.set_hvac_mode`, `climate.set_temperature`, `climate.set_fan_mode` | Multi-call: issues one service per non-null argument, in that order. |
| `ha_activate_scene(scene_entity)` | `scene.turn_on` | |
| `ha_trigger_script(script_entity, variables?)` | `script.turn_on` | `variables` are passed through as service kwargs. |
| `ha_trigger_automation(automation_entity)` | `automation.trigger` | Manually fires. Not the same as enabling. |
| `ha_toggle_automation(entity_id, enabled)` | `automation.turn_on` / `turn_off` | Enable/disable gating. Does not fire the automation. |
| `ha_send_notification(service, message, title?, target?)` | `notify.<service>` | Entity-less call — goes through `HomeAssistantClient.execute_service(entity_id=None, domain='notify', ...)`. |

### Registry + presence (Phase B)

These use the Home Assistant WebSocket API under the hood via
`HomeAssistantClient._ws_call` — HA's `config/*_registry/list` endpoints
are WS-only, so a short-lived WS connection is opened per call (auth
handshake → one command → close). Errors from the WS call surface as an
`error` key in the tool payload.

| Tool | WS / REST call | Notes |
|------|----------------|-------|
| `ha_list_areas()` | WS `config/area_registry/list` | Returns `[{area_id, name, aliases}]`. |
| `ha_get_entities_in_area(area, domains?)` | WS `config/area_registry/list` + `entity_registry/list` + `device_registry/list` | Resolves `area` by `area_id`, name, or alias (case-insensitive). Entities inherit their device's area when their own `area_id` is null. Disabled / hidden entities are filtered. Grouped by domain. |
| `ha_get_presence()` | REST `GET /api/states` | Buckets `person.*` and `device_tracker.*` into two lists with their state + `friendly_name`. |

### Timer / template / history / calendar (Phase C)

| Tool | HA endpoint | Notes |
|------|-------------|-------|
| `ha_set_timer(entity_id, duration?)` | `timer.start` | `duration` uses HA's `HH:MM:SS` format (e.g. `0:05:00`). Omit to use the timer helper's configured default. |
| `ha_cancel_timer(entity_id)` | `timer.cancel` | |
| `ha_evaluate_template(template)` | `POST /api/template` (text response) | Server-side Jinja2 render. Uses `_post_text` on the client since HA returns raw text here, not JSON. |
| `ha_get_entity_history(entity_id, hours?)` | `GET /api/history/period/<start>?filter_entity_id=<id>&minimal_response` | `hours` clamped to `[1, 168]` (one week). Dense series are downsampled to ~200 points. Returns `{entity_id, hours, total_points, points[], sampled?}`. |
| `ha_get_calendar_events(calendar_entity, days?)` | `GET /api/calendars/<entity>?start=…&end=…` | `days` clamped to `[1, 31]`. Normalizes `start`/`end` from `{dateTime, date}` dicts to flat values. |

### Media player transport

| Tool | Purpose |
|------|---------|
| `ha_control_media_player(action, device?, value?)` | Play/pause/stop/seek/volume/mute/power/source on any HA `media_player` entity. Plex / library search **is not here** — see [Media Control](../../../integrations/media-control.md). |

Value units by action:

- `volume_set` → integer 0–100 (percent)
- `seek` → integer seconds from start
- `select_source` → source name string (e.g. `"HDMI 1"`)
- `shuffle` / `repeat` → boolean or `'off'` / `'all'` / `'one'`

## Configuration

Server-relevant env vars (loaded via `selene_agent.utils.config` and
`shared/configs/shared_config.py`):

| Var | Required | Purpose |
|-----|----------|---------|
| `HAOS_URL` | yes | Base HA URL; may or may not include `/api`. Both `https://host:8123` and `https://host:8123/api` work. |
| `HAOS_TOKEN` | yes | Long-lived access token (`Authorization: Bearer …`). |
| `HAOS_USE_SSL` | no | Currently unused at runtime; SSL is derived from the URL scheme. |

Derived automatically (no env var):

- **REST base**: `HAOS_URL` with any trailing `/api` stripped.
- **WebSocket URL**: `ws://<host>/api/websocket` (or `wss://` when `HAOS_URL`
  is `https`). See `HA_WS_URL` in `selene_agent/utils/config.py`.

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "homeassistant",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_homeassistant_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **REST client is aiohttp-backed.** The older blocking
  `homeassistant_api.Client` was replaced to unblock the orchestrator event
  loop under tool calls. All methods open a short-lived session with a 15s
  timeout.
- **`execute_service` pre-flights entity existence.** HA's
  `POST /api/services/<domain>/<service>` returns HTTP 200 with an empty
  body `[]` for *both* non-existent entities and legitimate no-ops (e.g.
  `turn_off` on an already-off light), so the response alone can't
  distinguish the two. Before the service POST, the client does a
  `GET /api/states/<entity_id>`; a 404 raises `EntityNotFoundError` and
  each wrapper (`ha_control_light`, `ha_execute_service`, `ha_control_climate`,
  `ha_activate_scene`, `ha_trigger_script`, `ha_trigger_automation`,
  `ha_toggle_automation`, `ha_set_timer`, `ha_cancel_timer`) turns that into
  a `FAILED: <kind> '<entity_id>' does not exist in Home Assistant. ...`
  message that tells the LLM to call `ha_get_domain_entity_states` /
  `ha_get_entities_in_area` before retrying. This costs one extra round-trip
  per service call but closes a silent-failure path where guessed entity
  IDs looked like success. `ha_send_notification` skips the check (it
  calls with `entity_id=None`).
- **`execute_service` takes an optional `domain`.** Notify / mobile /
  script-less services pass `entity_id=None` and an explicit `domain`
  (e.g. `domain="notify"`). Other tools let the client derive the domain
  from the entity ID.
- **Registry WS calls are short-lived.** Each `_ws_call` opens a new
  connection, does the HA auth handshake, sends one command, and closes.
  `ha_get_entities_in_area` therefore costs three connections (area +
  entity + device registry). Fine in practice — these tools run
  interactively, not in a hot loop.
- **Area lookup is tolerant.** `ha_get_entities_in_area` accepts the raw
  `area_id`, the display name, or any alias — all case-insensitive. If no
  area matches, it returns `{"error": ..., "known_areas": [...]}` to help
  the LLM recover.
- **History is truncated, not unbounded.** Above 200 points the response is
  stride-sampled. This matters if you're asking about a very bursty
  sensor — use a shorter `hours` window to see more detail.

## Troubleshooting

### Every HA tool returns "Home Assistant unavailable: …"

Initialization failed. Check `docker compose logs agent` for the
`Failed to initialize HA clients` line. Common causes:

- Bad `HAOS_URL` (wrong host / port / scheme). Must be reachable **from
  inside the agent container**, not just the host.
- `HAOS_TOKEN` missing or revoked.
- Network boundary (the agent container can't reach the HA VLAN).

### A tool returns `FAILED: <kind> '...' does not exist in Home Assistant`

The pre-flight `GET /api/states/<entity_id>` in `HomeAssistantClient.execute_service`
returned 404. The service POST was **not** issued, so nothing changed in HA.
This typically means the LLM guessed an entity ID that doesn't exist; the
message instructs it to call `ha_get_domain_entity_states` or
`ha_get_entities_in_area` first and retry. If the entity *should* exist,
verify the exact ID in HA's **Developer Tools → States** — naming is
case-sensitive and exact (`light.kitchen_light_1`, not `light.kitchen`).

### Tools return `{"error": "..."}` with a stack-trace-like string

The HA server responded but the call failed. Check HA's logs for the
corresponding REST / WS request. Common causes:

- Entity ID typos (`light.livingroom` vs `light.living_room`).
- Service doesn't exist on that HA instance (e.g. `notify.mobile_app_pixel`
  when the integration is named differently). Use `ha_get_domain_services`
  with `domain="notify"` to enumerate.
- Calendar entity doesn't exist or isn't exposed via the `/api/calendars/…`
  endpoint.

### `ha_list_areas` / `ha_get_entities_in_area` return `{"error": "WS registry … call failed"}`

The WebSocket call to HA failed. Typical causes: HA is down, `HAOS_URL`
isn't reachable from the agent container, or `HAOS_TOKEN` is wrong (the
auth handshake will raise `HA WS auth failed: …`). Check `docker compose
logs agent` for the specific error and restart after fixing HA:

```bash
docker compose restart agent
```

### `ha_evaluate_template` returns the literal template string

You're in TEST MODE — `HAOS_URL` / `HAOS_TOKEN` are unset, so the server
short-circuits and echoes the input. Set the env vars and restart.

### Timer does nothing

`timer.*` helpers must be defined in HA's configuration (they are not a
dynamic domain). Discover existing timers via:

```text
ha_get_domain_entity_states(domain="timer")
```

If the list is empty, add a `timer:` block to HA's `configuration.yaml`.

## Related files

- `services/agent/selene_agent/modules/mcp_homeassistant_tools/mcp_server.py`
  — tool registrations + dispatch, aiohttp REST client.
- `services/agent/selene_agent/modules/mcp_homeassistant_tools/ha_media_controller.py`
  — REST-only `media_player` transport controller (play/pause/volume/power
  on any HA media_player entity).
- `services/agent/selene_agent/utils/config.py` — agent-side config passthrough,
  including `HA_WS_URL` derivation.
- `shared/configs/shared_config.py` — canonical env-var surface.

## See also

- [Home Assistant Integration](../../../integrations/home-assistant.md) — end-user setup, HA token generation, voice examples.
- [Media Control](../../../integrations/media-control.md) — Plex + HA split for TV playback.
- [Tool Development](development.md) — adding or modifying MCP tools.
