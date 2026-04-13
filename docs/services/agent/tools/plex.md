# MCP Server: Plex (`mcp_plex_tools`)

Server-side reference for the Plex MCP module — tool inventory, module
config, internals, and server-level troubleshooting. For the user-facing
setup walkthrough (TV requirements, Plex token acquisition, authoring
`PLEX_CLIENT_HA_MAP`, verification steps), see
[Media Control](../../../integrations/media-control.md). Media control as a topic (Plex today,
Music Assistant later, HA transport throughout) lives there; this doc
documents only the Plex MCP module itself.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_plex_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_plex_tools` |
| Transport | MCP stdio |
| Server name | `havencore-plex` |
| Plex client library | `plexapi` (sync — calls are offloaded via `asyncio.to_thread`) |
| HA client | Minimal aiohttp REST (`_HAServiceClient` in `plex_client.py`) — used only for wake/launch |
| Tool count | 5 |

The server replaced the old DLNA media-scanner that lived inside
`mcp_homeassistant_tools`. That module shrank from ~1430 lines to ~170;
everything library / playback is now here, and HA is involved only for
transport control and optional wake/launch.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `plex_search(query, media_type?, limit?)` | Library search across all sections. Returns items with `rating_key`, `title`, `year`, `type`, `summary` (trimmed to ~300 chars). `media_type` narrows by Plex section type (`movie`, `show`, `episode`, `track`, `album`, `artist`). |
| `plex_list_recent(media_type?, limit?)` | Recently added across sections. `media_type` is `movie`, `show`, or `music`. Sorted by `addedAt` descending. |
| `plex_list_on_deck(limit?)` | The server-side continue-watching queue (`library.onDeck()`). |
| `plex_list_clients()` | Player-capable devices discovered via `MyPlexAccount.devices()`. Returns `name`, `product`, `platform`, `provides`. Use the `name` as `client_name` for `plex_play`. |
| `plex_play(rating_key, client_name)` | Play a specific item on a specific client. Partial `client_name` matches are accepted (substring, case-insensitive). If a `PLEX_CLIENT_HA_MAP` mapping exists, the TV is woken and Plex is launched first. |

## Configuration

Env vars (loaded in `mcp_server.py` via `selene_agent.utils.config`):

| Var | Required | Purpose |
|-----|----------|---------|
| `PLEX_URL` | yes | Base URL of the Plex Media Server (e.g. `http://10.0.50.110:32400`). No trailing slash. |
| `PLEX_TOKEN` | yes | `X-Plex-Token`. |
| `PLEX_CLIENT_HA_MAP` | optional | JSON object keyed by Plex client name. Enables wake/launch before `plex_play`. Authoring walkthrough in [Media Control](../../../integrations/media-control.md#3-optional-configure-wakelaunch-mapping). |
| `HAOS_URL` / `HAOS_TOKEN` | conditional | Required only when `PLEX_CLIENT_HA_MAP` is set, since wake/launch talks to HA. Shared with the HA MCP server. |

If `PLEX_URL` or `PLEX_TOKEN` are unset (or the token is literally
`NO_PLEX_TOKEN_CONFIGURED`), the server starts in **stub mode**: every tool
returns `{"error": "PLEX_URL / PLEX_TOKEN not configured", "hint": …}`.

`PLEX_CLIENT_HA_MAP` fields per client (`plex_client.py`'s `_ensure_ready`):

| Field | Default | Purpose |
|-------|---------|---------|
| `state_entity` | — | HA `media_player.*` entity read for `on/off/app_id`. `media_player.turn_on` is called against it when the TV is off. |
| `adb_entity` | — | HA `androidtv` integration entity that accepts `androidtv.adb_command`. |
| `plex_app_id` | `com.plexapp.android` | Package ID used to decide whether Plex is already foregrounded. |
| `launch_command` | `monkey -p <plex_app_id> -c android.intent.category.LAUNCHER 1` | Full ADB command to launch Plex. Override only if the default doesn't work. |

Invalid JSON in `PLEX_CLIENT_HA_MAP` is logged and falls back to no-wake
behavior (the server starts fine; playback just won't pre-warm the TV).

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "plex",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_plex_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **Discovery is via cloud relay, not GDM.** `server.clients()` relies on
  Plex's GDM multicast and doesn't cross subnets. `MyPlexAccount.devices()`
  + filter on `"player" in provides` works regardless of network topology,
  but requires the TV's Plex app to be signed into the same Plex account
  as the server.
- **`plexapi` is synchronous.** All library / account calls are wrapped in
  `asyncio.to_thread` (`_do_search`, `_do_list_recent`, etc.) so the
  stdio event loop stays responsive.
- **`PlexServer` + `MyPlexAccount` are lazy-initialized and cached.** The
  first tool call pays the auth round-trip; subsequent calls reuse the
  connection.
- **Wake/launch sequence in `_ensure_ready`:**
  1. Read `state_entity`'s state. If `off` / `unavailable` / `unknown`,
     call `media_player.turn_on` and sleep 3 seconds.
  2. Re-read the entity; check `app_id` / `source`. If it isn't
     `plex_app_id` and `adb_entity` is set, fire
     `androidtv.adb_command` with `launch_command` and sleep 3 seconds.
  3. Proceed to `playMedia()`.

  A short human-readable note (`"woke <entity>; launched Plex via <adb>"`)
  comes back in the tool response under `readiness` when any action was
  taken.
- **HA is best-effort.** If the HA call fails, exceptions propagate and
  the tool returns an error — the server does not silently skip
  wake/launch.
- **`_resolve_device` does partial, casefolded matches.** First exact name
  match wins; otherwise the first substring match wins. Clients without
  `"player" in provides` are filtered out.

## Troubleshooting

User-level symptoms ("nothing played on the TV", empty client list,
mapping not working) are documented in
[Media Control → Troubleshooting](../../../integrations/media-control.md#troubleshooting). This
section covers server-internal symptoms only.

### Server starts but all tools return stub-mode error

`PLEX_URL` / `PLEX_TOKEN` aren't set, or `PLEX_TOKEN` is literally
`NO_PLEX_TOKEN_CONFIGURED`. Set them in `.env` and
`docker compose down && up -d` (env changes require a full recycle).

### `{"error": "<plexapi ExceptionName>: ..."}` in tool responses

`_dispatch` catches exceptions and returns them as structured errors.
Common ones:

- `Unauthorized` — wrong `PLEX_TOKEN`.
- `NotFound` — `rating_key` doesn't exist (item removed, or you passed a
  key from a different server).
- `ConnectionRefused` / `Timeout` — agent can't reach `PLEX_URL` from
  inside the Docker network. Verify with:
  ```bash
  docker compose exec agent curl -I $PLEX_URL
  ```

### `plex_play` returns `"played": false` with `available_clients`

`_resolve_device` didn't match `client_name`. Partial substring match is
allowed (casefolded) but a one-letter typo will still fail. The response
includes the full list of player-capable device names — pass one of them
verbatim. (User-facing discovery guidance: see Media-Control.)

### `readiness` note reports a wake action but playback still doesn't start

The HA side thinks it succeeded but the TV didn't actually come up in
time. `_ensure_ready` waits a fixed 3 seconds after each HA call; a slow
TV might need longer. Options:

- Author a custom `launch_command` that itself includes an `adb shell
  input keyevent KEYCODE_WAKEUP` and a longer internal delay.
- Warm the TV via a separate `ha_control_media_player(action="turn_on")`
  call before `plex_play`.

## Related files

- `services/agent/selene_agent/modules/mcp_plex_tools/plex_client.py` —
  async facade over `plexapi` + `_HAServiceClient` for wake/launch.
- `services/agent/selene_agent/modules/mcp_plex_tools/mcp_server.py` —
  tool registrations + dispatch.
- `services/agent/selene_agent/utils/config.py` — `PLEX_URL`, `PLEX_TOKEN`,
  `PLEX_CLIENT_HA_MAP` passthrough.

## See also

- [Media Control](../../../integrations/media-control.md) — topic doc: TV requirements, Plex
  token acquisition, mapping authoring walkthrough, user-facing
  troubleshooting, forward-looking Music Assistant plans.
- [MCP Home Assistant](home-assistant.md) — `ha_control_media_player`
  (transport) lives there.
