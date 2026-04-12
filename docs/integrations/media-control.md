# Media Control (Plex + Home Assistant)

How HavenCore finds and plays media on your TVs and speakers. Today this
means Plex + Home Assistant; Music Assistant is on the roadmap for
audio-only devices (soundbars, Google Home, Wyoming satellites). This
doc is the topic-level view — server-side references for each MCP
module live in [MCP Plex](../services/agent/tools/plex.md) and
[MCP Home Assistant](../services/agent/tools/home-assistant.md).

## How it works

Media control is split across multiple systems, each owning a specific
concern:

| System | Responsibility | Status |
|--------|----------------|--------|
| **Plex Media Server** | Video library search, metadata, recently-added / on-deck, and "play this item on that device" on Plex-capable clients. | Shipping. |
| **Home Assistant** | Turning the TV on, launching the Plex app via ADB when needed, and generic transport control (pause / volume / etc.) on any `media_player` entity. | Shipping. |
| **Music Assistant** | Audio-first library + playback across soundbars, smart speakers, Wyoming satellites, and multi-room groups. | Planned — not yet integrated. |

Only Plex clients (TVs running the Plex app with "Advertise as Player"
enabled) can be targets for library-initiated playback today. Audio-only
devices can still be controlled for transport / volume via
`ha_control_media_player` on their HA entity, but they won't show up in
`plex_list_clients` and can't be a target for `plex_play`.

### Tools exposed to the LLM

From `mcp_plex_tools` — see [MCP Plex](../services/agent/tools/plex.md) for full signatures
and server internals:

| Tool | Purpose |
|------|---------|
| `plex_search` | Title / keyword search across libraries. |
| `plex_list_recent` | "What's new on Plex?" |
| `plex_list_on_deck` | Continue-watching queue. |
| `plex_list_clients` | Player-capable Plex devices (names the LLM can pass to `plex_play`). |
| `plex_play` | Start playback on a named client. Runs HA wake/launch first if a mapping is configured. |

From `mcp_homeassistant_tools` — see
[MCP Home Assistant](../services/agent/tools/home-assistant.md):

| Tool | Purpose |
|------|---------|
| `ha_control_media_player` | Pause / resume / stop / seek / volume / mute / select_source / turn_on / turn_off on any HA `media_player` entity. |

### The cold-start problem

Plex's `playMedia()` tells an already-running Plex client to play
something. It cannot turn the TV on, and it cannot bring the Plex app to
the foreground if another app is running. If the TV is off or Plex is
backgrounded, `plex_play` succeeds on the Plex side but nothing happens
on the TV.

HavenCore solves this with an optional `PLEX_CLIENT_HA_MAP` config.
When a mapping exists for the target client, `plex_play` wakes the TV
via HA and launches the Plex app via ADB before playback. A short note
describing any action taken comes back in the tool response under
`readiness`. For the exact sequence and timing, see
[MCP Plex → Internals](../services/agent/tools/plex.md#internals-worth-knowing).

## Requirements

For `plex_play` to work on a given TV:

1. **Android TV (or equivalent)** running the Plex app. The wake/launch
   fallback uses ADB, which in practice means Android TV / Google TV / Fire TV
   devices. Non-Android smart TVs (Vizio SmartCast, LG webOS, Samsung Tizen)
   will not work with the automatic launch path.
2. **Plex app with "Advertise as Player" enabled.** Without this, the device
   will not show up in `plex_list_clients`. Some TV apps hide this option; if
   it isn't there, the device is out of scope.
3. **ADB debugging enabled on the TV** (only required if you want automatic
   wake/launch). Settings → Device preferences → Developer options → USB /
   Network debugging. You'll also need to pair HA's `androidtv` integration
   with the TV.
4. **HA `androidtv` integration** configured against the TV — this produces
   the `media_player.<name>_adb` entity that the mapping uses for
   `adb_command`.

TVs without ADB (or without "Advertise as Player") can still be used for
transport control via `ha_control_media_player`, but not for automatic
playback initiation.

## Setup

### 1. Get a Plex token

From the Plex web UI, open any media item, click the ⋯ menu → **Get Info** →
**View XML**. The URL will end with `?X-Plex-Token=...`. Copy the token.

Alternative: Plex web UI → Settings → Account → "Authorized devices" — or sign
in via `plexapi.myplex.MyPlexAccount(username, password).authenticationToken`.

### 2. Configure `.env`

```bash
PLEX_URL="http://10.0.50.110:32400"   # LAN URL, no trailing slash
PLEX_TOKEN="xxxxxxxxxxxxxxxxxxxx"
```

Confirm the agent can reach `PLEX_URL` from inside its container — the agent
service runs on the Docker network, not the host.

### 3. (Optional) Configure wake/launch mapping

This step is only needed if your TV sleeps between uses and you want Selene
to be able to wake it. Skip if your Plex app is always running.

First, identify the Plex client name and the two HA entities:

```bash
# Plex client name — run inside the agent container, or any box with plexapi
python -c "
from plexapi.myplex import MyPlexAccount
a = MyPlexAccount(token='YOUR_TOKEN')
for d in a.devices():
    if 'player' in (d.provides or ''):
        print(d.name, '|', d.product, '|', d.platform)
"
```

Then find, in Home Assistant → Developer Tools → States:

- The TV's main `media_player.*` entity (used for `turn_on` and to read the
  current `app_id`). This is created by whatever HA integration manages the
  TV — for Sony Bravia that's the Bravia integration; for a Shield/Chromecast
  it's that integration.
- The TV's `media_player.*_adb` entity from the `androidtv` integration.
  This is the one that accepts `androidtv.adb_command`.

Set `PLEX_CLIENT_HA_MAP` as a JSON object keyed by **Plex client name**:

```bash
PLEX_CLIENT_HA_MAP='{
  "BRAVIA 4K VH21": {
    "state_entity": "media_player.living_room_tv_bravia_4k_vh21",
    "adb_entity":   "media_player.living_room_bravia_adb"
  }
}'
```

Optional fields per client:

- `plex_app_id` — defaults to `com.plexapp.android` (the Android TV Plex
  package). Override for forks or other platforms.
- `launch_command` — full ADB command to launch Plex. Defaults to
  `monkey -p <plex_app_id> -c android.intent.category.LAUNCHER 1`.

Restart the agent after editing `.env`:

```bash
docker compose down && docker compose up -d
```

### 4. Verify

In the dashboard chat at `http://<host>:6002/chat`:

- "What's new on Plex?" → should call `plex_list_recent`.
- "List my Plex clients." → should call `plex_list_clients` and name your TV.
- "Play Dune on the living room TV." → `plex_search` then `plex_play`;
  the TV should wake, Plex should open, and playback should start.
- "Pause the living room TV." → `ha_control_media_player` with
  `action=pause`.

If `plex_play` returns `"played": false` with an `available_clients` list, the
`client_name` didn't match any cloud-discovered device (partial matches are
accepted, but case and spelling matter if there are several similar names).

## Known limitations

- **One-TV-per-household assumption in practice.** The mapping config scales
  to any number of clients, but each client needs its own Android TV + ADB
  + "Advertise as Player" story. A Vizio or other non-Android TV that lacks
  "Advertise as Player" can still be used for transport control via
  `ha_control_media_player`, but will not appear in `plex_list_clients`
  and cannot be a target for `plex_play`.
- **No playback-control tools in the Plex module.** Pause / resume / seek go
  through `ha_control_media_player` on the TV's HA entity, not through Plex.
  This is intentional — HA's transport surface is more reliable here.
- **Hand-authored mapping.** `PLEX_CLIENT_HA_MAP` is edited by hand in `.env`
  today. Auto-discovery (cross-referencing Plex clients against HA entities)
  is a future improvement.
- **Audio-only devices are out of scope.** Soundbars, Google Home speakers,
  and Wyoming satellites are deferred to a future Music Assistant module.

## Troubleshooting

User-facing symptoms. For server-internal errors (stub mode, plexapi
exceptions, connection errors from inside the container), see
[MCP Plex → Troubleshooting](../services/agent/tools/plex.md#troubleshooting).

| Symptom | Likely cause |
|---------|--------------|
| `plex_list_clients` returns an empty list | TV not signed into the same Plex account as the server, or "Advertise as Player" is off, or the Plex app isn't running on the TV right now. |
| `plex_play` succeeds but nothing happens on the TV | TV is off or Plex app not foregrounded, and no `PLEX_CLIENT_HA_MAP` entry exists for that client. Add the mapping, or turn the TV on and open Plex manually before asking Selene to play. |
| Wake fires but Plex never launches | `adb_entity` isn't paired / authorized — check the HA `androidtv` integration; you may need to re-accept the ADB fingerprint on the TV. |
| `plex_play` returns `"no player-capable device matches …"` | `client_name` didn't match any discovered device. Ask the assistant to "list my Plex clients" first and use one of those names verbatim. |
| Transport control works but playback initiation doesn't | Expected on non-Android / non-"Advertise as Player" TVs. Use the Plex app on the TV directly to start playback, then use Selene for transport control. |
| Audio-only devices are missing entirely | Expected today — see the Music Assistant row in the "How it works" table. They can still be controlled with `ha_control_media_player` on their HA entity. |

## See also

- [MCP Plex](../services/agent/tools/plex.md) — server reference for the Plex MCP module
  (tool signatures, internals, stub mode, plexapi error mapping).
- [MCP Home Assistant](../services/agent/tools/home-assistant.md) — `ha_control_media_player`
  transport tool.
- `.env.tmpl` — reference config with an example `PLEX_CLIENT_HA_MAP`
  mapping.
