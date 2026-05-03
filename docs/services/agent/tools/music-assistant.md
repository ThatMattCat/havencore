# MCP Server: Music Assistant (`mcp_music_assistant_tools`)

Server-side reference for the Music Assistant (MA) MCP module — tool
inventory, module config, internals, and server-level troubleshooting.
MA is the audio-only counterpart to the Plex module: it routes library
playback to Chromecasts, Google Homes, soundbars, and other MA-exposed
speakers. Topic-level context (which service owns what, division of
labor vs. Plex and HA) lives in
[Media Control](../../../integrations/media-control.md).

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_music_assistant_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_music_assistant_tools` |
| Transport | MCP stdio |
| Server name | `havencore-music-assistant` |
| MA client library | `music-assistant-client` (async WebSocket — the same client the HA MA integration uses) |
| HA client | None — MA talks directly to its own providers; transport control lives on `ha_control_media_player` |
| Tool count | 7 |

The server connects to a running Music Assistant instance over its
WebSocket API. MA itself handles the per-provider details (Plex for
library files, Chromecast for Google-cast speakers, the `hass_players`
bridge for HA-exposed `media_player` entities, etc.) — this module only
exposes MA's aggregated search/player/queue surface.

Transport control (pause, resume, volume) is intentionally **not** in
this module. Any MA-exposed speaker shows up in HA as a
`media_player.*` entity, so `ha_control_media_player` already does the
right thing. This module owns everything *queue-aware* (search,
playback start, queue inspection, shuffle/repeat, clearing).

## Tool inventory

| Tool | Purpose |
|------|---------|
| `mass_search(query, media_type?, limit?)` | Cross-provider library search. Returns rows with `uri` (opaque — pass to `mass_play_media`), `name`, `artist`, `album`, `media_type`, and `providers`. `media_type` narrows to `track` / `album` / `artist` / `playlist` / `radio`. |
| `mass_list_players(include_hidden?)` | Enumerate MA players. Returns `player_id`, `display_name`, `provider`, `available`, `powered`, `state`, `volume_level`, and `current_item`. `include_hidden=false` (default) filters MA's `hide_in_ui` players (e.g., the built-in web player). |
| `mass_play_media(uri, player_name, mode?)` | Start playback of a `uri` (from a prior search) on a named speaker. `mode` is `replace` (default — clear queue, play now), `next` (insert after current), or `add` (append). `player_name` resolves exact-first, then substring match. |
| `mass_get_queue(player_name, item_limit?)` | Read the active queue. Returns `state`, `shuffle`, `repeat`, `current`, `upcoming[…]`, and `total_items`. Powers "what's playing?" / "what's next?". |
| `mass_queue_clear(player_name)` | Empty the queue and stop playback. |
| `mass_play_announcement(player_name, url, volume?, pre_announce?)` | Play a short audio clip (typically TTS) on a named speaker. MA ducks any currently playing track and resumes it when the clip finishes. Used by the autonomy engine's `speak` delivery channel. |
| `mass_playback_control(player_name, action)` | Queue-level actions: `shuffle_on`, `shuffle_off`, `repeat_off`, `repeat_one`, `repeat_all`. Pause/resume/skip stay on `ha_control_media_player`. |

### Tool-design notes

- **URIs are opaque.** The LLM gets `uri` strings (e.g.
  `library://album/34`) back from `mass_search` and threads them into
  `mass_play_media`. MA returns two URI flavors — library-scoped
  (`library://...`) and provider-scoped (`plex--<id>://...`). Both play;
  the LLM doesn't need to distinguish.
- **Display names, not player IDs.** Tools accept the `display_name`
  from `mass_list_players`. The server's resolver does exact match →
  substring match → `player_id` match, all casefolded. A clean miss
  returns `{"error": "no player matches …", "available_players": [...]}`.
- **Search is title-biased.** MA's search matches titles more strongly
  than artist fields. The server runs two fallbacks when a typed search
  returns empty: first retry without the `media_type` filter, then
  retry with the longest single token of the query. This covers the
  common LLM pattern of gluing title+artist into one query
  (`"The Better Life 3 Doors Down"`).

## Configuration

Env vars (loaded via `selene_agent.utils.config`):

| Var | Required | Purpose |
|-----|----------|---------|
| `MASS_URL` | yes | HTTP base URL of the Music Assistant server (e.g. `http://10.0.50.101:8095`). No trailing slash. The client upgrades this to `ws://…/ws` internally. |
| `MASS_TOKEN` | yes | MA long-lived access token. Mint one in the MA web UI → Settings → Users → *Create long-lived token*. Schema v28+ enforces auth on the WS endpoint. |

If either is unset, the server starts in a degraded mode and every
tool returns
`{"error": "MASS_URL / MASS_TOKEN not configured", "hint": …}`. The
agent stays healthy without MA configured.

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "music_assistant",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_music_assistant_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **Persistent WS connection.** `MassAgent.connect()` opens one
  `music-assistant-client` session at startup and keeps its listener
  task alive for the life of the subprocess. Tool calls reuse the
  connection; no per-call auth round-trip.
- **Search fallbacks run server-side.** `mass_client.search` runs the
  typed query; if zero results and a `media_type` filter was set, it
  retries across all types; if still zero and the query has multiple
  tokens, it retries with the longest token. The LLM sees a single
  response — no retry bookkeeping on its side.
- **Player resolution is casefolded and order-sensitive.** Exact
  match wins over substring match, which wins over `player_id` match.
  This matters when MA exposes the same physical device twice (e.g.,
  once via the `chromecast` provider and again via `hass_players`): the
  name the user says is matched exactly first, so duplicates without a
  name collision don't cause a resolution ambiguity.
- **No reconnect loop.** If the WS drops, the next tool call surfaces
  an error; the agent's MCP-manager layer can restart the subprocess
  on repeated failure. Deliberate choice — reconnect logic inside the
  module was deemed overkill given that the agent already supervises
  its MCP children.

## Troubleshooting

### Every tool returns the "not configured" stub error

`MASS_URL` or `MASS_TOKEN` is unset. Set both in `.env` and
`docker compose down && up -d` (env changes require a full recycle).

### Connection fails on startup

The module logs
`Failed to init Music Assistant agent: <ExceptionName>: <msg>` and all
tools surface that same error. Common causes:

- **Token mismatch** — MA rejects the token as unauthorized. Mint a new
  long-lived token in the MA web UI and update `MASS_TOKEN`.
- **Wrong URL** — the agent container can't reach `MASS_URL`. Verify
  from inside the container:
  ```bash
  docker compose exec agent curl -I "$MASS_URL"
  ```
- **MA schema mismatch** — the bundled client version has to be
  compatible with the running MA server. Bumping MA without bumping
  `music-assistant-client` (in `services/agent/pyproject.toml`) can
  surface as init errors.

### `mass_play_media` returns `"played": false`

The response includes an `available_players` list when the name didn't
resolve. Pass one of those verbatim. If the name *does* match but
nothing plays, it's usually one of:

- The player is `available=false` (offline / not yet discovered by MA
  on this boot). Check `mass_list_players` first.
- The `uri` is stale — library/<n> URIs are stable per-library, but a
  provider-scoped URI whose provider has since been removed will 404.
  Re-run `mass_search`.

### Duplicate players (e.g. soundbar via two providers)

MA can expose the same physical device via multiple providers — most
commonly the native Chromecast provider and the `hass_players` bridge.
Both show up in `mass_list_players`; only one typically plays cleanly.
The cleanest lane is the native provider (`chromecast`,
`universal_player`); the `hass_players` dup is a fallback. The module
does not filter these — the LLM (or the user asking by name) picks the
target.

## Related files

- `services/agent/selene_agent/modules/mcp_music_assistant_tools/mass_client.py` —
  async facade over `music-assistant-client`: connection lifecycle,
  player resolver, search fallbacks, flatteners for JSON-over-MCP.
- `services/agent/selene_agent/modules/mcp_music_assistant_tools/mcp_server.py` —
  tool registrations + dispatch.
- `services/agent/selene_agent/utils/config.py` — `MASS_URL`,
  `MASS_TOKEN` passthrough.

## See also

- [Media Control](../../../integrations/media-control.md) — topic doc
  covering the division of labor between Plex (video), MA (audio), and
  HA (transport).
- [MCP Plex](plex.md) — the video-playback counterpart.
- [MCP Home Assistant](home-assistant.md) — `ha_control_media_player`
  handles pause/resume/volume on any MA-exposed `media_player` entity.
