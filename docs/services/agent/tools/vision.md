# MCP Server: Vision Tools (`mcp_vision_tools`)

Reference doc for the vision-tools MCP server. Layers five purpose-built
vision tools on top of the [`vllm-vision`](../../vllm-vision/README.md)
service so the LLM can ask high-leverage questions ("what's on the
backyard camera?", "what changed?", "transcribe this receipt") without
having to assemble image URLs and prompts manually.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_vision_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_vision_tools` |
| Transport | MCP stdio |
| Server name | `havencore-vision-tools` |
| Backing service | [vllm-vision (port 8001)](../../vllm-vision/README.md) |
| Tool count | 5 |

This module is **complementary** to the older `query_multimodal_api`
tool in [`mcp_general_tools`](general.md). `query_multimodal_api` is a
generic "image URL + prompt" pass-through; the tools here wrap that with
focused behavior — fetching a fresh camera snapshot, comparing two
images side-by-side, normalizing OCR prompts, etc. New code should
prefer these over `query_multimodal_api`.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `describe_image(image_url, prompt?)` | General-purpose: describe an image at an HTTP(S) URL. `prompt` is optional — omitted defaults to a "what's in this scene" prompt that names people, animals, vehicles, packages, and anything unusual. |
| `describe_camera_snapshot(camera_name, prompt?)` | One-shot "what's happening on the {camera} camera?". Triggers a fresh HA snapshot capture (via the same MQTT round-trip the `mcp_mqtt_tools` server uses), then runs vision on the matching frame. Replaces the two-step "snapshot then describe" chain. `camera_name` is fuzzy-matched against snapshot URLs server-side. |
| `compare_snapshots(image_url_a, image_url_b, focus?)` | Sends both images in one request and asks "what changed?". Useful for "did the package get picked up?", "did anyone enter the room?". Optional `focus` narrows the comparison ("the porch", "people"). |
| `identify_object(image_url, hint?)` | Focused "what is this thing?" — returns a concise name + one-sentence description. `hint` ("plant", "bug", "appliance brand") narrows the domain. |
| `read_text_in_image(image_url)` | OCR-flavored prompt with `temperature=0.1` and `max_tokens=1024`. Preserves rough layout where it matters; marks illegible regions `[illegible]`. For receipts, mail, error screenshots, whiteboards. |

All tools accept `http(s)://` and `data:` URLs — `vllm-vision` fetches
them itself.

## Internals worth knowing

### Single chokepoint via `/api/vision/ask_url`

Four of the five tools (`describe_image`, `describe_camera_snapshot`,
`identify_object`, `read_text_in_image`) post to
`http://agent:6002/api/vision/ask_url`, the same in-process FastAPI
endpoint `query_multimodal_api` uses. This keeps the `model` /
`VISION_SERVED_NAME` injection, the upstream URL, and any future logging
or auth shaping in **one place** rather than scattered across each tool.

### Why `compare_snapshots` is the exception

The `/api/vision/ask_url` request schema is single-image
(`{text, image_url, ...}`). To send two images in one call —
necessary for an actual side-by-side diff rather than two separate
descriptions — `compare_snapshots` posts directly to
`vllm-vision`'s OpenAI-compatible `/v1/chat/completions` with a
multi-part user message:

```json
{
  "type": "image_url", "image_url": {"url": "<A>"}
},
{
  "type": "image_url", "image_url": {"url": "<B>"}
}
```

Both code paths share the same `_post_json` HTTP helper for timeouts,
JSON parsing, and error normalization, so retry/timeout behavior is
identical regardless of endpoint.

### `describe_camera_snapshot` lazy-instantiates `HACamSnapper`

To capture a fresh frame, the tool reuses the `HACamSnapper` class from
[`mcp_mqtt_tools`](mqtt.md) (the one behind `get_camera_snapshots`). It
imports the class at first-use and instantiates a second snapshotter
inside this MCP server's process. That means there are briefly two MQTT
subscribers to `home/cameras/snapshots` — both receive every message,
but each only consumes URLs from snapshots **it triggered**, so they
don't race in practice.

`get_camera_snapshots` is parameterless on the HA side — it captures
**all** cameras in one shot. The tool then matches `camera_name`
against the returned URLs server-side:

1. Case-insensitive substring (e.g. `"backyard"` matches
   `…/snap/backyard_2026.jpg`).
2. Token-split fallback (e.g. `"front door"` splits to `["front",
   "door"]` and matches `…/snap/front_door_cam.jpg` if both tokens
   are present in the URL).

If nothing matches, the tool returns `{error, available_urls}` so the
LLM can disambiguate.

### Default prompts

Each tool ships with a default prompt biased toward HavenCore use cases
(camera scenes, OCR, receipts). Override-able per call via the optional
`prompt` / `focus` / `hint` argument; the override is appended to the
default rather than replacing it for `identify_object`, so the user
hint always lands in the prompt context.

## Configuration

### Required

| Var | What it does |
|-----|--------------|
| `VISION_API_BASE` | Used for the `compare_snapshots` direct path; same value as the agent's. |
| `VISION_SERVED_NAME` | Sent as `model` in the `compare_snapshots` chat-completions body. |

The single-chokepoint tools rely on `agent:6002` being reachable
in-cluster (it always is — same compose network).

### Optional

| Var | Default | Notes |
|-----|---------|-------|
| `VISION_ASK_URL_ENDPOINT` | `http://agent:6002/api/vision/ask_url` | Override only if you're bench-testing against a different agent host. |
| `VISION_HTTP_TIMEOUT_SEC` | `180` | Per-call timeout for both endpoints. Vision generation is sometimes slow on cold cache; 180 s leaves headroom. |
| `HAOS_URL`, `HAOS_TOKEN`, `MQTT_BROKER`, `MQTT_PORT` | (inherited from agent env) | Used by the lazy `HACamSnapper` for `describe_camera_snapshot`. Same values the `mcp_mqtt_tools` server reads — no separate config. |

### `MCP_SERVERS` entry

```json
{
  "name": "vision",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_vision_tools"],
  "enabled": true
}
```

### Autonomy gating

All five tool names are in the `observe`-tier allow-list at
`services/agent/selene_agent/autonomy/tool_gating.py`, so autonomy
turns running at `observe` (or higher) can call them. The face-trigger
seeds in
`services/agent/selene_agent/autonomy/seeds/camera_events.py` use
`query_multimodal_api` for the `scene_description` gather step today;
that path is unchanged. Future seeds can use `describe_camera_snapshot`
directly when they need to drive a snapshot themselves.

## Troubleshooting

### `describe_camera_snapshot` returns `"no snapshot URL matched 'X'"`

The HA snapshot script ran and produced URLs, but none contain `X`. The
returned `available_urls` field lists every URL `get_camera_snapshots`
emitted — pick the right name and pass that. Common causes:

- HA entity_ids and the friendly names stored in your URL paths don't
  align (e.g. snapshot file is `frontdoor_…jpg` but the LLM passed
  `"front door"`). Token-split match handles `"front door"` → `front_door`,
  but not `"frontdoor"` → `front door`.
- The script doesn't actually snapshot the camera you asked about —
  check the HA `script.capture_all_cameras` definition.

### `describe_camera_snapshot` returns `"snapshot capture failed: timeout..."`

The HA script triggered but no `home/cameras/snapshots` MQTT message
arrived within the timeout. Check:

```bash
docker compose logs --tail=50 mosquitto
docker compose exec agent bash -lc 'mosquitto_sub -h $MQTT_BROKER -p $MQTT_PORT -t home/cameras/snapshots -C 1 -W 15'
```

If the MQTT subscriber gets nothing, the HA-side automation that
publishes after `script.capture_all_cameras` is wedged.

### `compare_snapshots` fails with `"VISION_API_BASE is not configured"`

The direct-to-vllm-vision path requires the env var. Confirm the agent
container has it set:

```bash
docker compose exec agent bash -lc 'echo $VISION_API_BASE'
```

Empty string ⇒ `.env` is missing the `VISION_*` block (the same one
the rest of the vision pipeline needs). Add it from `.env.example` and
`docker compose down agent && docker compose up -d agent`.

### Any tool returns `"vision API error (5xx)"`

`vllm-vision` is unreachable or unhealthy. Run through the same
checklist as `query_multimodal_api`:

```bash
docker compose ps vllm-vision
docker compose exec agent curl -sf http://vllm-vision:8000/v1/models
curl -sf http://localhost:6002/api/vision/health
```

The agent's `/api/vision/ask_url` proxy returns the upstream error
verbatim, so a model-side OOM or schema mismatch shows up here.

## Related files

- `services/agent/selene_agent/modules/mcp_vision_tools/server.py` —
  `VisionMCPServer` class, all five tool implementations, both HTTP
  helpers, the camera-URL matcher.
- `services/agent/selene_agent/api/vision.py` — the FastAPI proxy
  endpoints (`/api/vision/ask`, `/api/vision/ask_url`,
  `/api/vision/health`) that this module routes through.
- `services/agent/tests/test_mcp_vision_tools.py` — unit tests
  (mocked HTTP) covering every tool's payload shape and the camera-URL
  matcher.

## See also

- [vllm-vision](../../vllm-vision/README.md) — backend service, model fit,
  fallback ladder.
- [General Tools](general.md) — `query_multimodal_api`, the lower-level
  tool these wrap.
- [MQTT Tools](mqtt.md) — `get_camera_snapshots`, the HA-script trigger
  `describe_camera_snapshot` reuses.
- [Tool Development](development.md) — module layout, MCP stdio
  handshake, registration in `MCP_SERVERS`.
