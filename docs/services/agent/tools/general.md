# MCP Server: General Tools (`mcp_general_tools`)

Reference doc for the "general-purpose" MCP server — weather, web search,
computational knowledge, Wikipedia, image generation, Signal messaging, and
the multimodal AI gateway. These are the assistant's external-world tools
that aren't specific to any other subsystem.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_general_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_general_tools` |
| Transport | MCP stdio |
| Server name | `havencore-general-tools` |
| Tool count | Up to 7 (some tools are conditional on credentials) |

Tool registration is **conditional on credentials**. The server enumerates
tools at `list_tools()` time and only includes the ones whose env vars are
set. This is by design — missing credentials silently drop the tool
instead of registering one that always errors.

## Tool inventory

| Tool | Needs | Purpose |
|------|-------|---------|
| `generate_image(prompt)` | (none) | Submits a prompt to the ComfyUI service at `text-to-image:8188` using the `default` workflow. Returns a filepath and a URL to the finished image. |
| `send_signal_message(message, attachments?)` | `SIGNAL_PHONE_NUMBER`, `SIGNAL_DEFAULT_RECIPIENT` | Sends a Signal message (text + optional image/video attachments) via the `signal-api` container (`signal-cli-rest-api`). Recipient is fixed to `SIGNAL_DEFAULT_RECIPIENT` — the tool is intentionally not a free-form "send to anyone". Attachments accept URLs (auto-downloaded, 50 MB cap) or local paths, are base64-encoded, and sent via `POST /v2/send`; per-attachment errors are tracked and only a full failure is surfaced. Video size cap is ~95 MB. |
| `query_multimodal_api(image_url, text?)` | (none) | Send an image URL (and optional text prompt) to the vision LLM (`vllm-vision`). POSTs JSON to the agent's own `/api/vision/ask_url` endpoint, which forwards to `vllm-vision` — the agent-side proxy is the single chokepoint for logging and authentication. Image-only by design (the URL endpoint is single-image). For higher-leverage tools — fresh camera snapshots, two-image diffs, OCR — prefer the dedicated [Vision Tools server](vision.md) (`mcp_vision_tools`); for video uploads, use the multipart `/api/vision/ask` endpoint or the dashboard playground. |
| `wolfram_alpha(query)` | `WOLFRAM_ALPHA_API_KEY` | Wolfram Alpha LLM API for factual + computational questions. 1000-char response cap, 30 s timeout. |
| `get_weather_forecast(location, date?)` | `WEATHER_API_KEY` | weatherapi.com forecast — current day by default, or a specific `YYYY-MM-DD` up to 365 days ahead. Returns temp, conditions, precip, wind, and astronomy (sunrise/sunset/moon phase). |
| `brave_search(query, count?)` | `BRAVE_SEARCH_API_KEY` | Brave Search web results. Usually paired with the `fetch` MCP (from `mcp_server_fetch`) to actually read one of the returned pages. |
| `search_wikipedia(search_string, sentences?)` | (none — public API) | Summary from Wikipedia. `sentences` controls summary length; defaults to the helper's default (~7). |

## Configuration

Env vars read directly via `os.getenv()` in `mcp_server.py`:

| Var | Enables | Notes |
|-----|---------|-------|
| `WEATHER_API_KEY` | `get_weather_forecast` | weatherapi.com. |
| `BRAVE_SEARCH_API_KEY` | `brave_search` | Brave Search API. |
| `WOLFRAM_ALPHA_API_KEY` | `wolfram_alpha` | Wolfram LLM API. |
| `TIMEZONE` | date math in weather | Used to resolve relative dates against local "today". Set to an IANA zone (e.g. `America/Chicago`). |
| `SIGNAL_API_URL` | `send_signal_message` | URL of the `signal-api` container. Default `http://signal-api:8080` (internal Docker hostname). |
| `SIGNAL_PHONE_NUMBER` | `send_signal_message` | Your Signal account number in E.164 (e.g. `+15551234567`). The container is linked to this account as a secondary device. |
| `SIGNAL_DEFAULT_RECIPIENT` | `send_signal_message` | Where messages go. Leave empty to default to `SIGNAL_PHONE_NUMBER` (Note to Self). |

ComfyUI image generation and the vision gateway use in-cluster routing
(`text-to-image:8188` directly; `query_multimodal_api` posts to the agent's
own `/api/vision/ask_url`, which then talks to `vllm-vision`) and require no
additional credentials — they only work when their respective services are
running. The agent reads `VISION_API_BASE`, `VISION_API_KEY`, and
`VISION_SERVED_NAME` for the upstream call; see
[Configuration](../../../configuration.md) and the
[vllm-vision service doc](../../vllm-vision/README.md).

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "general_tools",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_general_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **`generate_image` uses the `SimpleComfyUI` helper** in
  `comfyui_tools.py` with workflow `default`. The returned JSON contains
  both the filepath and a URL pointing at the ComfyUI service
  (`text-to-image:8188`) — the dashboard proxies these through
  `/api/comfy/view` on the agent (port 6002).
- **`query_multimodal_api` routes through the agent's own FastAPI proxy,
  not directly at `vllm-vision`.** The tool POSTs JSON to
  `http://agent:6002/api/vision/ask_url`, which fills in the served-model
  name (`VISION_SERVED_NAME`) and forwards to the `VISION_API_BASE`
  upstream. Single chokepoint for logging, auth, and any future
  request-shaping; matches the pattern the dashboard's `/api/vision/ask`
  multipart endpoint already uses.
- **`send_signal_message`'s attachment pipeline is robust but silent.** Failed
  attachments are collected into `attachment_errors` and only surface in
  the response if *all* attachments fail. Partial success sends the
  remaining attachments without prompting.
- **Signal attachments are base64-encoded bytes** passed in the
  `base64_attachments` array of `POST /v2/send`. Signal infers the content
  type from the data; filenames from the source URL/path are not preserved.
- **Weather date logic branches on lookahead window.** 0–14 days uses
  `/forecast.json` with `days` set appropriately; 15–365 days uses
  `/future.json` (single-day forecast). More than 365 days returns a
  plain-English error.
- **Brave results are formatted as a numbered list** with title / URL /
  description per result — intended to be passed verbatim to the LLM for
  URL selection before a `fetch` call.

## Troubleshooting

### A tool you expect isn't listed

The server only registers tools whose env vars are populated. Check
`/api/tools` on the agent or call `/mcp/status` to see what came through.
Common misses:

- `wolfram_alpha` → `WOLFRAM_ALPHA_API_KEY` unset.
- `send_signal_message` → `SIGNAL_PHONE_NUMBER` unset, or the `signal-api` container not linked yet.
- `brave_search` → `BRAVE_SEARCH_API_KEY` unset.

### `generate_image` returns a ComfyUI connection error

The `text-to-image` service isn't reachable from inside the agent
container. Verify with:

```bash
docker compose ps text-to-image
docker compose exec agent curl -I http://text-to-image:8188
```

### `query_multimodal_api` returns a 5xx or `Vision API error`

`vllm-vision` is down or unreachable, or the agent's `/api/vision/ask_url`
proxy can't reach the upstream URL configured in `VISION_API_BASE`. Check
in this order:

```bash
# 1. Is vllm-vision running and serving the model?
docker compose ps vllm-vision
docker compose exec agent curl -sf http://vllm-vision:8000/v1/models

# 2. Can the agent container reach the host-LAN URL configured in .env?
docker compose exec agent bash -lc 'curl -sf -m 3 -o /dev/null -w "%{http_code}\n" "$VISION_API_BASE/models"'

# 3. Agent-side proxy health (this is what query_multimodal_api actually calls):
curl -sf http://localhost:6002/api/vision/health
```

If (1) is fine but (2) is failing, the `VISION_API_BASE` value in `.env`
points somewhere the agent container can't route to (typical: a hostname
that resolves outside the docker network). Update it to a host IP that
matches the same convention as `LLM_API_BASE`, then
`docker compose up -d --force-recreate --no-deps agent` so the new env
is picked up.

The tool extracts `data["response"]` from the proxy's reply; any
structural deviation raises `ValueError`.

### `send_signal_message` returns "SIGNAL_PHONE_NUMBER is not configured"

Either `SIGNAL_PHONE_NUMBER` is unset in `.env`, or the `signal-api` container
hasn't been linked to your Signal account yet. See the
[Signal one-time setup](#signal-one-time-setup) section below.

### `send_signal_message` returns a `Signal API returned 400/500`

The `signal-api` container can reach Signal's servers but the send failed.
Most common causes:

- The container is not linked yet (or linking was abandoned mid-flow).
- The `number` field doesn't match the linked account number exactly —
  must be E.164 with the leading `+`, e.g. `+15551234567`.
- The recipient number is malformed or not reachable.

Check container logs:

```bash
docker compose logs signal-api
```

## Signal one-time setup

The `signal-api` service runs `signal-cli-rest-api`, which talks to Signal's
servers over an **outbound-only** connection. It does not accept incoming
connections from the internet — port 8080 is bound to `127.0.0.1` so only
the host (and other containers on the internal Docker network) can reach it.

Link the container as a secondary device on your existing Signal account:

```bash
# 1. Start the service.
docker compose up -d signal-api

# 2. Generate a link QR code (saved as PNG on the host).
curl -fsSL 'http://127.0.0.1:8080/v1/qrcodelink?device_name=HavenCore' \
  --output signal-qr.png

# 3. Open signal-qr.png, then in your Signal app on your phone:
#    Settings → Linked Devices → Link New Device → scan the QR.

# 4. Set SIGNAL_PHONE_NUMBER in .env to the phone number on your Signal
#    account (E.164, e.g. +15551234567), restart the agent.
docker compose up -d agent
```

Leaving `SIGNAL_DEFAULT_RECIPIENT` empty sends messages to your own number
— they arrive in the **Note to Self** chat on your phone, which is the
standard pattern for self-notifications.

To unlink or rotate: delete `./volumes/signal-cli-config/`, restart the
container, and repeat the QR-link step.

### `get_weather_forecast` returns "Weather API key not configured"

`WEATHER_API_KEY` is unset. The tool is registered only when the key is
present, so seeing this string usually means the key was set at startup
but unset later — re-check `.env` and `docker compose down && up -d`.

### Wolfram returns 403

Key is invalid or revoked. Wolfram Alpha LLM API keys are separate from
the standard AppID — verify you're using the LLM endpoint's key.

## Related files

- `services/agent/selene_agent/modules/mcp_general_tools/mcp_server.py`
  — all tool registrations and most handlers.
- `services/agent/selene_agent/modules/mcp_general_tools/comfyui_tools.py`
  — `SimpleComfyUI` helper.
- `services/agent/selene_agent/modules/mcp_general_tools/wiki_tools.py`
  — `query_wikipedia` helper.

## See also

- [Tool Development](development.md) — adding new MCP tools.
