# MCP Server: General Tools (`mcp_general_tools`)

Reference doc for the "general-purpose" MCP server — weather, web search,
computational knowledge, Wikipedia, image generation, email, and the
multimodal AI gateway. These are the assistant's external-world tools that
aren't specific to any other subsystem.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_general_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_general_tools` |
| Transport | MCP stdio |
| Server name | `havencore-general-tools` |
| Tool count | 5–8 depending on which API keys / credentials are configured |

Tool registration is **conditional on credentials**. The server enumerates
tools at `list_tools()` time and only includes the ones whose env vars are
set. This is by design — missing credentials silently drop the tool
instead of registering one that always errors.

## Tool inventory

| Tool | Needs | Purpose |
|------|-------|---------|
| `generate_image(prompt)` | (none) | Submits a prompt to the ComfyUI service at `text-to-image:8188` using the `default` workflow. Returns a filepath and a URL to the finished image. |
| `send_email(subject, body, attachments?)` | `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | Sends mail via Gmail SMTP (`aiosmtplib`, STARTTLS). Recipient defaults to `DEFAULT_RECIPIENT` — the tool is not exposed to the LLM as free-form "send to anyone" on purpose. Attachments accept URLs (auto-downloaded, 50 MB cap, 30 s timeout) or local paths; per-attachment errors are tracked and only a full failure is surfaced. |
| `query_multimodal_api(text?, image_url?, audio_url?, video_url?)` | (none) | POSTs to the internal `iav-to-text` vision LLM through `http://nginx/iav/api`. Use for image / audio / video analysis. At least one of the four inputs is required. |
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
| `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` | `send_email` | Gmail app-password (not your main password). |
| `SMTP_SERVER` | `send_email` | Default `smtp.gmail.com`. |
| `SMTP_PORT` | `send_email` | Default `587`. |
| `DEFAULT_RECIPIENT` | `send_email` | Fallback when no `to` is supplied. Also used to keep the LLM from arbitrary addressing. |

ComfyUI image generation and the multimodal gateway use in-cluster hostnames
(`text-to-image:8188`, `nginx`) and require no additional credentials — they
only work when their respective services are running.

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
  `comfyui_tools.py` with workflow `default`. Generated images are served
  at `http://<host>:6006/outputs/<filename>` (the ComfyUI static output
  endpoint) — the returned JSON contains both the filepath and the URL.
- **`query_multimodal_api` routes through nginx, not the service directly.**
  The endpoint is `http://nginx/iav/api`, which proxies to
  `iav-to-text:8100`. This keeps the routing configurable in one place.
- **`send_email`'s attachment pipeline is robust but silent.** Failed
  attachments are collected into `attachment_errors` and only surface in
  the response if *all* attachments fail. Partial success sends the
  remaining attachments without prompting.
- **Images in email are MIMEImage-typed** based on
  `mimetypes.guess_type()`; everything else falls back to base64-encoded
  `application/octet-stream`.
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
- `send_email` → `GMAIL_APP_PASSWORD` unset.
- `brave_search` → `BRAVE_SEARCH_API_KEY` unset.

### `generate_image` returns a ComfyUI connection error

The `text-to-image` service isn't reachable from inside the agent
container. Verify with:

```bash
docker compose ps text-to-image
docker compose exec agent curl -I http://text-to-image:8188
```

### `query_multimodal_api` returns a 5xx or `Unexpected response structure`

`iav-to-text` or `nginx` is down, or the vision LLM model didn't load.
Check:

```bash
docker compose logs iav-to-text
docker compose logs nginx
```

The tool extracts `data["choices"][0]["message"]["content"]`; any
structural deviation raises `ValueError`.

### `send_email` returns "Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD"

You haven't configured Gmail credentials, or you set your account password
instead of an **app password**. Gmail requires app-specific passwords for
SMTP — generate one at Google Account → Security → App passwords
(2FA required).

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

- [External Services](../../../External-Services.md) — if you're adding a new
  third-party API, this is where the lifecycle pattern is documented.
- [Tool Development](development.md) — adding new MCP tools.
