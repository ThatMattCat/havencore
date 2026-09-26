# MCP Tools Host

Serves all 11 of the agent's MCP tool modules over MCP Streamable HTTP
from one Starlette app — one mount per module at `/mcp/<name>`, each
behind its own static bearer token. The agent's `MCPClientManager`
connects to the mounts as an HTTP client; external MCP clients can
consume them the same way.

The service reuses the agent image with a different entrypoint
(`python -m selene_agent.mcp_http_app`) and listens on port 6010,
published on the host IP only. The nginx gateway also proxies `/mcp/` to
it so external clients can go through `http(s)://<host>/mcp/<name>`.

## Purpose

- Host the in-tree MCP tool modules out-of-process from the agent, so
  tool code can be reloaded without restarting the agent
- Enforce per-module bearer-token auth (the only authenticated surface
  in HavenCore)
- Expose the same tool surface to external MCP clients (custom
  connectors, IDE integrations) that the agent itself uses

## Mounts

| Module | Mount path | Token env var | Tools |
|--------|-----------|---------------|-------|
| `mcp_general_tools` | `/mcp/general_tools` | `MCP_TOKEN_GENERAL_TOOLS` | up to 8 (credential-gated) |
| `mcp_qdrant_tools` | `/mcp/mcp_server_qdrant` | `MCP_TOKEN_QDRANT` | 3 |
| `mcp_homeassistant_tools` | `/mcp/homeassistant` | `MCP_TOKEN_HOMEASSISTANT` | 20 |
| `mcp_mqtt_tools` | `/mcp/mqtt` | `MCP_TOKEN_MQTT` | 1 |
| `mcp_plex_tools` | `/mcp/plex` | `MCP_TOKEN_PLEX` | 5 |
| `mcp_music_assistant_tools` | `/mcp/music_assistant` | `MCP_TOKEN_MUSIC_ASSISTANT` | 7 |
| `mcp_github_tools` | `/mcp/github` | `MCP_TOKEN_GITHUB` | 7 |
| `mcp_face_tools` | `/mcp/face` | `MCP_TOKEN_FACE` | 5 |
| `mcp_vision_tools` | `/mcp/vision` | `MCP_TOKEN_VISION` | 5 |
| `mcp_reminder_tools` | `/mcp/reminder` | `MCP_TOKEN_REMINDER` | 3 |
| `mcp_device_action_tools` | `/mcp/device_action` | `MCP_TOKEN_DEVICE_ACTION` | 5 |

69 tools total with all credentials configured. Per-module reference
docs live under [`services/agent/tools/`](../agent/tools/README.md).

## Configuration

```yaml
mcp-tools:
  image: agent:${VERSION:-latest}          # reuses the agent image
  command: ["python", "-m", "selene_agent.mcp_http_app"]
  environment:
    - AGENT_API_BASE=http://agent:6002     # reminder module → autonomy API
  volumes:
    - ./services/agent/:/app               # module code, bind-mounted
    - github_repo_clone:/var/cache/havencore/repo_clone
  ports:
    - "${HOST_IP_ADDRESS}:6010:6010"       # host-IP-scoped on purpose
```

Env vars (all from the shared `.env` unless noted):

| Var | Purpose |
|-----|---------|
| `MCP_TOKEN_<NAME>` | Static bearer token per mount (see table above). Generate with `openssl rand -hex 24`. **A module whose token var is unset is not mounted** — nothing is ever served unauthenticated; it shows under `failed` in `/health` and in the agent's `failed_servers`. |
| `MCP_HTTP_ALLOWED_HOSTS` | Extra Host-header values the DNS-rebinding protection accepts, comma-separated. `mcp-tools`, `localhost`, `127.0.0.1`, and `HOST_IP_ADDRESS` are always allowed; add any DNS name used to reach nginx. |
| `MCP_HTTP_ALLOWED_ORIGINS` | Full Origin-header values accepted verbatim, comma-separated — for clients whose origin isn't `http(s)://<host>`. Browser-extension MCP clients need this: e.g. Island's connector sends `Origin: chrome-extension://<extension-id>` and gets 403 until that origin is listed. |
| `MCP_HTTP_PORT` | Listen port (default `6010`). |
| `AGENT_API_BASE` | Set in `compose.yaml` to `http://agent:6002` — the reminder module's route back to the agent's autonomy REST API. |

Plus each module's own config (`HAOS_URL`/`HAOS_TOKEN`, `MASS_URL`,
`GITHUB_TOKEN`, `QDRANT_HOST`, and so on) — the service reads the same
`.env` as the agent.

## Auth

Every request to a mount must carry `Authorization: Bearer <token>`
matching that module's `MCP_TOKEN_<NAME>` value; anything else is a 401.
Verification is the SDK's `BearerAuthBackend` plus a static
constant-time verifier — there is no OAuth flow, token issuance, or
expiry; rotate by changing the env var and recreating the service (and
the agent, which resolves `token_env` at connect time).

## Sessions and transport

Transport defaults are deliberate and the clients depend on them:
**stateful sessions** (`Mcp-Session-Id` header issued on `initialize`,
echoed by the client on subsequent requests) and **SSE response bodies**
(clients must send `Accept: application/json, text/event-stream`).

DNS-rebinding protection validates the Host and Origin headers of every
request. Hosts are checked against an allowlist (`mcp-tools`,
`localhost`, `127.0.0.1`, `HOST_IP_ADDRESS`, plus
`MCP_HTTP_ALLOWED_HOSTS`); a name not on the list is rejected with
**421** on every request. Origins are derived from that host list
(`http://` and `https://` per host) plus any values in
`MCP_HTTP_ALLOWED_ORIGINS` verbatim; an unlisted Origin is rejected
with **403** ("Invalid Origin header"). Requests without an Origin
header (curl, the agent's own client) always pass the origin check.

## Startup and healthcheck

Module init runs concurrently in the app lifespan before uvicorn binds,
with per-module isolation: a module that fails to init (or lacks its
token) is logged and skipped, and the rest of the service comes up
normally. The github module's clone/fetch can block startup for minutes
on a cold volume — the compose healthcheck has a 360s `start_period` for
this, and the agent waits on `service_healthy` before connecting.

```bash
curl "http://${HOST_IP_ADDRESS}:6010/health"
# {"status": "ok", "mounted": [...], "failed": {}}   (503 + "degraded" if nothing mounted)
```

## Operational notes

- **Module code changes**: the source is bind-mounted
  (`./services/agent/:/app`), so `docker compose restart mcp-tools`
  reloads them. The agent's sessions heal lazily — the first tool call
  after the restart fails, a bounded background reconnect kicks in, and
  a retry succeeds — or `docker compose restart agent` reconnects (and
  re-reads tool schemas) immediately.
- **Dependency changes** (`pyproject.toml`): rebuild the shared image —
  `docker compose down mcp-tools agent && docker compose build agent &&
  docker compose up -d`.
- **`.env` changes** (tokens, module credentials): recreate, don't
  restart — `docker compose down mcp-tools agent && docker compose up
  -d mcp-tools agent`.

## Connecting an external MCP client

Any MCP client that speaks Streamable HTTP can use the mounts:

- **URL**: `http://<host>/mcp/<name>` through the nginx gateway
  (`https://` where TLS is set up), or
  `http://<HOST_IP_ADDRESS>:6010/mcp/<name>` direct.
- **Auth**: bearer token — the value of that module's
  `MCP_TOKEN_<NAME>`.

That's the whole contract. For a connector UI (e.g. an enterprise
browser's custom MCP connectors), enter the mount URL and the token as a
Bearer credential. If the client reaches nginx via a DNS name, add that
name to `MCP_HTTP_ALLOWED_HOSTS` or every request 421s. A connector that
runs inside a browser extension (e.g. Island) also sends
`Origin: chrome-extension://<extension-id>` — add that full origin to
`MCP_HTTP_ALLOWED_ORIGINS` or every request 403s.

Manual smoke test:

```bash
curl -is "http://${HOST_IP_ADDRESS}:6010/mcp/reminder" \
  -H "Authorization: Bearer $MCP_TOKEN_REMINDER" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-test","version":"0.0"}}}'
```

## Troubleshooting

| Symptom | Meaning / fix |
|---------|---------------|
| `401 Unauthorized` | Missing or wrong bearer token. Compare the client's token with the module's `MCP_TOKEN_<NAME>` in `.env`; recreate `mcp-tools` + agent after changing it. |
| `421 Misdirected Request` on every call | Host failed the DNS-rebinding allowlist. Add the hostname you're connecting through to `MCP_HTTP_ALLOWED_HOSTS`. |
| `403` "Invalid Origin header" on every call | The client's Origin isn't derivable from the host allowlist — typical for browser-extension MCP clients (`chrome-extension://…`). Add the full origin to `MCP_HTTP_ALLOWED_ORIGINS`. |
| "Session terminated" / "Session not found" errors after a restart | Expected: the restarted host forgot old `Mcp-Session-Id`s. The agent classifies this as a transport death and reconnects in the background — retry the tool call. External clients should re-`initialize`. |
| A module missing from `mounted` | Its token env var is unset, or its init raised — `/health`'s `failed` map and `docker compose logs mcp-tools` carry the reason. The agent reports it under `failed_servers` in `/api/mcp/status`. |
| Container unhealthy for minutes after start | Usually the github module's clone/fetch on a cold volume; the healthcheck `start_period` (360s) allows for it. Check `docker compose logs mcp-tools`. |

## Related

- [Agent tools (MCP servers)](../agent/tools/README.md) — per-module
  reference docs and the tool-development guide.
- [API Reference → MCP tool mounts](../../api-reference.md#mcp-tool-mounts-streamable-http)
  — wire-level surface summary.
- [Configuration → MCP](../../configuration.md#mcp-model-context-protocol-configuration)
  — `MCP_SERVERS`, tokens, client-side tuning vars.
- `services/agent/selene_agent/mcp_http_app.py` — the host app
  (module registry, auth wiring, transport security).
