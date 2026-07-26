"""Origin policy for the agent's HTTP and WebSocket surface.

Why this exists
---------------
`/api/*`, `/ws/*` and `/v1/*` have **no auth gate** — the deployment model is
"LAN only, trusted household". That makes the browser's own cross-origin rules
the last line of defence: without them, any web page a household member opens
on a laptop that can reach the HavenCore LAN could `fetch()` `/api/chat` and
drive the full tool-calling loop (unlock doors, send Signal messages, file
GitHub issues) and read the response.

The app used to install `CORSMiddleware(allow_origins=["*"], ...)`, which threw
that defence away: with `allow_credentials=True` Starlette echoes *whatever*
Origin the caller sent back in `Access-Control-Allow-Origin`, so preflights are
auto-approved and responses are readable by the attacker page.

Two separate holes have to be plugged, because they are enforced in different
places:

1. **HTTP** — `CORSMiddleware` with an explicit origin allowlist, explicit
   methods/headers and `allow_credentials=False` (nothing here is
   cookie-authenticated).
2. **WebSockets** — CORS does **not** apply to WebSocket handshakes at all.
   `new WebSocket('ws://host:6002/ws/chat')` is exempt from preflight and from
   the same-origin policy, so a locked-down `CORSMiddleware` alone would leave
   the *primary* chat path wide open. `WebSocketOriginGuard` closes it.

Absent-Origin policy
--------------------
The guard rejects a handshake only when an `Origin` header is **present and not
allowlisted**. A missing `Origin` is allowed. Browsers always send `Origin` on a
WebSocket handshake; non-browser clients (the ESP32 satellite firmware, the
Android companion app's OkHttp socket, `websocat`, python scripts) never do.
That asymmetry is what lets this block the drive-by browser vector with **zero
client-side changes**.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

# The one port the agent binds (compose: "6002:6002") — dashboard, /api/*,
# /ws/* and /v1/* are all served from it.
AGENT_PORT = 6002

# Methods the dashboard actually issues (frontend/src/lib/api.ts uses GET plus
# POST/PUT/PATCH/DELETE), plus the two the browser needs for preflight and
# conditional GETs. Deliberately not "*".
ALLOWED_METHODS: List[str] = [
    "GET",
    "HEAD",
    "OPTIONS",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
]

# Request headers a browser client may send. The CORS safelist (Accept,
# Accept-Language, Content-Language, Content-Type) is added by Starlette on top
# of this. X-Session-Id / X-Idle-Timeout / X-Device-Name are the agent's own
# session-identity headers (see api/chat.py); Authorization is for
# OpenAI-compat callers hitting /v1/*.
ALLOWED_HEADERS: List[str] = [
    "Authorization",
    "Content-Type",
    "X-Session-Id",
    "X-Idle-Timeout",
    "X-Device-Name",
]

# Response headers a cross-origin caller is allowed to read. X-Session-Id is the
# session handle echoed by POST /api/chat; X-Visemes is the lip-sync timeline
# forwarded by the TTS proxy.
EXPOSED_HEADERS: List[str] = ["X-Session-Id", "X-Visemes"]


def _normalize(origin: str) -> str:
    """Canonical form for comparison: trimmed, no trailing slash, lowercased.

    Browsers already serialize Origin as lowercase scheme+host with the default
    port omitted; this only guards against sloppy hand-written .env entries.
    """
    return origin.strip().rstrip("/").lower()


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def default_origins(host_ip: Optional[str] = None) -> List[str]:
    """Origins the dashboard is legitimately served from, with no .env edit.

    Derived from HOST_IP_ADDRESS so a stock install keeps working:
      - ``http://<HOST_IP_ADDRESS>:6002`` — agent's own published port.
      - ``http://<HOST_IP_ADDRESS>``      — nginx on :80, which reverse-proxies
        ``/``, ``/api/``, ``/ws/`` and ``/v1/...`` to agent:6002.
      - the same pair for ``localhost`` / ``127.0.0.1`` (browsing from the
        Docker host itself).

    Deployments fronted by TLS or a hostname must set AGENT_CORS_ORIGINS —
    a hostname is not derivable from an IP.
    """
    if host_ip is None:
        host_ip = getattr(config, "HOST_IP_ADDRESS", "") or ""
    hosts = [h for h in (host_ip.strip(), "localhost", "127.0.0.1") if h]
    origins: List[str] = []
    for host in hosts:
        origins.append(f"http://{host}:{AGENT_PORT}")
        # Browsers omit the default port when serializing Origin, so nginx on
        # :80 shows up as a bare scheme://host.
        origins.append(f"http://{host}")
    return _dedupe(origins)


def allowed_origins() -> List[str]:
    """The effective allowlist: AGENT_CORS_ORIGINS if set, else the default set."""
    raw = (getattr(config, "AGENT_CORS_ORIGINS", "") or "").strip()
    if raw:
        configured = _dedupe(part.strip() for part in raw.split(","))
        if configured:
            return configured
    return default_origins()


def is_origin_allowed(origin: str, origins: Optional[Sequence[str]] = None) -> bool:
    """True if `origin` is allowlisted. `"*"` in the list allows everything."""
    allowed = list(origins) if origins is not None else allowed_origins()
    if any(o.strip() == "*" for o in allowed):
        return True
    return _normalize(origin) in {_normalize(o) for o in allowed}


def origin_from_scope(scope: Scope) -> Optional[str]:
    """Read the raw `Origin` header off an ASGI scope, or None if absent."""
    for name, value in scope.get("headers") or []:
        if name == b"origin":
            return value.decode("latin-1")
    return None


class WebSocketOriginGuard:
    """Pure-ASGI middleware rejecting cross-origin WebSocket handshakes.

    Installed once on the app, so it covers every current and future ``/ws/*``
    route (``/ws/chat``, ``/ws/logs``, ``/ws/autonomy/runs``) without per-route
    copy-paste. HTTP scopes pass straight through — `CORSMiddleware` owns those.

    A rejected handshake is closed with 1008 (policy violation) *before*
    ``websocket.accept``, which the ASGI server turns into an HTTP 403 on the
    upgrade request. See the module docstring for the absent-Origin rule.
    """

    def __init__(self, app: ASGIApp, origins: Optional[Sequence[str]] = None) -> None:
        self.app = app
        self._origins = list(origins) if origins is not None else None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "websocket":
            await self.app(scope, receive, send)
            return

        origin = origin_from_scope(scope)
        # Absent Origin => non-browser client => allow. Only a present-and-not-
        # allowlisted Origin is a cross-origin browser attempt.
        if origin is not None and not is_origin_allowed(origin, self._origins):
            logger.warning(
                "Rejected cross-origin WebSocket handshake path=%s origin=%s "
                "(not in AGENT_CORS_ORIGINS)",
                scope.get("path"),
                origin,
            )
            # Drain the queued `websocket.connect` before closing so every ASGI
            # server sees a well-formed reject.
            try:
                message = await receive()
            except Exception:  # pragma: no cover - client vanished mid-handshake
                return
            if message.get("type") != "websocket.connect":  # pragma: no cover
                return
            await send({"type": "websocket.close", "code": 1008})
            return

        await self.app(scope, receive, send)


def install_origin_policy(app) -> List[str]:
    """Install the HTTP CORS allowlist + the WebSocket origin guard.

    Both are frozen against the same snapshot of the allowlist, so HTTP and WS
    can never disagree. Returns the effective allowlist for logging.
    """
    origins = allowed_origins()
    app.add_middleware(WebSocketOriginGuard, origins=origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Nothing in this API is cookie- or session-cookie-authenticated; the
        # dashboard never sets `credentials: 'include'`. Keeping this False is
        # what stops an allowlisted-but-compromised origin from riding along on
        # ambient credentials.
        allow_credentials=False,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
        expose_headers=EXPOSED_HEADERS,
    )
    return origins
