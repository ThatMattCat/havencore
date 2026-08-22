"""Streamable HTTP host for the in-tree MCP tool modules.

One Starlette app serves all 11 modules, one mount per module at
``/mcp/<name>`` (same short names as the agent's ``MCP_SERVERS`` entries).
Runs as the ``mcp-tools`` compose service on port 6010; the agent's
``MCPClientManager`` connects to each mount over MCP Streamable HTTP
instead of spawning stdio subprocesses.

Transport defaults are deliberate and must not change casually:
``stateless_http=False`` (stateful sessions) and ``json_response=False``
(SSE response bodies) — the agent's client and the nginx ``/mcp/``
location are configured for exactly this shape.

Auth: every mount requires its own static bearer token, read at startup
from the env var named in ``MODULES`` (e.g. ``MCP_TOKEN_HOMEASSISTANT``).
A missing token fails that module's mount — nothing is ever served
unauthenticated. Wiring, per module shape: legacy shim modules pass
``streamable_http_app(token_verifier=...)``, which installs the SDK's
``RequireAuthMiddleware`` on the route (401 when unauthenticated);
converted ``MCPServer`` modules can't take a bare verifier there (only
full OAuth ``AuthSettings``), so the host wraps their sub-app in
``RequireAuthMiddleware`` itself. Either way that middleware only checks
``scope["user"]``, and because credential parsing is added by the SDK
only when OAuth ``AuthSettings`` are supplied, every sub-app is
additionally wrapped in Starlette's ``AuthenticationMiddleware`` with
the SDK's ``BearerAuthBackend`` so ``scope["user"]`` is actually
populated.

Module init runs concurrently in the lifespan with per-module
try/except: a module that fails to init (or lacks its token) is logged
and NOT mounted — the agent surfaces it in ``failed_servers`` — and the
rest of the service comes up normally. Successful modules' session
managers run under one ``AsyncExitStack`` (the SDK's documented
"multiple servers, one app" pattern; mounted sub-app lifespans never
execute, so the parent lifespan must run them).
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Set

import uvicorn
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp.server import MCPServer
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import TransportSecuritySettings

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


# --- Module registry --------------------------------------------------------
#
# ``name`` doubles as the mount path (/mcp/<name>) and MUST match the "name"
# field of the corresponding MCP_SERVERS entry in .env. Loaders replicate each
# module's own main() pre-run initialization; constructors run in a worker
# thread because several block on network work (qdrant/mqtt connect in
# __init__, github's clone/fetch can take minutes).


@dataclass
class LoadedModule:
    """A module instance that survived init and is ready to mount.

    Two shapes, distinguished by ``mcp``:

    - legacy shim modules: ``instance`` owns ``.server`` — the low-level mcp
      ``Server`` (via ``modules/_mcp_compat.py``); ``mcp`` is None.
    - converted modules (mcp 2.0 decorator API): ``mcp`` is the module's
      configured ``MCPServer`` (``instance`` still carries the module object
      for symmetry/cleanup).
    """
    name: str
    instance: Any
    cleanup: Optional[Callable[[], Awaitable[None]]] = None
    mcp: Optional[MCPServer] = None


async def _load_general_tools() -> LoadedModule:
    from selene_agent.modules.mcp_general_tools.mcp_server import GeneralToolsServer
    inst = await asyncio.to_thread(GeneralToolsServer)
    return LoadedModule("general_tools", inst, mcp=inst.mcp)


async def _load_qdrant() -> LoadedModule:
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import QdrantMCPServer
    inst = await asyncio.to_thread(QdrantMCPServer)  # network in __init__
    return LoadedModule("mcp_server_qdrant", inst, mcp=inst.mcp)


async def _load_homeassistant() -> LoadedModule:
    from selene_agent.modules.mcp_homeassistant_tools.mcp_server import HomeAssistantMCPServer
    inst = await asyncio.to_thread(HomeAssistantMCPServer)
    await inst.initialize_clients()  # opens the HA WebSocket (records init_error itself)
    return LoadedModule("homeassistant", inst)


async def _load_mqtt() -> LoadedModule:
    from selene_agent.modules.mcp_mqtt_tools.mcp_server import MQTTServer
    inst = await asyncio.to_thread(MQTTServer)  # MQTT connect in __init__
    return LoadedModule("mqtt", inst, mcp=inst.mcp)


async def _load_plex() -> LoadedModule:
    from selene_agent.modules.mcp_plex_tools.mcp_server import PlexMCPServer
    inst = await asyncio.to_thread(PlexMCPServer)
    await asyncio.to_thread(inst.initialize)  # synchronous Plex/HA client setup
    return LoadedModule("plex", inst, mcp=inst.mcp)


async def _load_music_assistant() -> LoadedModule:
    from selene_agent.modules.mcp_music_assistant_tools.mcp_server import MusicAssistantMCPServer
    inst = await asyncio.to_thread(MusicAssistantMCPServer)
    await inst.initialize()  # opens the Music Assistant WebSocket

    async def _cleanup():
        if inst.agent is not None:
            await inst.agent.disconnect()

    return LoadedModule("music_assistant", inst, cleanup=_cleanup, mcp=inst.mcp)


async def _load_github() -> LoadedModule:
    from selene_agent.modules.mcp_github_tools.github_mcp_server import GitHubMCPServer
    # __init__ can block for minutes on the clone/fetch — keep it off the loop.
    inst = await asyncio.to_thread(GitHubMCPServer)
    return LoadedModule("github", inst, mcp=inst.mcp)


async def _load_face() -> LoadedModule:
    from selene_agent.modules.mcp_face_tools.face_mcp_server import FaceMCPServer
    inst = await asyncio.to_thread(FaceMCPServer)
    return LoadedModule("face", inst, mcp=inst.mcp)


async def _load_vision() -> LoadedModule:
    from selene_agent.modules.mcp_vision_tools.server import VisionMCPServer
    inst = await asyncio.to_thread(VisionMCPServer)
    return LoadedModule("vision", inst, mcp=inst.mcp)


async def _load_reminder() -> LoadedModule:
    from selene_agent.modules.mcp_reminder_tools.mcp_server import ReminderToolsServer
    inst = await asyncio.to_thread(ReminderToolsServer)
    return LoadedModule("reminder", inst, mcp=inst.mcp)


async def _load_device_action() -> LoadedModule:
    from selene_agent.modules.mcp_device_action_tools.mcp_server import DeviceActionToolsServer
    inst = await asyncio.to_thread(DeviceActionToolsServer)
    return LoadedModule("device_action", inst, mcp=inst.mcp)


# (name, token env var, loader) — order mirrors MCP_SERVERS for readability.
MODULES: list[tuple[str, str, Callable[[], Awaitable[LoadedModule]]]] = [
    ("general_tools", "MCP_TOKEN_GENERAL_TOOLS", _load_general_tools),
    ("mcp_server_qdrant", "MCP_TOKEN_QDRANT", _load_qdrant),
    ("homeassistant", "MCP_TOKEN_HOMEASSISTANT", _load_homeassistant),
    ("mqtt", "MCP_TOKEN_MQTT", _load_mqtt),
    ("plex", "MCP_TOKEN_PLEX", _load_plex),
    ("music_assistant", "MCP_TOKEN_MUSIC_ASSISTANT", _load_music_assistant),
    ("github", "MCP_TOKEN_GITHUB", _load_github),
    ("face", "MCP_TOKEN_FACE", _load_face),
    ("vision", "MCP_TOKEN_VISION", _load_vision),
    ("reminder", "MCP_TOKEN_REMINDER", _load_reminder),
    ("device_action", "MCP_TOKEN_DEVICE_ACTION", _load_device_action),
]


# --- Auth -------------------------------------------------------------------


class StaticTokenVerifier:
    """SDK ``TokenVerifier`` that accepts exactly one pre-shared token."""

    def __init__(self, token: str, client_id: str):
        self._token = token
        self._client_id = client_id

    async def verify_token(self, token: str) -> AccessToken | None:
        if secrets.compare_digest(token, self._token):
            return AccessToken(token=token, client_id=self._client_id, scopes=[])
        return None


# --- Transport security (DNS-rebinding protection) --------------------------


def _transport_security() -> TransportSecuritySettings:
    """Host/Origin allowlist for the session manager's validation.

    Requests legitimately arrive with three different Host headers:
    ``mcp-tools:6010`` (the agent, direct), the LAN host IP or a name the
    operator uses (via nginx, which forwards the client's Host bare, no
    port), and ``localhost:6010`` / ``127.0.0.1:6010`` (testing). Extra
    names (e.g. a LAN DNS alias) go in MCP_HTTP_ALLOWED_HOSTS
    (comma-separated). Getting this wrong is a silent 421 on every
    request, so keep the list generous.
    """
    hosts: Set[str] = {
        "mcp-tools", "mcp-tools:*",
        "localhost", "localhost:*",
        "127.0.0.1", "127.0.0.1:*",
    }
    host_ip = (config.HOST_IP_ADDRESS or "").strip()
    if host_ip:
        hosts |= {host_ip, f"{host_ip}:*"}
    for extra in os.getenv("MCP_HTTP_ALLOWED_HOSTS", "").split(","):
        extra = extra.strip()
        if extra:
            hosts |= {extra, f"{extra}:*"}

    origins: Set[str] = set()
    for h in hosts:
        origins.add(f"http://{h}")
        origins.add(f"https://{h}")  # nginx may terminate TLS in front of us
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(hosts),
        allowed_origins=sorted(origins),
    )


# --- App assembly -----------------------------------------------------------


@dataclass
class _HostState:
    mounted: Set[str] = field(default_factory=set)
    failed: Dict[str, str] = field(default_factory=dict)
    mount_paths: Set[str] = field(default_factory=set)


STATE = _HostState()


async def _init_one(name: str, token_env: str,
                    loader: Callable[[], Awaitable[LoadedModule]]) -> Optional[LoadedModule]:
    """Init one module; on any failure record it and return None."""
    token = os.environ.get(token_env, "").strip()
    if not token:
        STATE.failed[name] = (
            f"token env var {token_env} is not set — refusing to serve this "
            f"module unauthenticated; not mounted"
        )
        logger.error(f"MCP HTTP host: {name}: {STATE.failed[name]}")
        return None
    try:
        loaded = await loader()
        logger.info(f"MCP HTTP host: initialized module {name}")
        return loaded
    except Exception as e:
        STATE.failed[name] = f"{type(e).__name__}: {e}"
        logger.error(f"MCP HTTP host: init failed for {name}: {STATE.failed[name]}")
        logger.debug(traceback.format_exc())
        return None


async def _health(request: Request) -> JSONResponse:
    ok = bool(STATE.mounted)
    return JSONResponse(
        {
            "status": "ok" if ok else "degraded",
            "mounted": sorted(STATE.mounted),
            "failed": STATE.failed,
        },
        status_code=200 if ok else 503,
    )


@contextlib.asynccontextmanager
async def _lifespan(app: Starlette):
    logger.info(f"MCP HTTP host starting: {len(MODULES)} modules to initialize")
    security = _transport_security()

    results = await asyncio.gather(
        *[_init_one(name, token_env, loader) for name, token_env, loader in MODULES]
    )

    async with contextlib.AsyncExitStack() as stack:
        for (name, token_env, _loader), loaded in zip(MODULES, results):
            if loaded is None:
                continue
            try:
                verifier = StaticTokenVerifier(
                    os.environ[token_env].strip(), client_id=f"mcp-{name}"
                )
                # streamable_http_path="/" because each sub-app is mounted at
                # its full /mcp/<name> prefix. Stateful + SSE defaults, spelled
                # out because the client side depends on them.
                if loaded.mcp is not None:
                    # Converted module (MCPServer decorator API). Its
                    # streamable_http_app() can only thread a token verifier
                    # through when full OAuth AuthSettings are configured, so
                    # for our static-token setup the host applies the same two
                    # SDK middlewares the low-level path gets: the sub-app is
                    # wrapped in RequireAuthMiddleware here (401 when
                    # unauthenticated), and both shapes get BearerAuthBackend
                    # below so scope["user"] is populated.
                    sub_app = loaded.mcp.streamable_http_app(
                        streamable_http_path="/",
                        json_response=False,
                        stateless_http=False,
                        transport_security=security,
                    )
                    session_manager = loaded.mcp.session_manager
                    guarded = RequireAuthMiddleware(sub_app, required_scopes=[])
                else:
                    # Legacy shim module: the low-level streamable_http_app
                    # accepts the verifier directly and installs
                    # RequireAuthMiddleware on its route itself.
                    sub_app = loaded.instance.server.streamable_http_app(
                        streamable_http_path="/",
                        json_response=False,
                        stateless_http=False,
                        transport_security=security,
                        token_verifier=verifier,
                    )
                    session_manager = loaded.instance.server.session_manager
                    guarded = sub_app
                # Mounted sub-apps never run their own lifespan; run the
                # session manager here (must happen on this task — its anyio
                # cancel scope binds to the entering task).
                await stack.enter_async_context(session_manager.run())
                if loaded.cleanup is not None:
                    stack.push_async_callback(loaded.cleanup)
                wrapped = AuthenticationMiddleware(
                    guarded, backend=BearerAuthBackend(verifier)
                )
                path = f"/mcp/{name}"
                app.router.routes.append(Mount(path, app=wrapped))
                STATE.mount_paths.add(path)
                STATE.mounted.add(name)
                logger.info(f"MCP HTTP host: mounted {name} at {path}")
            except Exception as e:
                STATE.failed[name] = f"mount failed: {type(e).__name__}: {e}"
                logger.error(f"MCP HTTP host: {name}: {STATE.failed[name]}")
                logger.debug(traceback.format_exc())

        if STATE.failed:
            logger.error(
                f"MCP HTTP host up with FAILURES: {len(STATE.mounted)} mounted "
                f"({', '.join(sorted(STATE.mounted))}), "
                f"{len(STATE.failed)} failed ({', '.join(sorted(STATE.failed))})"
            )
        else:
            logger.info(
                f"MCP HTTP host up: all {len(STATE.mounted)} modules mounted "
                f"({', '.join(sorted(STATE.mounted))})"
            )
        yield
        logger.info("MCP HTTP host shutting down")


class _SlashlessMountShim:
    """Serve ``/mcp/<name>`` at the exact URL the clients are configured with.

    Starlette answers a mount's bare path with a 307 to the trailing-slash
    form; the MCP clients would follow it, but that doubles every request
    (and every SSE GET) through nginx. Rewriting the path once here makes
    the exact URL first-class instead.
    """

    def __init__(self, app: Starlette, paths: Set[str]):
        self._app = app
        self._paths = paths  # shared, live set — populated during lifespan

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") in self._paths:
            scope = dict(scope)
            scope["path"] = scope["path"] + "/"
            raw = scope.get("raw_path")
            if isinstance(raw, bytes):
                scope["raw_path"] = raw + b"/"
        await self._app(scope, receive, send)


_starlette_app = Starlette(
    routes=[Route("/health", _health, methods=["GET"])],
    lifespan=_lifespan,
)
app = _SlashlessMountShim(_starlette_app, STATE.mount_paths)


# --- Entry point ------------------------------------------------------------


def main():
    port = int(os.getenv("MCP_HTTP_PORT", "6010"))
    # log_config=None for the same reason as selene_agent.py's entrypoint:
    # letting uvicorn run its own logging dictConfig races the Loki queue
    # handler's background thread across the global logging lock and can
    # deadlock startup. This process owns logging via custom_logger.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", log_config=None)


if __name__ == "__main__":
    main()
