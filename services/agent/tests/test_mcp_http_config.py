"""Tests for the Streamable HTTP MCP client config (Phase 2 cutover).

MCP_SERVERS entries may now be HTTP ({"name", "url", "token_env"}) as well
as the legacy stdio form ({"name", "command", "args"}). Tokens are resolved
by env-var indirection at connect time, never inlined in the JSON.
"""
from __future__ import annotations

import json

import pytest

from selene_agent.utils import config
from selene_agent.utils import mcp_client_manager as mcm
from selene_agent.utils.mcp_client_manager import (
    MCPClientManager,
    MCPServerConfig,
    MCPServerConnection,
)


# --- config entry parsing --------------------------------------------------


def _load(monkeypatch, entries) -> MCPClientManager:
    """Run selene_agent's loader against a scripted MCP_SERVERS value."""
    from selene_agent.selene_agent import _load_mcp_server_configs

    monkeypatch.setattr(config, "MCP_SERVERS", json.dumps(entries))
    mgr = MCPClientManager()
    _load_mcp_server_configs(mgr)
    return mgr


def test_http_entries_parse_url_and_token_env(monkeypatch):
    mgr = _load(monkeypatch, [
        {
            "name": "reminder",
            "url": "http://mcp-tools:6010/mcp/reminder",
            "token_env": "MCP_TOKEN_REMINDER",
            "enabled": True,
        },
    ])

    assert list(mgr.servers) == ["reminder"]
    cfg = mgr.servers["reminder"]
    assert cfg.transport == "http"
    assert cfg.url == "http://mcp-tools:6010/mcp/reminder"
    assert cfg.token_env == "MCP_TOKEN_REMINDER"
    assert cfg.command is None


def test_stdio_entries_still_parse(monkeypatch):
    """stdio retirement is a later phase — command entries must keep working."""
    mgr = _load(monkeypatch, [
        {
            "name": "mqtt",
            "command": "python",
            "args": ["-m", "selene_agent.modules.mcp_mqtt_tools"],
            "enabled": True,
        },
    ])

    cfg = mgr.servers["mqtt"]
    assert cfg.transport == "stdio"
    assert cfg.command == "python"
    assert cfg.url is None
    params = cfg.to_stdio_params()
    assert params.command == "python"


def test_mixed_transports_coexist(monkeypatch):
    mgr = _load(monkeypatch, [
        {"name": "reminder", "url": "http://mcp-tools:6010/mcp/reminder",
         "token_env": "MCP_TOKEN_REMINDER"},
        {"name": "mqtt", "command": "python", "args": ["-m", "x"]},
    ])

    assert mgr.servers["reminder"].transport == "http"
    assert mgr.servers["mqtt"].transport == "stdio"


def test_entry_without_url_or_command_is_skipped(monkeypatch):
    mgr = _load(monkeypatch, [
        {"name": "broken", "enabled": True},
        {"url": "http://mcp-tools:6010/mcp/nameless"},
        {"name": "reminder", "url": "http://mcp-tools:6010/mcp/reminder"},
    ])

    assert list(mgr.servers) == ["reminder"]


def test_disabled_http_entry_is_not_added(monkeypatch):
    mgr = _load(monkeypatch, [
        {"name": "reminder", "url": "http://mcp-tools:6010/mcp/reminder",
         "enabled": False},
    ])

    assert mgr.servers == {}


# --- bearer token resolution ------------------------------------------------


def test_bearer_token_resolves_from_env(monkeypatch):
    monkeypatch.setenv("MCP_TOKEN_REMINDER", "sekrit-token")
    cfg = MCPServerConfig(
        name="reminder",
        url="http://mcp-tools:6010/mcp/reminder",
        token_env="MCP_TOKEN_REMINDER",
    )
    assert cfg.bearer_token() == "sekrit-token"


def test_bearer_token_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("MCP_TOKEN_REMINDER", raising=False)
    cfg = MCPServerConfig(
        name="reminder",
        url="http://mcp-tools:6010/mcp/reminder",
        token_env="MCP_TOKEN_REMINDER",
    )
    with pytest.raises(RuntimeError, match="MCP_TOKEN_REMINDER"):
        cfg.bearer_token()


def test_bearer_token_none_without_token_env():
    cfg = MCPServerConfig(name="open", url="http://mcp-tools:6010/mcp/open")
    assert cfg.bearer_token() is None


async def test_http_connect_fails_cleanly_on_missing_token(monkeypatch):
    """A missing token env var must fail the connect (-> failed_servers),
    never fall through to an unauthenticated request."""
    monkeypatch.delenv("MCP_TOKEN_REMINDER", raising=False)
    cfg = MCPServerConfig(
        name="reminder",
        url="http://mcp-tools:6010/mcp/reminder",
        token_env="MCP_TOKEN_REMINDER",
    )
    conn = MCPServerConnection("reminder", cfg)
    with pytest.raises(RuntimeError, match="MCP_TOKEN_REMINDER"):
        await conn.connect()
    assert conn._httpx_client is None  # nothing half-opened


# --- HTTP transport-error classification ------------------------------------


def test_httpx_transport_errors_classify_as_transport():
    import httpx2

    assert mcm.is_transport_error(httpx2.ConnectError("refused")) is True
    assert mcm.is_transport_error(httpx2.ReadError("reset")) is True
    assert mcm.is_transport_error(
        httpx2.RemoteProtocolError("server disconnected")
    ) is True
    # Wrapped the way anyio task groups deliver them.
    assert mcm.is_transport_error(
        ExceptionGroup("tg", [httpx2.ConnectError("refused")])
    ) is True


def test_http_status_error_is_not_transport():
    """The server answered (401/500/...) — the connection itself is fine."""
    import httpx2

    request = httpx2.Request("POST", "http://mcp-tools:6010/mcp/reminder")
    response = httpx2.Response(401, request=request)
    exc = httpx2.HTTPStatusError("401", request=request, response=response)
    assert mcm.is_transport_error(exc) is False


def test_session_death_mcp_errors_are_transport():
    """A dead session after an mcp-tools restart surfaces as an MCPError in
    one of two spellings: "Session not found" (the SDK server's own 404
    body, relayed verbatim by the client — the one seen live) or "Session
    terminated" (client-synthesized for a bare 404). Both must trigger
    reconnect, unlike every other MCPError."""
    assert mcm._MCPError is not None
    assert mcm.is_transport_error(
        mcm._MCPError(-32600, "Session not found")
    ) is True
    assert mcm.is_transport_error(
        mcm._MCPError(-32600, "Session terminated")
    ) is True
    # Ordinary JSON-RPC errors still prove the server is alive.
    assert mcm.is_transport_error(
        mcm._MCPError(-32602, "Unknown tool")
    ) is False


# --- reconnect re-arm after cool-down ---------------------------------------


async def test_exhausted_reconnect_rearms_after_cooldown():
    """After an abandoned cycle, a later tool call past the cool-down re-arms
    a fresh bounded cycle instead of staying dead until an agent restart
    (needed for outages longer than one cycle, e.g. an mcp-tools restart)."""
    import anyio

    from tests.test_mcp_reconnect import FakeSession, _build_manager, _drain_reconnect, SERVER, TOOL

    mgr = _build_manager(behavior=lambda: anyio.BrokenResourceError())
    attempts = []

    async def always_failing_connect(server_name, server_config):
        attempts.append(server_name)
        raise RuntimeError("still down")

    mgr._connect_to_server = always_failing_connect  # type: ignore[assignment]

    await mgr.execute_tool(TOOL, {})
    await _drain_reconnect(mgr)
    assert len(attempts) == mgr.reconnect_max_attempts
    assert SERVER in mgr.failed_servers

    # Within the cool-down: fail fast, no re-arm.
    with pytest.raises(ValueError):
        await mgr.execute_tool(TOOL, {})
    await _drain_reconnect(mgr)
    assert len(attempts) == mgr.reconnect_max_attempts

    # Cool-down elapsed and the server is back: the next call re-arms and
    # the reconnect succeeds.
    mgr.reconnect_rearm_cooldown_seconds = 0.0

    async def now_working_connect(server_name, server_config):
        attempts.append(server_name)
        fresh = MCPServerConnection(server_name, mgr.servers[server_name])
        fresh.client_session = FakeSession()
        mgr.connections[server_name] = fresh

    mgr._connect_to_server = now_working_connect  # type: ignore[assignment]

    with pytest.raises(ValueError):
        await mgr.execute_tool(TOOL, {})
    await _drain_reconnect(mgr)

    assert len(attempts) == mgr.reconnect_max_attempts + 1
    assert mgr.get_server_status()["connected_servers"] == [SERVER]
    assert SERVER not in mgr.failed_servers
    assert await mgr.execute_tool(TOOL, {}) == "ok"
