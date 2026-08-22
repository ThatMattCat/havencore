"""Tests for MCP transport-death detection and bounded reconnect (issue #67).

A crashed MCP stdio subprocess used to be invisible: `is_connected()` only
checked that `client_session is not None`, so `/mcp/status` kept reporting the
dead server as healthy and every tool call burned the full
`MCP_TOOL_TIMEOUT_SECONDS` on a pipe nobody was reading.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import anyio
import pytest

from selene_agent.utils import config
from selene_agent.utils import mcp_client_manager as mcm
from selene_agent.utils.mcp_client_manager import (
    MCPClientManager,
    MCPServerConfig,
    MCPServerConnection,
    ToolSource,
    UnifiedTool,
)

SERVER = "homeassistant"
TOOL = "ha_turn_on"


class FakeResult:
    """Stand-in for CallToolResult (SDK 2.x snake_case field access)."""

    def __init__(self, text: str, is_error: bool = False):
        self.is_error = is_error
        self.content = [type("TextContent", (), {"text": text})()]


class FakeSession:
    """Minimal ClientSession stand-in: records calls, plays a scripted result."""

    def __init__(self, behavior=None):
        self.calls: List[Dict[str, Any]] = []
        self._behavior = behavior or (lambda: FakeResult("ok"))

    async def call_tool(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        outcome = self._behavior()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _build_manager(behavior=None) -> MCPClientManager:
    """A manager with one configured+connected server exposing one tool."""
    mgr = MCPClientManager()
    server_config = MCPServerConfig(name=SERVER, command="python", args=["-m", "x"])
    mgr.add_server(server_config)

    conn = MCPServerConnection(SERVER, server_config)
    conn.client_session = FakeSession(behavior)
    mgr.connections[SERVER] = conn

    mgr.mcp_tools[TOOL] = UnifiedTool(
        name=TOOL,
        description="turn something on",
        parameters={"type": "object", "properties": {}},
        source=ToolSource.MCP,
        server_name=SERVER,
    )
    mgr.server_tools[SERVER] = [TOOL]
    # Keep reconnect timing deterministic; the cap is what matters here.
    mgr.reconnect_backoff_seconds = 0

    # No test may spawn a real MCP subprocess: the default stub makes every
    # reconnect attempt fail instantly. Tests that exercise recovery override
    # this with their own fake.
    async def no_subprocess_connect(server_name, server_config):
        raise RuntimeError("stubbed: no real MCP subprocess in tests")

    mgr._connect_to_server = no_subprocess_connect  # type: ignore[assignment]
    return mgr


async def _drain_reconnect(mgr: MCPClientManager):
    """Await whatever reconnect task execute_tool scheduled (if any)."""
    task = mgr._reconnect_tasks.get(SERVER)
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


# --- classifier -----------------------------------------------------------


def test_classifier_separates_transport_from_tool_errors():
    assert mcm.is_transport_error(anyio.BrokenResourceError()) is True
    assert mcm.is_transport_error(anyio.ClosedResourceError()) is True
    assert mcm.is_transport_error(anyio.EndOfStream()) is True
    assert mcm.is_transport_error(BrokenPipeError(32, "Broken pipe")) is True
    assert mcm.is_transport_error(ConnectionResetError()) is True
    # Wrapped by an anyio task group / explicit chaining.
    assert mcm.is_transport_error(
        ExceptionGroup("tg", [anyio.BrokenResourceError()])
    ) is True
    chained = RuntimeError("write failed")
    chained.__cause__ = BrokenPipeError()
    assert mcm.is_transport_error(chained) is True

    # Ordinary tool failures must never be mistaken for a dead pipe.
    assert mcm.is_transport_error(ValueError("entity_id is required")) is False
    assert mcm.is_transport_error(RuntimeError("tool blew up")) is False
    assert mcm.is_transport_error(KeyError("missing")) is False
    assert mcm.is_transport_error(asyncio.TimeoutError()) is False
    # An MCPError means the server answered us — it is demonstrably alive.
    if mcm._MCPError is not None:
        assert mcm.is_transport_error(
            mcm._MCPError(-32602, "Unknown tool")
        ) is False


# --- detect ---------------------------------------------------------------


async def test_transport_failure_marks_server_not_connected():
    """A broken pipe during execute_tool must flip the server to dead and show
    up in get_server_status() instead of being reported as connected."""
    mgr = _build_manager(behavior=lambda: anyio.ClosedResourceError())
    conn = mgr.connections[SERVER]
    assert conn.is_connected() is True

    result = await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})

    assert "connection lost" in result
    assert conn.is_connected() is False

    status = mgr.get_server_status()
    assert status["connected_servers"] == []
    assert SERVER in status["disconnected_servers"]

    await _drain_reconnect(mgr)


async def test_ordinary_tool_error_does_not_mark_server_dead():
    """Regression guard: a tool that ran and failed must not take its server
    (and the other 19 tools on it) down with it."""
    mgr = _build_manager(behavior=lambda: RuntimeError("upstream 500 from HA"))
    conn = mgr.connections[SERVER]

    result = await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})

    assert "Error executing MCP tool" in result
    assert conn.is_connected() is True

    status = mgr.get_server_status()
    assert status["connected_servers"] == [SERVER]
    assert status["disconnected_servers"] == {}
    # No reconnect storm for something that was never a transport problem.
    assert mgr._reconnect_tasks == {}


async def test_tool_reported_error_result_does_not_mark_server_dead():
    """`isError=True` means the tool ran — the subprocess is demonstrably alive."""
    mgr = _build_manager(behavior=lambda: FakeResult("no such entity", is_error=True))
    conn = mgr.connections[SERVER]

    result = await mgr.execute_tool(TOOL, {"entity_id": "light.nope"})

    assert "MCP tool error" in result
    assert conn.is_connected() is True
    assert mgr.get_server_status()["disconnected_servers"] == {}
    assert mgr._reconnect_tasks == {}


# --- recover --------------------------------------------------------------


async def test_reconnect_restores_server_and_tools():
    """After a transport death the bounded reconnect re-establishes the server
    and tool calls work again."""
    mgr = _build_manager(behavior=lambda: anyio.BrokenResourceError())
    attempts: List[str] = []

    async def fake_connect(server_name, server_config):
        attempts.append(server_name)
        fresh = MCPServerConnection(server_name, server_config)
        fresh.client_session = FakeSession()  # healthy again
        mgr.connections[server_name] = fresh

    mgr._connect_to_server = fake_connect  # type: ignore[assignment]

    await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})
    assert mgr.get_server_status()["connected_servers"] == []

    await _drain_reconnect(mgr)

    assert attempts == [SERVER]
    assert mgr.get_server_status()["connected_servers"] == [SERVER]
    assert mgr.connections[SERVER].is_connected() is True

    # The tool works again over the fresh session.
    assert await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"}) == "ok"


async def test_reconnect_is_bounded():
    """A server module that will never come up must not be retried forever."""
    mgr = _build_manager(behavior=lambda: anyio.BrokenResourceError())
    attempts: List[str] = []

    async def always_failing_connect(server_name, server_config):
        attempts.append(server_name)
        raise RuntimeError("module import failed")

    mgr._connect_to_server = always_failing_connect  # type: ignore[assignment]

    await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})
    await _drain_reconnect(mgr)

    assert len(attempts) == mgr.reconnect_max_attempts
    assert SERVER in mgr.failed_servers

    # Further tool calls fail fast without re-arming the retry budget.
    for _ in range(3):
        with pytest.raises(ValueError):
            await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})
        await _drain_reconnect(mgr)

    assert len(attempts) == mgr.reconnect_max_attempts


async def test_reconnect_backoff_grows_between_attempts(monkeypatch):
    """Attempt 1 is immediate; later attempts back off exponentially."""
    mgr = _build_manager(behavior=lambda: anyio.BrokenResourceError())
    mgr.reconnect_backoff_seconds = 2.0
    delays: List[float] = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def always_failing_connect(server_name, server_config):
        raise RuntimeError("nope")

    mgr._connect_to_server = always_failing_connect  # type: ignore[assignment]

    await mgr.execute_tool(TOOL, {})
    await _drain_reconnect(mgr)

    assert delays == [2.0, 4.0]  # no sleep before attempt 1


# --- fail fast ------------------------------------------------------------


async def test_dead_server_call_fails_fast(monkeypatch):
    """Once a server is known dead, later calls must not burn the full
    MCP_TOOL_TIMEOUT_SECONDS on a pipe nobody is reading.

    Mirrors the real failure: the subprocess dies (call 1 raises a transport
    error), and from then on the dead pipe simply never answers.
    """
    state = {"calls": 0}

    async def dying_call_tool(name, arguments):
        state["calls"] += 1
        if state["calls"] == 1:
            raise anyio.ClosedResourceError()
        await asyncio.Event().wait()  # dead pipe: no response, ever

    mgr = _build_manager()
    conn = mgr.connections[SERVER]
    conn.client_session.call_tool = dying_call_tool  # type: ignore[assignment]
    # Isolate the fail-fast behavior from recovery.
    mgr._schedule_reconnect = lambda server_name: None  # type: ignore[assignment]
    # Short but non-trivial window: without the fix the call waits it out.
    monkeypatch.setattr(config, "MCP_TOOL_TIMEOUT_SECONDS", 2.0)

    await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})  # observes death

    started = time.monotonic()
    with pytest.raises(ValueError):
        await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert state["calls"] == 1  # never touched the dead pipe again


async def test_dead_server_call_schedules_reconnect():
    """Failing fast still arms recovery rather than giving up silently."""
    mgr = _build_manager()
    mgr.connections[SERVER].mark_dead("subprocess exited")

    reconnected = asyncio.Event()

    async def fake_connect(server_name, server_config):
        fresh = MCPServerConnection(server_name, server_config)
        fresh.client_session = FakeSession()
        mgr.connections[server_name] = fresh
        reconnected.set()

    mgr._connect_to_server = fake_connect  # type: ignore[assignment]

    with pytest.raises(ValueError):
        await mgr.execute_tool(TOOL, {"entity_id": "light.kitchen"})

    await _drain_reconnect(mgr)
    assert reconnected.is_set()
    assert mgr.get_server_status()["connected_servers"] == [SERVER]


# --- teardown -------------------------------------------------------------


async def test_cleanup_disconnects_connections_retired_by_reconnect():
    """Connections replaced by a reconnect still hold un-exited MCP contexts;
    cleanup() must unwind them too."""
    mgr = _build_manager(behavior=lambda: anyio.BrokenResourceError())
    old = mgr.connections[SERVER]
    old.disconnect = AsyncMock()  # type: ignore[assignment]

    async def fake_connect(server_name, server_config):
        fresh = MCPServerConnection(server_name, server_config)
        fresh.client_session = FakeSession()
        fresh.disconnect = AsyncMock()  # type: ignore[assignment]
        mgr.connections[server_name] = fresh

    mgr._connect_to_server = fake_connect  # type: ignore[assignment]

    await mgr.execute_tool(TOOL, {})
    await _drain_reconnect(mgr)
    new = mgr.connections[SERVER]

    await mgr.cleanup()

    old.disconnect.assert_awaited_once()
    new.disconnect.assert_awaited_once()
