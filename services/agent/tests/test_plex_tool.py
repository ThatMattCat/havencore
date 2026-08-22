"""Tests for the Plex MCP tools (selene_agent.modules.mcp_plex_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path compact-JSON wire format with explicit
nulls, and the module's ``{"error": ..., "hint": ...}`` / ``{"error":
"<Type>: <e>"}`` error contract (unchanged from the hand-dispatch era). The
Plex server is mocked at the ``PlexAgent`` layer — no playback is ever
started.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_plex_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Real defaults (limit=10), descriptions, types, and required
    lists must all still match the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


def _make_server():
    from selene_agent.modules.mcp_plex_tools.mcp_server import PlexMCPServer

    return PlexMCPServer()  # initialize() not called: agent stays None


@pytest.mark.asyncio
async def test_mcp_surface_matches_schema_baseline():
    baseline = json.loads(_FIXTURE.read_text())
    server = _make_server()
    tools = {t.name: t for t in await server.mcp.list_tools()}

    assert sorted(tools) == [t["name"] for t in baseline]  # fixture is name-sorted
    for expected in baseline:
        tool = tools[expected["name"]]
        assert tool.description == expected["description"]
        assert tool.output_schema is None  # structured_output=False: plain text results
        got = _strip_generator_noise(tool.input_schema)
        want = _strip_generator_noise(expected["inputSchema"])
        assert got == want, f"schema drift for {expected['name']}"


@pytest.mark.asyncio
async def test_mcp_call_tool_search_tolerates_explicit_null_and_defaults():
    """Explicit JSON null for the optional media_type must behave as "not
    provided" (LLMs send them routinely), omitted limit falls back to the
    schema default of 10, and the result stays one compact-JSON text block
    (json.dumps default=str) — this module's pre-MCPServer wire format."""
    server = _make_server()
    rows = [{"rating_key": "12345", "title": "The Matrix", "year": 1999,
             "type": "movie", "summary": "A hacker learns the truth."}]
    server.agent = MagicMock()
    server.agent.search = AsyncMock(return_value=rows)

    result = await server.mcp.call_tool("plex_search", {
        "query": "matrix",
        "media_type": None,
    })

    assert not result.is_error
    server.agent.search.assert_awaited_once_with(query="matrix", media_type=None, limit=10)
    payload = json.loads(result.content[0].text)
    assert payload == rows
    assert result.content[0].text == json.dumps(rows, default=str)


@pytest.mark.asyncio
async def test_mcp_call_tool_uninitialized_agent_keeps_hint_payload():
    """With no Plex credentials the agent stays None and every tool returns
    the exact configuration-hint error dict of the hand-dispatch era, in
    ordinary (non-isError) content."""
    server = _make_server()
    server.init_error = "PLEX_URL / PLEX_TOKEN not configured"

    result = await server.mcp.call_tool("plex_list_clients", {})

    assert not result.is_error
    assert json.loads(result.content[0].text) == {
        "error": "PLEX_URL / PLEX_TOKEN not configured",
        "hint": "Set PLEX_URL and PLEX_TOKEN in .env and restart the agent service.",
    }


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_payload():
    """An unexpected client exception must surface as the exact
    '{"error": "<Type>: <e>"}' compact JSON of the old handler."""
    server = _make_server()
    server.agent = MagicMock()
    server.agent.list_on_deck = AsyncMock(side_effect=RuntimeError("kaboom"))

    result = await server.mcp.call_tool("plex_list_on_deck", {"limit": 3})

    assert not result.is_error
    assert result.content[0].text == json.dumps({"error": "RuntimeError: kaboom"})
    server.agent.list_on_deck.assert_awaited_once_with(limit=3)


@pytest.mark.asyncio
async def test_mcp_call_tool_missing_required_param_is_validation_error():
    """Missing required params now fail pydantic validation; direct
    call_tool() re-raises as ToolError (over the wire: is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()
    server.agent = MagicMock()

    with pytest.raises(ToolError):
        await server.mcp.call_tool("plex_play", {"rating_key": "12345"})
