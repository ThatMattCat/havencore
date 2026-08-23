"""Tests for the Music Assistant MCP tools (selene_agent.modules.mcp_music_assistant_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path compact-JSON wire format with explicit
nulls, and the module's ``{"error": ..., "hint": ...}`` / ``{"error":
"<Type>: <e>"}`` error contract (unchanged from the hand-dispatch era). The
Music Assistant server is mocked at the ``MassAgent`` layer — no speaker is
ever actuated. (The client's own volume normalization is covered by
``test_mass_play_announcement.py``.)
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_music_assistant_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Real defaults (limit=5, include_hidden=false, mode='replace',
    item_limit=5), enums, descriptions, types, and required lists must all
    still match the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


def _make_server():
    from selene_agent.modules.mcp_music_assistant_tools.mcp_server import MusicAssistantMCPServer

    return MusicAssistantMCPServer()  # initialize() not called: agent stays None


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
    """Explicit JSON null for the optional media_type enum must behave as
    "not provided" (LLMs send them routinely), omitted limit falls back to
    the schema default of 5, and the result stays one compact-JSON text
    block (json.dumps default=str) — this module's pre-MCPServer wire
    format."""
    server = _make_server()
    rows = [{"uri": "library://track/42", "name": "Kryptonite",
             "artist": "3 Doors Down", "album": "The Better Life",
             "media_type": "track", "providers": ["plex"]}]
    server.agent = MagicMock()
    server.agent.search = AsyncMock(return_value=rows)

    result = await server.mcp.call_tool("mass_search", {
        "query": "Kryptonite",
        "media_type": None,
    })

    assert not result.is_error
    server.agent.search.assert_awaited_once_with(query="Kryptonite", media_type=None, limit=5)
    payload = json.loads(result.content[0].text)
    assert payload == rows
    assert result.content[0].text == json.dumps(rows, default=str)


@pytest.mark.asyncio
async def test_mcp_call_tool_list_players_null_include_hidden_reads_false():
    """mass_list_players is the one sync client call in the dispatch; an
    explicit null include_hidden coerces to False exactly like the old
    bool(args.get(..., False))."""
    server = _make_server()
    players = [{"player_id": "lr-001", "display_name": "Living Room",
                "available": True, "powered": True, "state": "idle",
                "volume_level": 30, "current_item": None}]
    server.agent = MagicMock()
    server.agent.list_players = MagicMock(return_value=players)

    result = await server.mcp.call_tool("mass_list_players", {"include_hidden": None})

    assert not result.is_error
    server.agent.list_players.assert_called_once_with(include_hidden=False)
    assert json.loads(result.content[0].text) == players


@pytest.mark.asyncio
async def test_mcp_call_tool_uninitialized_agent_keeps_hint_payload():
    """With no MA credentials the agent stays None and every tool returns
    the exact configuration-hint error dict of the hand-dispatch era, in
    ordinary (non-isError) content."""
    server = _make_server()
    server.init_error = "MASS_URL / MASS_TOKEN not configured"

    result = await server.mcp.call_tool("mass_queue_clear", {"player_name": "Living Room"})

    assert not result.is_error
    assert json.loads(result.content[0].text) == {
        "error": "MASS_URL / MASS_TOKEN not configured",
        "hint": "Set MASS_URL and MASS_TOKEN in .env and restart the agent service.",
    }


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_payload():
    """An unexpected client exception must surface as the exact
    '{"error": "<Type>: <e>"}' compact JSON of the old handler."""
    server = _make_server()
    server.agent = MagicMock()
    server.agent.get_queue = AsyncMock(side_effect=RuntimeError("kaboom"))

    result = await server.mcp.call_tool("mass_get_queue", {"player_name": "Living Room"})

    assert not result.is_error
    assert result.content[0].text == json.dumps({"error": "RuntimeError: kaboom"})
    server.agent.get_queue.assert_awaited_once_with(player_name="Living Room", item_limit=5)


@pytest.mark.asyncio
async def test_mcp_call_tool_rejects_invalid_action_via_enum():
    """The Literal enum now rejects unknown playback actions at validation;
    direct call_tool() re-raises as ToolError (over the wire:
    is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()
    server.agent = MagicMock()

    with pytest.raises(ToolError, match="action"):
        await server.mcp.call_tool("mass_playback_control", {
            "player_name": "Living Room",
            "action": "moonwalk",
        })
