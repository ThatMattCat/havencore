"""Tests for the general MCP tools (selene_agent.modules.mcp_general_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline (which in this deployment gates all seven tools ON), the
happy-path plain-text wire format with explicit nulls, the ``Error: ...``
error contract, and the pinned ``additionalProperties: false`` on
query_multimodal_api. External APIs are mocked at the ``requests`` /
``query_wikipedia`` layer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_general_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Real defaults (count=4), descriptions, item schemas, types,
    required lists, and additionalProperties must all still match the
    baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


def _make_server():
    from selene_agent.modules.mcp_general_tools.mcp_server import GeneralToolsServer

    return GeneralToolsServer()


@pytest.mark.asyncio
async def test_mcp_surface_matches_schema_baseline():
    """All seven baseline tools must register (every gate is ON in this
    environment) with fixture-identical schemas — including the pinned
    additionalProperties: false on query_multimodal_api."""
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

    assert tools["query_multimodal_api"].input_schema["additionalProperties"] is False


@pytest.mark.asyncio
async def test_brave_search_happy_path_null_count_falls_back_to_default():
    """Explicit JSON null for the optional count must behave as "not
    provided" (LLMs send them routinely) and fall back to 4, and the result
    stays one plain-text block — this module's pre-MCPServer wire format."""
    from selene_agent.modules.mcp_general_tools import mcp_server

    server = _make_server()
    resp = MagicMock()
    resp.json.return_value = {"web": {"results": [
        {"title": "HavenCore", "url": "http://x", "description": "smart home"},
    ]}}

    with patch.object(mcp_server.requests, "get", return_value=resp) as get:
        result = await server.mcp.call_tool("brave_search", {
            "query": "havencore",
            "count": None,
        })

    assert not result.is_error
    text = result.content[0].text
    assert text == "1. HavenCore\n   http://x\n   smart home"
    assert get.call_args.kwargs["params"] == {"q": "havencore", "count": 4}


@pytest.mark.asyncio
async def test_search_wikipedia_default_sentences_flow_through():
    """Omitted sentences falls back to 7, exactly like the old
    arguments.get("sentences", 7) dispatch."""
    from selene_agent.modules.mcp_general_tools import mcp_server

    server = _make_server()

    with patch.object(mcp_server, "query_wikipedia",
                      new=AsyncMock(return_value="A summary.")) as wiki:
        result = await server.mcp.call_tool("search_wikipedia", {
            "search_string": "Selene",
        })

    assert not result.is_error
    assert result.content[0].text == "A summary."
    wiki.assert_awaited_once_with("Selene", 7)


@pytest.mark.asyncio
async def test_impl_exception_becomes_error_text():
    """Any impl exception must surface as the plain 'Error: ...' text in
    ordinary content (not a protocol error) — the pre-MCPServer contract for
    this module. wolfram_alpha maps requests failures to ValueError itself."""
    from selene_agent.modules.mcp_general_tools import mcp_server

    server = _make_server()

    with patch.object(mcp_server, "query_wikipedia",
                      new=AsyncMock(side_effect=RuntimeError("wiki down"))):
        result = await server.mcp.call_tool("search_wikipedia", {
            "search_string": "Selene",
        })

    assert not result.is_error
    assert result.content[0].text == "Error: wiki down"


@pytest.mark.asyncio
async def test_call_tool_rejects_missing_required_param():
    """Required params are now enforced at validation; direct call_tool()
    re-raises as ToolError (over the wire: is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()

    with pytest.raises(ToolError, match="query"):
        await server.mcp.call_tool("brave_search", {})
