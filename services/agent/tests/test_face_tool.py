"""Tests for the face-recognition MCP tools (selene_agent.modules.mcp_face_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path wire format with explicit nulls, and the
module's ``{"error": ...}`` indented-JSON error contract (requests errors map
to the same friendly messages as the hand-dispatch era). The face-recognition
service is mocked at the ``FaceMCPServer._get`` HTTP-helper layer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_face_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Real defaults (hours=24), ranges, enums, descriptions, types,
    and required lists must all still match the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    return schema


def _make_server():
    from selene_agent.modules.mcp_face_tools.face_mcp_server import FaceMCPServer

    return FaceMCPServer()


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
async def test_mcp_call_tool_returns_json_text_and_tolerates_explicit_nulls():
    """Explicit JSON null for the optional camera filter must behave as "not
    provided" (LLMs send them routinely), and the result stays one
    indented-JSON text block — this module's pre-MCPServer wire format."""
    server = _make_server()
    rows = [
        {"person_name": "Matt", "camera": "camera.front_duo_3_fluent",
         "captured_at": "2026-08-22T10:00:00+00:00", "confidence": 0.91},
    ]

    with patch.object(server, "_get", return_value=rows) as get:
        result = await server.mcp.call_tool("face_recent_visitors", {
            "hours": 2,
            "camera": None,
        })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["count"] == 1
    assert payload["camera"] is None  # null filter -> no camera restriction
    assert payload["visitors"][0]["name"] == "Matt"
    # Null camera means no /api/cameras resolution round-trip happened.
    get.assert_called_once()
    assert get.call_args.args[0] == "/api/detections"
    assert get.call_args.kwargs["params"]["since_seconds_ago"] == 2 * 3600
    assert result.content[0].text == json.dumps(payload, indent=2)


@pytest.mark.asyncio
async def test_mcp_call_tool_default_hours_flow_through():
    """Omitted hours falls back to the schema default of 24, exactly like the
    old dict-get dispatch."""
    server = _make_server()

    with patch.object(server, "_get", return_value=[]) as get:
        result = await server.mcp.call_tool("face_recent_visitors", {})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["hours"] == 24
    assert get.call_args.kwargs["params"]["since_seconds_ago"] == 24 * 3600


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_payload():
    """An unexpected impl exception must surface as {"error": ...} indented
    JSON (ordinary content, not a protocol error) — the pre-MCPServer
    contract for this module."""
    server = _make_server()

    with patch.object(server, "_get", side_effect=RuntimeError("kaboom")):
        result = await server.mcp.call_tool("face_list_known_people", {})

    assert not result.is_error
    assert result.content[0].text == json.dumps({"error": "kaboom"}, indent=2)


@pytest.mark.asyncio
async def test_mcp_call_tool_network_error_keeps_friendly_message():
    """The requests.RequestException branch of the old handler survives:
    an unreachable face-recognition service maps to the same friendly text."""
    server = _make_server()

    exc = requests.ConnectionError("nope")
    with patch.object(server, "_get", side_effect=exc):
        result = await server.mcp.call_tool("face_who_is_at", {"camera": "front door"})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {"error": f"face-recognition service unreachable: {exc}"}


@pytest.mark.asyncio
async def test_mcp_call_tool_rejects_invalid_level_via_enum():
    """The Literal enum now rejects bad access levels at validation; direct
    call_tool() re-raises as ToolError (over the wire: is_error=True)."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()

    with pytest.raises(ToolError, match="level"):
        await server.mcp.call_tool("face_set_access_level", {
            "name": "Matt",
            "level": "supreme_overlord",
        })
