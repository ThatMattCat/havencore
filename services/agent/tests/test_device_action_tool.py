"""Tests for the device-action MCP tools (selene_agent.modules.mcp_device_action_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path compact-JSON wire format with explicit
nulls, the exact camera-tool fallback texts (the orchestrator answers these
tools in-process; the MCP handlers are deliberate no-op fallbacks), and the
``{"status": "error"}`` error contract.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_device_action_tools.tools.json"

CAMERA_TOOLS = (
    "take_photo",
    "identify_object_in_photo",
    "read_text_from_image",
    "who_is_in_view",
)


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default. Ranges (hour/minute/days_of_week items), descriptions, types,
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
    from selene_agent.modules.mcp_device_action_tools.mcp_server import DeviceActionToolsServer

    return DeviceActionToolsServer()


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
async def test_mcp_call_tool_set_alarm_tolerates_explicit_nulls():
    """Explicit JSON nulls for label/days_of_week must behave as "not
    provided" (LLMs send them routinely), and the result stays one
    compact-JSON text block — this module never indented."""
    server = _make_server()

    result = await server.mcp.call_tool("set_alarm", {
        "hour": 7,
        "minute": 30,
        "label": None,
        "days_of_week": None,
    })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {
        "status": "scheduled",
        "hour": 7,
        "minute": 30,
        "label": None,
        "days_of_week": [],  # null filter -> one-off alarm, exactly like absent
    }
    assert result.content[0].text == json.dumps(payload)


@pytest.mark.asyncio
async def test_mcp_call_tool_set_alarm_repeating_days_flow_through():
    server = _make_server()

    result = await server.mcp.call_tool("set_alarm", {
        "hour": 6,
        "minute": 0,
        "label": "gym",
        "days_of_week": [2, 4, 6],
    })

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "scheduled"
    assert payload["label"] == "gym"
    assert payload["days_of_week"] == [2, 4, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", CAMERA_TOOLS)
async def test_mcp_call_tool_camera_tools_keep_exact_fallback_text(tool_name):
    """The camera tools are deliberate no-op fallbacks — the orchestrator
    short-circuits them through COMPANION_UPLOAD_TOOLS and never reaches the
    MCP path. Anything that does must get the exact benign error, unchanged
    from the hand-dispatch era."""
    server = _make_server()

    result = await server.mcp.call_tool(tool_name, {})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload == {
        "status": "error",
        "error": (
            f"{tool_name} must be invoked through the orchestrator's "
            "companion-upload path; direct MCP invocation is not supported."
        ),
    }
    assert result.content[0].text == json.dumps(payload)


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_payload():
    """An unexpected impl exception must surface as a compact-JSON
    {"status": "error"} payload (ordinary content, not a protocol error) —
    the pre-MCPServer contract for this module."""
    server = _make_server()

    with patch.object(server, "companion_camera_fallback", side_effect=RuntimeError("kaboom")):
        result = await server.mcp.call_tool("take_photo", {"reason": "testing"})

    assert not result.is_error
    assert result.content[0].text == json.dumps({"status": "error", "error": "kaboom"})


@pytest.mark.asyncio
async def test_mcp_call_tool_rejects_out_of_range_hour_via_schema():
    """hour now carries ge/le constraints in the schema; direct call_tool()
    re-raises the validation failure as ToolError (over the wire:
    is_error=True) where the old dispatch returned the impl's error dict."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()

    with pytest.raises(ToolError, match="hour"):
        await server.mcp.call_tool("set_alarm", {"hour": 99, "minute": 0})
