"""Tests for the MQTT camera-snapshot MCP tool (selene_agent.modules.mcp_mqtt_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline, the happy-path wire format, the always-listed tool's
"MQTT not connected" fallback result, and the module's ``Error: <e>`` text
error contract (carried over byte-for-byte from the hand-dispatch era).
``HACamSnapper`` (paho thread + HA HTTP) is mocked out entirely.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_mqtt_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``default`` for optional params; it also omits an empty
    ``required`` list where the hand-written schema spelled out ``[]``
    (semantically identical). Everything else must match the baseline.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    if schema.get("required") == []:
        schema.pop("required")
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        prop.pop("default", None)
    return schema


def _make_server():
    from selene_agent.modules.mcp_mqtt_tools import mcp_server

    with patch.object(mcp_server, "HACamSnapper") as snapper_cls:
        server = mcp_server.MQTTServer()
    assert server.snapshotter is snapper_cls.return_value
    return server


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
async def test_mcp_call_tool_returns_indented_json_text():
    server = _make_server()
    server.snapshotter.mqtt_client.is_connected.return_value = True
    payload = {"success": True, "message": "Snapshots captured", "urls": ["http://x/1.jpg"]}
    server.snapshotter.get_camera_snapshots = AsyncMock(return_value=payload)

    result = await server.mcp.call_tool("get_camera_snapshots", {})

    assert not result.is_error
    assert result.content[0].text == json.dumps(payload, indent=2)


@pytest.mark.asyncio
async def test_mcp_call_tool_disconnected_returns_error_result():
    """The tool used to vanish from tools/list while MQTT was down; it is now
    always listed, and a broker outage comes back as an ordinary error result."""
    server = _make_server()
    server.snapshotter.mqtt_client.is_connected.return_value = False
    server.snapshotter.get_camera_snapshots = AsyncMock()

    result = await server.mcp.call_tool("get_camera_snapshots", {})

    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["success"] is False
    assert "MQTT not connected" in payload["error"]
    server.snapshotter.get_camera_snapshots.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_text():
    """An unexpected impl exception must surface as ``Error: <e>`` text in
    ordinary content (not a protocol error) — this module's pre-MCPServer
    contract (unlike reminder's JSON error payloads)."""
    server = _make_server()
    server.snapshotter.mqtt_client.is_connected.return_value = True
    server.snapshotter.get_camera_snapshots = AsyncMock(side_effect=RuntimeError("kaboom"))

    result = await server.mcp.call_tool("get_camera_snapshots", {})

    assert not result.is_error
    assert result.content[0].text == "Error: kaboom"
