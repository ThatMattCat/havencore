"""Tests for the Home Assistant MCP tools (selene_agent.modules.mcp_homeassistant_tools).

Covers the mcp 2.0 ``MCPServer`` decorator surface: schema parity against the
fixtures baseline (including the pinned union-type / default / hidden-alias
schemas), the happy-path wire format with explicit nulls, the module's
init-guard and error contracts (unchanged from the hand-dispatch era), and
the hidden tolerated aliases (``start_time`` / ``days_ahead``) autonomy
passes. The HA REST/WS layer is mocked at the ``HomeAssistantClient`` level —
no service is ever executed against a real Home Assistant.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "tool_schemas" / "mcp_homeassistant_tools.tools.json"


def _strip_generator_noise(schema: dict) -> dict:
    """Drop pydantic's additive schema noise so structure can be compared.

    The mcp 2.0 generator adds a model-level ``title``, a prettified ``title``
    per property, and ``"default": null`` on optionals that had no baseline
    default; the baseline sometimes spells out an empty ``"required": []``
    that the generator omits. Real defaults (hours=24, days=7), descriptions,
    types, enum orders, and non-empty required lists must all still match.
    """
    schema = copy.deepcopy(schema)
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
        if prop.get("default", "sentinel") is None:
            prop.pop("default")
    if schema.get("required") == []:
        schema.pop("required")
    return schema


def _make_server():
    from selene_agent.modules.mcp_homeassistant_tools.mcp_server import HomeAssistantMCPServer

    return HomeAssistantMCPServer()  # initialize_clients() not called: ha_client stays None


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
async def test_mcp_call_tool_list_services_happy_path():
    """Plain happy path: the result is the impl method's pretty-printed JSON
    text (indent=2), exactly the pre-MCPServer wire format."""
    server = _make_server()
    server.ha_client = MagicMock()
    server.ha_client.get_domain_services = AsyncMock(
        return_value={"turn_on": "Turn on", "turn_off": "Turn off"}
    )

    result = await server.mcp.call_tool("ha_list_services", {"domain": "light"})

    assert not result.is_error
    expected = {
        "domain": "light",
        "services": {"turn_on": "Turn on", "turn_off": "Turn off"},
        "count": 2,
    }
    assert result.content[0].text == json.dumps(expected, indent=2)
    server.ha_client.get_domain_services.assert_awaited_once_with("light")


@pytest.mark.asyncio
async def test_mcp_call_tool_control_light_tolerates_explicit_nulls():
    """Explicit JSON nulls for the optional color params must behave as "not
    provided" (LLMs send them routinely): only brightness_pct reaches the HA
    service call, and the plain-string result passes through untouched."""
    server = _make_server()
    server.ha_client = MagicMock()
    server.ha_client.execute_service = AsyncMock(
        return_value="Service light.turn_on executed on light.porch"
    )

    result = await server.mcp.call_tool("ha_control_light", {
        "entity_id": "light.porch",
        "state": "on",
        "brightness_pct": 50,
        "color_name": None,
        "color_temp_kelvin": None,
        "rgb_color": None,
        "hs_color": None,
        "hex_color": None,
    })

    assert not result.is_error
    assert result.content[0].text == "Service light.turn_on executed on light.porch"
    server.ha_client.execute_service.assert_awaited_once_with(
        "light.porch", "turn_on", brightness_pct=50
    )


@pytest.mark.asyncio
async def test_mcp_call_tool_uninitialized_client_keeps_guard_text():
    """Without initialize_clients() every tool returns the exact
    'Home Assistant unavailable: ...' guard text of the hand-dispatch era, in
    ordinary (non-isError) content."""
    server = _make_server()

    result = await server.mcp.call_tool("ha_get_presence", {})

    assert not result.is_error
    assert result.content[0].text == (
        "Home Assistant unavailable: Home Assistant clients not initialized"
    )


@pytest.mark.asyncio
async def test_mcp_call_tool_impl_exception_becomes_error_text():
    """An exception escaping an impl method must surface as the old outer
    handler's exact 'Error: <e>' text payload (is_error stays False)."""
    server = _make_server()
    server.ha_client = MagicMock()
    server._list_areas = AsyncMock(side_effect=RuntimeError("kaboom"))

    result = await server.mcp.call_tool("ha_list_areas", {})

    assert not result.is_error
    assert result.content[0].text == "Error: kaboom"


@pytest.mark.asyncio
async def test_mcp_call_tool_trigger_script_decodes_json_string_variables():
    """The advertised schema says object, but a JSON *string* for `variables`
    must still be decoded server-side (LLMs send those routinely) — the Any
    signature type plus the schema pin keep that pre-MCPServer tolerance."""
    server = _make_server()
    server.ha_client = MagicMock()
    server.ha_client.execute_service = AsyncMock(return_value="Service script.turn_on executed")

    result = await server.mcp.call_tool("ha_trigger_script", {
        "script_entity": "script.bedtime",
        "variables": '{"who": "matt"}',
    })

    assert not result.is_error
    assert result.content[0].text == "Service script.turn_on executed"
    server.ha_client.execute_service.assert_awaited_once_with(
        "script.bedtime", "turn_on", variables={"who": "matt"}
    )


@pytest.mark.asyncio
async def test_mcp_call_tool_hidden_aliases_still_honored():
    """`days_ahead` (ha_get_calendar_events) and `start_time`
    (ha_get_entity_history) are gone from the advertised schemas but must keep
    working — autonomy's anomaly/briefing handlers pass them."""
    server = _make_server()
    server.ha_client = MagicMock()

    async def fake_get(path):
        if path == "/api/calendars":
            return [{"entity_id": "calendar.family"}]
        return []

    server.ha_client._get = AsyncMock(side_effect=fake_get)

    result = await server.mcp.call_tool("ha_get_calendar_events", {"days_ahead": 1})
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["days"] == 1
    assert payload["calendar_entity"] == "calendar.family"

    start = (datetime.now(timezone.utc) - timedelta(hours=5, minutes=1)).isoformat()
    result = await server.mcp.call_tool("ha_get_entity_history", {
        "entity_ids": ["sensor.kitchen_temp"],
        "start_time": start,
    })
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert payload["hours"] == 6  # ceil-ish: seconds // 3600 + 1
    assert payload["entity_id"] == "sensor.kitchen_temp"


@pytest.mark.asyncio
async def test_mcp_call_tool_violated_enum_is_validation_error():
    """A state outside the on/off/toggle enum now fails pydantic validation;
    direct call_tool() re-raises as ToolError (over the wire: is_error=True).
    Same for a missing required param."""
    from mcp.server.mcpserver.exceptions import ToolError

    server = _make_server()
    server.ha_client = MagicMock()

    with pytest.raises(ToolError):
        await server.mcp.call_tool("ha_control_light", {
            "entity_id": "light.porch",
            "state": "dim",
        })

    with pytest.raises(ToolError):
        await server.mcp.call_tool("ha_get_state", {})
