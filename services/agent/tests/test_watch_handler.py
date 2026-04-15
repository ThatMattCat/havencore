"""Tests for the watch agenda handler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_watch_renders_body_template_from_payload():
    from selene_agent.autonomy.handlers import watch

    item = {
        "id": "w-1",
        "name": "Front door",
        "config": {
            "title": "Door opened",
            "body_template": "Door is {state}",
            "channel": "ha_push",
            "severity": "warn",
        },
        "_trigger_event": {
            "source": "mqtt",
            "topic": "home/door/front/state",
            "payload": {"state": "open"},
        },
    }
    result = await watch.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["_unusual"] is True
    assert result["_notify_body"] == "Door is open"
    assert result["severity"] == "warn"
    assert result["_notify_channel"] == "ha_push"
    assert result["signature_hash"]
    # Stable signature: two different items + same topic => different hash.
    item2 = dict(item, id="w-2")
    r2 = await watch.handle(
        item2, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["signature_hash"] != r2["signature_hash"]


@pytest.mark.asyncio
async def test_watch_body_template_survives_missing_keys():
    from selene_agent.autonomy.handlers import watch

    item = {
        "id": "w-3",
        "config": {
            "title": "fallback",
            "body_template": "Unknown {missing_key}",
            "channel": "ha_push",
        },
        "_trigger_event": {"source": "mqtt", "topic": "a/b", "payload": {}},
    }
    result = await watch.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    # On render failure the handler falls back to the title.
    assert result["_notify_body"] == "fallback"


@pytest.mark.asyncio
async def test_watch_condition_blocks_when_state_too_recent(monkeypatch):
    from datetime import datetime, timezone
    from selene_agent.autonomy.handlers import watch

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(
        return_value={"state": "on", "last_changed": datetime.now(timezone.utc).isoformat()}
    )
    item = {
        "id": "w-4",
        "config": {
            "condition": {"entity_id": "binary_sensor.door", "min_duration_sec": 300},
            "body_template": "x",
        },
        "_trigger_event": {"source": "mqtt", "topic": "a/b", "payload": {}},
    }
    result = await watch.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["status"] == "ok"
    assert result["summary"] == "condition not held"
    # condition-not-held path does not opt into the unusual notification flow.
    assert result.get("_unusual") is False


@pytest.mark.asyncio
async def test_watch_condition_passes_when_state_stale_enough(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from selene_agent.autonomy.handlers import watch

    mcp = MagicMock()
    old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
    mcp.execute_tool = AsyncMock(return_value={"state": "on", "last_changed": old})
    item = {
        "id": "w-5",
        "config": {
            "condition": {"entity_id": "binary_sensor.door", "min_duration_sec": 300},
            "body_template": "stale",
        },
        "_trigger_event": {"source": "mqtt", "topic": "a/b", "payload": {}},
    }
    result = await watch.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_unusual"] is True
    assert result["_notify_body"] == "stale"
