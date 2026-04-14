"""Tests for the watch_llm (LLM-judged reactive triage) handler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _turn_result(content: str):
    return MagicMock(content=content, messages=[], metrics={}, error=None, status="ok")


@pytest.mark.asyncio
async def test_unusual_high_sets_notify_fields(monkeypatch):
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"state": "open"})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "high", '
        '"summary": "front door open at 3am", '
        '"signature": "door-late-night", "evidence": ["no one home"]}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-1",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "subject": "front-door",
            "gather": {"entities": ["binary_sensor.front_door"], "memories_k": 0},
            "severity_floor": "low",
            "notify": {"channel": "signal", "to": "+15550000"},
        },
        "_trigger_event": {"source": "mqtt", "topic": "door", "payload": {"state": "open"}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_unusual"] is True
    assert result["severity"] == "high"
    assert result["_notify_channel"] == "signal"
    assert result["_notify_to"] == "+15550000"
    assert result["signature_hash"]
    assert "front door" in result["_notify_body"]


@pytest.mark.asyncio
async def test_below_severity_floor_drops_notification(monkeypatch):
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "low", '
        '"summary": "minor", "signature": "x", "evidence": []}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-2",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"memories_k": 0},
            "severity_floor": "high",
        },
        "_trigger_event": {"source": "mqtt", "topic": "a", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_unusual"] is False
    assert "below severity" in result["summary"]


@pytest.mark.asyncio
async def test_nominal_judgment_is_not_unusual(monkeypatch):
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": false, "severity": "none", '
        '"summary": "", "signature": "nominal", "evidence": []}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-3",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {"gather": {"memories_k": 0}},
        "_trigger_event": {"source": "mqtt", "topic": "t", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_unusual"] is False
    assert result["severity"] == "none"


@pytest.mark.asyncio
async def test_invalid_json_returns_error(monkeypatch):
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result("this is not JSON"))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-4",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {"gather": {"memories_k": 0}},
        "_trigger_event": {"source": "mqtt", "topic": "t", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "invalid JSON" in result["summary"]


@pytest.mark.asyncio
async def test_speaker_notify_cfg_passthrough(monkeypatch):
    """watch_llm must pass speaker device/voice/volume through _notify_cfg
    so the engine's _build_notifier can assemble a SpeakerNotifier."""
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "med", '
        '"summary": "s", "signature": "sig", "evidence": []}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-5",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"memories_k": 0},
            "notify": {
                "channel": "speaker",
                "device": "Living Room",
                "voice": "af_heart",
                "volume": 0.5,
            },
        },
        "_trigger_event": {"source": "mqtt", "topic": "t", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_notify_channel"] == "speaker"
    speaker = result["_notify_cfg"]["speaker"]
    assert speaker["device"] == "Living Room"
    assert speaker["voice"] == "af_heart"
    assert speaker["volume"] == 0.5
