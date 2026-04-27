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


# --- v2: LLM-picked channel + safety rails --------------------------------

@pytest.mark.asyncio
async def test_llm_silent_collapses_to_nominal(monkeypatch):
    """When the LLM emits channel=silent it's saying 'on reflection, nothing
    to do here' — handler must mark _unusual=False so the cooldown bookkeeping
    treats it consistently with a true nominal."""
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "med", '
        '"summary": "x", "signature": "sig", "evidence": [], '
        '"channel": "silent"}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-silent",
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
async def test_llm_picks_speaker_when_someone_home(monkeypatch):
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    # gather presence call returns persons w/ one home
    mcp.execute_tool = AsyncMock(return_value={
        "persons": [{"entity_id": "person.matt", "state": "home"}],
        "device_trackers": [],
    })
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "med", '
        '"summary": "stranger at door", "signature": "sig", "evidence": [], '
        '"channel": "speaker", "urgency": "alert"}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-spk",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"presence": True, "memories_k": 0},
            "notify": {"channel": "signal"},
        },
        "_trigger_event": {"source": "mqtt", "topic": "t", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_notify_channel"] == "speaker"
    assert result["_notify_urgency"] == "alert"


@pytest.mark.asyncio
async def test_speaker_downgrades_to_signal_when_no_one_home(monkeypatch):
    """Safety rail: if presence shows no resident is home, the speaker
    channel is downgraded so the alert still reaches the user via Signal."""
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={
        "persons": [{"entity_id": "person.matt", "state": "not_home"}],
        "device_trackers": [],
    })
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "high", '
        '"summary": "front door at 3am", "signature": "sig", "evidence": [], '
        '"channel": "speaker"}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-down",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"presence": True, "memories_k": 0},
            "notify": {"channel": "signal"},
        },
        "_trigger_event": {"source": "mqtt", "topic": "t", "payload": {}},
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    # Speaker would be silly if nobody's there to hear it — Signal carries
    # the alert instead.
    assert result["_notify_channel"] == "signal"


@pytest.mark.asyncio
async def test_snapshot_attachment_threaded_for_signal(monkeypatch):
    """When the inbound event carries a sensor_event.snapshot_url and
    attach_snapshot is enabled, the handler returns _notify_attachments so
    the engine forwards it to SignalNotifier."""
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={})
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "med", '
        '"summary": "unknown face", "signature": "sig", "evidence": [], '
        '"channel": "signal"}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-attach",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"memories_k": 0},
            "notify": {"channel": "signal"},
            "attach_snapshot": True,
        },
        "_trigger_event": {
            "source": "mqtt",
            "topic": "haven/face/unknown",
            "payload": {},
            "sensor_event": {
                "domain": "face",
                "kind": "unknown",
                "zone": "front_door",
                "snapshot_url": "http://agent:6002/api/face/detections/d/snapshot",
                "subject": {"type": "person", "identity": None},
                "raw": {},
            },
        },
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_notify_attachments"] == [
        "http://agent:6002/api/face/detections/d/snapshot"
    ]


@pytest.mark.asyncio
async def test_attach_snapshot_skipped_when_channel_is_speaker(monkeypatch):
    """Speaker can't carry a binary attachment; handler must not put a
    snapshot URL on a speaker run."""
    from selene_agent.autonomy.handlers import watch_llm

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={
        "persons": [{"entity_id": "person.matt", "state": "home"}],
    })
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_turn_result(
        '{"unusual": true, "severity": "med", '
        '"summary": "x", "signature": "sig", "evidence": [], '
        '"channel": "speaker"}'
    ))
    monkeypatch.setattr(watch_llm, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "wl-spk-no-attach",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "config": {
            "gather": {"presence": True, "memories_k": 0},
            "attach_snapshot": True,
        },
        "_trigger_event": {
            "source": "mqtt",
            "topic": "haven/face/unknown",
            "payload": {},
            "sensor_event": {
                "snapshot_url": "http://agent:6002/x.jpg",
                "raw": {},
            },
        },
    }
    result = await watch_llm.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["_notify_channel"] == "speaker"
    assert result["_notify_attachments"] is None
