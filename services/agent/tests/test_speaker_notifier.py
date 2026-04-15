"""Tests for SpeakerNotifier — the TTS → audio_store → MA pipeline."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_notifier(mcp_tool_return=None, synth_return=b"fake-mp3-bytes"):
    from selene_agent.autonomy.notifiers import SpeakerNotifier
    from selene_agent.services.audio_store import AudioStore

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(
        return_value=mcp_tool_return if mcp_tool_return is not None
        else {"success": True, "played": True}
    )
    tts = MagicMock()
    tts.synth = AsyncMock(return_value=synth_return)
    store = AudioStore(default_ttl_sec=60)
    notifier = SpeakerNotifier(
        mcp,
        device="Living Room",
        voice="af_heart",
        volume=0.4,
        tts_client=tts,
        audio_store=store,
        base_url="http://agent:6002",
    )
    return notifier, mcp, tts, store


@pytest.mark.asyncio
async def test_send_happy_path_invokes_tts_then_ma():
    notifier, mcp, tts, store = _make_notifier()
    ok = await notifier.send(title="Alarm", body="The door is open.")
    assert ok is True
    tts.synth.assert_awaited_once()
    mcp.execute_tool.assert_awaited_once()
    name, payload = mcp.execute_tool.await_args.args
    assert name == "mass_play_announcement"
    assert payload["player_name"] == "Living Room"
    assert payload["volume"] == 0.4
    assert payload["url"].startswith("http://agent:6002/api/tts/audio/")
    assert payload["url"].endswith(".mp3")


@pytest.mark.asyncio
async def test_send_no_device_returns_false():
    from selene_agent.autonomy.notifiers import SpeakerNotifier

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock()
    tts = MagicMock()
    tts.synth = AsyncMock(return_value=b"x")
    notifier = SpeakerNotifier(mcp, device="", tts_client=tts)
    ok = await notifier.send(title="t", body="b")
    assert ok is False
    mcp.execute_tool.assert_not_awaited()
    tts.synth.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_tts_failure_does_not_call_ma():
    from selene_agent.autonomy.notifiers import SpeakerNotifier
    from selene_agent.services.audio_store import AudioStore

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock()
    tts = MagicMock()
    tts.synth = AsyncMock(side_effect=RuntimeError("kokoro down"))
    notifier = SpeakerNotifier(
        mcp, device="Living Room", tts_client=tts, audio_store=AudioStore()
    )
    ok = await notifier.send(title="t", body="b")
    assert ok is False
    mcp.execute_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_ma_played_false_marks_failure():
    notifier, mcp, _tts, _store = _make_notifier(
        mcp_tool_return={"success": True, "played": False, "error": "player offline"}
    )
    ok = await notifier.send(title="t", body="b")
    assert ok is False


@pytest.mark.asyncio
async def test_staged_audio_readable_via_store():
    notifier, _mcp, _tts, store = _make_notifier()
    await notifier.send(title="Hi", body="there")
    # The token lived in the URL; extract and read it from the store.
    url = _mcp_last_url(notifier)
    assert url  # sanity
    token = url.rsplit("/", 1)[1].replace(".mp3", "")
    # Single-fetch: the router will pop, so reading here asserts it was stored.
    data = await store.get(token)
    assert data is not None
    assert data[0] == b"fake-mp3-bytes"


def _mcp_last_url(notifier):
    mcp = notifier.mcp_manager
    if not mcp.execute_tool.await_args:
        return None
    return mcp.execute_tool.await_args.args[1]["url"]
