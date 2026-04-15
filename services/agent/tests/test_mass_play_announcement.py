"""Tests for the MA client's play_announcement volume normalization.

We avoid spinning the real MusicAssistantClient; just prod the normalization
branch directly against a fake player resolver.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _fake_agent():
    from selene_agent.modules.mcp_music_assistant_tools import mass_client

    agent = mass_client.MassAgent.__new__(mass_client.MassAgent)
    client = MagicMock()
    client.players.play_announcement = AsyncMock()
    agent._client = client  # type: ignore[attr-defined]

    player = SimpleNamespace(
        name="Living Room", player_id="lr-001", hide_in_ui=False
    )
    agent._resolve_player = lambda name: player  # type: ignore[assignment]
    agent._require_client = lambda: client  # type: ignore[assignment]
    return agent, client


@pytest.mark.asyncio
async def test_volume_float_normalized_to_percent():
    agent, client = _fake_agent()
    result = await agent.play_announcement("Living Room", "http://x/a.mp3", volume=0.4)
    assert result["played"] is True
    kwargs = client.players.play_announcement.await_args.kwargs
    assert kwargs["volume_level"] == 40
    assert kwargs["url"] == "http://x/a.mp3"


@pytest.mark.asyncio
async def test_volume_int_passthrough():
    agent, client = _fake_agent()
    result = await agent.play_announcement("Living Room", "http://x/a.mp3", volume=75)
    assert result["played"] is True
    assert client.players.play_announcement.await_args.kwargs["volume_level"] == 75


@pytest.mark.asyncio
async def test_volume_none_leaves_unset():
    agent, client = _fake_agent()
    result = await agent.play_announcement("Living Room", "http://x/a.mp3")
    assert result["played"] is True
    assert client.players.play_announcement.await_args.kwargs["volume_level"] is None


@pytest.mark.asyncio
async def test_volume_clamped_to_range():
    agent, client = _fake_agent()
    await agent.play_announcement("Living Room", "http://x/a.mp3", volume=250)
    assert client.players.play_announcement.await_args.kwargs["volume_level"] == 100
    await agent.play_announcement("Living Room", "http://x/a.mp3", volume=-0.5)
    assert client.players.play_announcement.await_args.kwargs["volume_level"] == 0


@pytest.mark.asyncio
async def test_unknown_player_returns_error():
    from selene_agent.modules.mcp_music_assistant_tools import mass_client

    agent = mass_client.MassAgent.__new__(mass_client.MassAgent)
    client = MagicMock()
    client.players.play_announcement = AsyncMock()
    client.players = [SimpleNamespace(name="Kitchen", hide_in_ui=False)]
    agent._client = client  # type: ignore[attr-defined]
    agent._resolve_player = lambda name: None  # type: ignore[assignment]
    agent._require_client = lambda: client  # type: ignore[assignment]

    result = await agent.play_announcement("Basement", "http://x.mp3")
    assert result["played"] is False
    assert "Basement" in result["error"]
