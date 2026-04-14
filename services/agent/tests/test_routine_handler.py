"""Tests for the routine agenda handler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_routine_empty_prompt_returns_error():
    from selene_agent.autonomy.handlers import routine

    item = {"id": "r-0", "kind": "routine", "config": {}}
    result = await routine.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "prompt" in result["error"]


@pytest.mark.asyncio
async def test_routine_invalid_tools_override_returns_error(monkeypatch):
    from selene_agent.autonomy.handlers import routine

    item = {
        "id": "r-1",
        "kind": "routine",
        "autonomy_level": "notify",
        "config": {
            "prompt": "Summarize",
            # create_memory is in V1_DENY → outside the tier.
            "tools_override": ["create_memory"],
        },
    }
    result = await routine.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "tools_override" in result["summary"]


@pytest.mark.asyncio
async def test_routine_runs_turn_and_delivers_via_channel(monkeypatch):
    from selene_agent.autonomy.handlers import routine

    # Stub AutonomousTurn so we never touch the real LLM.
    fake_result = MagicMock(status="ok", content="weekly summary", messages=[{"role": "assistant"}], metrics={"llm_ms": 10}, error=None)
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(routine, "AutonomousTurn", lambda **kw: fake_turn)

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(routine, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-2",
        "kind": "routine",
        "autonomy_level": "notify",
        "name": "Weekly recap",
        "config": {
            "prompt": "Summarize this week",
            "deliver": {"channel": "email", "to": "me@example.com"},
        },
    }
    result = await routine.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "ok"
    assert result["notified_via"] == "email"
    assert result["messages"] == [{"role": "assistant"}]
    fake_turn.run.assert_awaited_once_with("Summarize this week")


@pytest.mark.asyncio
async def test_routine_marks_error_when_turn_returns_failure(monkeypatch):
    from selene_agent.autonomy.handlers import routine

    fake_result = MagicMock(status="error", content="", messages=[], metrics={}, error="timeout")
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(routine, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "r-3",
        "kind": "routine",
        "autonomy_level": "notify",
        "config": {"prompt": "anything", "deliver": {"channel": "email"}},
    }
    result = await routine.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert result["error"] == "timeout"
    assert result["notified_via"] is None
