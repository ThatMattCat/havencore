"""Tests for per-session idle_timeout_override and _apply_idle_timeout_override helper."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.api.chat import _apply_idle_timeout_override
from selene_agent.orchestrator import AgentOrchestrator
from selene_agent.utils import config


def _make_orch(override=None):
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-under-test",
    )
    if override is not None:
        orch.idle_timeout_override = override
    return orch


def test_override_field_defaults_none():
    orch = _make_orch()
    assert orch.idle_timeout_override is None
    assert orch.effective_timeout() == config.CONVERSATION_TIMEOUT


def test_override_takes_effect():
    orch = _make_orch(override=30)
    assert orch.effective_timeout() == 30


def test_override_zero_falls_back_to_default():
    # 0 is falsy → use default (semantically "no override").
    orch = _make_orch(override=0)
    assert orch.effective_timeout() == config.CONVERSATION_TIMEOUT


def test_apply_override_valid_string():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, "45")
    assert orch.idle_timeout_override == 45


def test_apply_override_valid_int():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, 120)
    assert orch.idle_timeout_override == 120


def test_apply_override_clamps_low():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, "5")
    assert orch.idle_timeout_override == config.CONVERSATION_TIMEOUT_MIN


def test_apply_override_clamps_high():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, "99999")
    assert orch.idle_timeout_override == config.CONVERSATION_TIMEOUT_MAX


def test_apply_override_garbage_ignored():
    orch = _make_orch(override=60)
    for bad in ("not-a-number", "", None, [], {}):
        _apply_idle_timeout_override(orch, bad)
    # Starting value is preserved, no exception.
    assert orch.idle_timeout_override == 60


def test_apply_override_negative_clamps_to_min():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, -10)
    assert orch.idle_timeout_override == config.CONVERSATION_TIMEOUT_MIN


def test_apply_override_negative_one_is_never_sentinel():
    """-1 means "never auto-summarize" and bypasses the MIN clamp."""
    orch = _make_orch()
    _apply_idle_timeout_override(orch, -1)
    assert orch.idle_timeout_override == -1


def test_apply_override_negative_one_string():
    orch = _make_orch()
    _apply_idle_timeout_override(orch, "-1")
    assert orch.idle_timeout_override == -1


def test_effective_timeout_returns_negative_one_for_never_sentinel():
    orch = _make_orch(override=-1)
    assert orch.effective_timeout() == -1


async def test_check_session_timeout_skips_never_sentinel():
    """Regression: _check_session_timeout runs at the start of every turn.
    With override=-1 and any elapsed time, it must NOT call _summarize_and_reset.
    Prior to the fix, `elapsed > -1` was always True and every new turn triggered
    a summary.
    """
    orch = _make_orch(override=-1)
    orch.last_query_time = time.time() - 600  # 10 minutes "idle"
    orch._summarize_and_reset = AsyncMock()  # type: ignore[assignment]

    await orch._check_session_timeout()
    orch._summarize_and_reset.assert_not_called()
