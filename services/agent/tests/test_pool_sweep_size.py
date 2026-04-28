"""Tests for size-aware behavior in SessionOrchestratorPool.

Covers two surfaces:
- ``idle_sweep`` resets sessions that exceed the size threshold (in addition
  to the idle window) and publishes a summary_reset notification with the
  size reason.
- ``_flush_one`` routes oversized eviction/shutdown flushes through
  ``_summarize_and_reset`` so the persisted row has metadata.rolling_summary
  rather than a bloated raw blob (closes the cold-resume replay gap).
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.orchestrator import AgentOrchestrator
from selene_agent.utils import config
from selene_agent.utils.session_pool import SessionOrchestratorPool


def _build_pool(provider=None):
    """Pool with a configurable provider_getter so size-resolution paths fire."""
    return SessionOrchestratorPool(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        max_size=8,
        provider_getter=(lambda: provider) if provider is not None else None,
    )


def _build_orch(session_id: str, *, last_q: float | None = None, override: int | None = None):
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id=session_id,
    )
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    orch.idle_timeout_override = override
    orch.last_query_time = last_q if last_q is not None else time.time()
    orch._user_turn_since_reset = True
    orch._summarize_and_reset = AsyncMock(return_value="rolling recap")  # type: ignore[assignment]
    return orch


async def test_size_sweep_resets_oversized_dashboard_session(monkeypatch):
    """A session with idle_timeout=-1 (dashboard sentinel) and oversized
    messages must be reset by the sweep with the size reason."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=1_000)  # threshold = 750
    pool = _build_pool(provider=provider)

    orch = _build_orch("sid-big", override=-1)  # never-idle dashboard session
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 16_000},  # ~4000 tokens
    ]
    pool._sessions["sid-big"] = orch
    pool._locks["sid-big"] = asyncio.Lock()

    published = []
    pool.publish = MagicMock(side_effect=lambda sid, ev: published.append((sid, ev)))

    await pool.idle_sweep()

    orch._summarize_and_reset.assert_awaited_once()
    args, kwargs = orch._summarize_and_reset.call_args
    assert kwargs.get("reason") == "context_size_summarize"
    assert published and published[0][1]["reason"] == "context_size_summarize"


async def test_size_sweep_skips_disabled_orchestrators(monkeypatch):
    """Autonomy orchestrators set context_size_check_enabled=False — even
    if they somehow ended up in the pool with a bloated buffer, the sweep
    must skip them."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=1_000)
    pool = _build_pool(provider=provider)

    orch = _build_orch("sid-auto", override=-1)
    orch.context_size_check_enabled = False
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 16_000},
    ]
    pool._sessions["sid-auto"] = orch
    pool._locks["sid-auto"] = asyncio.Lock()

    await pool.idle_sweep()
    orch._summarize_and_reset.assert_not_called()


async def test_size_sweep_noop_when_threshold_unavailable(monkeypatch):
    """No provider → resolve returns None → size path is skipped. Idle path
    still runs normally so existing idle behavior is preserved."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)

    pool = _build_pool(provider=None)
    orch = _build_orch("sid-no-provider", override=-1)
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 50_000},
    ]
    pool._sessions["sid-no-provider"] = orch
    pool._locks["sid-no-provider"] = asyncio.Lock()

    await pool.idle_sweep()
    orch._summarize_and_reset.assert_not_called()


async def test_idle_sweep_still_works_for_idle_sessions(monkeypatch):
    """Regression: introducing size awareness must not break the existing
    idle path."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)

    pool = _build_pool(provider=None)  # no provider → size path off
    now = time.time()
    orch = _build_orch("sid-idle", override=30, last_q=now - 45)
    pool._sessions["sid-idle"] = orch
    pool._locks["sid-idle"] = asyncio.Lock()

    await pool.idle_sweep()
    orch._summarize_and_reset.assert_awaited_once()
    args, kwargs = orch._summarize_and_reset.call_args
    assert kwargs.get("reason") == "idle_timeout_summarize"


async def test_flush_one_summarizes_oversized_session(monkeypatch):
    """Eviction/shutdown of an oversized session must route through
    _summarize_and_reset so the persisted row carries metadata.rolling_summary
    instead of a raw bloated buffer."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=1_000)
    pool = _build_pool(provider=provider)

    orch = _build_orch("sid-evict")
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 16_000},  # over threshold
    ]

    stored = []

    async def fake_store(messages, session_id=None, metadata=None):
        stored.append({"messages": messages, "metadata": metadata})
        return True

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.store_conversation_history",
        fake_store,
    )

    await pool._flush_one(orch, reason="lru_eviction")

    # The oversized branch routes through _summarize_and_reset (which would
    # internally persist with metadata.rolling_summary in production). The
    # raw store path must NOT have been called from _flush_one itself.
    orch._summarize_and_reset.assert_awaited_once()
    args, kwargs = orch._summarize_and_reset.call_args
    assert kwargs.get("reason") == "lru_eviction_size"
    assert stored == []  # raw path bypassed


async def test_flush_one_keeps_normal_path_when_under_threshold(monkeypatch):
    """A small/under-threshold session should keep using the original raw
    persist path — we only divert when the buffer is actually problematic."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=32_768)
    pool = _build_pool(provider=provider)

    orch = _build_orch("sid-tiny")
    orch.device_name = "Office"

    captured = {}

    async def fake_store(messages, session_id=None, metadata=None):
        captured["metadata"] = metadata
        return True

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.store_conversation_history",
        fake_store,
    )

    await pool._flush_one(orch, reason="shutdown_flush")

    orch._summarize_and_reset.assert_not_called()
    assert captured["metadata"]["device_name"] == "Office"
    assert captured["metadata"]["reset_reason"] == "shutdown_flush"
