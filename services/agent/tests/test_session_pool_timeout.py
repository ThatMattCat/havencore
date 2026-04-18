"""Tests for per-session timeout in SessionOrchestratorPool sweep + hydrate."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.orchestrator import AgentOrchestrator
from selene_agent.utils import config
from selene_agent.utils.session_pool import SessionOrchestratorPool


def _build_pool():
    return SessionOrchestratorPool(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        max_size=8,
    )


def _build_orch(session_id: str, override: int | None = None, last_q: float | None = None):
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id=session_id,
    )
    # Treat as non-empty session so the sweep picks it up.
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    orch.idle_timeout_override = override
    orch.last_query_time = last_q if last_q is not None else time.time()
    orch._summarize_and_reset = AsyncMock()  # type: ignore[assignment]
    return orch


async def test_sweep_uses_per_session_timeout():
    pool = _build_pool()
    now = time.time()

    # short-override session: 30s window, last query 45s ago → expired.
    short = _build_orch("short", override=30, last_q=now - 45)
    # default session: uses CONVERSATION_TIMEOUT (≥ 45s), last query 45s ago → fresh.
    default = _build_orch("default", override=None, last_q=now - 45)

    pool._sessions["short"] = short
    pool._sessions["default"] = default
    pool._locks["short"] = __import__("asyncio").Lock()
    pool._locks["default"] = __import__("asyncio").Lock()

    await pool.idle_sweep()

    short._summarize_and_reset.assert_awaited_once()
    default._summarize_and_reset.assert_not_called()


async def test_sweep_skips_fresh_sessions():
    pool = _build_pool()
    now = time.time()
    fresh = _build_orch("fresh", override=30, last_q=now - 5)
    pool._sessions["fresh"] = fresh
    pool._locks["fresh"] = __import__("asyncio").Lock()

    await pool.idle_sweep()
    fresh._summarize_and_reset.assert_not_called()


async def test_flush_one_stores_override_in_metadata(monkeypatch):
    pool = _build_pool()
    orch = _build_orch("s1", override=120)

    captured = {}

    async def fake_store(messages, session_id=None, metadata=None):
        captured["messages"] = messages
        captured["session_id"] = session_id
        captured["metadata"] = metadata
        return True

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.store_conversation_history",
        fake_store,
    )

    await pool._flush_one(orch, reason="unit_test")

    assert captured["session_id"] == "s1"
    assert captured["metadata"]["idle_timeout_override"] == 120
    assert captured["metadata"]["reset_reason"] == "unit_test"


async def test_hydrate_from_db_restores_override(monkeypatch):
    pool = _build_pool()

    async def fake_get(session_id, limit=1):
        return [{
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "prior"},
                {"role": "assistant", "content": "reply"},
            ],
            "metadata": {"idle_timeout_override": 240},
        }]

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.get_conversation_history",
        fake_get,
    )
    # Short-circuit prepare() so we don't need L4 infra.
    monkeypatch.setattr(
        "selene_agent.orchestrator.AgentOrchestrator.prepare",
        AsyncMock(),
    )

    orch = await pool._hydrate_from_db("s1")
    assert orch is not None
    assert orch.idle_timeout_override == 240


async def test_hydrate_clamps_out_of_range_override(monkeypatch):
    pool = _build_pool()

    async def fake_get(session_id, limit=1):
        return [{
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "prior"},
            ],
            "metadata": {"idle_timeout_override": 99_999},
        }]

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.get_conversation_history",
        fake_get,
    )
    monkeypatch.setattr(
        "selene_agent.orchestrator.AgentOrchestrator.prepare",
        AsyncMock(),
    )

    orch = await pool._hydrate_from_db("s1")
    assert orch is not None
    assert orch.idle_timeout_override == config.CONVERSATION_TIMEOUT_MAX


async def test_hydrate_ignores_garbage_override(monkeypatch):
    pool = _build_pool()

    async def fake_get(session_id, limit=1):
        return [{
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "prior"},
            ],
            "metadata": {"idle_timeout_override": "not-a-number"},
        }]

    monkeypatch.setattr(
        "selene_agent.utils.session_pool.conversation_db.get_conversation_history",
        fake_get,
    )
    monkeypatch.setattr(
        "selene_agent.orchestrator.AgentOrchestrator.prepare",
        AsyncMock(),
    )

    orch = await pool._hydrate_from_db("s1")
    assert orch is not None
    assert orch.idle_timeout_override is None
