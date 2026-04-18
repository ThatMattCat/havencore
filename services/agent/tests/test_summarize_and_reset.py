"""Tests for AgentOrchestrator._summarize_and_reset and helpers."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.orchestrator import AgentOrchestrator


def _build_orch(messages=None, summary_text="test summary"):
    client = MagicMock()
    if summary_text is None:
        # Simulate a failing client (raises generic exception).
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    else:
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=summary_text))]
        client.chat.completions.create = AsyncMock(return_value=resp)

    orch = AgentOrchestrator(
        client=client,
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-sum",
    )
    # Pre-populate a conversation that looks lived-in.
    orch.messages = messages if messages is not None else [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "what's the weather?"},
        {"role": "assistant", "content": "sunny and 72"},
        {"role": "user", "content": "turn on the lamp"},
        {"role": "assistant", "content": "done, bedroom lamp on"},
    ]
    return orch, client


async def test_summary_happy_path(monkeypatch):
    stored = {}

    async def fake_store(messages, session_id=None, metadata=None):
        stored["messages"] = list(messages)
        stored["session_id"] = session_id
        stored["metadata"] = metadata
        return True

    monkeypatch.setattr(
        "selene_agent.orchestrator.conversation_db.store_conversation_history",
        fake_store,
    )
    # Short-circuit initialize() so it doesn't try to build the L4 block.
    orig_init = AgentOrchestrator.initialize

    async def fake_init(self):
        self.messages = [{"role": "system", "content": "sys prompt"}]
        self._l4_pending = False

    monkeypatch.setattr(AgentOrchestrator, "initialize", fake_init)

    orch, client = _build_orch(summary_text="User asked about weather and turned on bedroom lamp.")
    prev_session_id = orch.session_id

    await orch._summarize_and_reset(reason="idle_timeout_summarize")

    # Session id is preserved.
    assert orch.session_id == prev_session_id
    # Messages: system prompt + summary + tail exchanges.
    assert orch.messages[0]["role"] == "system"
    # Second message should be our summary system injection.
    assert orch.messages[1]["role"] == "system"
    assert "Prior conversation summary" in orch.messages[1]["content"]
    assert "bedroom lamp" in orch.messages[1]["content"]
    # Tail contains the 2 most recent user/assistant exchanges (all 4 messages here).
    tail_roles = [m["role"] for m in orch.messages[2:]]
    assert tail_roles == ["user", "assistant", "user", "assistant"]
    # Persistence was called with full prior history and rolling_summary metadata.
    assert stored["session_id"] == prev_session_id
    assert stored["metadata"]["rolling_summary"] is not None
    assert "bedroom lamp" in stored["metadata"]["rolling_summary"]
    assert stored["metadata"]["reset_reason"] == "idle_timeout_summarize"


async def test_summary_llm_timeout_falls_back_to_tail_only(monkeypatch):
    async def fake_store(messages, session_id=None, metadata=None):
        return True

    monkeypatch.setattr(
        "selene_agent.orchestrator.conversation_db.store_conversation_history",
        fake_store,
    )

    async def fake_init(self):
        self.messages = [{"role": "system", "content": "sys prompt"}]
        self._l4_pending = False

    monkeypatch.setattr(AgentOrchestrator, "initialize", fake_init)

    orch, client = _build_orch()
    client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

    await orch._summarize_and_reset(reason="test")

    # No summary system message; tail preserved.
    assert orch.messages[0]["role"] == "system"
    # Next messages should be tail, not a summary injection.
    assert orch.messages[1]["role"] != "system"
    tail_roles = [m["role"] for m in orch.messages[1:]]
    assert tail_roles == ["user", "assistant", "user", "assistant"]


async def test_summary_llm_exception_falls_back_to_tail_only(monkeypatch):
    async def fake_store(messages, session_id=None, metadata=None):
        return True

    monkeypatch.setattr(
        "selene_agent.orchestrator.conversation_db.store_conversation_history",
        fake_store,
    )

    async def fake_init(self):
        self.messages = [{"role": "system", "content": "sys prompt"}]
        self._l4_pending = False

    monkeypatch.setattr(AgentOrchestrator, "initialize", fake_init)

    orch, _ = _build_orch(summary_text=None)  # client raises

    await orch._summarize_and_reset(reason="test")

    assert orch.messages[0]["role"] == "system"
    assert orch.messages[1]["role"] != "system"


async def test_empty_session_short_circuits(monkeypatch):
    called = {"store": 0, "llm": 0}

    async def fake_store(*a, **kw):
        called["store"] += 1
        return True

    monkeypatch.setattr(
        "selene_agent.orchestrator.conversation_db.store_conversation_history",
        fake_store,
    )

    async def fake_init(self):
        self.messages = [{"role": "system", "content": "sys"}]
        self._l4_pending = False

    monkeypatch.setattr(AgentOrchestrator, "initialize", fake_init)

    orch, client = _build_orch(messages=[{"role": "system", "content": "sys"}])

    async def counting(*a, **kw):
        called["llm"] += 1
        raise RuntimeError("should not be called")

    client.chat.completions.create = counting

    await orch._summarize_and_reset(reason="test")

    assert called["store"] == 0
    assert called["llm"] == 0


async def test_tail_exchanges_drops_orphaned_tool_messages():
    orch, _ = _build_orch()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "turn on lamp"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "ha", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "name": "ha", "content": "ok"},
        {"role": "assistant", "content": "lamp on"},
    ]
    tail = orch._tail_exchanges(msgs, 2)
    roles = [m["role"] for m in tail]
    # Two user turns plus their responses, with tool chain intact.
    assert roles.count("user") == 2
    assert roles[-1] == "assistant"
    assert roles[0] == "user"


async def test_tail_exchanges_drops_dangling_assistant_toolcalls():
    orch, _ = _build_orch()
    # Assistant declares a tool_call that has no matching tool response in the
    # tail. The dangling assistant should be trimmed.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "missing", "type": "function",
             "function": {"name": "x", "arguments": "{}"}}
        ]},
    ]
    tail = orch._tail_exchanges(msgs, 2)
    # The last assistant has unsatisfied tool_calls and should be trimmed.
    assert not (tail and tail[-1].get("role") == "assistant" and tail[-1].get("tool_calls"))


async def test_tail_exchanges_n_equals_one():
    orch, _ = _build_orch()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tail = orch._tail_exchanges(msgs, 1)
    assert [m["content"] for m in tail] == ["u2", "a2"]


async def test_summary_preserves_session_id(monkeypatch):
    async def fake_store(*a, **kw):
        return True

    monkeypatch.setattr(
        "selene_agent.orchestrator.conversation_db.store_conversation_history",
        fake_store,
    )

    async def fake_init(self):
        # Crucially: don't rotate session_id when pinned.
        self.messages = [{"role": "system", "content": "sys"}]
        self._l4_pending = False

    monkeypatch.setattr(AgentOrchestrator, "initialize", fake_init)

    orch, _ = _build_orch()
    before = orch.session_id
    await orch._summarize_and_reset(reason="test")
    assert orch.session_id == before


async def test_check_session_timeout_routes_to_summarize(monkeypatch):
    """_check_session_timeout should delegate to _summarize_and_reset when expired."""
    orch, _ = _build_orch()
    orch.last_query_time = 1.0  # way in the past (truthy) → expired under any positive timeout

    fake = AsyncMock()
    monkeypatch.setattr(
        AgentOrchestrator, "_summarize_and_reset", fake,
    )
    await orch._check_session_timeout()
    fake.assert_awaited_once()


async def test_check_session_timeout_noop_when_fresh(monkeypatch):
    import time as _time
    orch, _ = _build_orch()
    orch.last_query_time = _time.time()  # fresh

    fake = AsyncMock()
    monkeypatch.setattr(
        AgentOrchestrator, "_summarize_and_reset", fake,
    )
    await orch._check_session_timeout()
    fake.assert_not_called()
