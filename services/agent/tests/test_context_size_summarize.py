"""Tests for the context-size summarization gate inside AgentOrchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.orchestrator import AgentOrchestrator, EventType
from selene_agent.utils import config


def _build_orch(provider_max_model_len=32_768):
    """Build an orchestrator wired to a fake provider with a configurable
    max_model_len. The provider_getter is honored at every chat-completion
    call site, including _check_context_size."""
    client = MagicMock()
    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=provider_max_model_len)
    provider.chat_completion = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])
    )

    orch = AgentOrchestrator(
        client=client,
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-ctx",
        provider_getter=lambda: provider,
    )
    return orch, provider


async def test_check_context_size_triggers_summarize_when_oversized(monkeypatch):
    """When estimated message bytes exceed the threshold, the check must
    delegate to _summarize_and_reset with the size reason and stash the
    pending tuple for run() to yield."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    orch, provider = _build_orch(provider_max_model_len=1_000)
    # threshold = 1_000 * 0.75 = 750 tokens ≈ 3000 chars.
    big_blob = "x" * 16_000  # estimate ~4000 tokens
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": big_blob},
    ]

    fake_summary = AsyncMock(return_value="rolling recap")
    monkeypatch.setattr(AgentOrchestrator, "_summarize_and_reset", fake_summary)

    await orch._check_context_size()

    fake_summary.assert_awaited_once()
    args, kwargs = fake_summary.call_args
    assert kwargs.get("reason") == "context_size_summarize"
    assert orch._pending_summary_reset == ("context_size_summarize", "rolling recap")


async def test_check_context_size_noop_when_under_threshold(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    orch, _ = _build_orch(provider_max_model_len=32_768)
    # Tiny payload, well under the 24K-token threshold.
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    fake_summary = AsyncMock()
    monkeypatch.setattr(AgentOrchestrator, "_summarize_and_reset", fake_summary)
    await orch._check_context_size()
    fake_summary.assert_not_called()
    assert orch._pending_summary_reset is None


async def test_check_context_size_noop_when_disabled(monkeypatch):
    """Autonomy turns flip context_size_check_enabled off — even with a
    bloated buffer the check must early-return."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)

    orch, _ = _build_orch(provider_max_model_len=1_000)
    orch.context_size_check_enabled = False
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 16_000},
    ]

    fake_summary = AsyncMock()
    monkeypatch.setattr(AgentOrchestrator, "_summarize_and_reset", fake_summary)
    await orch._check_context_size()
    fake_summary.assert_not_called()


async def test_check_context_size_noop_when_threshold_unavailable(monkeypatch):
    """If the provider can't report max_model_len and no override is set,
    skip silently rather than synthesize a guess."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)

    orch, provider = _build_orch(provider_max_model_len=None)
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "x" * 50_000},
    ]
    fake_summary = AsyncMock()
    monkeypatch.setattr(AgentOrchestrator, "_summarize_and_reset", fake_summary)
    await orch._check_context_size()
    fake_summary.assert_not_called()


async def test_check_context_size_noop_on_empty_session():
    """One message (system only) should never trigger a reset."""
    orch, _ = _build_orch()
    orch.messages = [{"role": "system", "content": "sys"}]
    await orch._check_context_size()
    assert orch._pending_summary_reset is None


async def test_run_yields_summary_reset_with_size_reason(monkeypatch):
    """run() must yield a SUMMARY_RESET event whose data.reason is the
    actual trigger ("context_size_summarize"), not the legacy hardcoded
    idle reason."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 1)  # tiny → always over
    orch, provider = _build_orch(provider_max_model_len=32_768)
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    # Stub _summarize_and_reset to set the pending tuple deterministically,
    # mirroring the real path.
    async def fake_summarize(self, reason: str):
        return "size-triggered recap"

    monkeypatch.setattr(AgentOrchestrator, "_summarize_and_reset", fake_summarize)
    monkeypatch.setattr(AgentOrchestrator, "prepare", AsyncMock())
    monkeypatch.setattr(AgentOrchestrator, "_check_session_timeout", AsyncMock())
    monkeypatch.setattr(AgentOrchestrator, "_build_retrieval_block", AsyncMock(return_value=None))

    # End the loop on the first iteration with content.
    provider.chat_completion = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="final", tool_calls=None))]
        )
    )

    events = []
    async for ev in orch.run("hello"):
        events.append((ev.type, dict(ev.data)))

    summary_events = [e for e in events if e[0] == EventType.SUMMARY_RESET]
    assert len(summary_events) == 1
    assert summary_events[0][1]["reason"] == "context_size_summarize"
    assert summary_events[0][1]["summary"] == "size-triggered recap"


async def test_run_yields_summary_reset_with_idle_reason(monkeypatch):
    """Mirror of the previous test but for the idle path — the same yield
    site must carry the idle reason when that's what fired."""
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)

    orch, provider = _build_orch(provider_max_model_len=None)  # disables size path
    orch.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    async def fake_check_idle(self):
        self._pending_summary_reset = ("idle_timeout_summarize", "idle recap")

    monkeypatch.setattr(AgentOrchestrator, "_check_session_timeout", fake_check_idle)
    monkeypatch.setattr(AgentOrchestrator, "prepare", AsyncMock())
    monkeypatch.setattr(AgentOrchestrator, "_build_retrieval_block", AsyncMock(return_value=None))

    provider.chat_completion = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="final", tool_calls=None))]
        )
    )

    events = []
    async for ev in orch.run("hello"):
        events.append((ev.type, dict(ev.data)))

    summary_events = [e for e in events if e[0] == EventType.SUMMARY_RESET]
    assert len(summary_events) == 1
    assert summary_events[0][1]["reason"] == "idle_timeout_summarize"
    assert summary_events[0][1]["summary"] == "idle recap"
