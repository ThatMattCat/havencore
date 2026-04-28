"""Tests for REASONING event surfacing and the defensive <think>...</think> strip.

Covers:
- Provider-surfaced ``reasoning_content`` → REASONING event, persisted as
  ``reasoning_content`` on the stored assistant message (the field GLM-4.5-Air's
  chat_template.jinja reads when rendering <think>…</think> for assistant
  messages within the current in-progress turn).
- Raw ``<think>…</think>`` in content (vLLM parser miss) → defensive strip
  extracts reasoning, cleans content, yields REASONING event, and folds the
  salvaged reasoning into the same ``reasoning_content`` field.
- Both sources present → concatenated in the REASONING event and on the
  persisted message.
- Neither present → no REASONING event (no regression for Qwen-style models)
  and no empty ``reasoning_content`` field added.
- Legacy ``reasoning`` alias (vLLM glm45 parser's name) → never persisted; we
  normalize to ``reasoning_content`` so the chat template can find it.
- Provider lacking ``pop_last_reasoning`` → no crash (AttributeError guard).
- VLLMProvider._capture_reasoning unit: reads model_extra and reasoning_content
  attribute, and pop clears state after one read.
"""
from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.orchestrator import (
    AgentOrchestrator,
    EventType,
    strip_think_blocks,
)
from selene_agent.providers.vllm import VLLMProvider


# ---------- Helpers ----------


def _mk_message(content=None, tool_calls=None):
    """Build an assistant-message stand-in whose model_dump returns a plain dict.

    MagicMock.model_dump by default returns a MagicMock (which has no .pop), so
    tests that exercise the orchestrator's scrub-then-append path need a real
    dict coming out. model_dump reads the current attribute state at call time
    (mirrors real Pydantic behavior after ``msg.content = cleaned`` assignments).
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    def _dump():
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = msg.tool_calls
        return d

    msg.model_dump = _dump
    return msg


def _mk_response(message):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


class _FakeProvider:
    """Minimal provider that lets tests set the chat_completion response and
    the reasoning value returned by pop_last_reasoning independently."""

    name = "fake"

    def __init__(self, *, response, reasoning: Optional[str] = None):
        self._response = response
        self._reasoning = reasoning
        self.calls = 0

    async def chat_completion(self, **kwargs) -> Any:
        self.calls += 1
        return self._response

    def pop_last_cache_stats(self):
        return {"read": 0, "create": 0}

    def pop_last_reasoning(self) -> Optional[str]:
        v = self._reasoning
        self._reasoning = None
        return v


class _NoReasoningProvider(_FakeProvider):
    """Simulates an older provider that doesn't implement pop_last_reasoning.
    Orchestrator must fall through via the AttributeError guard."""

    def __getattribute__(self, name):
        if name == "pop_last_reasoning":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _build_orch(provider: _FakeProvider) -> AgentOrchestrator:
    client = MagicMock()
    orch = AgentOrchestrator(
        client=client,
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-reasoning",
        provider_getter=lambda: provider,
    )
    orch.messages = [{"role": "system", "content": "sys"}]
    return orch


async def _run_once(orch: AgentOrchestrator, user_message="hi") -> List:
    events: List = []
    async for ev in orch.run(user_message):
        events.append(ev)
    return events


# ---------- strip_think_blocks unit ----------


def test_strip_think_blocks_no_tags_returns_input_unchanged():
    clean, reasoning = strip_think_blocks("plain answer with no tags")
    assert clean == "plain answer with no tags"
    assert reasoning is None


def test_strip_think_blocks_single_block():
    clean, reasoning = strip_think_blocks(
        "<think>step 1 then step 2</think>final answer"
    )
    assert clean == "final answer"
    assert reasoning == "step 1 then step 2"


def test_strip_think_blocks_multiple_blocks_concatenated():
    clean, reasoning = strip_think_blocks(
        "<think>first thought</think>middle<think>second thought</think>end"
    )
    assert clean == "middleend"
    assert reasoning == "first thought\n\nsecond thought"


def test_strip_think_blocks_handles_multiline():
    src = "<think>line 1\nline 2\nline 3</think>answer"
    clean, reasoning = strip_think_blocks(src)
    assert clean == "answer"
    assert reasoning == "line 1\nline 2\nline 3"


def test_strip_think_blocks_empty_tags_produce_no_reasoning():
    clean, reasoning = strip_think_blocks("<think></think>just an answer")
    assert clean == "just an answer"
    assert reasoning is None


# ---------- VLLMProvider capture unit ----------


def _mk_llm_message(**fields: Any) -> MagicMock:
    """Build a message mock where only the named fields are strings; all others
    return None instead of MagicMock auto-attributes. The provider's capture
    logic probes ``reasoning`` and ``reasoning_content`` on both ``model_extra``
    and the message itself — auto-attributes would be truthy but non-string and
    muddy the test intent. Setting everything explicitly keeps the mock tight."""
    msg = MagicMock(spec=["model_extra", "reasoning", "reasoning_content"])
    msg.model_extra = fields.get("model_extra", {})
    msg.reasoning = fields.get("reasoning", None)
    msg.reasoning_content = fields.get("reasoning_content", None)
    return msg


def test_vllm_provider_captures_from_model_extra_reasoning_content():
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(model_extra={"reasoning_content": "cot from extras"})
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() == "cot from extras"
    # Second pop returns None (state cleared).
    assert provider.pop_last_reasoning() is None


def test_vllm_provider_captures_from_model_extra_reasoning_glm45():
    # vLLM's glm45 parser (observed in current builds) emits under ``reasoning``.
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(model_extra={"reasoning": "glm cot"})
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() == "glm cot"


def test_vllm_provider_captures_from_attribute_fallback_reasoning_content():
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(reasoning_content="cot via attribute")
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() == "cot via attribute"


def test_vllm_provider_captures_from_attribute_fallback_reasoning():
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(reasoning="cot via reasoning attr")
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() == "cot via reasoning attr"


def test_vllm_provider_reasoning_key_preferred_over_reasoning_content():
    # When both are present, ``reasoning`` (current glm45) wins over
    # ``reasoning_content`` (older naming). Not a load-bearing behavior — both
    # carry the same CoT when both are present — but pinning the order keeps
    # the capture deterministic.
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(
        model_extra={"reasoning": "new", "reasoning_content": "old"}
    )
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() == "new"


def test_vllm_provider_absent_reasoning_yields_none():
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message()
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() is None


def test_vllm_provider_whitespace_reasoning_treated_as_absent():
    provider = VLLMProvider.from_client(client=MagicMock(), model="test")
    msg = _mk_llm_message(model_extra={"reasoning": "   \n  "})
    response = MagicMock()
    response.choices = [MagicMock(message=msg)]
    provider._capture_reasoning(response)
    assert provider.pop_last_reasoning() is None


# ---------- Orchestrator run() cases ----------


async def test_reasoning_from_provider_surfaced_as_event():
    assistant_msg = _mk_message(content="The sky is blue due to Rayleigh scattering.")
    provider = _FakeProvider(
        response=_mk_response(assistant_msg),
        reasoning="I considered Mie vs Rayleigh; picked Rayleigh.",
    )
    orch = _build_orch(provider)

    events = await _run_once(orch, "why is the sky blue?")

    reasoning_events = [e for e in events if e.type == EventType.REASONING]
    assert len(reasoning_events) == 1
    assert reasoning_events[0].data["content"] == (
        "I considered Mie vs Rayleigh; picked Rayleigh."
    )
    assert reasoning_events[0].data["iteration"] == 1

    # Final DONE carries clean content.
    done = [e for e in events if e.type == EventType.DONE][0]
    assert done.data["content"] == "The sky is blue due to Rayleigh scattering."

    # Persisted history keeps reasoning under ``reasoning_content`` (the field
    # GLM-4.5-Air's chat_template.jinja reads). The legacy ``reasoning`` alias
    # is normalized away. The chat template only renders this back into the
    # prompt for assistant messages newer than the most recent user message
    # (in-progress agentic turn); for completed prior turns the template
    # auto-emits empty <think></think>, so retaining it is harmless.
    stored_assistant = [m for m in orch.messages if m.get("role") == "assistant"][0]
    assert "reasoning" not in stored_assistant
    assert stored_assistant.get("reasoning_content") == (
        "I considered Mie vs Rayleigh; picked Rayleigh."
    )
    assert stored_assistant["content"] == "The sky is blue due to Rayleigh scattering."


async def test_defensive_strip_pulls_reasoning_from_raw_content():
    # Simulates vLLM parser miss: raw <think>…</think> stayed in content, and
    # no separate reasoning_content field was provided.
    raw = "<think>checking weather tool isn't needed</think>It's sunny outside."
    assistant_msg = _mk_message(content=raw)
    provider = _FakeProvider(response=_mk_response(assistant_msg), reasoning=None)
    orch = _build_orch(provider)

    events = await _run_once(orch, "how's the weather?")

    reasoning_events = [e for e in events if e.type == EventType.REASONING]
    assert len(reasoning_events) == 1
    assert reasoning_events[0].data["content"] == "checking weather tool isn't needed"

    done = [e for e in events if e.type == EventType.DONE][0]
    assert done.data["content"] == "It's sunny outside."

    stored_assistant = [m for m in orch.messages if m.get("role") == "assistant"][0]
    assert stored_assistant["content"] == "It's sunny outside."
    assert "<think>" not in stored_assistant["content"]
    # Salvaged reasoning is folded into reasoning_content so the chat template
    # has a single canonical field to read.
    assert stored_assistant.get("reasoning_content") == "checking weather tool isn't needed"


async def test_reasoning_from_both_sources_combined():
    # Rare: parser extracted one block, another slipped through in content.
    assistant_msg = _mk_message(content="<think>leftover thought</think>done.")
    provider = _FakeProvider(
        response=_mk_response(assistant_msg),
        reasoning="parser-extracted thought",
    )
    orch = _build_orch(provider)

    events = await _run_once(orch, "x")

    reasoning_events = [e for e in events if e.type == EventType.REASONING]
    assert len(reasoning_events) == 1
    combined = reasoning_events[0].data["content"]
    # Both pieces preserved, separated by blank line.
    assert "parser-extracted thought" in combined
    assert "leftover thought" in combined

    done = [e for e in events if e.type == EventType.DONE][0]
    assert done.data["content"] == "done."

    # Both pieces are also persisted under reasoning_content for the template.
    stored_assistant = [m for m in orch.messages if m.get("role") == "assistant"][0]
    persisted = stored_assistant.get("reasoning_content", "")
    assert "parser-extracted thought" in persisted
    assert "leftover thought" in persisted


async def test_no_reasoning_yields_no_reasoning_event():
    # Baseline: Qwen-style model, clean content, no reasoning anywhere.
    assistant_msg = _mk_message(content="just a plain answer")
    provider = _FakeProvider(response=_mk_response(assistant_msg), reasoning=None)
    orch = _build_orch(provider)

    events = await _run_once(orch, "x")

    assert not any(e.type == EventType.REASONING for e in events)
    done = [e for e in events if e.type == EventType.DONE][0]
    assert done.data["content"] == "just a plain answer"
    # No reasoning anywhere → don't add an empty reasoning_content field.
    stored_assistant = [m for m in orch.messages if m.get("role") == "assistant"][0]
    assert "reasoning_content" not in stored_assistant
    assert "reasoning" not in stored_assistant


async def test_provider_without_pop_last_reasoning_does_not_crash():
    assistant_msg = _mk_message(content="answer")
    provider = _NoReasoningProvider(response=_mk_response(assistant_msg))
    orch = _build_orch(provider)

    events = await _run_once(orch, "x")

    assert not any(e.type == EventType.REASONING for e in events)
    done = [e for e in events if e.type == EventType.DONE][0]
    assert done.data["content"] == "answer"


async def test_reasoning_event_arrives_before_done():
    assistant_msg = _mk_message(content="final text")
    provider = _FakeProvider(
        response=_mk_response(assistant_msg),
        reasoning="my chain of thought",
    )
    orch = _build_orch(provider)

    events = await _run_once(orch, "x")
    types_in_order = [e.type for e in events]

    reasoning_idx = types_in_order.index(EventType.REASONING)
    done_idx = types_in_order.index(EventType.DONE)
    assert reasoning_idx < done_idx


async def test_legacy_reasoning_field_normalized_to_reasoning_content():
    """vLLM's glm45 parser surfaces CoT under ``reasoning`` (Pydantic
    model_extra), so model_dump() yields a dict like
    ``{"role": "assistant", "content": "...", "reasoning": "..."}``. GLM-4.5-Air's
    chat_template.jinja reads ``m.reasoning_content`` instead. The orchestrator
    must normalize: drop the legacy alias, write the value under
    ``reasoning_content`` so the template renders <think>…</think> on the next
    in-turn iteration."""
    msg = MagicMock()
    msg.content = "weather is fine"
    msg.tool_calls = None

    def _dump():
        # Simulates a real vLLM response where the parser populated
        # model_extra["reasoning"], which then rides along through model_dump.
        return {
            "role": "assistant",
            "content": msg.content,
            "reasoning": "checking the weather tool",
        }

    msg.model_dump = _dump
    provider = _FakeProvider(
        response=_mk_response(msg),
        reasoning="checking the weather tool",
    )
    orch = _build_orch(provider)

    await _run_once(orch, "weather?")

    stored = [m for m in orch.messages if m.get("role") == "assistant"][0]
    assert "reasoning" not in stored
    assert stored.get("reasoning_content") == "checking the weather tool"
