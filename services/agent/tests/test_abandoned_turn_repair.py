"""Regression tests for issue #39 — an abandoned turn must not poison the session.

run() appends the assistant message carrying ``tool_calls`` *before* it yields
TOOL_CALL/TOOL_RESULT and appends the matching ``role=tool`` replies. A turn
dropped in that window (WS client disconnect, REST request cancellation) used
to leave the pooled session with unanswered tool_calls — an invalid sequence
the provider rejects, so every later turn on that session_id failed.

Covers:
- ``_repair_dangling_tool_calls`` unit: well-formed history untouched; trailing
  gap filled; partial answers topped up; mid-history gap filled in place.
- run()'s ``finally``: closing the generator mid tool-loop (the disconnect
  shape) leaves ``orch.messages`` well-formed.
- Normal completion synthesizes nothing.
- A session cold-resumed with a pre-existing gap is healed before its first
  LLM call, not after.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional
from unittest.mock import MagicMock

from selene_agent.orchestrator import AgentOrchestrator, EventType


# ---------- Helpers ----------


def _mk_tool_call(tc_id: str, name: str = "ha_control_light", args: str = "{}"):
    """Provider-style tool_call object (attribute access, as the SDK returns)."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    return tc


def _mk_message(content=None, tool_calls=None):
    """Assistant-message stand-in whose model_dump returns plain dicts.

    Mirrors real pydantic: the attribute holds objects, model_dump() serializes
    tool_calls to dicts — which is what lands in orch.messages.
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls

    def _dump():
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    msg.model_dump = _dump
    return msg


def _mk_response(message):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


class _ScriptedProvider:
    """Returns the queued responses in order, one per chat_completion call."""

    name = "fake"

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls = 0

    async def chat_completion(self, **kwargs) -> Any:
        self.calls += 1
        return self._responses.pop(0)

    def pop_last_cache_stats(self):
        return {"read": 0, "create": 0}

    def pop_last_reasoning(self) -> Optional[str]:
        return None


def _build_orch(provider: Optional[_ScriptedProvider] = None) -> AgentOrchestrator:
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-abandon",
        provider_getter=(lambda: provider),
    )
    orch.messages = [{"role": "system", "content": "sys"}]
    orch._l4_pending = False
    orch.retrieval_enabled = False
    orch.context_size_check_enabled = False
    return orch


def _unanswered(messages: List[dict]) -> List[str]:
    """tool_call ids declared by an assistant with no matching role=tool reply."""
    answered = {
        m.get("tool_call_id")
        for m in messages
        if m.get("role") == "tool"
    }
    declared: List[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            declared += [tc["id"] for tc in (m.get("tool_calls") or [])]
    return [tcid for tcid in declared if tcid not in answered]


# ---------- _repair_dangling_tool_calls unit ----------


def test_repair_leaves_well_formed_history_untouched():
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "lights on"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "Done."},
    ]
    before = list(orch.messages)

    assert orch._repair_dangling_tool_calls() == 0
    assert orch.messages == before


def test_repair_fills_trailing_gap():
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "lights on"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
    ]

    assert orch._repair_dangling_tool_calls() == 1
    last = orch.messages[-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "c1"
    assert json.loads(last["content"])["error"] == "tool_call_abandoned"
    assert _unanswered(orch.messages) == []


def test_repair_tops_up_partially_answered_call_batch():
    """Disconnect after the first of two parallel tool calls resolved."""
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "lights and lock"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]

    assert orch._repair_dangling_tool_calls() == 1
    assert [m.get("tool_call_id") for m in orch.messages if m["role"] == "tool"] == ["c1", "c2"]
    assert _unanswered(orch.messages) == []


def test_repair_inserts_mid_history_gap_in_place():
    """Synthetic result goes right after its own assistant, not at the end."""
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "first"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "hi"},
    ]

    assert orch._repair_dangling_tool_calls() == 1
    roles = [(m["role"], m.get("tool_call_id") or m.get("content")) for m in orch.messages]
    assert roles[2] == ("assistant", None)
    assert roles[3] == ("tool", "c1")
    assert roles[4] == ("user", "second")


# ---------- run() finally ----------


async def test_generator_closed_mid_tool_loop_leaves_history_valid():
    """The disconnect shape: consumer stops iterating right after TOOL_CALL."""
    tc = _mk_tool_call("call_abandon_1")
    provider = _ScriptedProvider([_mk_response(_mk_message(tool_calls=[tc]))])
    orch = _build_orch(provider)

    stream = orch.run("turn on the kitchen lights")
    seen: List[EventType] = []
    async for ev in stream:
        seen.append(ev.type)
        if ev.type == EventType.TOOL_CALL:
            break  # client vanished; nothing drains the rest
    await stream.aclose()

    assert EventType.TOOL_CALL in seen
    assert any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in orch.messages
    ), "precondition: the assistant tool_calls message was already appended"
    assert _unanswered(orch.messages) == []


async def test_completed_turn_synthesizes_nothing():
    provider = _ScriptedProvider([_mk_response(_mk_message(content="All set."))])
    orch = _build_orch(provider)

    events = [ev async for ev in orch.run("hello")]

    assert events[-1].type == EventType.DONE
    assert not [m for m in orch.messages if m.get("role") == "tool"]


async def test_resumed_session_with_inherited_gap_is_healed_before_the_llm_call():
    """A history flushed mid-turn before this fix must not fail its next turn."""
    provider = _ScriptedProvider([_mk_response(_mk_message(content="Recovered."))])
    orch = _build_orch(provider)
    orch.messages += [
        {"role": "user", "content": "earlier query"},
        {"role": "assistant", "tool_calls": [{"id": "poisoned"}]},
    ]

    sent: List[List[dict]] = []
    original = provider.chat_completion

    async def _capture(**kwargs):
        sent.append(list(kwargs["messages"]))
        return await original(**kwargs)

    provider.chat_completion = _capture

    events = [ev async for ev in orch.run("are you there?")]

    assert events[-1].type == EventType.DONE
    assert sent, "provider was called"
    assert _unanswered(sent[0]) == [], "the repair happened before the LLM saw it"
    assert _unanswered(orch.messages) == []
