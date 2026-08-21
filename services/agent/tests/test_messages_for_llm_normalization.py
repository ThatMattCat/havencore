"""_messages_for_llm must never send a system message past index 0.

Qwen3.8's chat template raises ``System message must be at the beginning.``
(vLLM surfaces it as a 400) for any mid-history system message. The agent
injects those in three places — the per-turn retrieval block, the
summarize-reset recap, and the session pool's MCP-failure note — and the
latter two also persist, so cold-resumed histories can replay them. The
normalization re-tags them as user messages at send time only.

Covers:
- Retrieval block lands right before the last user message, as role=user.
- Mid-history system entries (recap / MCP note shapes) are re-tagged.
- messages[0] keeps its system role; self.messages is never mutated.
- run() end-to-end: the provider sees no system message past index 0.
"""
from __future__ import annotations

import copy
from typing import Any, List, Optional
from unittest.mock import MagicMock

from selene_agent.orchestrator import AgentOrchestrator, EventType


def _mk_message(content=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump = lambda: {"role": "assistant", "content": msg.content}
    return msg


def _mk_response(message):
    resp = MagicMock()
    resp.choices = [MagicMock(message=message)]
    return resp


class _ScriptedProvider:
    name = "fake"

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)

    async def chat_completion(self, **kwargs) -> Any:
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
        session_id="sess-sysnorm",
        provider_getter=(lambda: provider),
    )
    orch.messages = [{"role": "system", "content": "sys"}]
    orch._l4_pending = False
    orch.retrieval_enabled = False
    orch.context_size_check_enabled = False
    return orch


# ---------- _messages_for_llm unit ----------


def test_retrieval_block_inserted_before_last_user_as_user_role():
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "second"},
    ]

    msgs = orch._messages_for_llm("<retrieved_memories>...</retrieved_memories>")

    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "user"]
    assert msgs[3]["content"] == "<retrieved_memories>...</retrieved_memories>"
    assert msgs[4]["content"] == "second"


def test_mid_history_system_messages_are_retagged():
    orch = _build_orch()
    orch.messages += [
        {"role": "system", "content": "[Prior conversation summary]\nrecap"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "system", "content": "Note: an MCP module failed to start."},
    ]

    msgs = orch._messages_for_llm(None)

    assert msgs[0]["role"] == "system"
    assert all(m["role"] != "system" for m in msgs[1:])
    # Content survives the re-tag untouched.
    assert msgs[1]["content"].startswith("[Prior conversation summary]")
    assert msgs[4]["content"].startswith("Note: an MCP module failed")


def test_self_messages_not_mutated():
    orch = _build_orch()
    orch.messages += [
        {"role": "system", "content": "[Prior conversation summary]\nrecap"},
        {"role": "user", "content": "hello"},
    ]
    before = copy.deepcopy(orch.messages)

    orch._messages_for_llm("<retrieved_memories>x</retrieved_memories>")

    assert orch.messages == before


def test_clean_history_passes_through_unchanged():
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]

    assert orch._messages_for_llm(None) == orch.messages


def test_none_valued_keys_are_stripped_content_kept():
    """vLLM's schema rejects explicit ``"tool_calls": null`` — the exact shape
    model_dump() emits and older persisted histories carry."""
    orch = _build_orch()
    orch.messages += [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "annotations": None,
            "audio": None,
            "function_call": None,
            "tool_calls": None,
            "reasoning_content": "thought hard",
        },
        {"role": "user", "content": "try again"},
    ]

    msgs = orch._messages_for_llm(None)

    dumped = msgs[2]
    assert set(dumped) == {"role", "content", "reasoning_content"}
    assert dumped["content"] is None  # content survives even as None
    # Original history keeps the raw dump.
    assert "tool_calls" in orch.messages[2]


def test_real_tool_calls_survive_stripping():
    orch = _build_orch()
    tool_calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    orch.messages += [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]

    msgs = orch._messages_for_llm(None)

    assert msgs[2]["tool_calls"] == tool_calls


# ---------- run() end-to-end ----------


async def test_provider_never_sees_mid_history_system_messages():
    """The summarize-reset shape: recap at index 1, then a normal turn."""
    provider = _ScriptedProvider([_mk_response(_mk_message(content="All set."))])
    orch = _build_orch(provider)
    orch.messages += [
        {"role": "system", "content": "[Prior conversation summary]\nrecap"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
    ]

    sent: List[List[dict]] = []
    original = provider.chat_completion

    async def _capture(**kwargs):
        sent.append(list(kwargs["messages"]))
        return await original(**kwargs)

    provider.chat_completion = _capture

    events = [ev async for ev in orch.run("and now?")]

    assert events[-1].type == EventType.DONE
    assert sent, "provider was called"
    assert sent[0][0]["role"] == "system"
    assert all(m.get("role") != "system" for m in sent[0][1:])
    # The pooled history keeps its semantic system labeling for persistence.
    assert orch.messages[1]["role"] == "system"


# ---------- degenerate empty response (reasoning-only, e.g. budget exhausted) ----------


def _mk_empty_message():
    """content=None, tool_calls=None — the shape a reasoning model returns
    when the completion budget runs out inside the think block."""
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = None
    msg.model_dump = lambda: {
        "role": "assistant",
        "content": None,
        "function_call": None,
        "tool_calls": None,
    }
    return msg


def _mk_empty_response(finish_reason="stop"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=_mk_empty_message(), finish_reason=finish_reason)]
    return resp


async def test_empty_response_is_not_left_in_history():
    """The failed turn must not poison the session — a retry has to work."""
    provider = _ScriptedProvider([
        _mk_empty_response(),
        _mk_response(_mk_message(content="Recovered.")),
    ])
    orch = _build_orch(provider)

    events1 = [ev async for ev in orch.run("big question")]

    assert events1[-1].type == EventType.ERROR
    assert orch.messages[-1]["role"] == "user", "empty assistant dump was popped"

    events2 = [ev async for ev in orch.run("try again")]

    assert events2[-1].type == EventType.DONE
    assert events2[-1].data["content"] == "Recovered."


async def test_length_exhaustion_gets_specific_error():
    provider = _ScriptedProvider([_mk_empty_response(finish_reason="length")])
    orch = _build_orch(provider)

    events = [ev async for ev in orch.run("big question")]

    assert events[-1].type == EventType.ERROR
    assert "completion budget" in events[-1].data["error"]
