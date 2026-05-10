"""Round-trip translation tests for the Anthropic provider.

Exercises the pure translation helpers in ``providers.anthropic`` so we catch
regressions in tool_use / tool_result plumbing, system-message extraction,
stop-reason mapping, and unsupported-sampling-param stripping — without ever
hitting the network.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# ---------- request-side translation ----------

def test_translate_tools_openai_to_anthropic():
    from selene_agent.providers.anthropic import _translate_tools

    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    out = _translate_tools(oai_tools)
    assert out == [
        {
            "name": "get_weather",
            "description": "Look up weather",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]


def test_translate_tools_none_returns_none():
    from selene_agent.providers.anthropic import _translate_tools

    assert _translate_tools(None) is None
    assert _translate_tools([]) is None


def test_translate_tools_strips_top_level_oneof_anyof_allof():
    """Anthropic rejects ``oneOf``/``allOf``/``anyOf`` at the top of
    ``input_schema`` (iav_to_text tool hit this in prod)."""
    from selene_agent.providers.anthropic import _translate_tools

    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": "iav_to_text",
                "description": "Analyze media",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "image_url": {"type": "string"},
                    },
                    "required": [],
                    "additionalProperties": False,
                    "anyOf": [
                        {"required": ["text"]},
                        {"required": ["image_url"]},
                    ],
                },
            },
        }
    ]
    out = _translate_tools(oai_tools)
    schema = out[0]["input_schema"]
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert "allOf" not in schema
    # Non-forbidden fields are preserved.
    assert schema["type"] == "object"
    assert set(schema["properties"].keys()) == {"text", "image_url"}
    assert schema["additionalProperties"] is False


def test_mark_tools_for_caching_adds_cache_control_to_last_tool_only():
    from selene_agent.providers.anthropic import _mark_tools_for_caching

    tools = [
        {"name": "a", "description": "", "input_schema": {"type": "object", "properties": {}}},
        {"name": "b", "description": "", "input_schema": {"type": "object", "properties": {}}},
    ]
    out = _mark_tools_for_caching(tools)
    assert "cache_control" not in out[0]
    assert out[1]["cache_control"] == {"type": "ephemeral"}
    # Original list is not mutated.
    assert "cache_control" not in tools[1]


def test_mark_tools_for_caching_handles_empty_and_none():
    from selene_agent.providers.anthropic import _mark_tools_for_caching

    assert _mark_tools_for_caching(None) is None
    assert _mark_tools_for_caching([]) == []


def test_build_system_with_caching_returns_list_with_cache_control():
    from selene_agent.providers.anthropic import _build_system_with_caching

    out = _build_system_with_caching("You are Selene.")
    assert out == [
        {"type": "text", "text": "You are Selene.", "cache_control": {"type": "ephemeral"}}
    ]


def test_build_system_with_caching_empty_returns_none():
    from selene_agent.providers.anthropic import _build_system_with_caching

    assert _build_system_with_caching("") is None
    assert _build_system_with_caching("   ") is not None  # non-empty string still cached


def test_mark_last_message_for_caching_string_content_becomes_block_list():
    from selene_agent.providers.anthropic import _mark_last_message_for_caching

    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest"},
    ]
    out = _mark_last_message_for_caching(msgs)
    # Earlier messages untouched.
    assert out[0] == {"role": "user", "content": "first"}
    assert out[1] == {"role": "assistant", "content": "reply"}
    # Last message's content is a block list with cache_control on it.
    assert out[2]["role"] == "user"
    assert out[2]["content"] == [
        {"type": "text", "text": "latest", "cache_control": {"type": "ephemeral"}}
    ]


def test_mark_last_message_for_caching_list_content_marks_last_block():
    from selene_agent.providers.anthropic import _mark_last_message_for_caching

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "also ok"},
            ],
        }
    ]
    out = _mark_last_message_for_caching(msgs)
    content = out[0]["content"]
    assert "cache_control" not in content[0]
    assert content[1]["cache_control"] == {"type": "ephemeral"}


def test_mark_last_message_for_caching_empty_messages_noop():
    from selene_agent.providers.anthropic import _mark_last_message_for_caching

    assert _mark_last_message_for_caching([]) == []


def test_system_extracted_from_leading_message():
    from selene_agent.providers.anthropic import _translate_messages

    msgs = [
        {"role": "system", "content": "You are Selene."},
        {"role": "user", "content": "hi"},
    ]
    system, body = _translate_messages(msgs)
    assert system == "You are Selene."
    assert body == [{"role": "user", "content": "hi"}]


def test_multiple_system_messages_are_merged():
    from selene_agent.providers.anthropic import _translate_messages

    msgs = [
        {"role": "system", "content": "Part A"},
        {"role": "system", "content": "Part B"},
        {"role": "user", "content": "go"},
    ]
    system, body = _translate_messages(msgs)
    assert system == "Part A\n\nPart B"
    assert body == [{"role": "user", "content": "go"}]


def test_assistant_tool_calls_become_tool_use_blocks():
    from selene_agent.providers.anthropic import _translate_messages

    msgs = [
        {"role": "user", "content": "what's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Chicago"}),
                    },
                }
            ],
        },
    ]
    _, body = _translate_messages(msgs)
    assert body[1]["role"] == "assistant"
    blocks = body[1]["content"]
    tool_use = [b for b in blocks if b.get("type") == "tool_use"]
    assert len(tool_use) == 1
    assert tool_use[0]["id"] == "call_123"
    assert tool_use[0]["name"] == "get_weather"
    assert tool_use[0]["input"] == {"location": "Chicago"}


def test_adjacent_tool_results_merge_into_one_user_message():
    from selene_agent.providers.anthropic import _translate_messages

    msgs = [
        {"role": "user", "content": "combo"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "t1", "arguments": "{}"}},
                {"id": "c2", "type": "function",
                 "function": {"name": "t2", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result-1"},
        {"role": "tool", "tool_call_id": "c2", "content": "result-2"},
    ]
    _, body = _translate_messages(msgs)
    # Last message is the merged user-role tool_result batch.
    merged = body[-1]
    assert merged["role"] == "user"
    tool_results = [b for b in merged["content"] if b.get("type") == "tool_result"]
    assert [b["tool_use_id"] for b in tool_results] == ["c1", "c2"]
    assert [b["content"] for b in tool_results] == ["result-1", "result-2"]


def test_tool_choice_maps_to_anthropic_forms():
    from selene_agent.providers.anthropic import _translate_tool_choice

    assert _translate_tool_choice("auto", True) is None
    assert _translate_tool_choice(None, True) is None
    assert _translate_tool_choice("none", True) is None
    assert _translate_tool_choice("required", True) == {"type": "any"}
    assert _translate_tool_choice(
        {"type": "function", "function": {"name": "get_weather"}}, True
    ) == {"type": "tool", "name": "get_weather"}
    # Without tools, the choice is meaningless.
    assert _translate_tool_choice("required", False) is None


# ---------- response-side translation ----------

def _mk_text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _mk_tool_use_block(id: str, name: str, input_: dict):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input_)


def _mk_anthropic_msg(*, content, stop_reason, input_tokens=10, output_tokens=20, id="msg_1"):
    return SimpleNamespace(
        id=id,
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_response_text_only_becomes_openai_content():
    from selene_agent.providers.anthropic import _to_openai_response

    anth = _mk_anthropic_msg(
        content=[_mk_text_block("Hello there.")],
        stop_reason="end_turn",
    )
    out = _to_openai_response(anth, "claude-opus-4-7")
    assert out.choices[0].message.content == "Hello there."
    assert out.choices[0].message.tool_calls is None
    assert out.choices[0].finish_reason == "stop"
    assert out.usage.prompt_tokens == 10
    assert out.usage.completion_tokens == 20
    assert out.model == "claude-opus-4-7"


def test_response_tool_use_becomes_openai_tool_calls():
    from selene_agent.providers.anthropic import _to_openai_response

    anth = _mk_anthropic_msg(
        content=[
            _mk_text_block("I'll check that."),
            _mk_tool_use_block("toolu_abc", "get_weather", {"location": "NYC"}),
        ],
        stop_reason="tool_use",
    )
    out = _to_openai_response(anth, "claude-opus-4-7")
    msg = out.choices[0].message
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.id == "toolu_abc"
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"location": "NYC"}
    assert out.choices[0].finish_reason == "tool_calls"


def test_stop_reason_max_tokens_maps_to_length():
    from selene_agent.providers.anthropic import _to_openai_response

    anth = _mk_anthropic_msg(
        content=[_mk_text_block("cut short")], stop_reason="max_tokens"
    )
    out = _to_openai_response(anth, "claude-opus-4-7")
    assert out.choices[0].finish_reason == "length"


# ---------- integration: kwargs handed to anthropic SDK ----------

@pytest.mark.asyncio
async def test_opus_47_strips_temperature_and_top_p(monkeypatch):
    """Opus 4.7 rejects sampling params — the provider must drop them."""
    from selene_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test", model="claude-opus-4-7")

    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return _mk_anthropic_msg(
            content=[_mk_text_block("ok")], stop_reason="end_turn"
        )

    provider._client.messages = SimpleNamespace(create=_fake_create)

    await provider.chat_completion(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=500,
    )

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert captured["max_tokens"] == 500
    # System is rendered as a block list carrying a cache_control breakpoint.
    assert captured["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert captured["model"] == "claude-opus-4-7"


@pytest.mark.asyncio
async def test_older_model_forwards_temperature(monkeypatch):
    from selene_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test", model="claude-3-5-sonnet-latest")

    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return _mk_anthropic_msg(
            content=[_mk_text_block("ok")], stop_reason="end_turn"
        )

    provider._client.messages = SimpleNamespace(create=_fake_create)

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.4,
        top_p=0.8,
        max_tokens=100,
    )

    assert captured["temperature"] == 0.4
    assert captured["top_p"] == 0.8


@pytest.mark.asyncio
async def test_pop_last_cache_stats_captures_and_resets(monkeypatch):
    """Cache-token counts from the Anthropic usage block must be stashed on
    the provider and returned by pop_last_cache_stats exactly once."""
    from selene_agent.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="sk-ant-test", model="claude-opus-4-7")

    # Fresh provider — nothing to report yet.
    assert provider.pop_last_cache_stats() == {"read": 0, "create": 0}

    async def _fake_create(**kwargs):
        return SimpleNamespace(
            id="msg_x",
            content=[_mk_text_block("ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                cache_read_input_tokens=4321,
                cache_creation_input_tokens=77,
            ),
        )

    provider._client.messages = SimpleNamespace(create=_fake_create)

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        max_tokens=100,
    )

    assert provider.pop_last_cache_stats() == {"read": 4321, "create": 77}
    # Second pop is zeroed — no double-count.
    assert provider.pop_last_cache_stats() == {"read": 0, "create": 0}


def test_vllm_pop_last_cache_stats_initial_zero():
    from selene_agent.providers.vllm import VLLMProvider

    vllm = VLLMProvider(base_url="http://x", api_key="", model="gpt-3.5-turbo")
    # No request fired yet → nothing to report.
    assert vllm.pop_last_cache_stats() == {"read": 0, "create": 0}


@pytest.mark.asyncio
async def test_vllm_pop_last_cache_stats_captures_and_resets():
    """vLLM (>=0.6) reports prefix-cache hits via usage.prompt_tokens_details.
    The provider must surface those on pop_last_cache_stats so per-turn metrics
    show real hit rates instead of always-zero stubs."""
    from selene_agent.providers.vllm import VLLMProvider

    provider = VLLMProvider(base_url="http://x", api_key="", model="gpt-3.5-turbo")

    async def _fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", model_extra={}),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=12000,
                completion_tokens=1,
                total_tokens=12001,
                prompt_tokens_details=SimpleNamespace(cached_tokens=10240),
            ),
        )

    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=_fake_create)
    )

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=1,
    )

    # read = cached_tokens; create = prompt_tokens - cached_tokens.
    assert provider.pop_last_cache_stats() == {"read": 10240, "create": 1760}
    # Second pop is zeroed — no double-count.
    assert provider.pop_last_cache_stats() == {"read": 0, "create": 0}


@pytest.mark.asyncio
async def test_vllm_pop_last_cache_stats_handles_missing_details():
    """Older vLLM builds may omit prompt_tokens_details entirely. The provider
    must report read=0 / create=prompt_tokens in that case rather than crash."""
    from selene_agent.providers.vllm import VLLMProvider

    provider = VLLMProvider(base_url="http://x", api_key="", model="gpt-3.5-turbo")

    async def _fake_create(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok", model_extra={}),
                finish_reason="stop",
            )],
            usage=SimpleNamespace(
                prompt_tokens=500,
                completion_tokens=1,
                total_tokens=501,
            ),
        )

    provider._client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=_fake_create)
    )

    await provider.chat_completion(
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        max_tokens=1,
    )

    assert provider.pop_last_cache_stats() == {"read": 0, "create": 500}
