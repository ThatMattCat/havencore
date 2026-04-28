"""Tests for the chars/4 token estimator and provider-aware threshold resolver."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.utils import config
from selene_agent.utils.tokens import (
    estimate_messages_tokens,
    resolve_context_limit_tokens,
)


def test_estimate_empty_is_zero():
    assert estimate_messages_tokens([]) == 0


def test_estimate_is_monotonic_in_length():
    short = [{"role": "user", "content": "hi"}]
    longer = [{"role": "user", "content": "hi" + ("x" * 200)}]
    assert estimate_messages_tokens(longer) > estimate_messages_tokens(short)


def test_estimate_includes_tool_call_arguments():
    """tool_calls.arguments should count toward the budget — they ride into
    every chat-completion call along with the rest of the message bytes."""
    plain = [{"role": "assistant", "content": "ok"}]
    with_tool = [
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "tool",
                        "arguments": '{"a":' + ("9" * 500) + "}",
                    },
                }
            ],
        }
    ]
    assert estimate_messages_tokens(with_tool) > estimate_messages_tokens(plain)


def test_estimate_handles_unserializable_gracefully():
    """A message with a non-JSON-serializable payload must not crash —
    fall back to str()."""

    class Weird:
        def __repr__(self):
            return "x" * 100

    msgs = [{"role": "tool", "content": Weird()}]
    # default=str inside the helper handles the conversion.
    assert estimate_messages_tokens(msgs) > 0


async def test_resolve_uses_override_when_set(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 12_000)
    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=1_000_000)
    # Override wins, the provider is never consulted.
    assert await resolve_context_limit_tokens(provider) == 12_000
    provider.get_max_model_len.assert_not_called()


async def test_resolve_falls_back_to_fraction_of_max_model_len(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75)
    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=32_768)
    assert await resolve_context_limit_tokens(provider) == int(32_768 * 0.75)


async def test_resolve_returns_none_when_provider_cannot_report(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(return_value=None)
    assert await resolve_context_limit_tokens(provider) is None


async def test_resolve_returns_none_when_provider_raises(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    provider = MagicMock()
    provider.get_max_model_len = AsyncMock(side_effect=RuntimeError("boom"))
    assert await resolve_context_limit_tokens(provider) is None


async def test_resolve_returns_none_when_provider_missing_method(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)

    class Bare:
        pass

    assert await resolve_context_limit_tokens(Bare()) is None


async def test_resolve_returns_none_when_provider_is_none(monkeypatch):
    monkeypatch.setattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    assert await resolve_context_limit_tokens(None) is None
