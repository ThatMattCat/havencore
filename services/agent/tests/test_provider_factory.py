"""Tests for the provider factory — build_provider() resolution + fallback."""
from __future__ import annotations

import pytest


def test_build_vllm_returns_vllm_provider(monkeypatch):
    from selene_agent.providers import build_provider
    from selene_agent.providers.vllm import VLLMProvider

    monkeypatch.setattr(
        "selene_agent.utils.config.LLM_API_BASE", "http://vllm:8000/v1"
    )
    monkeypatch.setattr("selene_agent.utils.config.LLM_API_KEY", "dummy")
    p = build_provider("vllm", vllm_model="test-model")
    assert isinstance(p, VLLMProvider)
    assert p.name == "vllm"
    assert p.model == "test-model"


def test_build_anthropic_with_key(monkeypatch):
    from selene_agent.providers import build_provider
    from selene_agent.providers.anthropic import AnthropicProvider

    monkeypatch.setattr(
        "selene_agent.utils.config.ANTHROPIC_API_KEY", "sk-ant-test"
    )
    monkeypatch.setattr(
        "selene_agent.utils.config.ANTHROPIC_MODEL", "claude-opus-4-7"
    )
    p = build_provider("anthropic", vllm_model="ignored")
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"
    assert p.model == "claude-opus-4-7"


def test_build_anthropic_without_key_falls_back_to_vllm(monkeypatch):
    from selene_agent.providers import build_provider
    from selene_agent.providers.vllm import VLLMProvider

    monkeypatch.setattr("selene_agent.utils.config.ANTHROPIC_API_KEY", "")
    p = build_provider("anthropic", vllm_model="fallback-model")
    assert isinstance(p, VLLMProvider)
    assert p.model == "fallback-model"


def test_unknown_provider_falls_back_to_vllm():
    from selene_agent.providers import build_provider
    from selene_agent.providers.vllm import VLLMProvider

    p = build_provider("mystery-llm", vllm_model="m")
    assert isinstance(p, VLLMProvider)


def test_openai_stub_falls_back_to_vllm():
    from selene_agent.providers import build_provider
    from selene_agent.providers.vllm import VLLMProvider

    p = build_provider("openai", vllm_model="m")
    # OpenAI provider is stubbed (raises NotImplementedError on direct construction);
    # the factory wraps that and falls back to vLLM.
    assert isinstance(p, VLLMProvider)


def test_valid_providers_constant():
    from selene_agent.providers import VALID_PROVIDERS

    assert set(VALID_PROVIDERS) == {"vllm", "anthropic", "openai"}
