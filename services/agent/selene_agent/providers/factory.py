"""build_provider — resolve a provider name to a ready ``LLMProvider``.

Reads env via ``selene_agent.utils.config`` and falls back to vLLM (with a
WARN log) on unknown names or missing required config for the requested
provider. Callers can assume a usable provider is always returned.

The vLLM model is detected at lifespan startup from ``/v1/models`` (not env),
so callers must pass it in via ``vllm_model``.
"""
from __future__ import annotations

from typing import Optional

from selene_agent.providers.base import LLMProvider, VALID_PROVIDERS
from selene_agent.providers.vllm import VLLMProvider
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


def _build_vllm(vllm_model: str) -> LLMProvider:
    return VLLMProvider(
        base_url=config.LLM_API_BASE,
        api_key=config.LLM_API_KEY,
        model=vllm_model,
    )


def _build_anthropic(vllm_model: str) -> LLMProvider:
    from selene_agent.providers.anthropic import AnthropicProvider

    api_key = getattr(config, "ANTHROPIC_API_KEY", "") or ""
    model = getattr(config, "ANTHROPIC_MODEL", "") or "claude-opus-4-7"
    if not api_key:
        logger.warning(
            "[providers] ANTHROPIC_API_KEY is empty; falling back to vLLM."
        )
        return _build_vllm(vllm_model)
    return AnthropicProvider(api_key=api_key, model=model)


def _build_openai(vllm_model: str) -> LLMProvider:
    logger.warning(
        "[providers] 'openai' provider is not wired yet; falling back to vLLM."
    )
    return _build_vllm(vllm_model)


def build_provider(name: str, *, vllm_model: str) -> LLMProvider:
    """Return an ``LLMProvider`` for ``name``; falls back to vLLM on failure.

    ``vllm_model`` is the model id detected from the local vLLM ``/v1/models``
    endpoint at lifespan startup; it's used both for the direct vLLM provider
    and as the fallback target if the requested provider can't be built.
    """
    normalized = (name or "").strip().lower()
    if normalized not in VALID_PROVIDERS:
        logger.warning(
            f"[providers] unknown provider {name!r}; valid={VALID_PROVIDERS}; "
            "falling back to vLLM."
        )
        return _build_vllm(vllm_model)

    try:
        if normalized == "vllm":
            return _build_vllm(vllm_model)
        if normalized == "anthropic":
            return _build_anthropic(vllm_model)
        if normalized == "openai":
            return _build_openai(vllm_model)
    except Exception as e:
        logger.error(
            f"[providers] failed to build {normalized!r} provider ({e!r}); "
            "falling back to vLLM."
        )
        return _build_vllm(vllm_model)

    return _build_vllm(vllm_model)
