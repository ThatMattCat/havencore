"""Pluggable LLM provider seam.

Every call site in the agent (main orchestrator loop, session summary,
autonomy one-shot) should go through ``LLMProvider.chat_completion()``
instead of calling ``AsyncOpenAI.chat.completions.create()`` directly, so
the backing model can be swapped at runtime (local vLLM ↔ Anthropic Claude
↔ OpenAI ChatGPT) without changing orchestrator code.

Contract: providers accept OpenAI-shaped kwargs and return an
``openai.types.chat.ChatCompletion`` — non-OpenAI backends translate in
both directions.
"""
from selene_agent.providers.base import LLMProvider, VALID_PROVIDERS
from selene_agent.providers.factory import build_provider

__all__ = ["LLMProvider", "VALID_PROVIDERS", "build_provider"]
