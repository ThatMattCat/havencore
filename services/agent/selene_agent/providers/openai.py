"""OpenAI (api.openai.com) provider — stub, not wired yet.

Structurally identical to VLLMProvider (OpenAI-compat in, OpenAI-compat out);
the only differences are base_url, api_key, and model. Deferred until after
the Anthropic path is validated in production.
"""
from __future__ import annotations


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        raise NotImplementedError(
            "OpenAI (api.openai.com) provider is not yet wired. "
            "Use 'vllm' or 'anthropic' via the System-page toggle for now."
        )
