"""vLLM provider — thin wrapper over AsyncOpenAI pointed at the local vLLM endpoint.

This is the default provider. It forwards kwargs to the OpenAI SDK
unchanged, so existing orchestrator behavior (including the `<tool_call>`
content-tag fallback and native function-calling response shapes) is
preserved exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


class VLLMProvider:
    name = "vllm"

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "dummy-key")
        self.model = model

    @classmethod
    def from_client(cls, *, client: AsyncOpenAI, model: str) -> "VLLMProvider":
        """Construct a provider around an already-built AsyncOpenAI. Used by
        the orchestrator's default fallback so tests that pass a mock client
        actually exercise that mock (rather than a fresh AsyncOpenAI targeting
        ``LLM_API_BASE``)."""
        p = cls.__new__(cls)
        p._client = client
        p.model = model
        return p

    async def chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: int = 1024,
    ) -> ChatCompletion:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if top_p is not None:
            kwargs["top_p"] = top_p
        return await self._client.chat.completions.create(**kwargs)
