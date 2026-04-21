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
        self._last_reasoning: Optional[str] = None

    @classmethod
    def from_client(cls, *, client: AsyncOpenAI, model: str) -> "VLLMProvider":
        """Construct a provider around an already-built AsyncOpenAI. Used by
        the orchestrator's default fallback so tests that pass a mock client
        actually exercise that mock (rather than a fresh AsyncOpenAI targeting
        ``LLM_API_BASE``)."""
        p = cls.__new__(cls)
        p._client = client
        p.model = model
        p._last_reasoning = None
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
        response = await self._client.chat.completions.create(**kwargs)
        self._capture_reasoning(response)
        return response

    def _capture_reasoning(self, response: ChatCompletion) -> None:
        # vLLM with --reasoning-parser glm45 (and similar) returns chain-of-thought
        # in a non-standard field alongside the clean answer. Observed in the wild
        # under ``reasoning`` (glm45 parser in current vLLM builds); earlier docs
        # and other parsers use ``reasoning_content``. The OpenAI SDK doesn't type
        # either, so probe both names via the Pydantic model_extra dict (populated
        # when extras are parsed) and direct attribute access. Keep the first
        # non-empty *string* — attribute access on some mock/SDK shapes can yield
        # non-string truthy values that we must skip over.
        try:
            msg = response.choices[0].message
        except (AttributeError, IndexError):
            self._last_reasoning = None
            return
        extra = getattr(msg, "model_extra", None)
        if not isinstance(extra, dict):
            extra = {}
        picked: Optional[str] = None
        for key in ("reasoning", "reasoning_content"):
            for candidate in (extra.get(key), getattr(msg, key, None)):
                if isinstance(candidate, str) and candidate.strip():
                    picked = candidate
                    break
            if picked:
                break
        self._last_reasoning = picked

    def pop_last_cache_stats(self) -> Dict[str, int]:
        return {"read": 0, "create": 0}

    def pop_last_reasoning(self) -> Optional[str]:
        value = self._last_reasoning
        self._last_reasoning = None
        return value
