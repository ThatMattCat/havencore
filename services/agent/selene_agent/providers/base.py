"""LLMProvider protocol — the seam every chat-completion call goes through."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from openai.types.chat import ChatCompletion


VALID_PROVIDERS = ("vllm", "anthropic", "openai")


@runtime_checkable
class LLMProvider(Protocol):
    """A backend that answers OpenAI-shaped chat-completion requests.

    Implementations must return an ``openai.types.chat.ChatCompletion`` so
    the orchestrator can call ``.choices[0].message.model_dump()`` and
    iterate ``.tool_calls`` without caring which provider is active.
    """

    name: str
    model: str

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
        ...

    def pop_last_cache_stats(self) -> Dict[str, int]:
        """Return cache-token counts from the most recent chat_completion call
        and reset internal state, so repeated calls don't double-count.

        Shape: ``{"read": <int>, "create": <int>}``. Providers without prompt
        caching (vLLM, OpenAI stub) return zeros.
        """
        ...
