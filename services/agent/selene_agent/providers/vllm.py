"""vLLM provider — thin wrapper over AsyncOpenAI pointed at the local vLLM endpoint.

This is the default provider. It forwards kwargs to the OpenAI SDK
unchanged, so existing orchestrator behavior (including the `<tool_call>`
content-tag fallback and native function-calling response shapes) is
preserved exactly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion


logger = logging.getLogger(__name__)


class VLLMProvider:
    name = "vllm"

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "dummy-key")
        self.model = model
        self._last_reasoning: Optional[str] = None
        self._last_cache_read: int = 0
        self._last_cache_create: int = 0
        self._max_model_len: Optional[int] = None
        self._max_model_len_fetched: bool = False

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
        p._last_cache_read = 0
        p._last_cache_create = 0
        p._max_model_len = None
        p._max_model_len_fetched = False
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
        self._capture_cache_stats(response)
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

    def _capture_cache_stats(self, response: ChatCompletion) -> None:
        """Pull prefix-cache hit counts off the response.

        vLLM (>= 0.6) reports prompt-cache hits via the OpenAI-shaped
        ``usage.prompt_tokens_details.cached_tokens`` field. ``read`` is
        the cached-input slice; ``create`` is the rest of the prompt that
        had to be prefilled this call. Naming mirrors the Anthropic provider
        so the dashboard's per-turn metrics surface either backend uniformly.
        """
        self._last_cache_read = 0
        self._last_cache_create = 0
        try:
            usage = response.usage
        except (AttributeError, IndexError):
            return
        if usage is None:
            return
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            v = getattr(details, "cached_tokens", None)
            if isinstance(v, int):
                cached = v
            else:
                extra = getattr(details, "model_extra", None) or {}
                if isinstance(extra, dict):
                    v = extra.get("cached_tokens")
                    if isinstance(v, int):
                        cached = v
        if not cached:
            extra = getattr(usage, "model_extra", None) or {}
            if isinstance(extra, dict):
                nested = extra.get("prompt_tokens_details") or {}
                if isinstance(nested, dict):
                    v = nested.get("cached_tokens")
                    if isinstance(v, int):
                        cached = v
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        cached = max(0, int(cached or 0))
        self._last_cache_read = cached
        self._last_cache_create = max(0, prompt_tokens - cached)

    def pop_last_cache_stats(self) -> Dict[str, int]:
        out = {"read": self._last_cache_read, "create": self._last_cache_create}
        self._last_cache_read = 0
        self._last_cache_create = 0
        return out

    def pop_last_reasoning(self) -> Optional[str]:
        value = self._last_reasoning
        self._last_reasoning = None
        return value

    async def get_max_model_len(self) -> Optional[int]:
        """Fetch ``max_model_len`` from vLLM's ``/v1/models`` once and cache.

        vLLM surfaces the running ``--max-model-len`` value on each model
        record, so this scales automatically when compose changes. Failures
        cache ``None`` so we don't pound the endpoint when it's unhealthy.
        """
        if self._max_model_len_fetched:
            return self._max_model_len
        self._max_model_len_fetched = True
        try:
            resp = await self._client.models.list()
        except Exception as e:
            logger.warning("vLLM /v1/models query failed: %s", e)
            self._max_model_len = None
            return None
        # Pick the entry that matches our served model name when present;
        # otherwise fall back to the first model the endpoint reports.
        chosen = None
        try:
            for entry in resp.data:
                if getattr(entry, "id", None) == self.model:
                    chosen = entry
                    break
            if chosen is None and resp.data:
                chosen = resp.data[0]
        except AttributeError:
            chosen = None
        if chosen is None:
            self._max_model_len = None
            return None
        # vLLM puts ``max_model_len`` directly on the entry; the OpenAI SDK
        # doesn't type it, so probe model_extra and direct attrs.
        candidate: Optional[int] = None
        extra = getattr(chosen, "model_extra", None)
        if isinstance(extra, dict):
            v = extra.get("max_model_len")
            if isinstance(v, int) and v > 0:
                candidate = v
        if candidate is None:
            v = getattr(chosen, "max_model_len", None)
            if isinstance(v, int) and v > 0:
                candidate = v
        self._max_model_len = candidate
        if candidate is not None:
            logger.info("vLLM max_model_len = %d (model=%s)", candidate, self.model)
        else:
            logger.warning("vLLM /v1/models did not surface max_model_len for %s", self.model)
        return self._max_model_len
