"""Anthropic provider — wraps AsyncAnthropic and translates to/from the
OpenAI chat-completions shape the orchestrator expects.

On Opus 4.7 the following sampling params are REMOVED and will 400 if sent:
``temperature``, ``top_p``, ``top_k``, ``budget_tokens``. Extended thinking
on 4.7 only accepts ``thinking={"type": "adaptive"}``. We drop the unsupported
params silently for the default model (claude-opus-4-7) and forward them for
older Anthropic models that still accept them.

Prompt caching: three ephemeral breakpoints per call — last tool entry (caches
the tools array), system block (caches tools + system), and the last message's
last content block (caches the accumulated conversation). Render order is
``tools`` → ``system`` → ``messages``, so each breakpoint cascades over
everything before it. 5-minute TTL, refreshed on each hit.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage

logger = logging.getLogger(__name__)


# Models where legacy sampling params (temperature, top_p, top_k) still work.
# Opus 4.7 rejects them outright. Be conservative — only forward for models
# we know accept them.
_LEGACY_SAMPLING_OK_PREFIXES = (
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3",
)


def _accepts_sampling(model: str) -> bool:
    return model.startswith(_LEGACY_SAMPLING_OK_PREFIXES)


_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
    "refusal": "stop",
}


# Static context-window sizes (input tokens) for Anthropic models. The API
# doesn't expose this on a per-model endpoint, so we keep a small map of the
# models we actually run. Unknown models fall back to ``_ANTHROPIC_DEFAULT``,
# which matches the long-standing 200K Claude window.
_ANTHROPIC_DEFAULT_MAX_TOKENS = 200_000
_ANTHROPIC_MAX_TOKENS_BY_PREFIX = (
    ("claude-opus-4-7", 200_000),
    ("claude-opus-4-6", 200_000),
    ("claude-opus-4-5", 200_000),
    ("claude-opus-4", 200_000),
    ("claude-sonnet-4-6", 200_000),
    ("claude-sonnet-4-5", 200_000),
    ("claude-sonnet-4", 200_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-haiku-4", 200_000),
    ("claude-3-5", 200_000),
    ("claude-3-7", 200_000),
    ("claude-3", 200_000),
)


def _anthropic_max_tokens_for(model: str) -> int:
    if not model:
        return _ANTHROPIC_DEFAULT_MAX_TOKENS
    for prefix, size in _ANTHROPIC_MAX_TOKENS_BY_PREFIX:
        if model.startswith(prefix):
            return size
    return _ANTHROPIC_DEFAULT_MAX_TOKENS


_DISALLOWED_TOP_LEVEL_SCHEMA_KEYS = ("oneOf", "allOf", "anyOf")


def _sanitize_input_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic rejects ``oneOf``/``allOf``/``anyOf`` at the top level of
    ``input_schema``. Strip them — the model can infer cross-field constraints
    from the field descriptions. Deeper occurrences (inside ``properties``) are
    allowed and preserved."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    cleaned = {k: v for k, v in schema.items() if k not in _DISALLOWED_TOP_LEVEL_SCHEMA_KEYS}
    cleaned.setdefault("type", "object")
    cleaned.setdefault("properties", {})
    return cleaned


def _translate_tools(
    oai_tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """OpenAI function-tool schema → Anthropic tool schema."""
    if not oai_tools:
        return None
    out: List[Dict[str, Any]] = []
    for t in oai_tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", "") or "",
                "input_schema": _sanitize_input_schema(params),
            }
        )
    return out or None


def _translate_tool_choice(
    tc: Any, tools_present: bool
) -> Optional[Dict[str, Any]]:
    """OpenAI ``tool_choice`` → Anthropic ``tool_choice``.

    - ``None`` / ``"auto"``  → omit (Anthropic default is auto)
    - ``"none"``             → omit tools entirely (caller's job); we return None
    - ``"required"``         → ``{"type": "any"}``
    - ``{"type": "function", "function": {"name": X}}`` → ``{"type": "tool", "name": X}``
    """
    if not tools_present:
        return None
    if tc is None or tc == "auto":
        return None
    if tc == "required":
        return {"type": "any"}
    if tc == "none":
        return None
    if isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def _translate_messages(
    oai_messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """OpenAI messages → (system_text, anthropic_messages).

    - Collapse leading ``role=system`` messages into a single system string.
    - ``role=assistant`` with ``tool_calls`` → assistant with ``tool_use`` blocks.
    - ``role=tool`` → ``role=user`` with ``tool_result`` blocks. Adjacent tool
      results are merged into a single user message (Anthropic requires this).
    """
    system_parts: List[str] = []
    body: List[Dict[str, Any]] = []

    # Extract leading system messages; also absorb any stray system messages later.
    for m in oai_messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        body.append(m)

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(body):
        m = body[i]
        role = m.get("role")

        if role == "user":
            content = m.get("content")
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                out.append({"role": "user", "content": content})
            else:
                out.append({"role": "user", "content": ""})
            i += 1
            continue

        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    tool_input = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                        "name": fn.get("name") or "",
                        "input": tool_input,
                    }
                )
            if not blocks:
                # Anthropic disallows empty content; send a single space.
                blocks = [{"type": "text", "text": " "}]
            out.append({"role": "assistant", "content": blocks})
            i += 1
            continue

        if role == "tool":
            # Merge consecutive tool results into a single user message.
            tool_blocks: List[Dict[str, Any]] = []
            while i < len(body) and body[i].get("role") == "tool":
                tm = body[i]
                tool_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tm.get("tool_call_id") or "",
                        "content": _stringify_tool_content(tm.get("content")),
                    }
                )
                i += 1
            out.append({"role": "user", "content": tool_blocks})
            continue

        # Unknown role — skip.
        i += 1

    system_text = "\n\n".join(system_parts).strip()
    return system_text, out


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


_EPHEMERAL = {"type": "ephemeral"}


def _mark_tools_for_caching(
    anth_tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Attach ``cache_control`` to the last tool. Anthropic caches the entire
    tools array up to the breakpoint, so one mark covers them all."""
    if not anth_tools:
        return anth_tools
    out = list(anth_tools)
    out[-1] = {**out[-1], "cache_control": _EPHEMERAL}
    return out


def _build_system_with_caching(
    system_text: str,
) -> Optional[List[Dict[str, Any]]]:
    """Render ``system`` as a list-of-blocks so we can attach ``cache_control``.
    Caching the system block extends the cached prefix through tools + system."""
    if not system_text:
        return None
    return [{"type": "text", "text": system_text, "cache_control": _EPHEMERAL}]


def _mark_last_message_for_caching(
    anth_messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach ``cache_control`` to the last content block of the last message
    so the accumulated conversation gets cached. Normalizes string content to
    a single ``text`` block first — ``cache_control`` only attaches to blocks."""
    if not anth_messages:
        return anth_messages
    out = list(anth_messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {"type": "text", "text": content, "cache_control": _EPHEMERAL}
        ]
    elif isinstance(content, list) and content:
        new_content = list(content)
        new_content[-1] = {**new_content[-1], "cache_control": _EPHEMERAL}
        last["content"] = new_content
    else:
        # Empty / unexpected — skip caching this turn rather than mutate.
        return out
    out[-1] = last
    return out


def _to_openai_response(
    anthropic_msg: Any, model: str
) -> ChatCompletion:
    """Translate an Anthropic ``Message`` response into an OpenAI ``ChatCompletion``."""
    text_parts: List[str] = []
    tool_calls: List[ChatCompletionMessageToolCall] = []

    for block in getattr(anthropic_msg, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            try:
                args_json = json.dumps(getattr(block, "input", {}) or {})
            except (TypeError, ValueError):
                args_json = "{}"
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=getattr(block, "id", f"call_{uuid.uuid4().hex[:16]}"),
                    type="function",
                    function=Function(
                        name=getattr(block, "name", "") or "",
                        arguments=args_json,
                    ),
                )
            )
        # "thinking" blocks and others — not exposed to the orchestrator.

    content_str = "".join(text_parts) if text_parts else None
    # If there are tool calls, OpenAI convention is content=None.
    if tool_calls:
        content_str = content_str or None

    message = ChatCompletionMessage(
        role="assistant",
        content=content_str,
        tool_calls=tool_calls or None,
    )

    anth_stop = getattr(anthropic_msg, "stop_reason", None) or "end_turn"
    finish_reason = _STOP_REASON_MAP.get(anth_stop, "stop")

    choice = Choice(
        index=0,
        finish_reason=finish_reason,
        message=message,
        logprobs=None,
    )

    usage_obj = getattr(anthropic_msg, "usage", None)
    prompt_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0) if usage_obj else 0
    completion_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0) if usage_obj else 0
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )

    return ChatCompletion(
        id=getattr(anthropic_msg, "id", f"chatcmpl_{uuid.uuid4().hex[:16]}"),
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str):
        if not api_key:
            raise RuntimeError(
                "Anthropic provider requires ANTHROPIC_API_KEY; none provided."
            )
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self._last_cache_read = 0
        self._last_cache_create = 0

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
        system_text, anth_messages = _translate_messages(messages)
        anth_tools = _translate_tools(tools)
        anth_tool_choice = _translate_tool_choice(tool_choice, bool(anth_tools))

        anth_tools = _mark_tools_for_caching(anth_tools)
        system_param = _build_system_with_caching(system_text)
        anth_messages = _mark_last_message_for_caching(anth_messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "max_tokens": max_tokens,
        }
        if system_param is not None:
            kwargs["system"] = system_param
        if anth_tools:
            kwargs["tools"] = anth_tools
            if anth_tool_choice is not None:
                kwargs["tool_choice"] = anth_tool_choice

        if _accepts_sampling(self.model):
            kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
        # Opus 4.7+: sampling params are REMOVED — silently drop.

        resp = await self._client.messages.create(**kwargs)

        usage = getattr(resp, "usage", None)
        cache_read = 0
        cache_create = 0
        if usage is not None:
            cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            if cache_read or cache_create:
                logger.info(
                    "[anthropic] cache read=%d create=%d input=%d output=%d",
                    cache_read,
                    cache_create,
                    int(getattr(usage, "input_tokens", 0) or 0),
                    int(getattr(usage, "output_tokens", 0) or 0),
                )
        self._last_cache_read = cache_read
        self._last_cache_create = cache_create

        return _to_openai_response(resp, self.model)

    def pop_last_cache_stats(self) -> Dict[str, int]:
        stats = {"read": self._last_cache_read, "create": self._last_cache_create}
        self._last_cache_read = 0
        self._last_cache_create = 0
        return stats

    def pop_last_reasoning(self) -> Optional[str]:
        # Anthropic's ``thinking`` blocks aren't surfaced through the OpenAI
        # translation, so there's never anything to pop here. Conform to the
        # protocol so the orchestrator's ``pop_last_reasoning`` probe doesn't
        # AttributeError.
        return None

    async def get_max_model_len(self) -> Optional[int]:
        """Look up the model's static context window from the prefix map.

        Anthropic doesn't expose per-model max-token info on a public endpoint
        we can hit cheaply, so the map in this module is the source of truth.
        Unknown models return the conservative 200K default."""
        return _anthropic_max_tokens_for(self.model)
