"""Token counting helpers for context-budget gating.

The agent doesn't load a tokenizer — every call site that needs a token
estimate uses chars/4 against the JSON-serialized message list. The
constant matches the convention used in ``api/memory.py`` for the L4
block and is a deliberate floor rather than a precise count: under the
real BPE tokenization the same payload is usually fewer tokens, so we
gate slightly conservatively.

Threshold resolution layers two env vars on top of the active provider's
``get_max_model_len()``:

- ``CONVERSATION_CONTEXT_LIMIT_TOKENS`` (>0) — absolute override. Wins
  unconditionally when set.
- ``CONVERSATION_CONTEXT_LIMIT_FRACTION`` (default 0.75) — fraction of
  the provider's reported max-model-len. Lets the threshold scale
  automatically when ``--max-model-len`` changes in compose.

Returns ``None`` when the provider can't report a max length and no
override is configured — call sites should treat that as "skip the size
check this round" rather than synthesize a guess.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from selene_agent.utils import config


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """chars/4 over the JSON-serialized message list.

    Counts the full ``messages[0]`` system prompt (including the L4 block)
    so the threshold reflects what actually rides into every chat-completion
    call. Tool-call arguments and tool_call_id strings are included via the
    standard json.dumps walk.
    """
    if not messages:
        return 0
    total = 0
    for m in messages:
        try:
            serialized = json.dumps(m, default=str)
        except (TypeError, ValueError):
            serialized = str(m)
        total += len(serialized)
    return total // 4


async def resolve_context_limit_tokens(provider: Any) -> Optional[int]:
    """Compute the active context-size threshold.

    Order:
    1. ``CONVERSATION_CONTEXT_LIMIT_TOKENS`` override, if > 0.
    2. ``provider.get_max_model_len()`` × ``CONVERSATION_CONTEXT_LIMIT_FRACTION``.
    3. ``None`` (skip the check) if the provider can't report a value.
    """
    override = getattr(config, "CONVERSATION_CONTEXT_LIMIT_TOKENS_OVERRIDE", 0)
    if override and override > 0:
        return int(override)
    if provider is None:
        return None
    getter = getattr(provider, "get_max_model_len", None)
    if getter is None:
        return None
    try:
        mml = await getter()
    except Exception:
        return None
    if not isinstance(mml, int) or mml <= 0:
        return None
    fraction = float(getattr(config, "CONVERSATION_CONTEXT_LIMIT_FRACTION", 0.75))
    return max(1, int(mml * fraction))
