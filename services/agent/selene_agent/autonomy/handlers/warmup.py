"""vLLM prefix-cache warmup handler.

Sends a minimal chat completion to the chat model with the *same* system
prompt + tools schema the live ``AgentOrchestrator`` builds, so vLLM's
prefix cache treats the warmup ping and a real user turn as the same
prefix. ``max_tokens=1`` keeps the decode phase trivial; the ping's
purpose is the prefill, which is mostly a cache hit and refreshes the
prefix's LRU position.

Without this, autonomy turns (briefing, anomaly_sweep, watch, ...) build
their own handler-specific system prompts, which compete for KV-cache
slots and evict the chat prefix over a few minutes of idle chat — making
the next user turn pay a full prefill (the "first request slow after
waiting" symptom).

This is a direct vLLM call. It does NOT run an agent loop, does NOT
consume real tools, and does NOT touch session state.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


async def _build_chat_system_prompt() -> str:
    """Replicate ``AgentOrchestrator.initialize`` exactly.

    Diverging from the live path defeats the warmup — vLLM keys the prefix
    cache on rendered tokens, so any drift here means the warmup populates
    a different cache slot than real chat.
    """
    system_prompt = config.SYSTEM_PROMPT
    try:
        from selene_agent.utils.agent_state import get_agent_phase
        phase = await get_agent_phase()
        if phase == "learning":
            system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_LEARNING_ADDENDUM
        else:
            system_prompt = system_prompt + "\n" + config.SYSTEM_PROMPT_OPERATING_ADDENDUM
    except Exception as e:
        logger.warning(f"[warmup] phase lookup failed: {e}")

    try:
        from selene_agent.utils.l4_context import build_l4_block
        block = await build_l4_block()
        if block:
            system_prompt = block + "\n\n" + system_prompt
    except Exception as e:
        logger.warning(f"[warmup] L4 block build failed: {e}")
    return system_prompt


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
    provider_getter=None,
) -> Dict[str, Any]:
    if provider_getter is None:
        return {
            "status": "error",
            "summary": "warmup skipped — no provider_getter",
            "error": "provider_getter is None",
            "severity": "none",
            "signature_hash": None,
            "messages": [],
            "metrics": {},
        }

    system_prompt = await _build_chat_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "ping"},
    ]

    provider = provider_getter()
    started = time.perf_counter()
    try:
        response = await provider.chat_completion(
            messages=messages,
            tools=base_tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1,
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(f"[warmup] LLM call failed after {latency_ms}ms: {e}")
        return {
            "status": "error",
            "summary": "warmup ping failed",
            "error": f"{type(e).__name__}: {e}",
            "severity": "none",
            "signature_hash": None,
            "messages": [],
            "metrics": {"latency_ms": latency_ms},
        }
    latency_ms = int((time.perf_counter() - started) * 1000)

    cache_read = 0
    prompt_tokens = 0
    try:
        cache_stats = provider.pop_last_cache_stats()
        cache_read = int(cache_stats.get("read", 0) or 0)
    except AttributeError:
        pass
    try:
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    except AttributeError:
        pass

    # Latency is the reliable signal here. vLLM 0.19 doesn't populate
    # ``usage.prompt_tokens_details.cached_tokens`` on the OpenAI-compat
    # endpoint even though prefix caching is active — the per-request
    # field is reported as null. The wiring above stays in place so a
    # future vLLM upgrade (or the Anthropic provider) starts surfacing
    # real numbers without a code change. Until then: a 12k-token prefix
    # prefilled cold takes seconds; a warm hit lands in ~100 ms.
    hit_pct = (
        round(100.0 * cache_read / prompt_tokens, 1)
        if prompt_tokens > 0 else 0.0
    )
    if cache_read > 0:
        summary = (
            f"warmup ok ({latency_ms}ms, prefix cache "
            f"{cache_read}/{prompt_tokens} = {hit_pct}%)"
        )
    else:
        # Infer cold/warm from latency since vLLM didn't tell us.
        state = "warm" if latency_ms < 1000 else "cold"
        summary = (
            f"warmup ok ({latency_ms}ms, {prompt_tokens}-tok prefix, "
            f"inferred {state})"
        )
    logger.info(f"[warmup] {summary}")
    return {
        "status": "ok",
        "summary": summary,
        "severity": "none",
        "signature_hash": None,
        "messages": [],
        "metrics": {
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "cache_read_tokens": cache_read,
            "cache_creation_tokens": max(0, prompt_tokens - cache_read),
            "hit_pct": hit_pct,
        },
    }
