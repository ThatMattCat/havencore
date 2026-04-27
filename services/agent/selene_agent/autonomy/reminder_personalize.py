"""LLM rewrite step for reminder bodies.

A lightweight one-shot LLM call modeled on
``autonomy/memory_clustering.py:summarize_cluster`` — direct
``client.chat.completions.create()``, no AutonomousTurn, no tool gating.

The handler at fire time hands us the user-supplied title and body plus
the delivery channel, and we return ``{"body": str, "image_prompt":
str | None}``. ``body`` is always a non-empty string (falls back to the
original body on any failure). ``image_prompt`` is only suggested when
``channel == "signal"`` — for other channels the system prompt asks the
model to leave it null. The handler is responsible for actually
generating the image and attaching it.

On any error (timeout, JSON parse failure, empty body), we return the
original body and ``image_prompt=None`` so the handler can fall through
to deterministic delivery.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _build_system_prompt(channel: str) -> str:
    agent_name = config.AGENT_NAME or "Selene"
    image_clause = (
        "If a simple illustrative image would make the reminder more delightful, "
        "include a short image_prompt (3-12 words) describing what to render. "
        "Otherwise leave image_prompt null."
        if channel == "signal"
        else "Leave image_prompt null — this reminder is not delivered as a Signal message."
    )
    return (
        f"You are {agent_name}, a warm but concise home assistant. "
        f"Rewrite the user's reminder for them in your own voice — 1 to 2 short "
        f"sentences, no emojis, no markdown, no preamble. Keep all the "
        f"information from the original; do not invent details that weren't given. "
        f"{image_clause} "
        f'Respond with ONE JSON object and nothing else: '
        f'{{"body": "...", "image_prompt": "..." | null}}. '
        f"No prose, no code fences."
    )


def _user_prompt(title: str, body: str, channel: str) -> str:
    return (
        f"Channel: {channel}\n"
        f"Title: {title}\n"
        f"Original body: {body}\n\n"
        f"Output the JSON object only."
    )


async def personalize_reminder(
    *,
    client,
    model_name: str,
    title: str,
    body: str,
    channel: str,
    timeout_sec: float = 10.0,
    max_tokens: int = 250,
    temperature: float = 0.5,
) -> Dict[str, Any]:
    """Rewrite the reminder body in the agent's voice.

    Returns ``{"body": str, "image_prompt": str | None}``. ``body`` is
    guaranteed non-empty; on any failure it is the original ``body``
    argument and ``image_prompt`` is None.
    """
    fallback: Dict[str, Any] = {"body": body, "image_prompt": None}

    if not body and not title:
        return fallback

    system = _build_system_prompt(channel)
    user = _user_prompt(title, body, channel)

    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[reminder-personalize] LLM call timed out after {timeout_sec}s")
        return fallback
    except Exception as e:
        logger.warning(f"[reminder-personalize] LLM call failed: {e}")
        return fallback

    try:
        content = resp.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return fallback

    parsed = _extract_json(content)
    if parsed is None:
        logger.warning(f"[reminder-personalize] could not parse JSON from LLM output: {content[:200]!r}")
        return fallback

    new_body = parsed.get("body")
    if not isinstance(new_body, str) or not new_body.strip():
        return fallback

    image_prompt = parsed.get("image_prompt")
    if image_prompt is not None:
        if not isinstance(image_prompt, str) or not image_prompt.strip():
            image_prompt = None
        elif channel != "signal":
            # Defense in depth — the system prompt already says null for non-signal,
            # but never let a stray suggestion slip through to a channel that can't render it.
            image_prompt = None
        else:
            image_prompt = image_prompt.strip()[:240]

    return {"body": new_body.strip(), "image_prompt": image_prompt}
