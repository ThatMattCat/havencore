"""Routine handler — custom-prompt agentic turn + delivery.

Config shape::

    {
      "prompt": "Summarise this week's energy usage and weather.",
      "tools_override": ["get_weather_forecast", "ha_get_entity_history"],
      "deliver": {"channel": "signal"|"ha_push", "to": "..."},
      "system_prompt": "...",     # optional override
      "temperature": 0.4,         # optional
      "max_tokens": 800,          # optional
      "quiet_hours": {...}
    }

Tools_override is validated against the item's ``autonomy_level`` — any name
outside the tier allow-list raises ``ValueError`` (bubbled as a run error).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from selene_agent.autonomy.notifiers import (
    HAPushNotifier,
    NullNotifier,
    SignalNotifier,
    SpeakerNotifier,
)
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


DEFAULT_ROUTINE_SYSTEM_PROMPT = (
    "You are {agent_name}, running an autonomous routine on behalf of the "
    "primary resident. Use the tools provided to gather whatever context you "
    "need, then produce a concise plain-text answer. No markdown, no emojis. "
    "If the routine cannot be completed, say so briefly."
)


def _make_notifier(
    channel: str,
    to: str,
    mcp_manager: MCPClientManager,
    deliver_cfg: Dict[str, Any] | None = None,
):
    deliver_cfg = deliver_cfg or {}
    if channel in ("signal", "email"):
        return SignalNotifier(mcp_manager, default_to=to or config.AUTONOMY_BRIEFING_NOTIFY_TO)
    if channel == "ha_push":
        return HAPushNotifier(mcp_manager, target=to or config.AUTONOMY_HA_NOTIFY_TARGET)
    if channel == "speaker":
        return SpeakerNotifier(
            mcp_manager,
            device=deliver_cfg.get("device") or to or "",
            voice=deliver_cfg.get("voice") or "",
            volume=deliver_cfg.get("volume"),
        )
    return NullNotifier()


def _signature(item_id: str) -> str:
    raw = f"routine:{item_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager: MCPClientManager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
    provider_getter=None,
) -> Dict[str, Any]:
    cfg = item.get("config") or {}
    prompt = str(cfg.get("prompt") or "").strip()
    if not prompt:
        return {
            "status": "error",
            "summary": "routine has no prompt",
            "error": "empty prompt",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    system_prompt = cfg.get("system_prompt") or DEFAULT_ROUTINE_SYSTEM_PROMPT.format(
        agent_name=config.AGENT_NAME or "Selene"
    )
    tools_override = cfg.get("tools_override")

    try:
        turn = AutonomousTurn(
            client=client,
            mcp_manager=mcp_manager,
            model_name=model_name,
            base_tools=base_tools,
            autonomy_level=item.get("autonomy_level", "notify"),
            system_prompt=system_prompt,
            timeout_sec=int(cfg.get("timeout_sec") or config.AUTONOMY_TURN_TIMEOUT_SEC),
            temperature=float(cfg.get("temperature", 0.4)),
            max_tokens=int(cfg.get("max_tokens", 800)),
            tools_override=tools_override,
            provider_getter=provider_getter,
        )
    except ValueError as e:
        return {
            "status": "error",
            "summary": "tools_override invalid",
            "error": str(e),
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    result = await turn.run(prompt)

    if result.status != "ok" or not result.content:
        return {
            "status": "error",
            "summary": "routine generation failed",
            "error": result.error or "empty LLM output",
            "messages": result.messages,
            "metrics": result.metrics,
            "notified_via": None,
            "severity": "none",
            "signature_hash": _signature(item["id"]),
        }

    deliver = cfg.get("deliver") or {}
    channel = deliver.get("channel") or "signal"
    to = deliver.get("to") or ""
    notifier = _make_notifier(channel, to, mcp_manager, deliver)
    title = str(cfg.get("title") or item.get("name") or f"Routine: {item['kind']}").strip()
    delivered = await notifier.send(title=title, body=result.content)

    return {
        "status": "ok" if delivered else "error",
        "summary": (result.content.splitlines()[0] if result.content else title)[:200],
        "severity": "none",
        "signature_hash": _signature(item["id"]),
        "notified_via": channel if delivered else None,
        "messages": result.messages,
        "metrics": result.metrics,
        "error": None if delivered else f"{channel} delivery failed",
    }
