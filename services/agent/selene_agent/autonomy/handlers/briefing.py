"""Morning briefing handler.

Deterministic gather → single LLM summarize call → email notifier.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from selene_agent.autonomy.notifiers import EmailNotifier
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


BRIEFING_SYSTEM_PROMPT = (
    "You are {agent_name}, generating a short morning briefing email for your "
    "home's primary resident. Write in warm, plain text — no markdown, no "
    "emojis. Structure the email as one concise opening paragraph followed by "
    "2-6 terse bullet lines (prefix each with '- '). Cover only what's useful: "
    "today/tomorrow calendar events, notable weather, anything unusual from the "
    "overnight history. If a category has nothing to say, skip it. Do not ask "
    "questions. Do not call tools — all needed context is in the user message. "
    "End with a single blank line."
)


async def _safe_tool(mcp: MCPClientManager, name: str, args: Dict[str, Any]) -> Any:
    try:
        return await mcp.execute_tool(name, args)
    except Exception as e:
        logger.warning(f"[briefing] tool {name} failed: {e}")
        return f"<tool {name} failed: {e}>"


async def _gather(mcp: MCPClientManager, item_config: Dict[str, Any]) -> Dict[str, Any]:
    window_hours = int(item_config.get("window_hours", 10))
    start = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()

    # History entities: use configured camera entities + any watched binary_sensors
    # the operator may have named. If nothing is configured, this block is empty
    # and the briefing will note "no overnight activity data configured."
    history_entities: List[str] = list(item_config.get("camera_entities", []) or [])

    calls = {
        "calendar": _safe_tool(mcp, "ha_get_calendar_events", {"days_ahead": 2}),
        "weather": _safe_tool(mcp, "get_weather_forecast", {"location": config.CURRENT_LOCATION}),
    }
    if history_entities:
        calls["history"] = _safe_tool(
            mcp,
            "ha_get_entity_history",
            {"entity_ids": history_entities, "start_time": start},
        )

    results = await asyncio.gather(*calls.values(), return_exceptions=False)
    return dict(zip(calls.keys(), results))


def _render_user_prompt(state: Dict[str, Any]) -> str:
    lines = [
        "Compose today's morning briefing. Here is the state bundle:",
        "",
        "## Calendar (today + tomorrow)",
        str(state.get("calendar", "(unavailable)")),
        "",
        "## Weather",
        str(state.get("weather", "(unavailable)")),
    ]
    if "history" in state:
        lines += [
            "",
            "## Overnight history (configured entities)",
            str(state.get("history", "(unavailable)")),
        ]
    return "\n".join(lines)


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager: MCPClientManager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    item_config = item.get("config") or {}
    state = await _gather(mcp_manager, item_config)
    user_prompt = _render_user_prompt(state)

    turn = AutonomousTurn(
        client=client,
        mcp_manager=mcp_manager,
        model_name=model_name,
        base_tools=base_tools,
        autonomy_level=item.get("autonomy_level", "notify"),
        system_prompt=BRIEFING_SYSTEM_PROMPT.format(agent_name=config.AGENT_NAME or "Selene"),
        timeout_sec=config.AUTONOMY_TURN_TIMEOUT_SEC,
        temperature=0.5,
        max_tokens=800,
    )
    result = await turn.run(user_prompt)

    if result.status != "ok" or not result.content:
        return {
            "status": "error" if result.status != "ok" else "error",
            "summary": "briefing generation failed",
            "error": result.error or "empty LLM output",
            "messages": result.messages,
            "metrics": result.metrics,
            "notified_via": None,
            "severity": "none",
            "signature_hash": None,
        }

    notifier = EmailNotifier(mcp_manager)
    title = f"{config.AGENT_NAME or 'Selene'}: morning briefing"
    delivered = await notifier.send(title=title, body=result.content)

    return {
        "status": "ok" if delivered else "error",
        "summary": result.content.splitlines()[0][:200] if result.content else "briefing",
        "severity": "none",
        "signature_hash": "briefing:" + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "notified_via": "email" if delivered else None,
        "messages": result.messages,
        "metrics": result.metrics,
        "error": None if delivered else "email send failed",
    }
