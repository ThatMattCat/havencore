"""Ambient anomaly-sweep handler.

Deterministic state gather + memory context → single LLM judgment call
producing strict JSON → cooldown-aware HA push notification.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from selene_agent.autonomy.notifiers import HAPushNotifier
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


ANOMALY_SYSTEM_PROMPT = (
    "You are {agent_name}'s autonomous anomaly monitor. You will receive a "
    "state snapshot of the home and must decide whether anything is unusual "
    "enough to warrant notifying the resident. You are running autonomously. "
    "Do not ask questions. Do not call tools. Complete your assessment and "
    "exit. Respond with ONE JSON object and NOTHING else — no prose, no code "
    "fences, no preamble. Schema: "
    "{{"
    "\"unusual\": boolean, "
    "\"severity\": \"low\"|\"med\"|\"high\" (or \"none\" if unusual=false), "
    "\"summary\": string (<=120 chars, terse description; '' when nominal), "
    "\"signature\": string (stable slug like 'garage_open_gt_10min' for dedup; "
    "'nominal' when unusual=false), "
    "\"evidence\": array of up to 3 short bullet strings, "
    "\"suggested_action\": string or null (<=80 chars)"
    "}}. "
    "Be conservative: routine lights-on-at-night, normal presence patterns, "
    "and expected calendar events are NOT unusual. Trust prior-context "
    "memories. Prefer no notification over a noisy one."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_SEVERITY = {"none", "low", "med", "high"}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Direct parse first.
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Fall back to first {...} block.
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


async def _safe_tool(mcp: MCPClientManager, name: str, args: Dict[str, Any]) -> Any:
    try:
        return await mcp.execute_tool(name, args)
    except Exception as e:
        logger.warning(f"[anomaly] tool {name} failed: {e}")
        return f"<tool {name} failed: {e}>"


async def _gather(mcp: MCPClientManager, item_config: Dict[str, Any]) -> Dict[str, Any]:
    watch_domains: List[str] = list(item_config.get("watch_domains") or [])
    calls: Dict[str, Any] = {
        "presence": _safe_tool(mcp, "ha_get_presence", {}),
        "calendar": _safe_tool(mcp, "ha_get_calendar_events", {"days_ahead": 1}),
        "memories": _safe_tool(
            mcp,
            "search_memories",
            {"query": "household normal routine", "limit": 5},
        ),
    }
    for domain in watch_domains:
        calls[f"state_{domain}"] = _safe_tool(
            mcp, "ha_get_domain_entity_states", {"domain": domain}
        )
    # Check lights on at off-hours opportunistically; LLM sees local hour below.
    calls["state_light"] = _safe_tool(
        mcp, "ha_get_domain_entity_states", {"domain": "light"}
    )

    results = await asyncio.gather(*calls.values(), return_exceptions=False)
    return dict(zip(calls.keys(), results))


def _render_user_prompt(state: Dict[str, Any]) -> str:
    from zoneinfo import ZoneInfo

    local = datetime.now(ZoneInfo(config.CURRENT_TIMEZONE or "UTC"))
    lines = [
        f"Current local time: {local.strftime('%A %Y-%m-%d %H:%M %Z')}",
        "",
        "## Presence",
        str(state.get("presence", "(unavailable)")),
        "",
        "## Calendar (next 24h)",
        str(state.get("calendar", "(unavailable)")),
        "",
        "## Household-routine memories",
        str(state.get("memories", "(none)")),
    ]
    # Dump each state_<domain> block.
    for key, value in state.items():
        if key.startswith("state_"):
            lines += ["", f"## {key}", str(value)]

    lines += [
        "",
        "Output the JSON object only. No other text.",
    ]
    return "\n".join(lines)


def _signature_hash(sig: str) -> str:
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


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
        system_prompt=ANOMALY_SYSTEM_PROMPT.format(agent_name=config.AGENT_NAME or "Selene"),
        timeout_sec=config.AUTONOMY_TURN_TIMEOUT_SEC,
        temperature=0.2,
        max_tokens=400,
    )
    result = await turn.run(user_prompt)

    parsed = _extract_json(result.content)
    if parsed is None:
        logger.warning("[anomaly] LLM did not produce valid JSON; recording error run")
        return {
            "status": "error",
            "summary": "anomaly: invalid JSON output",
            "severity": None,
            "signature_hash": None,
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": result.error or f"unparseable: {result.content[:200]}",
        }

    unusual = bool(parsed.get("unusual"))
    severity = str(parsed.get("severity") or "none").lower()
    if severity not in _VALID_SEVERITY:
        severity = "none"
    summary_text = str(parsed.get("summary") or "").strip()
    signature = str(parsed.get("signature") or "").strip() or "unnamed"
    sig_hash = _signature_hash(signature)

    if not unusual:
        return {
            "status": "ok",
            "summary": "nominal",
            "severity": "none",
            "signature_hash": _signature_hash("nominal"),
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": None,
            # Internal: engine uses these to decide notification path
            "_unusual": False,
        }

    evidence = parsed.get("evidence") or []
    body_lines = [summary_text or "Unusual condition detected."]
    if evidence:
        body_lines.append("")
        for e in evidence[:3]:
            body_lines.append(f"- {str(e)[:200]}")
    suggested = parsed.get("suggested_action")
    if suggested:
        body_lines += ["", f"Suggested: {str(suggested)[:120]}"]

    return {
        "status": "ok",
        "summary": (summary_text or "unusual condition")[:200],
        "severity": severity,
        "signature_hash": sig_hash,
        "signature_raw": signature,
        "notified_via": None,  # filled in by engine after cooldown decision
        "messages": result.messages,
        "metrics": result.metrics,
        "error": None,
        "_unusual": True,
        "_notify_body": "\n".join(body_lines),
        "_notify_title": f"{config.AGENT_NAME or 'Selene'}: {severity}",
    }
