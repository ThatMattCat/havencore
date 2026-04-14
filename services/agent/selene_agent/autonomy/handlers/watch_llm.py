"""LLM-judged reactive triage handler.

Fires from an MQTT / webhook event (same trigger_spec matcher as ``watch``),
but where ``watch`` is deterministic ("if payload.state == 'open' notify"),
``watch_llm`` hands the event + a bounded state gather to the LLM and asks
for a single JSON judgment. Intended for cases where "unusual enough to
notify" can't be expressed as a simple condition.

Config::

    {
      "subject": "front-door-after-midnight",
      "gather": {
          "entities": ["binary_sensor.front_door", "person.matt"],
          "memories_k": 3
      },
      "notify": {
          "channel": "signal" | "ha_push" | "speaker",
          "to": "...",
          "device": "...",      # speaker only
          "voice": "...",       # speaker only
          "volume": 0.5         # speaker only
      },
      "severity_floor": "low" | "med" | "high",
      "cooldown_min": 15
    }

Returns the same ``_unusual`` / ``_notify_*`` / ``signature_hash`` shape as
``anomaly_sweep`` so the engine's shared cooldown + notification path picks
it up without special-casing.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime
from typing import Any, Dict, List

from selene_agent.autonomy.handlers.anomaly import _extract_json, _VALID_SEVERITY
from selene_agent.autonomy.turn import AutonomousTurn
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


WATCH_LLM_SYSTEM_PROMPT = (
    "You are {agent_name}'s autonomous event triage. You will receive an "
    "inbound event plus a short state gather of relevant entities and "
    "memories. Decide whether this event warrants notifying the resident. "
    "You are running autonomously. Do not ask questions. Do not call tools. "
    "Respond with ONE JSON object and NOTHING else — no prose, no code "
    "fences. Schema: "
    "{{"
    "\"unusual\": boolean, "
    "\"severity\": \"low\"|\"med\"|\"high\" (or \"none\" when unusual=false), "
    "\"summary\": string (<=120 chars, terse; '' when nominal), "
    "\"signature\": string (stable slug for dedup; 'nominal' when unusual=false), "
    "\"evidence\": array of up to 3 short bullet strings"
    "}}. "
    "Be conservative. Routine events that match the expected pattern "
    "should be marked unusual=false."
)


_SEVERITY_RANK = {"none": 0, "low": 1, "med": 2, "high": 3}


async def _safe_tool(mcp: MCPClientManager, name: str, args: Dict[str, Any]) -> Any:
    try:
        return await mcp.execute_tool(name, args)
    except Exception as e:
        logger.warning(f"[watch_llm] tool {name} failed: {e}")
        return f"<tool {name} failed: {e}>"


async def _gather(
    mcp: MCPClientManager, item_config: Dict[str, Any]
) -> Dict[str, Any]:
    gather_cfg = item_config.get("gather") or {}
    entities: List[str] = [e for e in (gather_cfg.get("entities") or []) if e]
    memories_k = int(gather_cfg.get("memories_k") or 3)
    subject = str(item_config.get("subject") or "event context").strip()

    calls: Dict[str, Any] = {}
    for ent in entities:
        # Prefer a bounded history window when available, else fall back to
        # a single current-state read via get_domain_entity_states.
        domain = ent.split(".", 1)[0] if "." in ent else ""
        if domain:
            calls[f"state_{ent}"] = _safe_tool(
                mcp,
                "ha_get_entity_history",
                {"entity_id": ent, "hours": 6, "limit": 20},
            )
    if memories_k > 0:
        calls["memories"] = _safe_tool(
            mcp, "search_memories", {"query": subject, "limit": memories_k}
        )

    if not calls:
        return {}
    results = await asyncio.gather(*calls.values(), return_exceptions=False)
    return dict(zip(calls.keys(), results))


def _render_user_prompt(
    subject: str, event: Dict[str, Any], state: Dict[str, Any]
) -> str:
    from zoneinfo import ZoneInfo

    tz = config.CURRENT_TIMEZONE or "UTC"
    local = datetime.now(ZoneInfo(tz))
    lines = [
        f"Subject: {subject or '(none)'}",
        f"Current local time: {local.strftime('%A %Y-%m-%d %H:%M %Z')}",
        "",
        "## Inbound event",
        f"source={event.get('source')!r} topic={event.get('topic')!r} "
        f"name={event.get('name')!r}",
        f"payload={event.get('payload')!r}",
    ]
    for key, value in state.items():
        if key == "memories":
            lines += ["", "## Relevant memories", str(value)]
        else:
            lines += ["", f"## {key}", str(value)]
    lines += ["", "Output the JSON object only. No other text."]
    return "\n".join(lines)


def _signature_hash(item_id: str, sig: str) -> str:
    raw = f"{item_id}:{sig}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def handle(
    item: Dict[str, Any],
    *,
    client,
    mcp_manager: MCPClientManager,
    model_name: str,
    base_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = item.get("config") or {}
    event = item.get("_trigger_event") or {}
    subject = str(cfg.get("subject") or item.get("name") or "watch_llm").strip()
    severity_floor = str(cfg.get("severity_floor") or "low").lower()
    if severity_floor not in _SEVERITY_RANK:
        severity_floor = "low"

    state = await _gather(mcp_manager, cfg)
    user_prompt = _render_user_prompt(subject, event, state)

    try:
        turn = AutonomousTurn(
            client=client,
            mcp_manager=mcp_manager,
            model_name=model_name,
            base_tools=base_tools,
            autonomy_level=item.get("autonomy_level", "notify"),
            system_prompt=WATCH_LLM_SYSTEM_PROMPT.format(
                agent_name=config.AGENT_NAME or "Selene"
            ),
            timeout_sec=int(cfg.get("timeout_sec") or config.AUTONOMY_TURN_TIMEOUT_SEC),
            temperature=0.2,
            max_tokens=400,
        )
    except ValueError as e:
        return {
            "status": "error",
            "summary": "tier invalid for watch_llm",
            "error": str(e),
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    result = await turn.run(user_prompt)
    parsed = _extract_json(result.content)
    if parsed is None:
        logger.warning(
            f"[watch_llm] LLM did not produce valid JSON for {item.get('id')}"
        )
        return {
            "status": "error",
            "summary": "watch_llm: invalid JSON output",
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
    sig_hash = _signature_hash(item["id"], signature)

    if not unusual:
        return {
            "status": "ok",
            "summary": "nominal",
            "severity": "none",
            "signature_hash": _signature_hash(item["id"], "nominal"),
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": None,
            "_unusual": False,
        }

    # Severity floor gate.
    if _SEVERITY_RANK[severity] < _SEVERITY_RANK[severity_floor]:
        return {
            "status": "ok",
            "summary": f"below severity floor ({severity} < {severity_floor})",
            "severity": severity,
            "signature_hash": sig_hash,
            "notified_via": None,
            "messages": result.messages,
            "metrics": result.metrics,
            "error": None,
            "_unusual": False,
        }

    evidence = parsed.get("evidence") or []
    body_lines = [summary_text or "Unusual event detected."]
    if evidence:
        body_lines.append("")
        for e in evidence[:3]:
            body_lines.append(f"- {str(e)[:200]}")

    notify = cfg.get("notify") or {}
    channel = notify.get("channel") or "signal"
    to = notify.get("to") or ""
    return {
        "status": "ok",
        "summary": (summary_text or "unusual event")[:200],
        "severity": severity,
        "signature_hash": sig_hash,
        "signature_raw": signature,
        "notified_via": None,
        "messages": result.messages,
        "metrics": result.metrics,
        "error": None,
        "_unusual": True,
        "_notify_title": f"{config.AGENT_NAME or 'Selene'}: {subject}",
        "_notify_body": "\n".join(body_lines)[:1000],
        "_notify_channel": channel,
        "_notify_to": to,
        # Pass speaker sub-config through so the engine's _build_notifier
        # can hand device/voice/volume into SpeakerNotifier.
        "_notify_cfg": {
            "speaker": {
                "device": notify.get("device"),
                "voice": notify.get("voice"),
                "volume": notify.get("volume"),
            }
        },
    }
