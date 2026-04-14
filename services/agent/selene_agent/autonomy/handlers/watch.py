"""Watch handler — deterministic event-triggered notify with optional condition.

Fires only from an event (MQTT or webhook). Config shape::

    {
      "title": "Front door opened at night",
      "body_template": "Door {state} at {_ts}",   # str.format_map over payload
      "channel": "signal" | "ha_push",
      "to": "...",                                # optional channel target
      "condition": {                              # optional — must hold at fire time
          "entity_id": "binary_sensor.front_door",
          "min_duration_sec": 60
      },
      "cooldown_min": 30,                         # reuses v1 signature cooldown
      "quiet_hours": {...}
    }

Returns ``signature_hash`` so the engine's v1 cooldown path can dedupe
per-item fires within the cooldown window.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from selene_agent.autonomy.notifiers import SignalNotifier, HAPushNotifier, NullNotifier
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


def _signature_hash(item_id: str, event: Dict[str, Any]) -> str:
    topic_or_name = event.get("topic") or event.get("name") or ""
    raw = f"{item_id}:{topic_or_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _make_notifier(channel: str, to: str, mcp_manager: MCPClientManager):
    if channel in ("signal", "email"):
        return SignalNotifier(mcp_manager, default_to=to or config.AUTONOMY_BRIEFING_NOTIFY_TO)
    if channel == "ha_push":
        return HAPushNotifier(mcp_manager, target=to or config.AUTONOMY_HA_NOTIFY_TARGET)
    return NullNotifier()


def _flatten_for_format(event: Dict[str, Any]) -> Dict[str, Any]:
    """Merge payload keys + event meta so body_template sees a flat namespace."""
    flat: Dict[str, Any] = {}
    payload = event.get("payload")
    if isinstance(payload, dict):
        flat.update(payload)
    flat["_topic"] = event.get("topic")
    flat["_name"] = event.get("name")
    flat["_source"] = event.get("source")
    flat["_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return flat


async def _condition_holds(
    condition: Dict[str, Any], mcp_manager: MCPClientManager
) -> bool:
    """Optional duration gate against Home Assistant state.

    If ``min_duration_sec`` is set, require ``entity_id`` to have been in its
    current state at least that long. Absent HA tool calls fail closed (False).
    """
    entity_id = condition.get("entity_id")
    if not entity_id:
        return True
    try:
        result = await mcp_manager.execute_tool(
            "ha_get_entity_state", {"entity_id": entity_id}
        )
    except Exception as e:
        logger.warning(f"[watch] condition tool call failed for {entity_id}: {e}")
        return False

    min_sec = condition.get("min_duration_sec")
    if not min_sec:
        return True

    changed_at_str: str = ""
    if isinstance(result, dict):
        changed_at_str = result.get("last_changed") or result.get("last_updated") or ""
    if not changed_at_str:
        return False
    try:
        changed_at = datetime.fromisoformat(changed_at_str.replace("Z", "+00:00"))
    except ValueError:
        return False
    if changed_at.tzinfo is None:
        changed_at = changed_at.replace(tzinfo=timezone.utc)
    elapsed = datetime.now(timezone.utc) - changed_at
    return elapsed >= timedelta(seconds=int(min_sec))


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

    condition = cfg.get("condition")
    if isinstance(condition, dict):
        if not await _condition_holds(condition, mcp_manager):
            return {
                "status": "ok",
                "summary": "condition not held",
                "severity": "none",
                "signature_hash": _signature_hash(item["id"], event),
                "notified_via": None,
                "messages": [],
                "metrics": {"condition_held": False},
                "error": None,
                "_unusual": False,
            }

    title = str(cfg.get("title") or item.get("name") or "Watch").strip()
    template = cfg.get("body_template") or title
    try:
        body = template.format_map(_flatten_for_format(event))
    except Exception as e:
        logger.warning(f"[watch] body_template render failed: {e}")
        body = title

    channel = cfg.get("channel") or "ha_push"
    to = cfg.get("to") or ""
    sig_hash = _signature_hash(item["id"], event)

    # Engine applies cooldown against this signature; return _unusual to opt
    # into the anomaly-style notification path (HA push with severity).
    return {
        "status": "ok",
        "summary": title[:200],
        "severity": cfg.get("severity") or "low",
        "signature_hash": sig_hash,
        "notified_via": None,  # stamped by engine after cooldown/notifier path
        "messages": [],
        "metrics": {"channel": channel, "trigger": event.get("source")},
        "error": None,
        "_unusual": True,
        "_notify_title": title,
        "_notify_body": body[:1000],
        "_notify_channel": channel,
        "_notify_to": to,
    }
