"""Watch handler — deterministic event-triggered notify with optional condition.

Fires only from an event (MQTT or webhook). Config shape::

    {
      "title": "Front door opened at night",
      "body_template": "Door {state} at {_ts}",   # str.format_map over payload
      "channel": "signal" | "ha_push",
      "to": "...",                                # optional channel target
      "condition": {                              # optional — must hold at fire time
          "entity_id": "binary_sensor.front_door",
          "state": "on",                          # optional — required current state
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
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from selene_agent.autonomy.notifiers import (
    HAPushNotifier,
    NullNotifier,
    SignalNotifier,
    SpeakerNotifier,
)
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
) -> Tuple[bool, Optional[str]]:
    """Optional state/duration gate against Home Assistant.

    Requires ``entity_id`` to currently be in ``state`` (when given) and to
    have held it for at least ``min_duration_sec`` (when given).

    Returns ``(held, error)``. A non-None ``error`` means the gate could not be
    evaluated — the caller reports that as a failed run rather than a healthy
    "condition not held", which previously hid the breakage entirely.
    """
    entity_id = condition.get("entity_id")
    if not entity_id:
        return True, None
    try:
        # ha_get_state, not ha_get_entity_state — the latter has never existed,
        # so every conditioned watch item failed closed here.
        raw = await mcp_manager.execute_tool("ha_get_state", {"entity_id": entity_id})
    except Exception as e:
        logger.warning(f"[watch] condition tool call failed for {entity_id}: {e}")
        return False, f"condition lookup failed: {e}"

    # execute_tool always hands back JSON text, never a dict.
    if isinstance(raw, dict):
        result = raw
    else:
        try:
            result = json.loads(raw)
        except (TypeError, ValueError) as e:
            logger.warning(f"[watch] condition state unparseable for {entity_id}: {e}")
            return False, f"condition lookup returned unparseable state: {raw!r}"
    if not isinstance(result, dict):
        return False, f"condition lookup returned unexpected shape: {type(result).__name__}"
    if result.get("error"):
        return False, f"condition lookup failed: {result['error']}"

    want_state = condition.get("state")
    if want_state not in (None, ""):
        if str(result.get("state")) != str(want_state):
            return False, None

    min_sec = condition.get("min_duration_sec")
    if not min_sec:
        return True, None

    changed_at_str = result.get("last_changed") or result.get("last_updated") or ""
    if not changed_at_str:
        return False, f"entity {entity_id} reported no last_changed/last_updated"
    try:
        changed_at = datetime.fromisoformat(str(changed_at_str).replace("Z", "+00:00"))
    except ValueError:
        return False, f"unparseable last_changed for {entity_id}: {changed_at_str!r}"
    if changed_at.tzinfo is None:
        changed_at = changed_at.replace(tzinfo=timezone.utc)
    elapsed = datetime.now(timezone.utc) - changed_at
    return elapsed >= timedelta(seconds=int(min_sec)), None


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
    event = item.get("_trigger_event") or {}

    condition = cfg.get("condition")
    if isinstance(condition, dict):
        held, cond_error = await _condition_holds(condition, mcp_manager)
        if not held:
            return {
                # A gate we couldn't evaluate is a failed run, not a healthy
                # "nothing to report" — otherwise the dashboard shows only ok
                # runs while the item silently never notifies.
                "status": "error" if cond_error else "ok",
                "summary": cond_error or "condition not held",
                "severity": "none",
                "signature_hash": _signature_hash(item["id"], event),
                "notified_via": None,
                "messages": [],
                "metrics": {"condition_held": False},
                "error": cond_error,
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
