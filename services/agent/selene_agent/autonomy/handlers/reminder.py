"""Reminder handler — deterministic one-shot / recurring notify.

No LLM, no tools, no AutonomousTurn. Config shape::

    {
      "title": "Switch laundry to dryer",
      "body":  "...",           # optional; falls back to title
      "channel": "signal" | "ha_push",
      "to": "...",              # optional override (signal: phone number, ha_push: notify.xxx target)
      "one_shot": false,        # optional; disables the item after a successful fire
      "quiet_hours": {...}      # optional; handled by engine, not here
    }
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from selene_agent.autonomy import db as autonomy_db
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


def _make_notifier(
    channel: str,
    to: str,
    mcp_manager: MCPClientManager,
    cfg: Dict[str, Any] | None = None,
):
    cfg = cfg or {}
    if channel in ("signal", "email"):  # "email" retained as legacy alias for existing DB items
        return SignalNotifier(mcp_manager, default_to=to or config.AUTONOMY_BRIEFING_NOTIFY_TO)
    if channel == "ha_push":
        return HAPushNotifier(mcp_manager, target=to or config.AUTONOMY_HA_NOTIFY_TARGET)
    if channel == "speaker":
        return SpeakerNotifier(
            mcp_manager,
            device=cfg.get("device") or to or "",
            voice=cfg.get("voice") or "",
            volume=cfg.get("volume"),
        )
    return NullNotifier()


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
    title = str(cfg.get("title") or item.get("name") or "Reminder").strip()
    body = str(cfg.get("body") or title).strip()
    channel = cfg.get("channel") or "ha_push"
    to = cfg.get("to") or ""

    if not body:
        return {
            "status": "error",
            "summary": "reminder has no body",
            "error": "empty body",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    notifier = _make_notifier(channel, to, mcp_manager, cfg)
    delivered = await notifier.send(title=title, body=body)
    notified_via = channel if delivered else None

    if delivered and bool(cfg.get("one_shot")):
        try:
            await autonomy_db.update_item(item["id"], {"enabled": False})
        except Exception as e:
            logger.warning(f"[reminder] failed to disable one_shot item {item['id']}: {e}")

    return {
        "status": "ok" if delivered else "error",
        "summary": title[:200],
        "severity": "none",
        "signature_hash": f"reminder:{item['id']}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
        "notified_via": notified_via,
        "messages": [],
        "metrics": {"channel": channel, "delivered": delivered},
        "error": None if delivered else f"{channel} delivery failed",
    }
