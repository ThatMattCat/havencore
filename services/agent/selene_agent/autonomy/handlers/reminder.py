"""Reminder handler — one-shot / recurring notify with optional LLM personalization.

Config shape::

    {
      "title": "Switch laundry to dryer",
      "body":  "...",           # optional; falls back to title
      "channel": "signal" | "ha_push" | "speaker",
      "to": "...",              # optional override (signal: phone number, ha_push: notify.xxx target, speaker: MA device)
      "one_shot": false,        # optional; engine deletes the agenda row after a successful fire
      "personalize": true,      # optional; default True — LLM rewrites the body in the agent's voice at fire time
      "quiet_hours": {...}      # optional; handled by engine, not here
    }

When ``personalize`` is true (the default), the handler runs a small
LLM rewrite via :func:`autonomy.reminder_personalize.personalize_reminder`
before delivery. For ``channel == "signal"`` the rewrite step may also
suggest a short ``image_prompt``; if it does, the handler invokes the
``generate_image`` MCP tool and attaches the resulting image to the
Signal message. Any failure in the personalization or image-gen steps
falls back gracefully to deterministic delivery.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from selene_agent.autonomy import db as autonomy_db
from selene_agent.autonomy.notifiers import (
    HAPushNotifier,
    NullNotifier,
    SignalNotifier,
    SpeakerNotifier,
)
from selene_agent.autonomy.reminder_personalize import personalize_reminder
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


IMAGE_GEN_TIMEOUT_SEC = 25.0


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


async def _generate_signal_image(
    mcp_manager: MCPClientManager,
    *,
    image_prompt: str,
    item_id: str,
) -> Optional[str]:
    """Call generate_image and return a local file path, or None on failure.

    Bounded by ``IMAGE_GEN_TIMEOUT_SEC`` so a stalled ComfyUI doesn't hold
    up reminder delivery indefinitely. ``mcp_manager.execute_tool`` may
    return either a string (typical) or an already-decoded dict; both shapes
    are handled.
    """
    try:
        raw = await asyncio.wait_for(
            mcp_manager.execute_tool("generate_image", {"prompt": image_prompt}),
            timeout=IMAGE_GEN_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[reminder] image gen timed out after {IMAGE_GEN_TIMEOUT_SEC}s for {item_id}"
        )
        return None
    except Exception as e:
        logger.warning(f"[reminder] generate_image tool call failed for {item_id}: {e}")
        return None

    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            logger.warning(
                f"[reminder] could not parse generate_image output for {item_id}: {raw[:200]!r}"
            )
            return None

    if not isinstance(data, dict):
        return None
    images = data.get("images") or []
    for img in images:
        if isinstance(img, dict):
            path = img.get("path")
            if isinstance(path, str) and path.strip():
                return path
    return None


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
    # Default to True for any item missing the field — including pre-existing
    # rows scheduled before personalization landed.
    personalize = bool(cfg.get("personalize", True))

    if not body:
        return {
            "status": "error",
            "summary": "reminder has no body",
            "error": "empty body",
            "messages": [],
            "metrics": {},
            "notified_via": None,
        }

    personalized = False
    image_attached = False
    attachments: Optional[List[str]] = None

    if personalize and client is not None and model_name:
        rewrite = await personalize_reminder(
            client=client,
            model_name=model_name,
            title=title,
            body=body,
            channel=channel,
        )
        if rewrite["body"] and rewrite["body"] != body:
            body = rewrite["body"]
            personalized = True
        image_prompt = rewrite.get("image_prompt")
        if image_prompt and channel == "signal":
            path = await _generate_signal_image(
                mcp_manager,
                image_prompt=image_prompt,
                item_id=item["id"],
            )
            if path:
                attachments = [path]
                image_attached = True

    notifier = _make_notifier(channel, to, mcp_manager, cfg)
    # When the user (or the LLM scheduling on their behalf) didn't supply a
    # distinct body, the MCP tool defaults body to title — both fields hold
    # the same string. SignalNotifier and SpeakerNotifier concatenate title
    # and body, which would render the reminder twice in one message / TTS
    # synthesis. Drop the title in that case so the text is rendered once.
    # HAPushNotifier uses title as a separate header field; an empty header
    # is harmless. (If personalization rewrote the body, title still differs
    # and we keep both.)
    notify_title = title if title.strip() != body.strip() else ""
    send_kwargs: Dict[str, Any] = {"title": notify_title, "body": body}
    if attachments and channel == "signal":
        send_kwargs["attachments"] = attachments
    delivered = await notifier.send(**send_kwargs)
    notified_via = channel if delivered else None

    # One-shot reminders that fired successfully are removed from agenda_items
    # so they don't accumulate in the dashboard's Agenda list. The audit trail
    # lives in autonomy_runs (FK is ON DELETE SET NULL). The engine performs
    # the actual delete after insert_run so the FK reference is valid.
    delete_after_run = delivered and bool(cfg.get("one_shot"))

    return {
        "status": "ok" if delivered else "error",
        "summary": title[:200],
        "severity": "none",
        "signature_hash": f"reminder:{item['id']}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
        "notified_via": notified_via,
        "messages": [],
        "metrics": {
            "channel": channel,
            "delivered": delivered,
            "personalized": personalized,
            "image_attached": image_attached,
        },
        "error": None if delivered else f"{channel} delivery failed",
        "_delete_after_run": delete_after_run,
    }
