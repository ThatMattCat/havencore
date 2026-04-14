"""Notifier abstraction for the autonomy engine.

Notifiers are invoked directly from handlers after the LLM turn — they do
**not** flow through the LLM's tool-calling surface. This keeps tier gating
honest: the LLM cannot choose whether/where to notify.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Protocol, Tuple

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


def _tool_result_ok(result: Any) -> Tuple[bool, str]:
    """Inspect an MCP tool result for a ``success: false`` envelope.

    MCP tools return strings; our general_tools / HA tools wrap replies as
    ``{"success": bool, ...}`` JSON. A non-raising call does not imply delivery
    — SMTP/HA failures come back as ``success: false`` payloads.
    Returns ``(ok, detail)`` where ``detail`` is the tool's error message
    when ``ok`` is False, or a short trace of the payload otherwise.
    """
    if result is None:
        return False, "empty result"
    if isinstance(result, dict):
        payload = result
    else:
        text = str(result)
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            return True, text[:200]
        if not isinstance(payload, dict):
            return True, text[:200]
    if payload.get("success") is False:
        return False, str(payload.get("error") or payload)[:300]
    return True, str(payload)[:200]


class Notifier(Protocol):
    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
    ) -> bool: ...


class NullNotifier:
    """No-op — used for observe-tier runs and tests."""

    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
    ) -> bool:
        logger.info(f"[NullNotifier] title={title!r} severity={severity} body_len={len(body)}")
        return True


class SignalNotifier:
    """Wraps the ``send_signal_message`` MCP tool.

    Recipient resolution order:
      1. ``to`` explicitly passed to ``send()``
      2. ``default_to`` supplied at construction
      3. ``config.AUTONOMY_BRIEFING_NOTIFY_TO``
      4. Whatever ``SIGNAL_DEFAULT_RECIPIENT`` env the general_tools MCP server reads
         (which itself falls back to ``SIGNAL_PHONE_NUMBER`` — i.e. Note to Self)

    The ``title`` argument is prepended to the body as a first line, since
    Signal messages have no subject field.
    """

    def __init__(self, mcp_manager: MCPClientManager, *, default_to: str = ""):
        self.mcp_manager = mcp_manager
        self.default_to = default_to or config.AUTONOMY_BRIEFING_NOTIFY_TO

    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
        to: Optional[str] = None,
    ) -> bool:
        message = f"{title}\n\n{body}" if title and body else (title or body or "")
        payload: dict = {"message": message}
        if attachments:
            payload["attachments"] = attachments
        recipient = to or self.default_to
        if recipient:
            payload["to"] = recipient
        try:
            result = await self.mcp_manager.execute_tool("send_signal_message", payload)
        except Exception as e:
            logger.error(f"[SignalNotifier] failed: {e}")
            return False
        ok, detail = _tool_result_ok(result)
        if ok:
            logger.info(f"[SignalNotifier] sent: {detail}")
        else:
            logger.error(f"[SignalNotifier] tool reported failure: {detail}")
        return ok


class HAPushNotifier:
    """Wraps the ``ha_send_notification`` MCP tool."""

    def __init__(self, mcp_manager: MCPClientManager, *, target: str = ""):
        self.mcp_manager = mcp_manager
        self.target = target or config.AUTONOMY_HA_NOTIFY_TARGET

    def _service_name(self) -> str:
        """Strip a leading ``notify.`` from the configured target."""
        t = self.target or ""
        return t[len("notify."):] if t.startswith("notify.") else t

    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
    ) -> bool:
        service = self._service_name()
        if not service:
            logger.warning("[HAPushNotifier] no AUTONOMY_HA_NOTIFY_TARGET configured; dropping notification")
            return False
        payload = {
            "service": service,
            "message": body,
            "title": title,
        }
        try:
            result = await self.mcp_manager.execute_tool("ha_send_notification", payload)
        except Exception as e:
            logger.error(f"[HAPushNotifier] failed: {e}")
            return False
        ok, detail = _tool_result_ok(result)
        if ok:
            logger.info(f"[HAPushNotifier] sent: {detail}")
        else:
            logger.error(f"[HAPushNotifier] tool reported failure: {detail}")
        return ok
