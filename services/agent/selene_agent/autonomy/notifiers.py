"""Notifier abstraction for the autonomy engine.

Notifiers are invoked directly from handlers after the LLM turn — they do
**not** flow through the LLM's tool-calling surface. This keeps tier gating
honest: the LLM cannot choose whether/where to notify.
"""
from __future__ import annotations

from typing import Any, List, Optional, Protocol

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.mcp_client_manager import MCPClientManager

logger = custom_logger.get_logger('loki')


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


class EmailNotifier:
    """Wraps the ``send_email`` MCP tool.

    Recipient resolution order:
      1. ``to`` explicitly passed to ``send()``
      2. ``config.AUTONOMY_BRIEFING_EMAIL_TO``
      3. Whatever ``DEFAULT_RECIPIENT`` env the general_tools MCP server reads

    ``send_email``'s MCP schema currently only accepts subject/body/attachments
    and uses ``DEFAULT_RECIPIENT`` env for the recipient. If the caller wants a
    different recipient, set ``DEFAULT_RECIPIENT`` on the agent container — or,
    when full per-notification ``to`` support lands, flip ``pass_to=True`` here.
    """

    def __init__(self, mcp_manager: MCPClientManager, *, default_to: str = ""):
        self.mcp_manager = mcp_manager
        self.default_to = default_to or config.AUTONOMY_BRIEFING_EMAIL_TO

    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
        to: Optional[str] = None,
    ) -> bool:
        payload = {"subject": title, "body": body}
        if attachments:
            payload["attachments"] = attachments
        try:
            result = await self.mcp_manager.execute_tool("send_email", payload)
            logger.info(f"[EmailNotifier] sent: {str(result)[:200]}")
            return True
        except Exception as e:
            logger.error(f"[EmailNotifier] failed: {e}")
            return False


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
            logger.info(f"[HAPushNotifier] sent: {str(result)[:200]}")
            return True
        except Exception as e:
            logger.error(f"[HAPushNotifier] failed: {e}")
            return False
