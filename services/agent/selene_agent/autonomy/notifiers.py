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
    """Inspect an MCP tool result for delivery failure.

    MCP tools return strings (or sometimes dicts). Our general_tools and HA
    tools mostly wrap replies as ``{"success": bool, ...}`` JSON, but the HA
    tools occasionally return a bare ``"Error sending notification ..."``
    string when the upstream service raises — those need to surface as a
    failure too, otherwise the notifier records ``notified_via=ha_push`` for
    a delivery that never happened.

    Returns ``(ok, detail)`` where ``detail`` is the tool's error message
    when ``ok`` is False, or a short trace of the payload otherwise.
    """
    if result is None:
        return False, "empty result"
    if isinstance(result, dict):
        payload = result
    else:
        text = str(result)
        # Plain-string error envelopes from HA tools (not JSON).
        stripped = text.strip()
        if stripped.lower().startswith("error") or "error sending notification" in stripped.lower():
            return False, stripped[:300]
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


class SpeakerNotifier:
    """Render body via TTS, stage the audio under ``/api/tts/audio/{token}.mp3``,
    then call ``mass_play_announcement`` on a Music Assistant player.

    The notifier is "delivery" — same surface as SignalNotifier/HAPushNotifier —
    so any handler that selects ``channel="speaker"`` gets spoken playback
    instead of a push/text message.
    """

    def __init__(
        self,
        mcp_manager: MCPClientManager,
        *,
        device: str = "",
        voice: str = "",
        volume: Optional[float] = None,
        tts_client=None,
        audio_store=None,
        base_url: str = "",
    ):
        self.mcp_manager = mcp_manager
        self.device = device or getattr(config, "AUTONOMY_SPEAKER_DEFAULT_DEVICE", "") or ""
        self.voice = voice or getattr(config, "AUTONOMY_SPEAKER_DEFAULT_VOICE", "af_heart")
        default_vol = getattr(config, "AUTONOMY_SPEAKER_DEFAULT_VOLUME", None)
        self.volume = volume if volume is not None else default_vol
        self._tts = tts_client  # lazy-init below if None
        self._store = audio_store
        self._base_url = (
            base_url
            or getattr(config, "AGENT_INTERNAL_BASE_URL", "")
            or "http://agent:6002"
        )

    def _get_tts(self):
        if self._tts is None:
            from selene_agent.services.tts_client import TTSClient
            self._tts = TTSClient()
        return self._tts

    def _get_store(self):
        if self._store is None:
            from selene_agent.services.audio_store import get_audio_store
            self._store = get_audio_store()
        return self._store

    def _render_text(self, title: str, body: str) -> str:
        if title and body:
            return f"{title}. {body}"
        return title or body or ""

    async def send(
        self,
        *,
        title: str,
        body: str,
        severity: str = "none",
        attachments: Optional[List[Any]] = None,
    ) -> bool:
        if not self.device:
            logger.warning("[SpeakerNotifier] no device configured; dropping announcement")
            return False
        text = self._render_text(title, body)
        if not text.strip():
            logger.warning("[SpeakerNotifier] empty text after render; dropping")
            return False
        # 1. Synth audio.
        try:
            audio_bytes = await self._get_tts().synth(
                text, voice=self.voice, response_format="mp3"
            )
        except Exception as e:
            logger.error(f"[SpeakerNotifier] TTS failed: {e}")
            return False
        # 2. Stage behind a short-lived token.
        try:
            token = await self._get_store().put(audio_bytes, content_type="audio/mpeg")
        except Exception as e:
            logger.error(f"[SpeakerNotifier] audio_store.put failed: {e}")
            return False
        url = f"{self._base_url.rstrip('/')}/api/tts/audio/{token}.mp3"
        # 3. Dispatch via MA MCP tool.
        payload = {"player_name": self.device, "url": url}
        if self.volume is not None:
            payload["volume"] = self.volume
        try:
            result = await self.mcp_manager.execute_tool(
                "mass_play_announcement", payload
            )
        except Exception as e:
            logger.error(f"[SpeakerNotifier] MA tool failed: {e}")
            return False
        ok, detail = _tool_result_ok(result)
        # The MA tool returns ``{"played": true|false, ...}``; _tool_result_ok
        # checks for success:false JSON. Add a played:false check on top.
        if ok and isinstance(result, (dict, str)):
            payload_check: Any = result
            if isinstance(result, str):
                try:
                    payload_check = json.loads(result)
                except (ValueError, TypeError):
                    payload_check = {}
            if isinstance(payload_check, dict) and payload_check.get("played") is False:
                logger.error(
                    f"[SpeakerNotifier] MA reported not played: {payload_check.get('error')}"
                )
                return False
        if ok:
            logger.info(f"[SpeakerNotifier] announcement queued on {self.device}: {detail}")
        else:
            logger.error(f"[SpeakerNotifier] tool reported failure: {detail}")
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
