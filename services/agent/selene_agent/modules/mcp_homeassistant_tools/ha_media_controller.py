"""Home Assistant media_player transport control.

Slim REST-based replacement for the old DLNA-scanning HAMediaLibrary. Library
search and playback now live in `mcp_plex_tools`; this module only exposes
generic transport / volume / power control over any HA media_player entity
via the `ha_control_media_player` MCP tool.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp

from selene_agent.utils.logger import get_logger

logger = get_logger("loki")


class ActionType(Enum):
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    TOGGLE = "toggle"
    NEXT = "next"
    PREVIOUS = "previous"
    SEEK = "seek"
    SHUFFLE = "shuffle"
    REPEAT = "repeat"

    VOLUME_SET = "volume_set"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    UNMUTE = "unmute"

    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"

    SELECT_SOURCE = "select_source"


def _normalize_volume(value: Any) -> float:
    """Accept 0-100 ints or 0.0-1.0 floats; return HA's expected 0.0-1.0."""
    if value is None:
        return 0.5
    v = float(value)
    return v / 100.0 if v > 1.0 else v


# action -> (HA media_player service name, payload builder from the caller-supplied `value`)
_SERVICE_MAP = {
    ActionType.PLAY: ("media_play", lambda v: {}),
    ActionType.PAUSE: ("media_pause", lambda v: {}),
    ActionType.STOP: ("media_stop", lambda v: {}),
    ActionType.TOGGLE: ("media_play_pause", lambda v: {}),
    ActionType.NEXT: ("media_next_track", lambda v: {}),
    ActionType.PREVIOUS: ("media_previous_track", lambda v: {}),
    ActionType.SEEK: ("media_seek", lambda v: {"seek_position": int(v) if v is not None else 0}),
    ActionType.SHUFFLE: ("shuffle_set", lambda v: {"shuffle": bool(v) if isinstance(v, bool) else True}),
    ActionType.REPEAT: ("repeat_set", lambda v: {"repeat": v if v in ("off", "all", "one") else "all"}),
    ActionType.VOLUME_SET: ("volume_set", lambda v: {"volume_level": _normalize_volume(v)}),
    ActionType.VOLUME_UP: ("volume_up", lambda v: {}),
    ActionType.VOLUME_DOWN: ("volume_down", lambda v: {}),
    ActionType.MUTE: ("volume_mute", lambda v: {"is_volume_muted": True}),
    ActionType.UNMUTE: ("volume_mute", lambda v: {"is_volume_muted": False}),
    ActionType.TURN_ON: ("turn_on", lambda v: {}),
    ActionType.TURN_OFF: ("turn_off", lambda v: {}),
    ActionType.SELECT_SOURCE: ("select_source", lambda v: {"source": str(v)}),
}


class MediaController:
    """REST-only control over HA media_player entities."""

    def __init__(self, ha_url: str, ha_token: str):
        # Accept either http/https REST URL or legacy ws/wss websocket URL — normalize to REST base.
        base = (ha_url or "").strip().rstrip("/")
        if base.startswith("wss://"):
            base = "https://" + base[len("wss://"):]
        elif base.startswith("ws://"):
            base = "http://" + base[len("ws://"):]
        if base.endswith("/api/websocket"):
            base = base[: -len("/api/websocket")]
        if base.endswith("/api"):
            base = base[: -len("/api")]
        self._base = base
        self._headers = {
            "Authorization": f"Bearer {ha_token}",
            "Content-Type": "application/json",
        }
        self._timeout = aiohttp.ClientTimeout(total=15)

    async def initialize(self, device_ids: Optional[List[str]] = None) -> None:
        """No-op. Kept for interface compatibility with the prior MediaController."""
        return

    async def _get_states(self) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as s:
            async with s.get(f"{self._base}/api/states") as r:
                r.raise_for_status()
                return await r.json()

    async def _call_service(self, service: str, entity_id: str, **data: Any) -> None:
        payload = {"entity_id": entity_id, **data}
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as s:
            async with s.post(
                f"{self._base}/api/services/media_player/{service}", json=payload
            ) as r:
                r.raise_for_status()

    async def get_media_player_statuses(self) -> Dict[str, Any]:
        try:
            states = await self._get_states()
            players = []
            for s in states:
                eid = s.get("entity_id", "")
                if not eid.startswith("media_player."):
                    continue
                attrs = s.get("attributes") or {}
                players.append({
                    "entity_id": eid,
                    "name": attrs.get("friendly_name") or eid,
                    "state": s.get("state"),
                    "source": attrs.get("source"),
                    "volume_level": attrs.get("volume_level"),
                    "is_volume_muted": attrs.get("is_volume_muted"),
                    "media_title": attrs.get("media_title"),
                    "media_artist": attrs.get("media_artist"),
                    "app_id": attrs.get("app_id"),
                    "source_list": attrs.get("source_list"),
                })
            return {"success": True, "players": players}
        except Exception as e:
            logger.error(f"get_media_player_statuses failed: {e}")
            return {"success": False, "error": str(e)}

    async def _resolve_device(self, device: Optional[str]) -> Optional[str]:
        if not device:
            return None
        if device.startswith("media_player."):
            return device
        status = await self.get_media_player_statuses()
        if not status.get("success"):
            return None
        needle = device.lower()
        for p in status["players"]:
            if needle in p["entity_id"].lower():
                return p["entity_id"]
            name = p.get("name") or ""
            if needle in name.lower():
                return p["entity_id"]
        return None

    async def control_media_player(
        self,
        action: str,
        device: Optional[str] = None,
        value: Optional[Union[int, float, str, bool]] = None,
        target_device: Optional[str] = None,
    ) -> Dict[str, Any]:
        entity_id = await self._resolve_device(device)
        if not entity_id:
            return {
                "success": False,
                "error": f"Device {device!r} not found",
                "existing_devices": (await self.get_media_player_statuses()).get("players", []),
            }

        try:
            action_type = ActionType(action.lower())
        except ValueError:
            return {"success": False, "error": f"Unknown action: {action}"}

        service, payload_fn = _SERVICE_MAP[action_type]
        try:
            await self._call_service(service, entity_id, **payload_fn(value))
            return {"success": True, "action": action, "entity_id": entity_id}
        except Exception as e:
            logger.error(f"control_media_player {action} on {entity_id}: {e}")
            return {"success": False, "error": str(e)}
