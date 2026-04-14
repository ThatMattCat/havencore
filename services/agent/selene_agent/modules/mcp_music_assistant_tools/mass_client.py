"""Music Assistant facade for the MCP server.

Wraps `music-assistant-client` with:
- Persistent WS connection (connect once, keep listener alive).
- Display-name → player_id resolution (LLM-friendly inputs).
- Flattened search result rows (drop provider internals the LLM doesn't need).
- Queue summary shaped for "what's playing / what's next" questions.

Reconnect-on-disconnect handling is deliberately minimal: if the WS drops,
the next tool call returns an error; the agent's MCP-manager layer can
restart the module on repeated failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import is_dataclass, fields
from typing import Any, Dict, List, Optional

import aiohttp
from music_assistant_client import MusicAssistantClient
from music_assistant_models.enums import MediaType, QueueOption, RepeatMode

_MEDIA_TYPE_MAP: Dict[str, MediaType] = {
    "track": MediaType.TRACK,
    "album": MediaType.ALBUM,
    "artist": MediaType.ARTIST,
    "playlist": MediaType.PLAYLIST,
    "radio": MediaType.RADIO,
}

_QUEUE_MODE_MAP: Dict[str, QueueOption] = {
    "replace": QueueOption.REPLACE,
    "next": QueueOption.NEXT,
    "add": QueueOption.ADD,
}

_REPEAT_MODE_MAP: Dict[str, RepeatMode] = {
    "off": RepeatMode.OFF,
    "one": RepeatMode.ONE,
    "all": RepeatMode.ALL,
}


class MassUnavailableError(RuntimeError):
    """Raised when the client is not connected."""


class MassAgent:
    """Async facade over `music-assistant-client` used by the MCP handlers."""

    def __init__(self, url: str, token: str):
        self._url = url
        self._token = token
        self._session: Optional[aiohttp.ClientSession] = None
        self._client: Optional[MusicAssistantClient] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._init_ready: Optional[asyncio.Event] = None

    # ---- lifecycle ----

    async def connect(self) -> None:
        if self._client is not None:
            return
        self._session = aiohttp.ClientSession()
        self._client = MusicAssistantClient(self._url, self._session, token=self._token)
        await self._client.connect()
        self._init_ready = asyncio.Event()
        self._listen_task = asyncio.create_task(
            self._client.start_listening(self._init_ready)
        )
        await self._init_ready.wait()

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session is not None:
            await self._session.close()
        self._client = None
        self._session = None
        self._listen_task = None
        self._init_ready = None

    # ---- public API ----

    async def search(
        self, query: str, media_type: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        client = self._require_client()
        all_types = list(_MEDIA_TYPE_MAP.values())
        typed = [_MEDIA_TYPE_MAP[media_type.lower()]] if media_type else all_types

        rows = await self._run_search(client, query, typed, limit)

        # Fallback 1: drop the media_type filter. MA's search is title-biased,
        # so "<album title> <artist>" queries often miss when typed=album but
        # hit a matching track or artist row that points at the same work.
        if not rows and media_type:
            rows = await self._run_search(client, query, all_types, limit)

        # Fallback 2: multi-token queries that still miss — retry with just
        # the longest single token (usually the distinctive word). Handles
        # the LLM gluing title+artist into one query.
        if not rows and len(query.split()) > 1:
            longest = max(query.split(), key=len)
            if len(longest) >= 4 and longest.lower() != query.lower():
                rows = await self._run_search(client, longest, all_types, limit)

        return rows

    @staticmethod
    async def _run_search(client, query: str, media_types, limit: int) -> List[Dict[str, Any]]:
        results = await client.music.search(
            search_query=query, media_types=media_types, limit=limit,
        )
        rows: List[Dict[str, Any]] = []
        for bucket in ("tracks", "albums", "artists", "playlists", "radio"):
            for item in getattr(results, bucket, []) or []:
                rows.append(_flatten_media_item(item))
        return rows

    def list_players(self, include_hidden: bool = False) -> List[Dict[str, Any]]:
        client = self._require_client()
        out: List[Dict[str, Any]] = []
        for p in client.players:
            if not include_hidden and getattr(p, "hide_in_ui", False):
                continue
            out.append(_flatten_player(p))
        return out

    async def play_media(
        self, uri: str, player_name: str, mode: str = "replace"
    ) -> Dict[str, Any]:
        client = self._require_client()
        player = self._resolve_player(player_name)
        if player is None:
            return {
                "played": False,
                "error": f"no player matches {player_name!r}",
                "available_players": [p.name for p in client.players if not getattr(p, "hide_in_ui", False)],
            }
        option = _QUEUE_MODE_MAP.get(mode.lower(), QueueOption.REPLACE)
        await client.player_queues.play_media(
            queue_id=player.player_id, media=uri, option=option,
        )
        return {
            "played": True,
            "player": player.name,
            "player_id": player.player_id,
            "uri": uri,
            "mode": option.value,
        }

    async def get_queue(self, player_name: str, item_limit: int = 5) -> Dict[str, Any]:
        client = self._require_client()
        player = self._resolve_player(player_name)
        if player is None:
            return {"error": f"no player matches {player_name!r}"}
        queue = await client.player_queues.get_active_queue(player.player_id)
        if queue is None:
            return {"player": player.name, "state": "idle", "current": None, "upcoming": []}

        items = await client.player_queues.get_queue_items(
            player.player_id, limit=max(item_limit + 1, 1),
        )
        current_idx = queue.current_index if queue.current_index is not None else 0
        current = items[current_idx] if 0 <= current_idx < len(items) else None
        upcoming = items[current_idx + 1 : current_idx + 1 + item_limit] if current else items[:item_limit]

        return {
            "player": player.name,
            "state": str(queue.state.value) if queue.state is not None else None,
            "shuffle": bool(queue.shuffle_enabled),
            "repeat": str(queue.repeat_mode.value) if queue.repeat_mode is not None else None,
            "current": _flatten_queue_item(current) if current else None,
            "upcoming": [_flatten_queue_item(i) for i in upcoming],
            "total_items": queue.items,
        }

    async def clear_queue(self, player_name: str) -> Dict[str, Any]:
        client = self._require_client()
        player = self._resolve_player(player_name)
        if player is None:
            return {"cleared": False, "error": f"no player matches {player_name!r}"}
        await client.player_queues.clear(player.player_id)
        return {"cleared": True, "player": player.name}

    async def play_announcement(
        self,
        player_name: str,
        url: str,
        volume: Optional[float] = None,
        pre_announce: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Play a short announcement URL on a player. Ducks current playback
        on MA's side, resumes after.

        ``volume`` accepts either a 0.0-1.0 float (treated as a level) or a
        0-100 int percentage. The MA API wants ``volume_level: int | None``
        (percentage), so we normalize.
        """
        client = self._require_client()
        player = self._resolve_player(player_name)
        if player is None:
            return {
                "played": False,
                "error": f"no player matches {player_name!r}",
                "available_players": [
                    p.name for p in client.players if not getattr(p, "hide_in_ui", False)
                ],
            }
        volume_level: Optional[int] = None
        if volume is not None:
            try:
                v = float(volume)
            except (TypeError, ValueError):
                v = None  # type: ignore[assignment]
            if v is not None:
                # 0.0-1.0 treated as a fraction, anything else clamped into 0-100.
                volume_level = int(round(v * 100 if 0.0 <= v <= 1.0 else v))
                volume_level = max(0, min(100, volume_level))
        await client.players.play_announcement(
            player.player_id,
            url=url,
            volume_level=volume_level,
            pre_announce=pre_announce,
        )
        return {
            "played": True,
            "player": player.name,
            "player_id": player.player_id,
            "url": url,
            "volume_level": volume_level,
        }

    async def playback_control(
        self, player_name: str, action: str
    ) -> Dict[str, Any]:
        client = self._require_client()
        player = self._resolve_player(player_name)
        if player is None:
            return {"ok": False, "error": f"no player matches {player_name!r}"}

        action_lc = action.lower().strip()
        if action_lc in ("shuffle_on", "shuffle_off"):
            await client.player_queues.shuffle(player.player_id, action_lc == "shuffle_on")
            return {"ok": True, "player": player.name, "action": action_lc}
        if action_lc.startswith("repeat_"):
            mode_key = action_lc.removeprefix("repeat_")
            mode = _REPEAT_MODE_MAP.get(mode_key)
            if mode is None:
                return {"ok": False, "error": f"unknown repeat mode {mode_key!r}"}
            await client.player_queues.repeat(player.player_id, mode)
            return {"ok": True, "player": player.name, "action": action_lc}
        return {"ok": False, "error": f"unknown action {action!r}; supported: shuffle_on, shuffle_off, repeat_off, repeat_one, repeat_all"}

    # ---- internals ----

    def _require_client(self) -> MusicAssistantClient:
        if self._client is None:
            raise MassUnavailableError("Music Assistant client not connected")
        return self._client

    def _resolve_player(self, name: str):
        client = self._require_client()
        target = (name or "").strip().casefold()
        if not target:
            return None
        players = list(client.players)
        for p in players:
            if (p.name or "").casefold() == target:
                return p
        for p in players:
            if target in (p.name or "").casefold():
                return p
        for p in players:
            if target == (p.player_id or "").casefold():
                return p
        return None


# ---- flatteners (plain dicts for JSON over MCP) ----

def _flatten_player(p) -> Dict[str, Any]:
    current = None
    if getattr(p, "current_media", None):
        cm = p.current_media
        current = {
            "title": getattr(cm, "title", None),
            "artist": getattr(cm, "artist", None),
            "album": getattr(cm, "album", None),
            "uri": getattr(cm, "uri", None),
        }
    return {
        "player_id": p.player_id,
        "display_name": p.name,
        "provider": p.provider,
        "available": p.available,
        "powered": p.powered,
        "state": str(p.playback_state.value) if p.playback_state is not None else None,
        "volume_level": p.volume_level,
        "current_item": current,
    }


def _flatten_media_item(item) -> Dict[str, Any]:
    mappings = list(getattr(item, "provider_mappings", None) or [])
    providers = sorted({getattr(m, "provider_domain", None) for m in mappings if getattr(m, "provider_domain", None)})
    row: Dict[str, Any] = {
        "uri": getattr(item, "uri", None),
        "name": getattr(item, "name", None),
        "media_type": str(getattr(getattr(item, "media_type", None), "value", "") or ""),
        "providers": providers,
    }
    artists = getattr(item, "artists", None)
    if artists:
        row["artist"] = ", ".join(a.name for a in artists if getattr(a, "name", None))
    album = getattr(item, "album", None)
    if album is not None:
        row["album"] = getattr(album, "name", None)
    return row


def _flatten_queue_item(qi) -> Optional[Dict[str, Any]]:
    if qi is None:
        return None
    media = getattr(qi, "media_item", None)
    return {
        "queue_item_id": getattr(qi, "queue_item_id", None),
        "title": getattr(media, "name", None) if media else None,
        "media_type": str(getattr(getattr(media, "media_type", None), "value", "") or "") if media else None,
        "uri": getattr(media, "uri", None) if media else None,
        "duration": getattr(qi, "duration", None),
    }
