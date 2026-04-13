"""Plex facade for the MCP server: cloud-relay playback + optional HA wake/launch.

plexapi is synchronous; callers run blocking calls via `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp
from plexapi.myplex import MyPlexDevice
from plexapi.server import PlexServer


class _HAServiceClient:
    """Minimal aiohttp client for HA REST — just state reads and service calls."""

    def __init__(self, url: str, token: str):
        base = url.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        self._base = base
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._timeout = aiohttp.ClientTimeout(total=10)

    async def get_state(self, entity_id: str):
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as s:
            async with s.get(f"{self._base}/api/states/{entity_id}") as r:
                if r.status == 404:
                    return None, {}
                r.raise_for_status()
                data = await r.json()
                return data.get("state"), data.get("attributes") or {}

    async def call_service(self, domain: str, service: str, entity_id: Optional[str] = None, **data):
        payload = {k: v for k, v in data.items() if v is not None}
        if entity_id:
            payload["entity_id"] = entity_id
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as s:
            async with s.post(f"{self._base}/api/services/{domain}/{service}", json=payload) as r:
                r.raise_for_status()


def _fmt(item: Any) -> Dict[str, Any]:
    out = {
        "rating_key": str(getattr(item, "ratingKey", "")),
        "type": getattr(item, "type", ""),
        "title": getattr(item, "title", ""),
        "year": getattr(item, "year", None),
        "summary": ((getattr(item, "summary", "") or "").strip() or None),
    }
    if out["summary"] and len(out["summary"]) > 300:
        out["summary"] = out["summary"][:299] + "…"
    if out["type"] == "episode":
        out["show"] = getattr(item, "grandparentTitle", None)
        out["season"] = getattr(item, "parentIndex", None)
        out["episode"] = getattr(item, "index", None)
    elif out["type"] == "track":
        out["artist"] = getattr(item, "grandparentTitle", None)
        out["album"] = getattr(item, "parentTitle", None)
    return out


def _section_matches(section_type: str, media_type: Optional[str]) -> bool:
    if not media_type:
        return True
    mt = media_type.lower()
    if mt in ("movie",):
        return section_type == "movie"
    if mt in ("show", "episode", "season"):
        return section_type == "show"
    if mt in ("artist", "album", "track", "music"):
        return section_type == "artist"
    return True


class PlexAgent:
    """Async facade over plexapi used by the MCP tool handlers."""

    def __init__(
        self,
        plex_url: str,
        plex_token: str,
        ha_url: Optional[str] = None,
        ha_token: Optional[str] = None,
        client_ha_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._plex_url = plex_url
        self._plex_token = plex_token
        self._client_ha_map = client_ha_map or {}
        self._ha = _HAServiceClient(ha_url, ha_token) if ha_url and ha_token else None
        self._server: Optional[PlexServer] = None
        self._account = None

    def _server_sync(self) -> PlexServer:
        if self._server is None:
            self._server = PlexServer(self._plex_url, self._plex_token)
        return self._server

    def _account_sync(self):
        if self._account is None:
            self._account = self._server_sync().myPlexAccount()
        return self._account

    # ---- sync library ops (called via to_thread) ----

    def _do_search(self, query: str, media_type: Optional[str], limit: int) -> List[Dict[str, Any]]:
        results = self._server_sync().search(query, mediatype=media_type, limit=limit)
        return [_fmt(r) for r in results]

    def _do_list_recent(self, media_type: Optional[str], limit: int) -> List[Dict[str, Any]]:
        items = []
        for section in self._server_sync().library.sections():
            if not _section_matches(section.type, media_type):
                continue
            try:
                items.extend(section.recentlyAdded(maxresults=limit))
            except Exception:
                continue
        items.sort(key=lambda i: getattr(i, "addedAt", None) or 0, reverse=True)
        return [_fmt(i) for i in items[:limit]]

    def _do_list_on_deck(self, limit: int) -> List[Dict[str, Any]]:
        items = self._server_sync().library.onDeck()
        return [_fmt(i) for i in items[:limit]]

    def _do_list_clients(self) -> List[Dict[str, Any]]:
        devices = [d for d in self._account_sync().devices() if "player" in (d.provides or [])]
        return [
            {
                "name": d.name,
                "product": d.product,
                "platform": d.platform,
                "provides": d.provides,
            }
            for d in devices
        ]

    def _resolve_device(self, name: str) -> Optional[MyPlexDevice]:
        t = (name or "").casefold()
        devices = [d for d in self._account_sync().devices() if "player" in (d.provides or [])]
        for d in devices:
            if (d.name or "").casefold() == t:
                return d
        for d in devices:
            if t and t in (d.name or "").casefold():
                return d
        return None

    def _do_play(self, rating_key: str, device: MyPlexDevice) -> Dict[str, Any]:
        media = self._server_sync().fetchItem(int(rating_key))
        client = device.connect()
        client.playMedia(media)
        return {
            "played": True,
            "title": getattr(media, "title", ""),
            "type": getattr(media, "type", ""),
            "client": device.name,
        }

    # ---- public async API ----

    async def search(self, query: str, media_type: Optional[str] = None, limit: int = 10):
        return await asyncio.to_thread(self._do_search, query, media_type, limit)

    async def list_recent(self, media_type: Optional[str] = None, limit: int = 10):
        return await asyncio.to_thread(self._do_list_recent, media_type, limit)

    async def list_on_deck(self, limit: int = 10):
        return await asyncio.to_thread(self._do_list_on_deck, limit)

    async def list_clients(self):
        return await asyncio.to_thread(self._do_list_clients)

    async def play(self, rating_key: str, client_name: str) -> Dict[str, Any]:
        device = await asyncio.to_thread(self._resolve_device, client_name)
        if device is None:
            available = [c["name"] for c in await self.list_clients()]
            return {
                "played": False,
                "error": f"no player-capable device matches {client_name!r}",
                "available_clients": available,
            }

        readiness_note = await self._ensure_ready(device.name)
        result = await asyncio.to_thread(self._do_play, rating_key, device)
        if readiness_note:
            result["readiness"] = readiness_note
        return result

    async def _ensure_ready(self, device_name: str) -> Optional[str]:
        """If a HA mapping exists, wake the TV and launch Plex if needed.

        Returns a short human-readable note describing any HA action taken, or None.
        """
        # TODO: PLEX_CLIENT_HA_MAP is currently hand-authored in .env (Plex client name
        # -> {state_entity, adb_entity}). Replace with auto-discovery — e.g. cross-reference
        # Plex clients against HA media_player entities by IP/MAC, or a chat/web setup wizard
        # that walks the user through mapping each TV. Adding a new device today requires
        # manually editing env + restart.
        if not self._ha:
            return None
        mapping = self._client_ha_map.get(device_name)
        if not mapping:
            return None

        state_entity = mapping.get("state_entity")
        adb_entity = mapping.get("adb_entity")
        plex_app_id = mapping.get("plex_app_id", "com.plexapp.android")
        launch_cmd = mapping.get(
            "launch_command",
            f"monkey -p {plex_app_id} -c android.intent.category.LAUNCHER 1",
        )

        actions = []

        if state_entity:
            state, attrs = await self._ha.get_state(state_entity)
            if state in (None, "off", "unavailable", "unknown"):
                await self._ha.call_service("media_player", "turn_on", entity_id=state_entity)
                actions.append(f"woke {state_entity}")
                await asyncio.sleep(3)
                state, attrs = await self._ha.get_state(state_entity)

            current_app = attrs.get("app_id") or attrs.get("source")
            if current_app != plex_app_id and adb_entity:
                await self._ha.call_service(
                    "androidtv", "adb_command",
                    entity_id=adb_entity, command=launch_cmd,
                )
                actions.append(f"launched Plex via {adb_entity}")
                await asyncio.sleep(3)

        return "; ".join(actions) if actions else None
