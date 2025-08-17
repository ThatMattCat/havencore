import asyncio
import json
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
import aiohttp
from datetime import datetime, timedelta
import base64
from enum import Enum
import config


# TODO: merge these media-specific controls into original haos tooling

class MediaType(Enum):
    """Media content types for Plex and other services."""
    MOVIE = "movie"
    SHOW = "show"
    SEASON = "season"
    EPISODE = "episode"
    ARTIST = "artist"
    ALBUM = "album"
    TRACK = "track"
    PLAYLIST = "playlist"
    VIDEO = "video"
    MUSIC = "music"
    IMAGE = "image"
    URL = "url"

class HomeAssistantMediaControl:
    """
    A comprehensive class to interact with Home Assistant for media control and streaming.
    Focuses on Plex, camera streams, image casting, and complete media management.
    """
    
    def __init__(self, host: str = None, token: str = None, port: int = 8123, use_ssl: bool = False):
        """
        Initialize the Home Assistant connection.
        
        Args:
            host: Home Assistant host (e.g., "192.168.1.100" or "homeassistant.local")
            token: Long-lived access token from Home Assistant
            port: Port number (default 8123)
            use_ssl: Whether to use HTTPS/WSS (default False for local)
        """
        self.host = host or config.HAOS_HOST
        self.port = port or 8123
        self.token = token or config.HAOS_TOKEN
        self.use_ssl = use_ssl or config.HAOS_USE_SSL

        # Build URLs
        protocol = "https" if self.use_ssl else "http"
        ws_protocol = "wss" if self.use_ssl else "ws"
        self.base_url = f"{protocol}://{self.host}:{self.port}"
        self.ws_url = f"{ws_protocol}://{self.host}:{self.port}/api/websocket"
        
        # WebSocket connection
        self.ws = None
        self.ws_session = None
        self.msg_id = 1
        self._connected = False
        self._lock = asyncio.Lock()
        
        # Cache for entity states
        self._state_cache = {}
        self._cache_timeout = 5  # seconds
        self._last_cache_update = {}
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Establish WebSocket connection to Home Assistant."""
        async with self._lock:
            try:
                if self._connected and self.ws and not self.ws.closed:
                    return True
                    
                if self.ws_session:
                    await self.ws_session.close()
                    self.ws_session = None
                    self.ws = None
                    self._connected = False
                
                self.ws_session = aiohttp.ClientSession()
                self.ws = await self.ws_session.ws_connect(self.ws_url)
                
                msg = await self.ws.receive_json()
                if msg["type"] == "auth_required":
                    self.logger.info("Authenticating with Home Assistant WebSocket")
                    await self.ws.send_json({
                        "type": "auth",
                        "access_token": self.token
                    })
                    
                    auth_result = await self.ws.receive_json()
                    if auth_result["type"] != "auth_ok":
                        self.logger.error(f"Authentication failed: {auth_result}")
                        await self.ws_session.close()
                        self.ws_session = None
                        self.ws = None
                        self._connected = False
                        return False
                
                self.logger.info("Successfully connected to Home Assistant WebSocket")
                self._connected = True
                return True
                
            except Exception as e:
                self.logger.error(f"WebSocket connection error: {e}")
                if self.ws_session:
                    await self.ws_session.close()
                self.ws_session = None
                self.ws = None
                self._connected = False
                return False
        
    async def disconnect(self):
        """Close WebSocket connection."""
        async with self._lock:
            if self.ws:
                await self.ws.close()
                self.ws = None
            if self.ws_session:
                await self.ws_session.close()
                self.ws_session = None
            self._connected = False

    async def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        return self._connected and self.ws and not self.ws.closed

    async def _send_ws_command(self, command: Dict[str, Any], 
                              timeout: int = 30) -> Dict[str, Any]:
        """
        Send command with timeout and reconnection logic.
        """
        # Ensure we're connected
        if not await self.is_connected():
            connected = await self.connect()
            if not connected:
                raise ConnectionError("Failed to connect to Home Assistant")
        
        command["id"] = self.msg_id
        self.msg_id += 1
        
        try:
            await self.ws.send_json(command)
            
            start_time = asyncio.get_event_loop().time()
            while True:
                remaining_time = timeout - (asyncio.get_event_loop().time() - start_time)
                if remaining_time <= 0:
                    raise asyncio.TimeoutError(f"Command timeout: {command}")
                
                try:
                    msg = await asyncio.wait_for(
                        self.ws.receive_json(),
                        timeout=min(remaining_time, 5)
                    )
                    
                    if msg.get("id") == command["id"]:
                        return msg
                    
                    if msg.get("type") == "event":
                        self.logger.debug(f"Received event: {msg}")
                        
                except asyncio.TimeoutError:
                    continue
                    
        except (aiohttp.ClientError, ConnectionError) as e:
            self.logger.error(f"WebSocket command error: {e}")
            self._connected = False
            connected = await self.connect()
            if connected:
                return await self._send_ws_command(command, timeout)
            else:
                raise ConnectionError(f"Lost connection and failed to reconnect: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error sending command: {e}")
            raise

                
    async def _call_service(self, domain: str, service: str, 
                          service_data: Dict[str, Any] = None,
                          target: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a Home Assistant service via WebSocket."""
        command = {
            "type": "call_service",
            "domain": domain,
            "service": service
        }
        
        if service_data:
            command["service_data"] = service_data
        if target:
            command["target"] = target
            
        return await self._send_ws_command(command)
    
    # ========== Entity State Management ==========
    
    async def get_entity_state(self, entity_id: str, 
                              use_cache: bool = True) -> Dict[str, Any]:
        """
        Get the current state of any entity.
        
        Args:
            entity_id: Entity ID to query
            use_cache: Whether to use cached state (5 second timeout)
        """
        now = datetime.now()
        
        if use_cache and entity_id in self._state_cache:
            last_update = self._last_cache_update.get(entity_id)
            if last_update and (now - last_update).total_seconds() < self._cache_timeout:
                return self._state_cache[entity_id]
        
        command = {
            "type": "get_states"
        }
        
        result = await self._send_ws_command(command)
        
        if result.get("success"):
            states = result.get("result", [])
            for state in states:
                if state["entity_id"] == entity_id:
                    self._state_cache[entity_id] = state
                    self._last_cache_update[entity_id] = now
                    return state
                    
        return None
    
    async def get_media_player_status(self, entity_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a media player including playback state and current media.
        
        Returns dict with:
            - state: playing, paused, idle, off, etc.
            - media_title: Current media title
            - media_artist: Current artist (for music)
            - media_series_title: TV show name (for episodes)
            - media_season: Season number
            - media_episode: Episode number
            - media_duration: Total duration in seconds
            - media_position: Current position in seconds
            - volume_level: Current volume (0.0 to 1.0)
            - is_volume_muted: Boolean
            - shuffle: Boolean
            - repeat: off, all, one
            - source: Current input source
            - source_list: Available sources
        """
        state = await self.get_entity_state(entity_id)
        
        if not state:
            return {"error": f"Entity {entity_id} not found"}
            
        attributes = state.get("attributes", {})
        
        return {
            "state": state.get("state"),
            "media_title": attributes.get("media_title"),
            "media_artist": attributes.get("media_artist"),
            "media_album_name": attributes.get("media_album_name"),
            "media_series_title": attributes.get("media_series_title"),
            "media_season": attributes.get("media_season"),
            "media_episode": attributes.get("media_episode"),
            "media_duration": attributes.get("media_duration"),
            "media_position": attributes.get("media_position"),
            "media_position_updated_at": attributes.get("media_position_updated_at"),
            "volume_level": attributes.get("volume_level"),
            "is_volume_muted": attributes.get("is_volume_muted"),
            "shuffle": attributes.get("shuffle"),
            "repeat": attributes.get("repeat"),
            "source": attributes.get("source"),
            "source_list": attributes.get("source_list"),
            "app_name": attributes.get("app_name"),
            "entity_picture": attributes.get("entity_picture"),
            "supported_features": attributes.get("supported_features")
        }
    
    # ========== Plex Search and Browse ==========
    async def search_plex_media(self, query: str, 
                               media_player_entity: str = "media_player.plex_for_android_tv_living_room_tv",
                               media_types: List[MediaType] = None,
                               limit: int = 5,
                               deep_search: bool = True) -> List[Dict[str, Any]]:
        """
        FIXED: Enhanced search that can do deep searching through Plex hierarchy.
        """
        if not media_player_entity:
            players = await self.get_all_plex_players()
            if not players:
                return {"error": "No Plex media player found"}
            # TODO: better logic to identify best player choice
            media_player_entity = players[0]['entity_id']
        
        results = []
        query_lower = query.lower()
        
        browse_result = await self.browse_media(
            media_player_entity,
            media_content_id=""
        )
        
        if not browse_result.get("success"):
            return []
        
        async def search_recursive(items, depth=0, max_depth=3):
            nonlocal results
            
            for item in items:
                if self._matches_search_enhanced(item, query_lower):
                    results.append({
                        "title": item.get("title"),
                        "media_content_id": item.get("media_content_id"),
                        "media_content_type": item.get("media_content_type"),
                        "thumbnail": item.get("thumbnail"),
                        "can_play": item.get("can_play", False),
                        "can_expand": item.get("can_expand", False),
                        "children_media_class": item.get("children_media_class"),
                        # Add more metadata
                        "media_class": item.get("media_class"),
                        "media_artist": item.get("media_artist"),
                        "media_album_name": item.get("media_album_name"),
                        "media_series_title": item.get("media_series_title"),
                        "media_season": item.get("media_season"),
                        "media_episode": item.get("media_episode")
                    })
                    
                    if len(results) >= limit:
                        return
                
                if deep_search and item.get("can_expand") and depth < max_depth:
                    if media_types and not self._matches_media_type_enhanced(item, media_types):
                        continue
                        
                    sub_browse = await self.browse_media(
                        media_player_entity,
                        media_content_type=item.get("media_content_type", ""),
                        media_content_id=item.get("media_content_id", "")
                    )
                    
                    if sub_browse.get("success"):
                        sub_items = sub_browse.get("result", {}).get("children", [])
                        await search_recursive(sub_items, depth + 1, max_depth)
                        
                        if len(results) >= limit:
                            return
        
        root_items = browse_result.get("result", {}).get("children", [])
        await search_recursive(root_items)
        
        return results[:limit]
    
    async def get_all_plex_players(self) -> List[Dict[str, Any]]:
        """
        NEW: Get all Plex media players with better detection.
        """
        players = await self.get_all_media_players()
        plex_players = []
        
        for player in players:
            app_name = player.get("app_name", "")
            name = player.get("name","")
            source = player.get("source","")
            is_plex = any([
                "plex" in player["entity_id"].lower(),
                app_name is not None and app_name.lower() == "plex",
                name is not None and "plex" in str(name).lower(),
                source is not None and "plex" in str(source).lower()
            ])
            
            if is_plex:
                plex_players.append(player)
                
        return plex_players
    
    def _matches_search_enhanced(self, item: Dict[str, Any], query: str) -> bool:
        """
        Enhanced search matching with fuzzy logic.
        """
        query_lower = query.lower()
        
        search_fields = [
            "title", "media_title", "media_artist", "media_album_name",
            "media_series_title", "original_title", "sort_title"
        ]
        
        for field in search_fields:
            value = item.get(field, "")
            if value and query_lower in str(value).lower():
                return True
        
        query_words = query_lower.split()
        title = item.get("title", "").lower()
        if all(word in title for word in query_words):
            return True
            
        return False
    
    def _matches_media_type_enhanced(self, item: Dict[str, Any], 
                                    media_types: List[MediaType]) -> bool:
        """
        Enhanced media type matching.
        """
        type_fields = ["media_class", "media_content_type", "children_media_class", "type"]
        
        for field in type_fields:
            item_type = str(item.get(field, "")).lower()
            for media_type in media_types:
                if media_type.value in item_type:
                    return True
                if media_type == MediaType.MOVIE and "film" in item_type:
                    return True
                if media_type == MediaType.SHOW and any(x in item_type for x in ["series", "tv"]):
                    return True
                    
        return False
    async def search_plex_by_type(self, 
                                 media_type: MediaType,
                                 title: str = None,
                                 artist: str = None,
                                 album: str = None,
                                 show: str = None,
                                 season: int = None,
                                 episode: int = None,
                                 media_player_entity: str = None) -> Dict[str, Any]:
        """
        Search for specific Plex content with detailed parameters.
        
        Args:
            media_type: Type of media to search for
            title: Title of movie, episode, or track
            artist: Artist name (for music)
            album: Album name (for music)
            show: TV show name
            season: Season number
            episode: Episode number
            media_player_entity: Plex media player entity
            
        Returns:
            First matching media item with full details
        """
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        query_parts = []
        if title:
            query_parts.append(title)
        if artist:
            query_parts.append(artist)
        if album:
            query_parts.append(album)
        if show:
            query_parts.append(show)
            
        query = " ".join(query_parts)
        
        results = await self.search_plex_media(
            query=query,
            media_player_entity=media_player_entity,
            media_types=[media_type],
            limit=20
        )
        
        for result in results:
            if media_type in [MediaType.EPISODE, MediaType.SEASON] and show:
                if result.get("can_expand"):
                    expanded = await self.browse_media(
                        media_player_entity,
                        media_content_type=result.get("media_content_type"),
                        media_content_id=result.get("media_content_id")
                    )
                    
                    if expanded.get("success"):
                        children = expanded.get("result", {}).get("children", [])
                        for child in children:
                            if self._matches_episode(child, season, episode):
                                return child
            else:
                return result
                
        return None
    
    async def get_plex_recently_added(self, 
                                     media_player_entity: str = None,
                                     media_type: MediaType = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently added media from Plex.
        
        Args:
            media_player_entity: Plex media player entity
            media_type: Filter by media type (optional)
            limit: Maximum number of items
        """
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        browse_result = await self.browse_media(
            media_player_entity,
            media_content_type="recent",
            media_content_id="recent"
        )
        
        if not browse_result.get("success"):
            browse_result = await self.browse_media(
                media_player_entity,
                media_content_type="",
                media_content_id=""
            )
        
        items = []
        if browse_result.get("success"):
            children = browse_result.get("result", {}).get("children", [])
            for child in children[:limit]:
                if not media_type or self._matches_media_type(child, [media_type]):
                    items.append(child)
                    
        return items
    
    async def browse_media(self, media_player_entity: str,
                          media_content_type: str = "",
                          media_content_id: str = "") -> Dict[str, Any]:
        """
        Browse media content available on a media player.
        
        Args:
            media_player_entity: Entity ID of media player
            media_content_type: Type to browse
            media_content_id: ID for deeper browsing
        """
        command = {
            "type": "media_player/browse_media",
            "entity_id": media_player_entity
        }
        
        if media_content_type:
            command["media_content_type"] = media_content_type
        if media_content_id:
            command["media_content_id"] = media_content_id
            
        return await self._send_ws_command(command)
    
    # ========== Media Player Controls ==========
    
    async def play(self, entity_id: str) -> Dict[str, Any]:
        """Resume playback on a media player."""
        return await self._call_service(
            "media_player",
            "media_play",
            target={"entity_id": entity_id}
        )
    
    async def pause(self, entity_id: str) -> Dict[str, Any]:
        """Pause playback on a media player."""
        return await self._call_service(
            "media_player",
            "media_pause",
            target={"entity_id": entity_id}
        )
    
    async def stop(self, entity_id: str) -> Dict[str, Any]:
        """Stop playback on a media player."""
        return await self._call_service(
            "media_player",
            "media_stop",
            target={"entity_id": entity_id}
        )
    
    async def play_pause_toggle(self, entity_id: str) -> Dict[str, Any]:
        """Toggle play/pause on a media player."""
        return await self._call_service(
            "media_player",
            "media_play_pause",
            target={"entity_id": entity_id}
        )
    
    async def next_track(self, entity_id: str) -> Dict[str, Any]:
        """Skip to next track/episode."""
        return await self._call_service(
            "media_player",
            "media_next_track",
            target={"entity_id": entity_id}
        )
    
    async def previous_track(self, entity_id: str) -> Dict[str, Any]:
        """Go to previous track/episode."""
        return await self._call_service(
            "media_player",
            "media_previous_track",
            target={"entity_id": entity_id}
        )
    
    async def seek(self, entity_id: str, position: int) -> Dict[str, Any]:
        """
        Seek to specific position in media.
        
        Args:
            entity_id: Media player entity ID
            position: Position in seconds
        """
        return await self._call_service(
            "media_player",
            "media_seek",
            service_data={"seek_position": position},
            target={"entity_id": entity_id}
        )
    
    async def set_shuffle(self, entity_id: str, shuffle: bool) -> Dict[str, Any]:
        """Enable or disable shuffle."""
        return await self._call_service(
            "media_player",
            "shuffle_set",
            service_data={"shuffle": shuffle},
            target={"entity_id": entity_id}
        )
    
    async def set_repeat(self, entity_id: str, repeat: str) -> Dict[str, Any]:
        """
        Set repeat mode.
        
        Args:
            entity_id: Media player entity ID
            repeat: "off", "all", or "one"
        """
        return await self._call_service(
            "media_player",
            "repeat_set",
            service_data={"repeat": repeat},
            target={"entity_id": entity_id}
        )
    
    async def clear_playlist(self, entity_id: str) -> Dict[str, Any]:
        """Clear the current playlist."""
        return await self._call_service(
            "media_player",
            "clear_playlist",
            target={"entity_id": entity_id}
        )
    
    # ========== Volume Controls ==========
    
    async def set_volume(self, entity_id: str, 
                         volume_level: float) -> Dict[str, Any]:
        """
        Set volume level for a media player.
        
        Args:
            entity_id: Media player entity ID
            volume_level: Volume level between 0.0 and 1.0
        """
        return await self._call_service(
            "media_player",
            "volume_set",
            service_data={"volume_level": volume_level},
            target={"entity_id": entity_id}
        )
    
    async def volume_up(self, entity_id: str) -> Dict[str, Any]:
        """Increase volume."""
        return await self._call_service(
            "media_player",
            "volume_up",
            target={"entity_id": entity_id}
        )
    
    async def volume_down(self, entity_id: str) -> Dict[str, Any]:
        """Decrease volume."""
        return await self._call_service(
            "media_player",
            "volume_down",
            target={"entity_id": entity_id}
        )
    
    async def mute(self, entity_id: str, mute: bool = True) -> Dict[str, Any]:
        """Mute or unmute a media player."""
        return await self._call_service(
            "media_player",
            "volume_mute",
            service_data={"is_volume_muted": mute},
            target={"entity_id": entity_id}
        )
    
    # ========== Media Playback ==========
    
    async def play_media(self, media_player_entity: str,
                        media_content_id: str,
                        media_content_type: str,
                        announce: bool = False,
                        extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Play specific media on a media player.
        
        Args:
            media_player_entity: Entity ID of the media player
            media_content_id: Content ID (URL, Plex ID, etc.)
            media_content_type: Type of content
            announce: If True, play as announcement and resume previous
            extra: Additional parameters
        """
        service_data = {
            "media_content_id": media_content_id,
            "media_content_type": media_content_type
        }
        
        if announce:
            service_data["announce"] = announce
        if extra:
            service_data["extra"] = extra
            
        return await self._call_service(
            "media_player",
            "play_media",
            service_data=service_data,
            target={"entity_id": media_player_entity}
        )
    
    async def play_plex_media(self, media_player_entity: str, 
                             media_content_id: str = None,
                             search_query: str = None,
                             media_type: MediaType = None) -> Dict[str, Any]:
        """
        Play Plex media on a specified media player.
        Can use either a known content ID or search for content.
        
        Args:
            media_player_entity: Entity ID of the media player
            media_content_id: Plex media ID (if known)
            search_query: Search for content by name
            media_type: Type of media when searching
        """
        # If no content ID provided, search for it
        if not media_content_id and search_query:
            search_results = await self.search_plex_media(
                query=search_query,
                media_player_entity=media_player_entity,
                media_types=[media_type] if media_type else None,
                limit=1
            )
            
            if search_results:
                media_content_id = search_results[0].get("media_content_id")
                media_content_type = search_results[0].get("media_content_type")
            else:
                return {"error": f"No media found for query: {search_query}"}
        else:
            media_content_type = media_type.value if media_type else "movie"
        
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=media_content_id,
            media_content_type=media_content_type
        )
    
    async def queue_media(self, media_player_entity: str,
                         media_content_id: str,
                         media_content_type: str,
                         enqueue: str = "add") -> Dict[str, Any]:
        """
        Add media to queue instead of playing immediately.
        
        Args:
            media_player_entity: Entity ID
            media_content_id: Content to queue
            media_content_type: Type of content
            enqueue: "add" (end of queue), "next" (play next), "play" (play now)
        """
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=media_content_id,
            media_content_type=media_content_type,
            extra={"enqueue": enqueue}
        )
    
    # ========== Camera Controls ==========
    
    async def stream_camera(self, media_player_entity: str, 
                           camera_entity: str) -> Dict[str, Any]:
        """
        Stream a camera feed to a media player.
        
        Args:
            media_player_entity: Entity ID of the media player
            camera_entity: Entity ID of the camera
        """
        stream_url = await self.get_camera_stream_url(camera_entity)
        
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=stream_url,
            media_content_type="video"
        )
    
    async def get_camera_stream_url(self, camera_entity: str) -> str:
        """Get the stream URL for a camera entity."""
        result = await self._send_ws_command({
            "type": "camera/stream",
            "entity_id": camera_entity
        })
        
        if result.get("success"):
            return result.get("result", {}).get("url", "")
        else:
            return f"{self.base_url}/api/camera_proxy_stream/{camera_entity}?token={self.token}"
    
    async def create_camera_snapshot(self, camera_entity: str,
                                    filename: str = None) -> str:
        """Create a snapshot from a camera and get its URL."""
        if not filename:
            filename = f"snapshot_{camera_entity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        result = await self._call_service(
            "camera",
            "snapshot",
            service_data={
                "filename": f"/config/www/{filename}"
            },
            target={"entity_id": camera_entity}
        )
        
        return f"{self.base_url}/local/{filename}"
    
    # ========== Image Casting ==========
    
    async def cast_image_url(self, media_player_entity: str, 
                            image_url: str,
                            title: str = "AI Generated Image",
                            duration: int = None) -> Dict[str, Any]:
        """
        Cast an image from a URL to a media player/display.
        
        Args:
            media_player_entity: Entity ID of the display/media player
            image_url: URL of the image
            title: Optional title for the image
            duration: Display duration in seconds (if supported)
        """
        extra = {"title": title}
        if duration:
            extra["duration"] = duration
            
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=image_url,
            media_content_type="image",
            extra=extra
        )
    
    async def cast_local_image(self, media_player_entity: str,
                              image_path: str,
                              title: str = "Local Image") -> Dict[str, Any]:
        """Cast a local image file through Home Assistant's media source."""
        media_content_id = f"media-source://media_source/local/{image_path}"
        
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=media_content_id,
            media_content_type="image",
            extra={"title": title}
        )
    
    # ========== Source/Input Management ==========
    
    async def select_source(self, entity_id: str, source: str) -> Dict[str, Any]:
        """
        Select input source on a media player.
        
        Args:
            entity_id: Media player entity ID
            source: Source name (must be in source_list)
        """
        return await self._call_service(
            "media_player",
            "select_source",
            service_data={"source": source},
            target={"entity_id": entity_id}
        )
    
    async def select_sound_mode(self, entity_id: str, 
                               sound_mode: str) -> Dict[str, Any]:
        """Select sound mode (if supported by device)."""
        return await self._call_service(
            "media_player",
            "select_sound_mode",
            service_data={"sound_mode": sound_mode},
            target={"entity_id": entity_id}
        )
    
    # ========== Device Discovery ==========
    
    async def get_all_media_players(self) -> List[Dict[str, Any]]:
        """Get all media player entities with their current states."""
        command = {"type": "get_states"}
        result = await self._send_ws_command(command)
        
        players = []
        if result.get("success"):
            states = result.get("result", [])
            for state in states:
                if state["entity_id"].startswith("media_player."):
                    players.append({
                        "entity_id": state["entity_id"],
                        "name": state.get("attributes", {}).get("friendly_name"),
                        "state": state["state"],
                        "app_name": state.get("attributes", {}).get("app_name"),
                        "source": state.get("attributes", {}).get("source"),
                        "media_title": state.get("attributes", {}).get("media_title"),
                        "volume_level": state.get("attributes", {}).get("volume_level"),
                        "is_plex": "plex" in state["entity_id"].lower() or 
                                  state.get("attributes", {}).get("app_name", "").lower() == "plex"
                    })
        return players
    
    async def get_all_cameras(self) -> List[Dict[str, Any]]:
        """Get all camera entities."""
        command = {"type": "get_states"}
        result = await self._send_ws_command(command)
        
        cameras = []
        if result.get("success"):
            states = result.get("result", [])
            for state in states:
                if state["entity_id"].startswith("camera."):
                    cameras.append({
                        "entity_id": state["entity_id"],
                        "name": state.get("attributes", {}).get("friendly_name"),
                        "state": state["state"],
                        "brand": state.get("attributes", {}).get("brand"),
                        "model": state.get("attributes", {}).get("model"),
                        "is_streaming": state.get("attributes", {}).get("is_streaming", False)
                    })
        return cameras
    
    # ========== TTS and Notifications ==========
    # TODO: implement Selene's TTS or figure out how to add that to HA
    async def send_tts_message(self, media_player_entity: str,
                              message: str,
                              language: str = "en-US",
                              announce: bool = True) -> Dict[str, Any]:
        """Send a text-to-speech message to a media player."""
        return await self._call_service(
            "tts",
            "cloud_say",  # or "google_translate_say"
            service_data={
                "entity_id": media_player_entity,
                "message": message,
                "language": language,
                "options": {"announce": announce}
            }
        )
    
    async def send_image_notification(self, image_url: str,
                                     message: str,
                                     title: str = "Notification",
                                     target: str = None) -> Dict[str, Any]:
        """Send a notification with an image."""
        service_data = {
            "message": message,
            "title": title,
            "data": {"image": image_url}
        }
        
        return await self._call_service(
            "notify",
            target or "notify",
            service_data=service_data
        )
    
    # ========== Helper Methods ==========
    
    async def _find_plex_media_player(self) -> Optional[str]:
        """Auto-detect a Plex media player entity."""
        players = await self.get_all_media_players()
        for player in players:
            if player.get("is_plex"):
                return player["entity_id"]
        return None
    
    def _matches_media_type(self, item: Dict[str, Any], 
                           media_types: List[MediaType]) -> bool:
        """Check if an item matches any of the specified media types."""
        item_type = item.get("media_class", item.get("media_content_type", ""))
        for media_type in media_types:
            if media_type.value in item_type.lower():
                return True
        return False
    
    def _matches_search(self, item: Dict[str, Any], query: str) -> bool:
        """Check if an item matches a search query."""
        query_lower = query.lower()
        title = item.get("title", "").lower()
        
        if query_lower in title:
            return True
            
        for key in ["media_artist", "media_album_name", "media_series_title"]:
            if key in item and query_lower in str(item[key]).lower():
                return True
                
        return False
    
    def _matches_episode(self, item: Dict[str, Any], 
                        season: int = None, 
                        episode: int = None) -> bool:
        """Check if an item matches specific season/episode."""
        if season and item.get("media_season") != season:
            return False
        if episode and item.get("media_episode") != episode:
            return False
        return True
    
    # ========== Playlist Management ==========
    
    async def get_current_playlist(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get the current playlist/queue for a media player.
        
        Returns:
            List of items in the queue with metadata
        """
        browse_result = await self.browse_media(
            entity_id,
            media_content_type="playlist",
            media_content_id="current"
        )
        
        if browse_result.get("success"):
            return browse_result.get("result", {}).get("children", [])
        
        state = await self.get_entity_state(entity_id)
        if state:
            return state.get("attributes", {}).get("media_playlist", [])
            
        return []
    
    async def create_playlist(self, name: str, 
                            items: List[str],
                            media_player_entity: str = None) -> Dict[str, Any]:
        """
        Create a new playlist.
        
        Args:
            name: Playlist name
            items: List of media content IDs
            media_player_entity: Target media player
        """
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        return await self._call_service(
            "media_player",
            "create_playlist",
            service_data={
                "name": name,
                "items": items
            },
            target={"entity_id": media_player_entity}
        )
    
    # ========== Advanced Plex Features ==========
    
    async def get_plex_library_sections(self, 
                                       media_player_entity: str = None) -> List[Dict[str, Any]]:
        """
        Get all Plex library sections (Movies, TV Shows, Music, etc.).
        
        Returns:
            List of library sections with their IDs and types
        """
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        browse_result = await self.browse_media(
            media_player_entity,
            media_content_type="",
            media_content_id=""
        )
        
        sections = []
        if browse_result.get("success"):
            children = browse_result.get("result", {}).get("children", [])
            for child in children:
                sections.append({
                    "title": child.get("title"),
                    "library_id": child.get("media_content_id"),
                    "type": child.get("media_content_type"),
                    "can_play": child.get("can_play", False),
                    "can_expand": child.get("can_expand", True)
                })
                
        return sections
    
    async def get_plex_on_deck(self, media_player_entity: str = None) -> List[Dict[str, Any]]:
        """Get Plex 'On Deck' items (continue watching)."""
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        browse_result = await self.browse_media(
            media_player_entity,
            media_content_type="on_deck",
            media_content_id="on_deck"
        )
        
        if browse_result.get("success"):
            return browse_result.get("result", {}).get("children", [])
        return []
    
    async def mark_plex_watched(self, media_content_id: str,
                               media_player_entity: str = None) -> Dict[str, Any]:
        """Mark Plex content as watched."""
        if not media_player_entity:
            media_player_entity = await self._find_plex_media_player()
            
        return await self._call_service(
            "plex",
            "mark_watched",
            service_data={"media_content_id": media_content_id},
            target={"entity_id": media_player_entity}
        )
    
    # ========== Smart TV / Display Controls ==========
    
    async def turn_on_display(self, entity_id: str) -> Dict[str, Any]:
        """Turn on a media player/display."""
        return await self._call_service(
            "media_player",
            "turn_on",
            target={"entity_id": entity_id}
        )
    
    async def turn_off_display(self, entity_id: str) -> Dict[str, Any]:
        """Turn off a media player/display."""
        return await self._call_service(
            "media_player",
            "turn_off",
            target={"entity_id": entity_id}
        )
    
    async def cast_dashboard(self, media_player_entity: str,
                           dashboard_path: str,
                           view_path: str = None) -> Dict[str, Any]:
        """Cast a Home Assistant dashboard to a display."""
        url = f"{self.base_url}/{dashboard_path}"
        if view_path:
            url += f"/{view_path}"
            
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=url,
            media_content_type="url"
        )
    
    # ========== Chromecast Specific ==========
    
    async def cast_youtube(self, media_player_entity: str,
                          video_id: str) -> Dict[str, Any]:
        """Cast a YouTube video to a Chromecast device."""
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=video_id,
            media_content_type="cast",
            extra={"app_name": "youtube"}
        )
    
    async def cast_spotify(self, media_player_entity: str,
                          spotify_uri: str) -> Dict[str, Any]:
        """Cast Spotify content to a device."""
        return await self.play_media(
            media_player_entity=media_player_entity,
            media_content_id=spotify_uri,
            media_content_type="music",
            extra={"app_name": "spotify"}
        )


# ========== Example Usage and Tool Definitions ==========

async def example_usage():
    """Comprehensive examples of using the media control class."""
    
    ha = HomeAssistantMediaControl(
        host="192.168.1.100",
        token="YOUR_LONG_LIVED_TOKEN",
        use_ssl=False
    )
    
    try:
        await ha.connect()
        
        print("\n=== Discovering Media Players ===")
        players = await ha.get_all_media_players()
        for player in players:
            print(f"Found: {player['name']} ({player['entity_id']}) - State: {player['state']}")
            if player['media_title']:
                print(f"  Currently playing: {player['media_title']}")
        
        print("\n=== Media Player Status ===")
        status = await ha.get_media_player_status("media_player.living_room_tv")
        print(f"State: {status['state']}")
        print(f"Playing: {status['media_title']}")
        print(f"Position: {status['media_position']}/{status['media_duration']} seconds")
        
        print("\n=== Searching Plex ===")
        
        movie_results = await ha.search_plex_media(
            query="Inception",
            media_types=[MediaType.MOVIE]
        )
        if movie_results:
            print(f"Found movie: {movie_results[0]['title']}")
            await ha.play_plex_media(
                media_player_entity="media_player.living_room_tv",
                media_content_id=movie_results[0]['media_content_id']
            )
        
        show_result = await ha.search_plex_by_type(
            media_type=MediaType.EPISODE,
            show="The Office",
            season=2,
            episode=1
        )
        if show_result:
            print(f"Found episode: {show_result['title']}")
        
        music_results = await ha.search_plex_by_type(
            media_type=MediaType.TRACK,
            artist="Pink Floyd",
            album="Dark Side of the Moon",
            title="Time"
        )
        
        print("\n=== Controlling Playback ===")
        
        await ha.pause("media_player.living_room_tv")
        await asyncio.sleep(2)
        
        await ha.play("media_player.living_room_tv")
        
        await ha.next_track("media_player.living_room_tv")
        
        await ha.seek("media_player.living_room_tv", 600)
        
        await ha.set_volume("media_player.living_room_tv", 0.5)
        
        print("\n=== Casting AI Image ===")
        await ha.cast_image_url(
            media_player_entity="media_player.kitchen_display",
            image_url="http://192.168.1.50:8080/ai_generated.png",
            title="Today's AI Art",
            duration=30  # Display for 30 seconds
        )
        
        print("\n=== Streaming Camera ===")
        cameras = await ha.get_all_cameras()
        if cameras:
            await ha.stream_camera(
                media_player_entity="media_player.bedroom_tv",
                camera_entity=cameras[0]['entity_id']
            )
        
        print("\n=== Recently Added to Plex ===")
        recent = await ha.get_plex_recently_added(limit=5)
        for item in recent:
            print(f"- {item['title']}")
        
        print("\n=== Plex Libraries ===")
        sections = await ha.get_plex_library_sections()
        for section in sections:
            print(f"- {section['title']} ({section['type']})")
        
    finally:
        await ha.disconnect()

if __name__ == "__main__":
    asyncio.run(example_usage())