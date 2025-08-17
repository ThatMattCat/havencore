"""
MediaControllerWrapper - Simplified interface for AI tool-calling
Consolidates multiple operations into logical grouped functions
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from datetime import datetime
import traceback

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from haos.lib_ha_media_tools import HomeAssistantMediaControl, MediaType
import config
import shared.scripts.logger as logger_module

logger = logger_module.get_logger('loki')


class ActionType(Enum):
    """Consolidated action types for media control."""

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
    
    QUEUE_ADD = "queue_add"
    QUEUE_NEXT = "queue_next"
    QUEUE_CLEAR = "queue_clear"


class HaMediaController:
    """
    Fixed wrapper with better connection management
    """
    
    def __init__(self, host: str = None, token: str = None, port: int = 8123, use_ssl: bool = False):
        """Initialize the wrapper with Home Assistant connection."""
        self.ha = HomeAssistantMediaControl(host, token, port, use_ssl)
        self.logger = logging.getLogger(__name__)
        self._device_cache = {}
        self._cache_timestamp = None
        self._cache_duration = 300  # 5 minutes
        
    async def connect(self):
        """Establish connection to Home Assistant."""
        connected = await self.ha.connect()
        if not connected:
            raise ConnectionError("Failed to connect to Home Assistant")
        logger.debug(f"Connect called. Connected: {connected}")
        return connected

    async def disconnect(self):
        """Close connection to Home Assistant."""
        await self.ha.disconnect()
        logger.debug("Disconnect called")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    # ========== CONSOLIDATED FUNCTIONS FOR AI TOOL-CALLING ==========
    
    async def control_media(self,
                           action: str,
                           device: str = None,
                           value: Any = None,
                           target_device: str = None) -> Dict[str, Any]:
        """
        Universal media control function - handles all playback, volume, and power actions.
        
        Args:
            action: Action from ActionType enum (play, pause, volume_set, etc.)
            device: Device entity ID or friendly name (auto-detects if None)
            value: Optional value for actions like seek position, volume level, source name
            target_device: For queue operations, the device to add content to
            
        Returns:
            Success status and any relevant data
            
        Examples:
            control_media("play", "living_room_tv")
            control_media("volume_set", "bedroom_speaker", value=50)
            control_media("seek", "living_room_tv", value=600)  # Seek to 10 minutes
            control_media("select_source", "living_room_tv", value="HDMI 1")
        """
        await self.connect()
        logger.debug(f"Executing action: {action} on media device: {device} with value: {value} and target_device: {target_device}")

        entity_id = await self._resolve_device(device)
        if not entity_id:
            return {"success": False, "error": f"Device '{device}' not found"}
        
        try:
            action_type = ActionType(action.lower())
            
            if action_type == ActionType.PLAY:
                result = await self.ha.play(entity_id)
            elif action_type == ActionType.PAUSE:
                result = await self.ha.pause(entity_id)
            elif action_type == ActionType.STOP:
                result = await self.ha.stop(entity_id)
            elif action_type == ActionType.TOGGLE:
                result = await self.ha.play_pause_toggle(entity_id)
            elif action_type == ActionType.NEXT:
                result = await self.ha.next_track(entity_id)
            elif action_type == ActionType.PREVIOUS:
                result = await self.ha.previous_track(entity_id)
            elif action_type == ActionType.SEEK:
                if value is None:
                    return {"success": False, "error": "Seek position required"}
                result = await self.ha.seek(entity_id, int(value))
            elif action_type == ActionType.SHUFFLE:
                shuffle_on = value if isinstance(value, bool) else True
                result = await self.ha.set_shuffle(entity_id, shuffle_on)
            elif action_type == ActionType.REPEAT:
                repeat_mode = value if value in ["off", "all", "one"] else "all"
                result = await self.ha.set_repeat(entity_id, repeat_mode)
                
            elif action_type == ActionType.VOLUME_SET:
                if value is None:
                    return {"success": False, "error": "Volume level required (0-100)"}
                # Convert percentage to 0.0-1.0 range
                volume = float(value) / 100 if value > 1 else float(value)
                result = await self.ha.set_volume(entity_id, volume)
            elif action_type == ActionType.VOLUME_UP:
                result = await self.ha.volume_up(entity_id)
            elif action_type == ActionType.VOLUME_DOWN:
                result = await self.ha.volume_down(entity_id)
            elif action_type == ActionType.MUTE:
                result = await self.ha.mute(entity_id, True)
            elif action_type == ActionType.UNMUTE:
                result = await self.ha.mute(entity_id, False)
                
            elif action_type == ActionType.TURN_ON:
                result = await self.ha.turn_on_display(entity_id)
            elif action_type == ActionType.TURN_OFF:
                result = await self.ha.turn_off_display(entity_id)
                
            elif action_type == ActionType.SELECT_SOURCE:
                if value is None:
                    return {"success": False, "error": "Source name required"}
                result = await self.ha.select_source(entity_id, value)
                
            elif action_type == ActionType.QUEUE_CLEAR:
                result = await self.ha.clear_playlist(entity_id)
            else:
                return {"success": False, "error": f"Unsupported action: {action}"}
                
            return {
                "success": result.get("success", True),
                "action": action,
                "device": entity_id,
                "value": value
            }
            
        except Exception as e:
            self.logger.error(f"Control media error: {e}")
            return {"success": False, "error": str(e)}
    
    async def play_content(self,
                          content: str,
                          device: str = None,
                          content_type: str = "auto",
                          search_type: str = None,
                          enqueue: bool = False,
                          **kwargs) -> Dict[str, Any]:
        """
        Universal content playback function - handles all media types.
        
        Args:
            content: Content to play (search query, URL, content ID, camera name, etc.)
            device: Target device (auto-detects if None)
            content_type: "auto", "search", "url", "camera", "image", "tts", "plex_id"
            search_type: When searching - "movie", "show", "episode", "music", "artist", "album"
            enqueue: Add to queue instead of playing immediately
            **kwargs: Additional parameters (season, episode, duration for images, etc.)
            
        Returns:
            Success status and playback details
            
        Examples:
            play_content("Inception", device="living_room_tv")
            play_content("The Office", device="bedroom_tv", search_type="show", season=2, episode=1)
            play_content("front door camera", device="kitchen_display", content_type="camera")
            play_content("https://example.com/image.jpg", device="hallway_display", content_type="image", duration=30)
            play_content("Good morning! Today's weather is sunny.", device="bedroom_speaker", content_type="tts")
        """
        await self.connect()
        logger.debug(f"Playing content: {content} on device: {device} with type: {content_type} and search_type: {search_type}")
        
        entity_id = await self._resolve_device(device)
        if not entity_id:
            entity_id = await self._auto_detect_device(content_type)
            if not entity_id:
                return {"success": False, "error": "No suitable device found"}
        
        try:
            if content_type == "auto":
                content_type = self._detect_content_type(content)
            
            if content_type == "tts":
                result = await self.ha.send_tts_message(
                    entity_id,
                    content,
                    language=kwargs.get("language", "en-US"),
                    announce=True
                )
                
            elif content_type == "camera":
                camera_entity = await self._resolve_camera(content)
                if not camera_entity:
                    return {"success": False, "error": f"Camera '{content}' not found"}
                result = await self.ha.stream_camera(entity_id, camera_entity)
                
            elif content_type == "image":
                duration = kwargs.get("duration", 30)
                title = kwargs.get("title", "Image")
                result = await self.ha.cast_image_url(entity_id, content, title, duration)
                
            elif content_type == "url":
                result = await self.ha.play_media(
                    entity_id,
                    media_content_id=content,
                    media_content_type="video" if any(ext in content for ext in ['.mp4', '.avi', '.mkv']) else "audio"
                )
                
            elif content_type == "plex_id":
                result = await self.ha.play_media(
                    entity_id,
                    media_content_id=content,
                    media_content_type=kwargs.get("media_type", "video")
                )

            elif content_type == "plex_video":
                result = await self.ha.play_media(
                    entity_id,
                    media_content_id=f"plex://{{\"library_name\": \"Movies\", \"title\": \"{content}\"}}",
                    media_content_type="video"
                )
                
            elif content_type == "search" or search_type:
                result = await self._search_and_play(
                    entity_id,
                    content,
                    search_type=search_type,
                    enqueue=enqueue,
                    **kwargs
                )


            else:
                result = await self._search_and_play(
                    entity_id,
                    content,
                    search_type=search_type if search_type else content_type,
                    enqueue=enqueue,
                    **kwargs
                )
                
            return result
            
        except Exception as e:
            self.logger.error(f"Play content error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_media_player_statuses(self, 
                        device: str = None,
                        info_type: str = "full") -> Dict[str, Any]:
        """
        Get status information about devices or currently playing media.
        
        Args:
            device: Specific device or "all" for all devices
            info_type: "full", "playing", "devices", "current"
            
        Returns:
            Status information based on request type
            
        Examples:
            get_media_player_statuses()  # Get all media players and their status
            get_media_player_statuses("living_room_tv")  # Get specific device status
            get_media_player_statuses(info_type="playing")  # Get all currently playing devices
        """
        await self.connect()
        logger.debug(f"Getting media player statuses for device: {device} with info_type: {info_type}")
        
        try:
            if device and device.lower() != "all":
                entity_id = await self._resolve_device(device)
                if not entity_id:
                    return {"success": False, "error": f"Device '{device}' not found"}
                    
                status = await self.ha.get_media_player_status(entity_id)
                
                if info_type == "full":
                    state = await self.ha.get_entity_state(entity_id)
                    if state:
                        status["supported_features"] = state.get("attributes", {}).get("supported_features")
                        status["source_list"] = state.get("attributes", {}).get("source_list")
                        
                return {"success": True, "device": entity_id, "status": status}
                
            else:
                players = await self.ha.get_all_media_players()
                
                if info_type == "playing":
                    if players:
                        players = [p for p in players if p.get("state", "") in ["playing", "paused"]]

                elif info_type == "devices":
                    return {
                        "success": True,
                        "devices": [
                            {
                                "id": p.get("entity_id", ""),
                                "name": p.get("name", ""),
                                "state": p.get("state", ""),
                                "type": "plex" if p.get("is_plex") else "generic"
                            }
                            for p in players
                        ]
                    }
                    
                return {"success": True, "players": players}
                
        except Exception as e:
            self.logger.error(f"Get status error: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_media(self,
                          query: str,
                          media_type: str = None,
                          limit: int = 5,
                          source: str = "plex") -> Dict[str, Any]:
        """
        Search for media content across different sources.
        
        Args:
            query: Search query
            media_type: "movie", "show", "episode", "music", "artist", "album", etc.
            limit: Maximum results
            source: "plex", "youtube", "spotify" (currently only Plex implemented)
            
        Returns:
            Search results with content IDs and metadata
            
        Examples:
            search_media("Inception")
            search_media("The Office", media_type="show")
            search_media("Pink Floyd", media_type="artist")
        """
        await self.connect()

        logger.debug(f"Searching media with query: {query}, media_type: {media_type}, limit: {limit}, source: {source}")
        try:
            if source.lower() == "plex":
                media_types = None
                if media_type:
                    if media_type.lower() in [e.value for e in MediaType]:
                        media_types = [MediaType(media_type.lower())]

                results = await self.ha.search_plex_media(
                    query=query,
                    media_types=media_types,
                    limit=limit
                )
                
                return {
                    "success": True,
                    "source": source,
                    "query": query,
                    "results": results,
                    "count": len(results)
                }
            else:
                return {"success": False, "error": f"Source '{source}' not implemented"}
                
        except Exception as e:
            self.logger.error(f"Search media error: {e} - {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    async def manage_queue(self,
                          action: str,
                          device: str = None,
                          content: str = None,
                          position: str = "end") -> Dict[str, Any]:
        """
        Manage playback queue/playlist.
        
        Args:
            action: "add", "remove", "clear", "show", "save"
            device: Target device
            content: Content to add (for "add" action)
            position: "end", "next" (for "add" action)
            
        Returns:
            Queue operation result
            
        Examples:
            manage_queue("show", "living_room_tv")
            manage_queue("add", "living_room_tv", "Inception", position="next")
            manage_queue("clear", "living_room_tv")
        """
        await self.connect()

        logger.debug(f"Manage queue called with action: {action}, device: {device}, content: {content}, position: {position}")
        entity_id = await self._resolve_device(device)
        if not entity_id:
            return {"success": False, "error": f"Device '{device}' not found"}
        
        try:
            if action == "show":
                queue = await self.ha.get_current_playlist(entity_id)
                return {"success": True, "queue": queue, "count": len(queue)}
                
            elif action == "clear":
                result = await self.ha.clear_playlist(entity_id)
                return {"success": result.get("success", True), "action": "cleared"}
                
            elif action == "add" and content:
                enqueue = "next" if position == "next" else "add"
                
                search_results = await self.ha.search_plex_media(content, limit=1)
                if not search_results:
                    return {"success": False, "error": f"Content '{content}' not found"}
                    
                result = await self.ha.queue_media(
                    entity_id,
                    search_results[0]["media_content_id"],
                    search_results[0]["media_content_type"],
                    enqueue=enqueue
                )
                return {"success": result.get("success", True), "added": content, "position": position}
                
            else:
                return {"success": False, "error": f"Invalid queue action: {action}"}
                
        except Exception as e:
            self.logger.error(f"Manage queue error: {e}")
            return {"success": False, "error": str(e)}
    
    # ========== HELPER METHODS ==========
    
    async def _resolve_device(self, device: str) -> Optional[str]:
        """Resolve device name to entity_id."""
        if not device:
            return None
            
        if device.startswith("media_player."):
            return device
            
        players = await self.ha.get_all_media_players()
        device_lower = device.lower()
        for player in players:
            if device_lower in player["entity_id"].lower() and player.get("state","") in ["playing", "paused"]:
                return player["entity_id"]
            if player.get("name") and device_lower in player["name"].lower() and player.get("state","") in ["playing", "paused"]:
                return player["entity_id"]
        for player in players:
            if device_lower in player["entity_id"].lower():
                return player["entity_id"]
            if player.get("name") and device_lower in player["name"].lower():
                return player["entity_id"]
                
        return None
    
    async def _resolve_camera(self, camera: str) -> Optional[str]:
        """Resolve camera name to entity_id."""
        if camera.startswith("camera."):
            return camera
            
        cameras = await self.ha.get_all_cameras()
        camera_lower = camera.lower()
        
        for cam in cameras:
            if camera_lower in cam["entity_id"].lower():
                return cam["entity_id"]
            if cam.get("name") and camera_lower in cam["name"].lower():
                return cam["entity_id"]
                
        return None
    
    async def _auto_detect_device(self, content_type: str) -> Optional[str]:
        """Auto-detect the best device for content type."""
        players = await self.ha.get_all_media_players()
        # TODO: Implement more sophisticated device detection logic
        available = [p for p in players if p["state"] in ["idle", "off", "standby"]]
        if not available:
            available = players
            
        if content_type in ["search", "plex_id", "plex_video"]:
            plex_players = [p for p in available if p.get("is_plex")]
            if plex_players:
                return plex_players[0]["entity_id"]
                
        return available[0]["entity_id"] if available else None
    
    def _detect_content_type(self, content: str) -> str:
        """Auto-detect content type from content string."""
        content_lower = content.lower()
        
        if content.startswith(("http://", "https://", "rtsp://")):
            if any(ext in content_lower for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                return "image"
            return "url"
            
        if any(word in content_lower for word in ["camera", "cam", "surveillance"]):
            return "camera"
            
        return "search"
    
    async def _search_and_play(self, 
                              entity_id: str,
                              query: str,
                              search_type: str = None,
                              enqueue: bool = False,
                              **kwargs) -> Dict[str, Any]:
        """Internal method to search and play content."""

        if search_type in ["show", "episode"] and "season" in kwargs:
            result = await self.ha.search_plex_by_type(
                media_type=MediaType.EPISODE,
                show=query,
                season=kwargs.get("season"),
                episode=kwargs.get("episode"),
                media_player_entity=entity_id
            )
            
            if result:
                if enqueue:
                    play_result = await self.ha.queue_media(
                        entity_id,
                        result["media_content_id"],
                        result["media_content_type"],
                        enqueue="add"
                    )
                else:
                    play_result = await self.ha.play_media(
                        entity_id,
                        result["media_content_id"],
                        result["media_content_type"]
                    )
                    
                return {
                    "success": play_result.get("success", True),
                    "playing": result.get("title"),
                    "type": "episode"
                }

        else:
            media_type = None
            if search_type:
                type_map = {
                    "movie": MediaType.MOVIE,
                    "music": MediaType.TRACK,
                    "artist": MediaType.ARTIST,
                    "album": MediaType.ALBUM
                }
                media_type = type_map.get(search_type)
                
            results = await self.ha.search_plex_media(
                query=query,
                media_player_entity=entity_id,
                media_types=[media_type] if media_type else None,
                limit=1
            )
            
            if not results:
                return {"success": False, "error": f"No results found for '{query}'"}
                
            # Play first result
            content = results[0]
            if enqueue:
                play_result = await self.ha.queue_media(
                    entity_id,
                    content["media_content_id"],
                    content["media_content_type"],
                    enqueue="add"
                )
            else:
                play_result = await self.ha.play_media(
                    entity_id,
                    content["media_content_id"],
                    content["media_content_type"]
                )
                
            return {
                "success": play_result.get("success", True),
                "playing": content.get("title"),
                "type": content.get("media_content_type")
            }


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Minimal set of tool definitions for AI function calling.
    Only 6 main functions to cover all functionality.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "control_media",
                "description": "Control media playback, volume, power, and sources on any device",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["play", "pause", "stop", "toggle", "next", "previous", 
                                    "seek", "shuffle", "repeat", "volume_set", "volume_up", 
                                    "volume_down", "mute", "unmute", "turn_on", "turn_off", 
                                    "select_source"],
                            "description": "Action to perform"
                        },
                        "device": {
                            "type": "string",
                            "description": "Device name or entity_id (optional, auto-detects)"
                        },
                        "value": {
                            "type": ["number", "string", "boolean"],
                            "description": "Value for the action (volume level 0-100, seek position in seconds, source name, etc.)"
                        }
                    },
                    "required": ["action"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_media_player_statuses",
                "description": "Get status of media players or currently playing content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device": {
                            "type": "string",
                            "description": "Specific device or 'all' for all devices"
                        },
                        "info_type": {
                            "type": "string",
                            "enum": ["full", "playing", "devices", "current"],
                            "description": "Type of information to return"
                        }
                    }
                }
            }
        }#,
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "search_media",
        #         "description": "Search for movies, TV shows, music, etc. in media libraries",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "query": {
        #                     "type": "string",
        #                     "description": "Search query"
        #                 },
        #                 "media_type": {
        #                     "type": "string",
        #                     "enum": ["movie", "show", "episode", "music", "artist", "album"],
        #                     "description": "Type of media to search for"
        #                 },
        #                 "limit": {
        #                     "type": "integer",
        #                     "description": "Maximum number of results (default: 10)"
        #                 },
        #                 "source": {
        #                     "type": "string",
        #                     "enum": ["plex", "youtube", "spotify"],
        #                     "description": "Media source (default: plex)"
        #                 }
        #             },
        #             "required": ["query"]
        #         }
        #     }
        # },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "play_content",
        #         "description": "Play any type of content (movies, shows, music, cameras, images, URLs, or text-to-speech)",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "content": {
        #                     "type": "string",
        #                     "description": "Content to play (search query, URL, camera name, or text for TTS)"
        #                 },
        #                 "device": {
        #                     "type": "string",
        #                     "description": "Target device (optional, auto-detects)"
        #                 },
        #                 "content_type": {
        #                     "type": "string",
        #                     "enum": ["auto", "search", "url", "camera", "image", "tts", "plex_id"],
        #                     "description": "Type of content (default: auto)"
        #                 },
        #                 "search_type": {
        #                     "type": "string",
        #                     "enum": ["movie", "show", "episode", "music", "artist", "album"],
        #                     "description": "Type when searching media"
        #                 },
        #                 "season": {
        #                     "type": "integer",
        #                     "description": "Season number for TV episodes"
        #                 },
        #                 "episode": {
        #                     "type": "integer",
        #                     "description": "Episode number for TV episodes"
        #                 },
        #                 "duration": {
        #                     "type": "integer",
        #                     "description": "Display duration in seconds (for images)"
        #                 },
        #                 "enqueue": {
        #                     "type": "boolean",
        #                     "description": "Add to queue instead of playing immediately"
        #                 }
        #             },
        #             "required": ["content"]
        #         }
        #     }
        # },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "manage_queue",
        #         "description": "Manage the playback queue or playlist",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "action": {
        #                     "type": "string",
        #                     "enum": ["add", "remove", "clear", "show"],
        #                     "description": "Queue management action"
        #                 },
        #                 "device": {
        #                     "type": "string",
        #                     "description": "Target device"
        #                 },
        #                 "content": {
        #                     "type": "string",
        #                     "description": "Content to add to queue (for 'add' action)"
        #                 },
        #                 "position": {
        #                     "type": "string",
        #                     "enum": ["end", "next"],
        #                     "description": "Where to add in queue (default: end)"
        #                 }
        #             },
        #             "required": ["action"]
        #         }
        #     }
        # }
    ]


# ========== Example Usage ==========

async def example_wrapper_usage():
    """Examples of using the wrapper class."""
    
    # Initialize wrapper
    controller = HaMediaController(
        host="192.168.1.100",
        token="YOUR_TOKEN"
    )
    
    try:
        await controller.control_media("pause", "living_room_tv")
        
        await controller.play_content(
            "Inception",
            device="living_room_tv"
        )
        
        await controller.play_content(
            "The Office",
            device="bedroom_tv",
            search_type="episode",
            season=2,
            episode=1
        )
        
        await controller.play_content(
            "front door camera",
            device="kitchen_display",
            content_type="camera"
        )
        
        await controller.play_content(
            "Good morning! The weather today is sunny with a high of 75 degrees.",
            device="bedroom_speaker",
            content_type="tts"
        )
        
        await controller.control_media(
            "volume_set",
            "living_room_tv",
            value=50
        )
        
        status = await controller.get_media_player_statuses(info_type="playing")
        logger.debug(f"Currently playing on {len(status['players'])} devices")
        
        results = await controller.search_media(
            "Star Wars",
            media_type="movie"
        )
        print(f"Found {results['count']} Star Wars movies")

        await controller.manage_queue(
            "add",
            "living_room_tv",
            "The Matrix",
            position="next"
        )
        
    finally:
        await controller.disconnect()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_wrapper_usage())