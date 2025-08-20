import asyncio
import websockets
import json
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import re
import logging

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import shared.scripts.logger as logger_module
logger = logger_module.get_logger('loki')
import config



class MediaType(Enum):
    """Media type enumeration"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PLAYLIST = "playlist"
    UNKNOWN = "unknown"

@dataclass
class MediaItem:
    """Represents a playable media item with metadata"""
    media_id: str
    title: str
    media_type: MediaType
    server_id: str
    server_name: str
    path: List[str]
    compatible_devices: Set[str] = field(default_factory=set)
    
    # Optional metadata
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[int] = None  # in seconds
    resolution: Optional[str] = None
    file_size: Optional[int] = None
    date_added: Optional[datetime] = None
    media_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['media_type'] = self.media_type.value
        d['compatible_devices'] = list(self.compatible_devices)
        if self.date_added:
            d['date_added'] = self.date_added.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediaItem':
        """Create from dictionary"""
        data = data.copy()
        data['media_type'] = MediaType(data['media_type'])
        data['compatible_devices'] = set(data.get('compatible_devices', []))
        if data.get('date_added'):
            data['date_added'] = datetime.fromisoformat(data['date_added'])
        return cls(**data)

@dataclass
class MediaServer:
    """Represents a DLNA server"""
    server_id: str
    name: str
    media_count: int = 0
    last_scan: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'server_id': self.server_id,
            'name': self.name,
            'media_count': self.media_count,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None
        }

@dataclass
class PlaybackDevice:
    """Represents a media player device used for playback"""
    entity_id: str
    name: str
    supported_types: Set[MediaType] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'supported_types': [t.value for t in self.supported_types]
        }

class HAMediaLibrary:
    """Main class for managing Home Assistant media library"""
    
    def __init__(self, ha_url: str, ha_token: str):
        """
        Initialize the Home Assistant Media Library

        Args:
            ha_url: Home Assistant WebSocket URL
            ha_token: Home Assistant long-lived access token
        """
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.websocket = None
        self.message_id = 1
        self._connected = False
        
        # Data storage
        self.media_items: Dict[str, MediaItem] = {}  # media_id -> MediaItem
        self.servers: Dict[str, MediaServer] = {}  # server_id -> MediaServer
        self.devices: Dict[str, PlaybackDevice] = {}  # entity_id -> PlaybackDevice

        # Indexing for fast queries
        self.title_index: Dict[str, List[str]] = {}  # lowercase title -> [media_ids]
        self.type_index: Dict[MediaType, List[str]] = {t: [] for t in MediaType}
        self.server_index: Dict[str, List[str]] = {}  # server_id -> [media_ids]
        self.genre_index: Dict[str, List[str]] = {}  # genre -> [media_ids]
        
        # Scan tracking
        self.visited_ids: Set[str] = set()
        self.scan_stats = {
            'total_scanned': 0,
            'folders_explored': 0,
            'errors': 0,
            'last_scan_duration': 0
        }
    
    async def connect(self):
        """Connect to Home Assistant WebSocket API"""

        # TODO: Migrate to aiohttp ClientConnection (reference lib_ha_media_tools)
        self.websocket = await websockets.connect(self.ha_url)
        
        auth_msg = await self.websocket.recv()
        auth_data = json.loads(auth_msg)
        
        auth = {
            "type": "auth",
            "access_token": self.ha_token
        }
        await self.websocket.send(json.dumps(auth))
        
        auth_result = await self.websocket.recv()
        result_data = json.loads(auth_result)
        self._connected = True
        if result_data["type"] != "auth_ok":
            self._connected = False
            raise Exception(f"Authentication failed: {auth_result}")
    
    async def disconnect(self):
        """Disconnect from Home Assistant"""
        if self.websocket:
            await self.websocket.close()
            self._connected = False

    async def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        return self._connected and self.websocket

    async def _send_ws_command(self, command: Dict[str, Any], 
                              timeout: int = 30) -> Dict[str, Any]:
        """
        Send command with timeout and reconnection logic.
        """
        # TODO: Replace with aiohttp and enhanced response logic checks/etc
        # Ensure we're connected
        if not await self.is_connected():
            await self.connect()
        
        command["id"] = self.message_id
        self.message_id += 1

        try:
            await self.websocket.send(json.dumps(command))
            response = await self.websocket.recv()
            return json.loads(response)

        except Exception as e:
            logger.error(f"Unexpected error sending command: {e}")
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

### Media Library Extraction and Interaction ###

    async def _browse_media(self, entity_id: str, media_content_type: Optional[str] = None, 
                           media_content_id: Optional[str] = None) -> Dict[str, Any]:
        """Internal method to browse media"""
        message = {
            "type": "media_player/browse_media",
            "entity_id": entity_id
        }
        
        if media_content_id is not None:
            message["media_content_id"] = media_content_id
            message["media_content_type"] = media_content_type if media_content_type is not None else ""
        elif media_content_type is not None:
            message["media_content_type"] = media_content_type
            message["media_content_id"] = ""
        
        return await self._send_ws_command(message)
        # await self.websocket.send(json.dumps(message))
        # self.message_id += 1
        
        # response = await self.websocket.recv()
        # return json.loads(response)
    
    def _determine_media_type(self, item: Dict[str, Any]) -> MediaType:
        """Determine media type from item data"""
        media_content_type = item.get("media_content_type", "").lower()
        media_class = item.get("media_class", "").lower()
        title = item.get("title", "").lower()
        
        video_types = ["video","movie","episode"]
        image_types = ["image","photo","picture"]
        audio_types = ["audio","music","song"]
        if any(vid in media_content_type for vid in video_types):
            return MediaType.VIDEO
        if any(aud in media_content_type for aud in audio_types):
            return MediaType.AUDIO
        if any(img in media_content_type for img in image_types):
            return MediaType.IMAGE
        elif "playlist" in media_content_type:
            return MediaType.PLAYLIST
        
        video_classes = ["video","movie","tv"]
        audio_classes = ["music","audio"]
        image_classes = ["image","photo"]
        if any(vid in media_class for vid in video_classes):
            return MediaType.VIDEO
        if any(aud in media_class for aud in audio_classes):
            return MediaType.AUDIO
        if any(img in media_class for img in image_classes):
            return MediaType.IMAGE
        
        video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
        audio_exts = ['.mp3', '.flac', '.wav', '.m4a', '.ogg', '.wma', '.aac']
        image_exts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
        
        for ext in video_exts:
            if title.endswith(ext):
                return MediaType.VIDEO
        for ext in audio_exts:
            if title.endswith(ext):
                return MediaType.AUDIO
        for ext in image_exts:
            if title.endswith(ext):
                return MediaType.IMAGE
        
        return MediaType.UNKNOWN
    
    def _extract_metadata(self, item: Dict[str, Any], path: List[str]) -> Dict[str, Any]:
        """Extract metadata from item and path"""
        metadata = {}
        
        # Try to extract from path
        path_str = " ".join(path).lower()
        
        # Extract year (look for 4-digit years)
        year_match = re.search(r'\b(19|20)\d{2}\b', " ".join(path))
        if year_match:
            metadata['year'] = int(year_match.group())
        
        # Extract resolution
        res_match = re.search(r'\b(1080p|720p|4k|2160p|480p)\b', path_str)
        if res_match:
            metadata['resolution'] = res_match.group().upper()
        
        # Extract genre from path (common folder names)
        # TODO: Audio has genres too..could implement all of this in directed search instead
        genres = ['action', 'comedy', 'drama', 'horror', 'sci-fi', 'romance', 
                  'thriller', 'documentary', 'animation', 'fantasy', 'adventure']
        for genre in genres:
            if genre in path_str:
                metadata['genre'] = genre.title()
                break
        
        # Get media class
        if item.get('media_class'):
            metadata['media_class'] = item.get('media_class')
        
        return metadata

    async def _scan_recursive(self, entity_id: str, device: PlaybackDevice, 
                             server: MediaServer, item: Dict[str, Any], 
                             path: List[str], max_depth: int, current_depth: int,
                             max_items_per_level: Optional[int]) -> None:
        """Recursively scan for media items"""
        if current_depth >= max_depth:
            return
        
        item_id = item.get("media_content_id", "")
        item_title = item.get("title", "Unknown")
        
        if item_id in self.visited_ids:
            return
        
        if item_id:
            self.visited_ids.add(item_id)
        
        self.scan_stats['total_scanned'] += 1
        
        # Check if playable
        if item.get("can_play", False):
            media_type = self._determine_media_type(item)
            metadata = self._extract_metadata(item, path)
            compat_devices = {device.entity_id}
            if self.media_items.get(item_id):
                compat_devices.add(self.media_items[item_id].compatible_devices)
    
            media_item = MediaItem(
                media_id=item_id,
                title=item_title,
                media_type=media_type,
                server_id=server.server_id,
                server_name=server.name,
                path=path.copy(),
                compatible_devices=compat_devices,
                **metadata
            )
            
            # Store and index
            self.media_items[item_id] = media_item
            
            # Update indices
            title_lower = item_title.lower()
            if title_lower not in self.title_index:
                self.title_index[title_lower] = []
            self.title_index[title_lower].append(item_id)
            
            self.type_index[media_type].append(item_id)
            
            if server.server_id not in self.server_index:
                self.server_index[server.server_id] = []
            self.server_index[server.server_id].append(item_id)
            
            if metadata.get('genre'):
                genre = metadata['genre'].lower()
                if genre not in self.genre_index:
                    self.genre_index[genre] = []
                self.genre_index[genre].append(item_id)

            #logger.debug(f"  {'  ' * current_depth}✅ Found {media_type.value}: {item_title}")

        # Check if expandable
        if item.get("can_expand", False):
            #logger.debug(f"  {'  ' * current_depth}📁 Exploring: {item_title}")
            self.scan_stats['folders_explored'] += 1
            
            try:
                result = await self._browse_media(
                    entity_id=entity_id,
                    media_content_type="",
                    media_content_id=item_id
                )
                
                if result.get("success"):
                    content_data = result.get("result", {})
                    children = content_data.get("children", [])
                    
                    if children:
                        items_to_explore = children[:max_items_per_level] if max_items_per_level else children
                        new_path = path + [item_title]
                        
                        for child in items_to_explore:
                            await self._scan_recursive(
                                entity_id, device, server, child, new_path,
                                max_depth, current_depth + 1, max_items_per_level
                            )
                    
            except Exception as e:
                self.scan_stats['errors'] += 1
                #logger.debug(f"  {'  ' * current_depth}❌ Error: {e}")
    
    async def scan_cameras(self, device_entity_ids: Optional[List[str]] = None,
                             max_depth: int = 10, 
                             max_items_per_level: Optional[int] = None) -> None:
        """
        Scan for available Cameras
        
        Args:
            device_entity_ids: List of device entity IDs to scan (None = scan all)
            max_depth: Maximum folder depth to scan
            max_items_per_level: Maximum items per folder (None = all)
        """
        start_time = datetime.now()

        cam_device = MediaServer(
            server_id="media-source://camera",
            name="Cameras"
        )

        if not device_entity_ids:
            raise ValueError("Please provide device_entity_ids")
        
        # logger.debug(f"\n🔍 Starting comprehensive media scan")
        # logger.debug(f"   Devices: {len(device_entity_ids)}")
        # logger.debug(f"   Max depth: {max_depth}")
        # logger.debug(f"   Max items per level: {max_items_per_level or 'All'}")
        # logger.debug("=" * 50)
        
        for device_id in device_entity_ids:
            logger.info(f"📱 Scanning device: {device_id}")

            device = PlaybackDevice(
                entity_id=device_id,
                name=device_id.split('.')[-1].replace('_', ' ').title(),
                supported_types={MediaType.VIDEO, MediaType.AUDIO, MediaType.IMAGE}  # Assume all for now
            )
            self.devices[device_id] = device
            
            try:
                result = await self._browse_media(
                    entity_id=device_id,
                    media_content_type="",
                    media_content_id=cam_device.server_id
                )
                
                if result.get("success"):
                    cameras = result.get("result", {}).get("children", [])

                    for item in cameras:
                        item_id = item.get("media_content_id", "")
                        item_title = item.get("title", "Unknown")

                        logger.debug(f"💾 Found camera: {item_title}")

                        media_type = self._determine_media_type(item)
                        metadata = self._extract_metadata(item, ["camera"])
                        media_item = MediaItem(
                            media_id=item_id,
                            title=item_title,
                            media_type=media_type,
                            server_id=cam_device.server_id,
                            server_name=cam_device.name,
                            path="",
                            compatible_devices={device.entity_id},
                            **metadata
                        )
                        if self.media_items.get(item_id) and self.media_items[item_id].compatible_devices:
                            for compat_device in self.media_items[item_id].compatible_devices:
                                media_item.compatible_devices.add(compat_device)

                        self.media_items[item_id] = media_item

                        # Update indices
                        title_lower = item_title.lower()
                        if title_lower not in self.title_index:
                            self.title_index[title_lower] = []
                        self.title_index[title_lower].append(item_id)
                        
                        self.type_index[media_type].append(item_id)
                        
                        if cam_device.server_id not in self.server_index:
                            self.server_index[cam_device.server_id] = []
                        self.server_index[cam_device.server_id].append(item_id)

                        if metadata.get('genre'):
                            genre = metadata['genre'].lower()
                            if genre not in self.genre_index:
                                self.genre_index[genre] = []
                            self.genre_index[genre].append(item_id)

                        # self.media_items[item_id] = media_item

            except Exception as e:
                logger.warning(f"❌ Error scanning device: {e}")
                self.scan_stats['errors'] += 1
        
        self.scan_stats['last_scan_duration'] = (datetime.now() - start_time).total_seconds()
        
        # logger.debug(f"\n{'=' * 50}")
        # logger.debug(f"📊 SCAN COMPLETE")
        # logger.debug(f"   Total media items: {len(self.media_items)}")
        # logger.debug(f"   Total servers: {len(self.servers)}")
        # logger.debug(f"   Folders explored: {self.scan_stats['folders_explored']}")
        # logger.debug(f"   Errors: {self.scan_stats['errors']}")
        # logger.debug(f"   Duration: {self.scan_stats['last_scan_duration']:.2f} seconds")
        # logger.debug(f"{'=' * 50}") 

    async def scan_all_dlna_media(self, device_entity_ids: Optional[List[str]] = None,
                             max_depth: int = 10, 
                             max_items_per_level: Optional[int] = None) -> None:
        """
        Scan all available media from all DLNA servers
        
        Args:
            device_entity_ids: List of device entity IDs to scan (None = scan all)
            max_depth: Maximum folder depth to scan
            max_items_per_level: Maximum items per folder (None = all)
        """
        start_time = datetime.now()
        
        if not device_entity_ids:
            raise ValueError("Please provide device_entity_ids")
        
        # logger.debug(f"\n🔍 Starting comprehensive media scan")
        # logger.debug(f"   Devices: {len(device_entity_ids)}")
        # logger.debug(f"   Max depth: {max_depth}")
        # logger.debug(f"   Max items per level: {max_items_per_level or 'All'}")
        # logger.debug("=" * 50)
        
        for device_id in device_entity_ids:
            logger.debug(f"\n📱 Scanning device: {device_id}")
            
            device = PlaybackDevice(
                entity_id=device_id,
                name=device_id.split('.')[-1].replace('_', ' ').title(),
                supported_types={MediaType.VIDEO, MediaType.AUDIO, MediaType.IMAGE}  # Assume all for now
            )
            self.devices[device_id] = device
            
            # Get DLNA servers
            try:
                result = await self._browse_media(
                    entity_id=device_id,
                    media_content_type="",
                    media_content_id="media-source://dlna_dms"
                )
                
                if result.get("success"):
                    servers = result.get("result", {}).get("children", [])
                    
                    for server_data in servers:
                        server_id = server_data.get("media_content_id", "")
                        server_name = server_data.get("title", "Unknown")
                        
                        logger.debug(f"💾 Found server: {server_name}")
                        
                        server = MediaServer(
                            server_id=server_id,
                            name=server_name,
                            last_scan=datetime.now()
                        )
                        
                        # Reset visited for each server
                        self.visited_ids = set()
                        
                        # Scan server
                        root_item = {
                            "media_content_id": server_id,
                            "title": server_name,
                            "can_expand": True,
                            "can_play": False
                        }
                        
                        await self._scan_recursive(
                            device_id, device, server, root_item, [],
                            max_depth, 0, max_items_per_level
                        )
                        
                        server.media_count = len(self.server_index.get(server_id, []))
                        self.servers[server_id] = server
                        
                        logger.debug(f"  ✅ Server scan complete: {server.media_count} items")
                        
            except Exception as e:
                logger.warning(f"  ❌ Error scanning device: {e}")
                self.scan_stats['errors'] += 1
        
        self.scan_stats['last_scan_duration'] = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "total_media_items": len(self.media_items),
            "total_servers": len(self.servers),
            "folders_explored": self.scan_stats['folders_explored'],
            "errors": self.scan_stats['errors'],
            "last_scan_duration": self.scan_stats['last_scan_duration']
        }
        logger.debug(f"📊 SCAN COMPLETE: {json.dumps(stats)}")

    # ========== Media Player Controls ==========
    
    async def play_media(self, media_item: MediaItem, device_entity_id: Optional[str] = None) -> bool:
        """
        Play a media item on a device
        
        Args:
            media_item: The MediaItem to play
            device_entity_id: Target device (None = use first compatible)
        
        Returns:
            True if successful, False otherwise
        """
        if not device_entity_id:
            if media_item.compatible_devices:
                device_entity_id = sorted(media_item.compatible_devices)[0]
            else:
                logger.debug("No compatible devices found")
                return False
        
        logger.debug(f"🎬 Playing '{media_item.title}' on {device_entity_id}")
        
        message = {
            "type": "call_service",
            "domain": "media_player",
            "service": "play_media",
            "target": {
                "entity_id": device_entity_id
            },
            "service_data": {
                "media_content_id": media_item.media_id,
                "media_content_type": media_item.media_type.value
            }
        }

        
        response_data = await self._send_ws_command(message)
    #    response_data = json.loads(response)
        
        if response_data.get("success"):
            logger.debug("✅ Playback started successfully")
            return True
        else:
            error = response_data.get("error", {})
            logger.debug(f"❌ Playback failed: {error.get('message', 'Unknown error')}")
            return False
 
    async def get_entity_state(self, entity_id: str, 
                              use_cache: bool = True) -> Dict[str, Any]:
        """
        Get the current state of any entity.
        
        Args:
            entity_id: Entity ID to query
            use_cache: Whether to use cached state (5 second timeout)
        """
        
        command = {
            "type": "get_states"
        }
        
        result = await self._send_ws_command(command)
        
        if result.get("success"):
            states = result.get("result", [])
            for state in states:
                if state["entity_id"] == entity_id:
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


    # ========== Media Player Controls ==========

    def find_media_items(self, title: Optional[str] = None,
             media_type: Optional[MediaType] = None,
             server: Optional[str] = None,
             genre: Optional[str] = None,
             year: Optional[int] = None,
             limit: Optional[int] = None) -> List[MediaItem]:
        """
        Query media items with various filters
        
        Args:
            title: Title search (partial match)
            media_type: Filter by media type
            server: Filter by server ID or name
            genre: Filter by genre
            year: Filter by year
            limit: Maximum results to return
        
        Returns:
            List of matching MediaItem objects
        """
        results = set()
        
        if not any([title, media_type, server, genre, year]):
            results = set(self.media_items.keys())
        
        if title:
            title_lower = title.lower()
            for stored_title, media_ids in self.title_index.items():
                if title_lower in stored_title:
                    results.update(media_ids) # if not results else results.intersection_update(media_ids)
        
        if media_type:
            type_items = set(self.type_index[media_type])
            results = results.intersection(type_items) if results else type_items
        
        if server:
            server_items = set()
            for server_id, items in self.server_index.items():
                if server in server_id or server in self.servers.get(server_id, MediaServer("", "")).name:
                    server_items.update(items)
            results = results.intersection(server_items) if results else server_items
        
        if genre:
            genre_lower = genre.lower()
            genre_items = set(self.genre_index.get(genre_lower, []))
            results = results.intersection(genre_items) if results else genre_items
        
        if year:
            year_items = {mid for mid in results if self.media_items[mid].year == year}
            results = year_items
        
        items = [self.media_items[mid] for mid in results if mid in self.media_items]
        
        if limit:
            items = items[:limit]
        
        return items
    
    def get_mediaitem_by_id(self, media_id: str) -> Optional[MediaItem]:
        """Get a specific media item by ID"""
        return self.media_items.get(media_id, None)
    
    def list_servers(self) -> List[MediaServer]:
        """List all DLNA servers"""
        return list(self.servers.values())
    
    def list_devices(self) -> List[PlaybackDevice]:
        """List all playback devices"""
        return list(self.devices.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get library statistics"""
        stats = {
            'total_items': len(self.media_items),
            'total_servers': len(self.servers),
            'total_devices': len(self.devices),
            'items_by_type': {t.value: len(self.type_index[t]) for t in MediaType},
            'items_by_server': {s.name: len(self.server_index.get(s.server_id, [])) 
                               for s in self.servers.values()},
            'total_genres': len(self.genre_index),
            'scan_stats': self.scan_stats
        }
        return stats
    
    def save_to_file(self, filename: str = "dlna_library.json") -> None:
        """Save library to JSON file"""
        data = {
            'media_items': {k: v.to_dict() for k, v in self.media_items.items()},
            'servers': {k: v.to_dict() for k, v in self.servers.items()},
            'devices': {k: v.to_dict() for k, v in self.devices.items()},
            'stats': self.get_stats(),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"💾 Library saved to {filename}")
    
    def load_from_file(self, filename: str = "dlna_library.json") -> bool:
        """Load library from JSON file"""
        if not filename.endswith('.json'):
            logger.debug("Invalid file format. Please provide a .json file.")
            return False
        if not os.path.exists(filename):
            logger.debug(f"File not found: {filename}")
            return False
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.media_items.clear()
            self.servers.clear()
            self.devices.clear()
            self.title_index.clear()
            self.type_index = {t: [] for t in MediaType}
            self.server_index.clear()
            self.genre_index.clear()
            
            for media_id, item_data in data.get('media_items', {}).items():
                item = MediaItem.from_dict(item_data)
                self.media_items[media_id] = item
                
                # Rebuild indices
                title_lower = item.title.lower()
                if title_lower not in self.title_index:
                    self.title_index[title_lower] = []
                self.title_index[title_lower].append(media_id)
                
                self.type_index[item.media_type].append(media_id)
                
                if item.server_id not in self.server_index:
                    self.server_index[item.server_id] = []
                self.server_index[item.server_id].append(media_id)
                
                if item.genre:
                    genre = item.genre.lower()
                    if genre not in self.genre_index:
                        self.genre_index[genre] = []
                    self.genre_index[genre].append(media_id)
            
            for server_id, server_data in data.get('servers', {}).items():
                self.servers[server_id] = MediaServer(
                    server_id=server_data['server_id'],
                    name=server_data['name'],
                    media_count=server_data['media_count'],
                    last_scan=datetime.fromisoformat(server_data['last_scan']) if server_data.get('last_scan') else None
                )
            
            for device_id, device_data in data.get('devices', {}).items():
                self.devices[device_id] = PlaybackDevice(
                    entity_id=device_data['entity_id'],
                    name=device_data['name'],
                    supported_types={MediaType(t) for t in device_data.get('supported_types', [])}
                )
        except Exception as e:
            logger.debug(f"Error loading library from {filename}: {e}")
            self.media_items.clear()
            self.servers.clear()
            self.devices.clear()
            self.title_index.clear()
            self.type_index = {t: [] for t in MediaType}
            self.server_index.clear()
            self.genre_index.clear()
            return False

        logger.debug(f"📂 Library loaded from {filename}")
        logger.debug(f"   Items: {len(self.media_items)}")
        logger.debug(f"   Servers: {len(self.servers)}")
        logger.debug(f"   Devices: {len(self.devices)}")

        return True


############## TO TEST AND INTEGRATE ##################

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
                if state["entity_id"].startswith("media_player.") and not "hideme" in state["entity_id"]:
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

########### END TEST SECTION ###############


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

class MediaController:

    def __init__(self, ha_url: str, ha_token: str):
        self.library_manager = HAMediaLibrary(ha_url, ha_token)

    async def initialize(self, device_ids: list[str] = None) -> None:
        """Initialize the media controller including library load/scan."""

        await self.library_manager.connect()

        all_players_full = await self.library_manager.get_all_media_players()
        if not device_ids:
            device_ids = [player["entity_id"] for player in all_players_full]

        if not self.library_manager.load_from_file("ha_media_library.json"):
            logger.warning("❌ Failed to load media library file, scanning devices...")
            await self.library_manager.scan_all_dlna_media(
                device_entity_ids=device_ids,
                max_depth=10,
            )
            await self.library_manager.scan_cameras(
                device_entity_ids=device_ids,
                max_depth=5,
                max_items_per_level=5
            )
            self.library_manager.save_to_file("ha_media_library.json")
            logger.info("✅ Media library scanned and saved.") 
        else:
            logger.info("✅ Saved media library loaded successfully.")

    async def play_media(self, media_item_id: str, playback_device_id: str = None) -> str:
        """Play a media item on the selected device."""

        media_item = self.library_manager.get_mediaitem_by_id(media_item_id)
        if not media_item:
            search_results = self.library_manager.find_media_items(title=media_item_id, limit=1)
            logger.debug(f"Search for '{media_item_id}' found {len(search_results)} results")
            media_item = search_results[0] if search_results else None
        if playback_device_id:
            device = self.library_manager.devices.get(playback_device_id, None)
        if media_item and device:
            success = await self.library_manager.play_media(media_item, device_entity_id=device.entity_id)
            if success:
                logger.info(f"✅ Playing: {media_item.title}")
                return {"status": "success", "message": f"Playing: {media_item.title}", "device": device.name}
            else:
                logger.error(f"❌ Failed to play: {media_item.title}")
                return {"status": "error", "message": f"Failed to play: {media_item.title}", "device": device.name}
        elif media_item:
            logger.info(f"Guessing best device to play {media_item.title} on, since no device ID was provided")
            success = await self.library_manager.play_media(media_item)
            if success:
                logger.debug(f"✅ Playing: {media_item.title} on guessed device")
                return {"status": "success", "message": f"Playing: {media_item.title} on guessed device"}
            else:
                logger.debug(f"❌ Failed to play: {media_item.title} on guessed device")
                return {"status": "error", "message": f"Failed to play: {media_item.title} on guessed device"}
        return {"status": "error", "message": "No media item found", "device": device.name}

    async def find_media_items(self, query: str, query_type: str = None, media_type: str = None, limit: int = 5) -> str:
        media_types = ["video", "audio", "image", "playlist"]
        query_types = ["title", "genre", "year"]

        if media_type and media_type.lower() not in media_types:
            return f"❌ Invalid media type. Valid types are: {media_types}"
        if query_type and query_type.lower() not in query_types:
            return f"❌ Invalid query type. Valid types are: {query_types}"

        media_type_enum = MediaType(media_type.lower()) if media_type else None

        results =[]
        if query_type:
            if query_type.lower() == "title":
                results = self.library_manager.find_media_items(title=query, media_type=media_type_enum, limit=limit)
            if query_type.lower() == "genre":
                results = self.library_manager.find_media_items(genre=query, media_type=media_type_enum, limit=limit)
            if query_type.lower() == "year":
                results = self.library_manager.find_media_items(year=query, media_type=media_type_enum, limit=limit)
        else:
            results = self.library_manager.find_media_items(title=query, media_type=media_type_enum, limit=limit)

        if results:
            result_lines = []
            result_lines.append(f"### Found {len(results)} result(s) for \"{query}\":\n")
            for idx, result in enumerate(results, 1):
                result_lines.append(
                    f"**{idx}. {result.title}**\n"
                    f"- Media ID: `{result.media_id}`\n"
                    f"- Media Type: `{result.media_type.value}`\n"
                    f"- Server: `{result.server_name}`\n"
                    f"- Path: {' > '.join(result.path) if result.path else '(root)'}\n"
                    f"- Compatible Devices: {', '.join(result.compatible_devices) if result.compatible_devices else 'None'}\n"
                )
            return "\n".join(result_lines)
        return f"No results found for query: \"{query}\""
    
    async def control_media_player(self,
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
            control_media_player("play", "living_room_tv")
            control_media_player("volume_set", "bedroom_speaker", value=50)
            control_media_player("seek", "living_room_tv", value=600)  # Seek to 10 minutes
            control_media_player("select_source", "living_room_tv", value="HDMI 1")
        """
        logger.info(f"Executing action: {action} on media device: {device} with value: {value} and target_device: {target_device}")

        entity_id = await self._resolve_device(device)
        if not entity_id:
            return {"success": False, "error": f"Device '{device}' not found", "existing_devices_and_states": self.get_media_player_statuses()}
        
        try:
            action_type = ActionType(action.lower())
            
            if action_type == ActionType.PLAY:
                result = await self.library_manager.play(entity_id)
            elif action_type == ActionType.PAUSE:
                result = await self.library_manager.pause(entity_id)
            elif action_type == ActionType.STOP:
                result = await self.library_manager.stop(entity_id)
            elif action_type == ActionType.TOGGLE:
                result = await self.library_manager.play_pause_toggle(entity_id)
            elif action_type == ActionType.NEXT:
                result = await self.library_manager.next_track(entity_id)
            elif action_type == ActionType.PREVIOUS:
                result = await self.library_manager.previous_track(entity_id)
            elif action_type == ActionType.SEEK:
                if value is None:
                    return {"success": False, "error": "Seek position required"}
                result = await self.library_manager.seek(entity_id, int(value))
            elif action_type == ActionType.SHUFFLE:
                shuffle_on = value if isinstance(value, bool) else True
                result = await self.library_manager.set_shuffle(entity_id, shuffle_on)
            elif action_type == ActionType.REPEAT:
                repeat_mode = value if value in ["off", "all", "one"] else "all"
                result = await self.library_manager.set_repeat(entity_id, repeat_mode)
                
            elif action_type == ActionType.VOLUME_SET:
                if value is None:
                    return {"success": False, "error": "Volume level required (0-100)"}
                # Convert percentage to 0.0-1.0 range
                volume = float(value) / 100 if value > 1 else float(value)
                result = await self.library_manager.set_volume(entity_id, volume)
            elif action_type == ActionType.VOLUME_UP:
                result = await self.library_manager.volume_up(entity_id)
            elif action_type == ActionType.VOLUME_DOWN:
                result = await self.library_manager.volume_down(entity_id)
            elif action_type == ActionType.MUTE:
                result = await self.library_manager.mute(entity_id, True)
            elif action_type == ActionType.UNMUTE:
                result = await self.library_manager.mute(entity_id, False)
                
            elif action_type == ActionType.TURN_ON:
                result = await self.library_manager.turn_on_display(entity_id)
            elif action_type == ActionType.TURN_OFF:
                result = await self.library_manager.turn_off_display(entity_id)
                
            elif action_type == ActionType.SELECT_SOURCE:
                if value is None:
                    return {"success": False, "error": "Source name required"}
                result = await self.library_manager.select_source(entity_id, value)
                
            elif action_type == ActionType.QUEUE_CLEAR:
                result = await self.library_manager.clear_playlist(entity_id)
            else:
                return {"success": False, "error": f"Unsupported action: {action}"}
                
            return {
                "success": result.get("success", True),
                "action": action,
                "device": entity_id,
                "value": value
            }
            
        except Exception as e:
            logger.error(f"Control media error: {e}")
            return {"success": False, "error": str(e)}

    async def get_media_player_statuses(self) -> Dict[str, Any]:
        """
        Get status information about all media_player devices in Home Assistant.
        """
        #logger.info(f"Getting media player statuses for device: {device} with info_type: {info_type}")
        
        try:
            players = await self.library_manager.get_all_media_players()
            return {"success": True, "players": players}

        except Exception as e:
            logger.error(f"Get status error: {e}")
            return {"success": False, "error": str(e)}

        # try:
        #     if device and device.lower() != "all":
        #         entity_id = await self._resolve_device(device)
        #         if not entity_id:
        #             return {"success": False, "error": f"Device '{device}' not found"}
                    
        #         status = await self.library_manager.get_media_player_status(entity_id)
                
        #         if info_type == "full":
        #             state = await self.library_manager.get_entity_state(entity_id)
        #             if state:
        #                 status["supported_features"] = state.get("attributes", {}).get("supported_features")
        #                 status["source_list"] = state.get("attributes", {}).get("source_list")
                        
        #         return {"success": True, "device": entity_id, "status": status}
                
        #     else:
        #         players = await self.library_manager.get_all_media_players()
                
        #         if info_type == "playing":
        #             if players:
        #                 players = [p for p in players if p.get("state", "") in ["playing", "paused"]]

        #         elif info_type == "devices":
        #             return {
        #                 "success": True,
        #                 "devices": [
        #                     {
        #                         "id": p.get("entity_id", ""),
        #                         "name": p.get("name", ""),
        #                         "state": p.get("state", ""),
        #                         "type": "plex" if p.get("is_plex") else "generic"
        #                     }
        #                     for p in players
        #                 ]
        #             }
                    
        #         return {"success": True, "players": players}
                
        # except Exception as e:
        #     logger.error(f"Get status error: {e}")
        #     return {"success": False, "error": str(e)}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Tool definitions for AI function c.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "control_media_player",
                    "description": "Control a media player's playback, volume, power, and sources on any device",
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
                                "description": "Media Player device name or entity_id (optional, auto-detects)"
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
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "play_media",
                    "description": "Play a media item on the selected device. Can search by title if media ID is not found.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "media_item_id": {
                                "type": "string",
                                "description": "The ID(preferred) or title of the media item to play. If an ID is not found, will search for items matching this as a title."
                            },
                            "playback_device_id": {
                                "type": "string",
                                "description": "The ID(preferred) of the playback device to use. If not provided, will attempt to guess the best device."
                            }
                        },
                        "required": ["media_item_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_media_items",
                    "description": "Search for media items in the library by title, genre, or year, with optional filtering by media type",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string (e.g., movie title, genre name, or year)"
                            },
                            "query_type": {
                                "type": "string",
                                "enum": ["title", "genre", "year"],
                                "description": "Type of search to perform. Defaults to 'title' if not specified"
                            },
                            "media_type": {
                                "type": "string",
                                "enum": ["video", "audio", "image", "playlist"],
                                "description": "Filter results by media type. If not specified, returns all types"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                                "minimum": 1
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]


######## Internal Helpers #########

    async def _resolve_device(self, device: str) -> Optional[str]:
        """Resolve device name to entity_id."""
        if not device:
            return None
            
        if device.startswith("media_player."):
            return device
            
        players = await self.library_manager.get_all_media_players()
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
            
        cameras = await self.library_manager.get_all_cameras()
        camera_lower = camera.lower()
        
        for cam in cameras:
            if camera_lower in cam["entity_id"].lower():
                return cam["entity_id"]
            if cam.get("name") and camera_lower in cam["name"].lower():
                return cam["entity_id"]
                
        return None
    

async def main():
    """Example usage"""
    
    device_ids = ["media_player.living_room_tv","media_player.bedroom_speaker"]
    
    controller = MediaController(config.HA_WS_URL, config.HAOS_TOKEN)
    await controller.initialize(device_ids)

    logger.info(f"Searching media for 'bob' titles:\n")
    search_results = await controller.find_media_items("star wars")
    logger.info(search_results)

    media_player_statuses = await controller.get_media_player_statuses()
    for media_player in media_player_statuses.get("players", []):
        logger.info(f"Media Player: {media_player['name']} ({media_player['entity_id']}) - State: {media_player['state']}")

    # logger.debug(f"Pausing living room tv")
    # await controller.control_media_player("play", "living_room_tv")
    await controller.control_media_player("unmute", device="media_player.living_room_tv_gcast")

if __name__ == "__main__":
    asyncio.run(main())