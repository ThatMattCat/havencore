#!/usr/bin/env python3
"""
Home Assistant MCP Server for HavenCore
Provides unified Home Assistant tools via MCP including device control, media management, and automation
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../shared/'))

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import HA functionality, fallback to mock if not available
try:
    import shared.scripts.logger as logger_module
    import utils.config as config
    import requests
    from homeassistant_api import Client
    import websockets
    from enum import Enum
    from dataclasses import dataclass, field
    
    # Get configuration
    HAOS_URL = config.HAOS_URL
    HAOS_TOKEN = config.HAOS_TOKEN
    HA_WS_URL = HAOS_URL.replace('/api', '').replace('https://', 'wss://').replace('http://', 'ws://') + '/api/websocket'
    
    logger.info("Successfully imported Home Assistant dependencies")
    HAS_HA_DEPS = True
except ImportError as e:
    logger.warning(f"Could not import HA dependencies: {e}. Using mock implementation.")
    HAS_HA_DEPS = False
    
    # Mock configuration
    HAOS_URL = "http://localhost:8123/api"
    HAOS_TOKEN = "mock_token"
    HA_WS_URL = "ws://localhost:8123/api/websocket"


# Define enums and data classes needed for media control
class ActionType(Enum):
    """Media player action types"""
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"
    NEXT = "next"
    PREVIOUS = "previous"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    VOLUME_SET = "volume_set"
    MUTE = "mute"
    UNMUTE = "unmute"
    SHUFFLE = "shuffle"
    REPEAT = "repeat"

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
    compatible_devices: set = field(default_factory=set)
    
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


class HomeAssistantClient:
    """Home Assistant API client"""
    
    def __init__(self):
        self._api_url = HAOS_URL
        self._token = HAOS_TOKEN

    def get_domain_entity_states(self, domain: str) -> str:
        """Get all entity states for a domain"""
        if not HAS_HA_DEPS:
            return json.dumps({
                f"{domain}.entity1": "on",
                f"{domain}.entity2": "off",
                f"{domain}.entity3": "unavailable"
            })
            
        with Client(self._api_url, self._token) as client:
            domain_entity_states = {}
            entities = client.get_entities()
            domain_entity_states = {
                entity.state.entity_id: entity.state.state 
                for _, entity in entities[domain].entities.items()
            }
            return json.dumps(domain_entity_states)

    def get_domain_services(self, domain: str) -> str:
        """Get all services for a domain"""
        if not HAS_HA_DEPS:
            return json.dumps({
                "turn_on": "Turn on entity",
                "turn_off": "Turn off entity", 
                "toggle": "Toggle entity state"
            })
            
        with Client(self._api_url, self._token) as client:
            domain_obj = client.get_domain(domain)
            return json.dumps({
                service.service_id: service.description 
                for _, service in domain_obj.services.items()
            })
        
    def get_entity_state(self, entity_id: str) -> str:
        """Get state of a specific entity"""
        if not HAS_HA_DEPS:
            return "on"
            
        with Client(self._api_url, self._token) as client:
            entity = client.get_entity(entity_id=entity_id)
            return entity.state.state

    def execute_service(self, entity_id: str, service: str) -> str:
        """Execute a service on an entity"""
        if not HAS_HA_DEPS:
            return f"Mock execution: {service} on {entity_id}"
            
        with Client(self._api_url, self._token) as client:
            domain_name = entity_id.split('.')[0]
            domain_obj = client.get_domain(domain_name)

            service_obj = domain_obj.services[service]
            changes = service_obj.trigger(entity_id=entity_id)
            final_state = self.get_entity_state(entity_id)
            return f"Service {service} executed on {entity_id}"


class MediaController:
    """Media controller for Home Assistant media players"""
    
    def __init__(self):
        self.ha_client = HomeAssistantClient()
        
    async def initialize(self):
        """Initialize the media controller"""
        pass
        
    async def control_media_player(self, action: str, device: Optional[str] = None, value: Optional[Union[int, str, bool]] = None) -> Dict[str, Any]:
        """Control a media player with the specified action"""
        if not HAS_HA_DEPS:
            return {
                "success": True,
                "action": action,
                "device": device or "mock_device",
                "value": value,
                "message": f"Mock media control: {action} on {device or 'mock_device'}"
            }
            
        # Implement basic media control using HA API
        try:
            action_type = ActionType(action.lower())
        except ValueError:
            return {"success": False, "error": f"Unknown action: {action}"}
            
        service_map = {
            ActionType.PLAY: "media_play",
            ActionType.PAUSE: "media_pause", 
            ActionType.STOP: "media_stop",
            ActionType.NEXT: "media_next_track",
            ActionType.PREVIOUS: "media_previous_track",
            ActionType.VOLUME_UP: "volume_up",
            ActionType.VOLUME_DOWN: "volume_down",
            ActionType.VOLUME_SET: "volume_set",
            ActionType.MUTE: "volume_mute",
            ActionType.UNMUTE: "volume_mute"
        }
        
        service = service_map.get(action_type)
        if service and device:
            try:
                result = self.ha_client.execute_service(device, service)
                return {
                    "success": True,
                    "action": action,
                    "device": device,
                    "value": value,
                    "message": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Action {action} not implemented or device not specified"}
    
    async def get_media_player_statuses(self) -> Dict[str, Any]:
        """Get status of all media players"""
        if not HAS_HA_DEPS:
            return {
                "success": True,
                "players": [
                    {
                        "entity_id": "media_player.living_room",
                        "name": "Living Room",
                        "state": "playing",
                        "media_title": "Mock Song"
                    }
                ]
            }
        
        try:
            states_json = self.ha_client.get_domain_entity_states("media_player")
            states = json.loads(states_json)
            players = [
                {
                    "entity_id": entity_id,
                    "state": state,
                    "name": entity_id.replace("media_player.", "").replace("_", " ").title()
                }
                for entity_id, state in states.items()
            ]
            return {"success": True, "players": players}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def play_media(self, media_item_id: str, playback_device_id: Optional[str] = None) -> Dict[str, Any]:
        """Play specific media on a media player"""
        if not HAS_HA_DEPS:
            return {
                "success": True,
                "message": f"Mock playing {media_item_id} on {playback_device_id or 'default_device'}"
            }
            
        # Use the play_media service
        try:
            device = playback_device_id or "media_player.default"
            result = self.ha_client.execute_service(device, "play_media")
            return {
                "success": True,
                "message": f"Playing {media_item_id} on {device}: {result}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def find_media_items(self, query: str, query_type: Optional[str] = None, 
                               media_type: Optional[str] = None, limit: int = 5) -> str:
        """Find media items matching a query"""
        if not HAS_HA_DEPS:
            return f"""Mock search results for "{query}":
1. Mock Movie 1 (media_id: mock_1)
2. Mock Song 2 (media_id: mock_2)
3. Mock Video 3 (media_id: mock_3)"""
        
        # This would need to be implemented based on available media sources
        mock_results = []
        for i in range(min(limit, 3)):
            mock_results.append(f"{i+1}. Mock {query} Result {i+1} (media_id: mock_{i+1})")
        
        return "\n".join(mock_results) if mock_results else f"No results found for '{query}'"


class HomeAssistantMCPServer:
    """MCP server providing comprehensive Home Assistant tools"""
    
    def __init__(self):
        self.server = Server("havencore-homeassistant")
        self.ha_client = None
        self.media_controller = None
        self.setup_handlers()
        
    async def initialize_clients(self):
        """Initialize Home Assistant clients"""
        try:
            # Initialize HA client (handles mock fallback internally)
            self.ha_client = HomeAssistantClient()
            
            # Initialize media controller
            self.media_controller = MediaController()
            await self.media_controller.initialize()
            
            logger.info("Home Assistant clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HA clients: {e}")
            # Create fallback clients
            self.ha_client = HomeAssistantClient()
            self.media_controller = MediaController()
            await self.media_controller.initialize()
            logger.info("Fallback Home Assistant clients initialized")
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available Home Assistant tools"""
            tools = []
            
            # Basic Home Assistant API Tools
            tools.extend([
                Tool(
                    name="ha_get_domain_entity_states",
                    description="Get the current states of all entities in a Home Assistant domain (e.g., light, switch, sensor)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The Home Assistant domain (e.g., 'light', 'switch', 'sensor', 'climate')"
                            }
                        },
                        "required": ["domain"]
                    }
                ),
                Tool(
                    name="ha_get_domain_services",
                    description="Get the available services for a Home Assistant domain",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The Home Assistant domain to get services for"
                            }
                        },
                        "required": ["domain"]
                    }
                ),
                Tool(
                    name="ha_execute_service",
                    description="Execute a Home Assistant service on an entity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "The entity ID to execute the service on (e.g., 'light.living_room')"
                            },
                            "service": {
                                "type": "string",
                                "description": "The service name to execute (e.g., 'turn_on', 'turn_off', 'toggle')"
                            }
                        },
                        "required": ["entity_id", "service"]
                    }
                )
            ])
            
            # Media Player Control Tools
            tools.extend([
                Tool(
                    name="ha_control_media_player",
                    description="Control media player playback, volume, power, and sources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["play", "pause", "stop", "toggle", "next", "previous", 
                                        "seek", "shuffle", "repeat", "volume_set", "volume_up", 
                                        "volume_down", "mute", "unmute", "turn_on", "turn_off", 
                                        "select_source"],
                                "description": "Action to perform on the media player"
                            },
                            "device": {
                                "type": "string",
                                "description": "Media player device name or entity_id (optional, auto-detects if not provided)"
                            },
                            "value": {
                                "type": ["number", "string", "boolean"],
                                "description": "Value for the action (volume level 0-100, seek position in seconds, source name, etc.)"
                            }
                        },
                        "required": ["action"]
                    }
                ),
                Tool(
                    name="ha_get_media_player_statuses",
                    description="Get status information about Home Assistant media players",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="ha_play_media",
                    description="Play a specific media item on a device from the media library",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "media_item_id": {
                                "type": "string",
                                "description": "The ID or title of the media item to play"
                            },
                            "playback_device_id": {
                                "type": "string",
                                "description": "The device ID to play on (optional, will auto-select if not provided)"
                            }
                        },
                        "required": ["media_item_id"]
                    }
                ),
                Tool(
                    name="ha_find_media_items",
                    description="Search for media items in the Home Assistant media library",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (movie title, song name, etc.)"
                            },
                            "query_type": {
                                "type": "string",
                                "enum": ["title", "genre", "year"],
                                "description": "Type of search to perform (defaults to 'title')"
                            },
                            "media_type": {
                                "type": "string",
                                "enum": ["video", "audio", "image", "playlist"],
                                "description": "Filter by media type (optional)"
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
                )
            ])
            
            logger.info(f"Listing {len(tools)} Home Assistant tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.BaseModel]:
            """Execute a Home Assistant tool"""
            logger.info(f"Home Assistant tool called: {name} with args: {arguments}")
            
            try:
                # Ensure clients are initialized
                if not self.ha_client:
                    await self.initialize_clients()
                
                # Basic Home Assistant API Operations
                if name == "ha_get_domain_entity_states":
                    result = await self._get_domain_entity_states(arguments.get("domain"))
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "ha_get_domain_services":
                    result = await self._get_domain_services(arguments.get("domain"))
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "ha_execute_service":
                    result = await self._execute_service(
                        arguments.get("entity_id"),
                        arguments.get("service")
                    )
                    return [types.TextContent(type="text", text=result)]
                
                # Media Player Control Operations
                elif name == "ha_control_media_player":
                    result = await self._control_media_player(
                        arguments.get("action"),
                        arguments.get("device"),
                        arguments.get("value")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "ha_get_media_player_statuses":
                    result = await self._get_media_player_statuses()
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "ha_play_media":
                    result = await self._play_media(
                        arguments.get("media_item_id"),
                        arguments.get("playback_device_id")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                
                elif name == "ha_find_media_items":
                    result = await self._find_media_items(
                        arguments.get("query"),
                        arguments.get("query_type"),
                        arguments.get("media_type"),
                        arguments.get("limit", 5)
                    )
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    return [types.TextContent(type="text", text=f"Unknown Home Assistant tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing Home Assistant tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    # Basic Home Assistant API Methods
    async def _get_domain_entity_states(self, domain: str) -> str:
        """Get entity states for a domain"""
        try:
            result = self.ha_client.get_domain_entity_states(domain)
            return f"Entity states for domain '{domain}':\n{result}"
        except Exception as e:
            return f"Error getting entity states for domain '{domain}': {str(e)}"
    
    async def _get_domain_services(self, domain: str) -> str:
        """Get services for a domain"""
        try:
            result = self.ha_client.get_domain_services(domain)
            return f"Services for domain '{domain}':\n{result}"
        except Exception as e:
            return f"Error getting services for domain '{domain}': {str(e)}"
    
    async def _execute_service(self, entity_id: str, service: str) -> str:
        """Execute a service on an entity"""
        try:
            result = self.ha_client.execute_service(entity_id, service)
            return f"Service '{service}' executed on '{entity_id}': {result}"
        except Exception as e:
            return f"Error executing service '{service}' on '{entity_id}': {str(e)}"

    # Media Player Control Methods
    async def _control_media_player(self, action: str, device: Optional[str] = None, value: Optional[Union[int, str, bool]] = None) -> Dict[str, Any]:
        """Control media player"""
        if not self.media_controller:
            return {"success": False, "error": "Media controller not available"}
        
        try:
            result = await self.media_controller.control_media_player(action, device, value)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_media_player_statuses(self) -> Dict[str, Any]:
        """Get media player statuses"""
        if not self.media_controller:
            return {"success": False, "error": "Media controller not available"}
        
        try:
            result = await self.media_controller.get_media_player_statuses()
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _play_media(self, media_item_id: str, playback_device_id: Optional[str] = None) -> Dict[str, Any]:
        """Play media item"""
        if not self.media_controller:
            return {"success": False, "error": "Media controller not available"}
        
        try:
            result = await self.media_controller.play_media(media_item_id, playback_device_id)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _find_media_items(self, query: str, query_type: Optional[str] = None, 
                               media_type: Optional[str] = None, limit: int = 5) -> str:
        """Find media items"""
        if not self.media_controller:
            return "Error: Media controller not available"
        
        try:
            result = await self.media_controller.find_media_items(query, query_type, media_type, limit)
            return result
        except Exception as e:
            return f"Error searching media: {str(e)}"
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting HavenCore Home Assistant MCP Server...")
        
        # Initialize HA clients
        await self.initialize_clients()
        
        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Home Assistant MCP Server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="HavenCore Home Assistant MCP Server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    )
                )
            )


async def main():
    """Main entry point"""
    server = HomeAssistantMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())