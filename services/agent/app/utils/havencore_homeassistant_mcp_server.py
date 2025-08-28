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
    from utils.haos.haos import HomeAssistant
    from utils.haos.ha_media_controller import MediaController, ActionType, MediaType
    
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


class MockHomeAssistant:
    """Mock Home Assistant client for testing"""
    
    def get_domain_entity_states(self, domain: str) -> str:
        return json.dumps({
            f"{domain}.entity1": "on",
            f"{domain}.entity2": "off",
            f"{domain}.entity3": "unavailable"
        })
    
    def get_domain_services(self, domain: str) -> str:
        return json.dumps({
            "turn_on": "Turn on entity",
            "turn_off": "Turn off entity", 
            "toggle": "Toggle entity state"
        })
    
    def execute_service(self, entity_id: str, service: str) -> str:
        return f"Mock execution: {service} on {entity_id}"


class MockMediaController:
    """Mock Media Controller for testing"""
    
    async def initialize(self):
        pass
    
    async def control_media_player(self, action: str, device: Optional[str] = None, value: Optional[Union[int, str, bool]] = None) -> Dict[str, Any]:
        return {
            "success": True,
            "action": action,
            "device": device or "mock_device",
            "value": value,
            "message": f"Mock execution: {action} on {device or 'mock_device'}"
        }
    
    async def get_media_player_statuses(self) -> Dict[str, Any]:
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
    
    async def play_media(self, media_item_id: str, playback_device_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": True,
            "message": f"Mock playing {media_item_id} on {playback_device_id or 'default_device'}"
        }
    
    async def find_media_items(self, query: str, query_type: Optional[str] = None, 
                               media_type: Optional[str] = None, limit: int = 5) -> str:
        return f"""Mock search results for "{query}":
1. Mock Movie 1 (media_id: mock_1)
2. Mock Song 2 (media_id: mock_2)
3. Mock Video 3 (media_id: mock_3)"""


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
            if HAS_HA_DEPS:
                # Initialize real HA client
                self.ha_client = HomeAssistant()
                
                # Initialize media controller
                self.media_controller = MediaController(HA_WS_URL, HAOS_TOKEN)
                await self.media_controller.initialize()
                
                logger.info("Home Assistant clients initialized successfully")
            else:
                # Use mock clients
                self.ha_client = MockHomeAssistant()
                self.media_controller = MockMediaController()
                await self.media_controller.initialize()
                
                logger.info("Mock Home Assistant clients initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HA clients: {e}")
            # Fallback to mock clients
            self.ha_client = MockHomeAssistant()
            self.media_controller = MockMediaController()
            await self.media_controller.initialize()
            logger.info("Fallback to mock Home Assistant clients")
        
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