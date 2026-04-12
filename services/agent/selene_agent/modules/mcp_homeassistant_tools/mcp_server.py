#!/usr/bin/env python3
"""
Home Assistant MCP Server for HavenCore
Provides unified Home Assistant tools via MCP including device control, media management, and automation
"""

import json
import asyncio
from typing import Any, Dict, List, Optional, Union

import aiohttp

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool
from mcp.server.models import InitializationOptions
from . import ha_media_controller

from selene_agent.utils.logger import get_logger
from selene_agent.utils import config as agent_config

logger = get_logger('loki')

HAOS_URL = agent_config.HAOS_URL
HAOS_TOKEN = agent_config.HAOS_TOKEN
HA_WS_URL = agent_config.HA_WS_URL

TEST_MODE = not HAOS_URL or not HAOS_TOKEN
if TEST_MODE:
    logger.warning("Running in TEST MODE with mock Home Assistant data (HAOS_URL/HAOS_TOKEN not set)")
    HAOS_URL = "http://localhost:8123/api"
    HAOS_TOKEN = "mock_token"
    HA_WS_URL = "ws://localhost:8123/api/websocket"


class HomeAssistantClient:
    """Async Home Assistant REST API client using aiohttp."""

    def __init__(self):
        base = HAOS_URL.rstrip("/")
        if base.endswith("/api"):
            base = base[:-4]
        self._base = base
        self._headers = {
            "Authorization": f"Bearer {HAOS_TOKEN}",
            "Content-Type": "application/json",
        }
        self._timeout = aiohttp.ClientTimeout(total=15)

    async def _get(self, path: str) -> Any:
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as session:
            async with session.get(f"{self._base}{path}") as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as session:
            async with session.post(f"{self._base}{path}", json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def get_domain_entity_states(self, domain: str) -> str:
        if TEST_MODE:
            return json.dumps({
                f"{domain}.entity1": "on",
                f"{domain}.entity2": "off",
                f"{domain}.entity3": "unavailable",
            })
        states = await self._get("/api/states")
        prefix = f"{domain}."
        filtered = {
            s["entity_id"]: s["state"]
            for s in states
            if s.get("entity_id", "").startswith(prefix)
        }
        return json.dumps(filtered)

    async def get_domain_services(self, domain: str) -> str:
        if TEST_MODE:
            return json.dumps({
                "turn_on": "Turn on entity",
                "turn_off": "Turn off entity",
                "toggle": "Toggle entity state",
            })
        services = await self._get("/api/services")
        for entry in services:
            if entry.get("domain") == domain:
                return json.dumps({
                    name: meta.get("description", "")
                    for name, meta in entry.get("services", {}).items()
                })
        return json.dumps({})

    async def get_entity_state(self, entity_id: str) -> str:
        if TEST_MODE:
            return "on"
        data = await self._get(f"/api/states/{entity_id}")
        return data.get("state", "unknown")

    async def execute_service(
        self,
        entity_id: Optional[str],
        service: str,
        domain: Optional[str] = None,
        **service_data,
    ) -> str:
        if entity_id and not domain:
            domain = entity_id.split(".")[0]
        if not domain:
            raise ValueError("execute_service requires entity_id or explicit domain")
        if TEST_MODE:
            target = entity_id or f"{domain}.*"
            return f"Mock execution: {domain}.{service} on {target}"
        payload = {**service_data}
        if entity_id:
            payload["entity_id"] = entity_id
        await self._post(f"/api/services/{domain}/{service}", payload)
        suffix = f" on {entity_id}" if entity_id else ""
        return f"Service {domain}.{service} executed{suffix}"

class HomeAssistantMCPServer:
    """MCP server providing comprehensive Home Assistant tools"""
    
    def __init__(self):
        self.server = Server("havencore-homeassistant")
        self.ha_client: Optional[HomeAssistantClient] = None
        self.media_controller: Optional[ha_media_controller.MediaController] = None
        self.init_error: Optional[str] = None
        self.setup_handlers()

    async def initialize_clients(self):
        """Initialize Home Assistant clients. Records init_error on failure."""
        try:
            self.ha_client = HomeAssistantClient()
            self.media_controller = ha_media_controller.MediaController(ha_url=HA_WS_URL, ha_token=HAOS_TOKEN)
            await self.media_controller.initialize()
            self.init_error = None
            logger.info("Home Assistant clients initialized successfully")
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to initialize HA clients: {e}")
        
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
                    description="Get the current states of all entities in a Home Assistant domain (e.g., media_player, light, switch, sensor)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The Home Assistant domain (e.g., 'media_player', 'light', 'switch', 'sensor', 'climate')"
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
                    description=(
                        "Execute a Home Assistant service on an entity. "
                        "Use service_data to pass service parameters such as brightness, color, "
                        "temperature, volume, etc. Example: service='turn_on' with "
                        "service_data={'brightness_pct': 50, 'color_name': 'blue'} on a light entity."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "The entity ID to execute the service on (e.g., 'light.living_room')"
                            },
                            "service": {
                                "type": "string",
                                "description": "The service name to execute (e.g., 'turn_on', 'turn_off', 'toggle', 'set_temperature')"
                            },
                            "service_data": {
                                "type": "object",
                                "description": (
                                    "Optional service-specific parameters as a JSON object. "
                                    "Common examples: {'brightness_pct': 50}, {'color_name': 'red'}, "
                                    "{'temperature': 72, 'hvac_mode': 'heat'}, {'volume_level': 0.5}."
                                ),
                                "additionalProperties": True
                            }
                        },
                        "required": ["entity_id", "service"]
                    }
                )
            ])
            
            # Device & Automation Control Tools
            tools.extend([
                Tool(
                    name="ha_control_light",
                    description=(
                        "Turn a light on/off/toggle with optional brightness, color, and color "
                        "temperature. Only include optional fields when the user requested that "
                        "attribute; turn_off and toggle ignore brightness/color."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Light entity ID (e.g. 'light.living_room_lamp')"
                            },
                            "state": {
                                "type": "string",
                                "enum": ["on", "off", "toggle"],
                                "description": "Desired state"
                            },
                            "brightness_pct": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Brightness percentage 0-100 (on only)"
                            },
                            "color_name": {
                                "type": "string",
                                "description": "CSS color name, e.g. 'red', 'warm_white' (on only)"
                            },
                            "color_temp_kelvin": {
                                "type": "integer",
                                "description": "Color temperature in Kelvin, e.g. 2700-6500 (on only)"
                            }
                        },
                        "required": ["entity_id", "state"]
                    }
                ),
                Tool(
                    name="ha_control_climate",
                    description=(
                        "Control a climate entity (thermostat). Supply any combination of "
                        "temperature, hvac_mode, and fan_mode — each issued as a separate "
                        "HA service call."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Climate entity ID (e.g. 'climate.living_room')"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Target temperature (unit matches the entity's configured unit)"
                            },
                            "hvac_mode": {
                                "type": "string",
                                "enum": ["off", "heat", "cool", "auto", "heat_cool", "dry", "fan_only"],
                                "description": "HVAC mode"
                            },
                            "fan_mode": {
                                "type": "string",
                                "description": "Fan mode (entity-specific, e.g. 'auto', 'low', 'high')"
                            }
                        },
                        "required": ["entity_id"]
                    }
                ),
                Tool(
                    name="ha_activate_scene",
                    description="Activate a Home Assistant scene (scene.turn_on).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "scene_entity": {
                                "type": "string",
                                "description": "Scene entity ID (e.g. 'scene.movie_night')"
                            }
                        },
                        "required": ["scene_entity"]
                    }
                ),
                Tool(
                    name="ha_trigger_script",
                    description=(
                        "Run a Home Assistant script (script.turn_on). Optional 'variables' "
                        "object is passed as script variables."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "script_entity": {
                                "type": "string",
                                "description": "Script entity ID (e.g. 'script.bedtime')"
                            },
                            "variables": {
                                "type": "object",
                                "description": "Optional script variables",
                                "additionalProperties": True
                            }
                        },
                        "required": ["script_entity"]
                    }
                ),
                Tool(
                    name="ha_trigger_automation",
                    description=(
                        "Manually trigger a Home Assistant automation (automation.trigger). "
                        "Use ha_toggle_automation to enable/disable instead."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "automation_entity": {
                                "type": "string",
                                "description": "Automation entity ID (e.g. 'automation.goodnight')"
                            }
                        },
                        "required": ["automation_entity"]
                    }
                ),
                Tool(
                    name="ha_toggle_automation",
                    description=(
                        "Enable or disable a Home Assistant automation (automation.turn_on/turn_off). "
                        "This controls whether the automation runs on its triggers; it does NOT "
                        "manually fire the automation — use ha_trigger_automation for that."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Automation entity ID"
                            },
                            "enabled": {
                                "type": "boolean",
                                "description": "true = turn_on (enable), false = turn_off (disable)"
                            }
                        },
                        "required": ["entity_id", "enabled"]
                    }
                ),
                Tool(
                    name="ha_send_notification",
                    description=(
                        "Send a notification through a Home Assistant notify.* service. "
                        "Call ha_get_domain_services with domain='notify' to discover the "
                        "available service names on this HA instance."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service": {
                                "type": "string",
                                "description": "notify service name, e.g. 'mobile_app_pixel_7' (without the 'notify.' prefix)"
                            },
                            "message": {
                                "type": "string",
                                "description": "Notification body"
                            },
                            "title": {
                                "type": "string",
                                "description": "Optional notification title"
                            },
                            "target": {
                                "type": ["string", "array"],
                                "items": {"type": "string"},
                                "description": "Optional target(s) — string or array, service-dependent"
                            }
                        },
                        "required": ["service", "message"]
                    }
                ),
                Tool(
                    name="ha_list_areas",
                    description=(
                        "List Home Assistant areas (rooms / zones). Useful before "
                        "ha_get_entities_in_area when the user names a room."
                    ),
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="ha_get_entities_in_area",
                    description=(
                        "List entities assigned to an area. Accepts area_id or case-insensitive "
                        "area name. Entities inherit from their device's area when they have no "
                        "direct area assignment. Results are grouped by domain."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "area": {
                                "type": "string",
                                "description": "Area ID or name (e.g. 'kitchen' or 'Kitchen')"
                            },
                            "domains": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional domain filter, e.g. ['light','switch']"
                            }
                        },
                        "required": ["area"]
                    }
                ),
                Tool(
                    name="ha_get_presence",
                    description=(
                        "Summarize presence: state of all person.* and device_tracker.* entities. "
                        "Typical states: 'home', 'not_home', or a zone name."
                    ),
                    inputSchema={"type": "object", "properties": {}}
                )
            ])

            # Media Player Control Tools
            tools.extend([
                Tool(
                    name="ha_control_media_player",
                    description=(
                        "Control a media player: playback (play/pause/stop/next/previous/seek), "
                        "volume, power, or input source. The 'value' field is required for "
                        "volume_set, seek, and select_source; ignored otherwise."
                    ),
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
                                "description": (
                                    "Value for the action. Units by action: "
                                    "volume_set = integer 0-100 (percent); "
                                    "seek = integer seconds from start; "
                                    "select_source = source name string (e.g. 'HDMI 1'); "
                                    "shuffle/repeat = boolean or 'off'/'all'/'one'."
                                )
                            }
                        },
                        "required": ["action"]
                    }
                ),
                Tool(
                    name="ha_stream_media",
                    description="Stream a specific media item to a device from the media library",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "media_item_id": {
                                "type": "string",
                                "description": "The media ID of the media item to stream"
                            },
                            "playback_device_id": {
                                "type": "string",
                                "description": "The device ID to stream on (optional, will auto-select if not provided)"
                            }
                        },
                        "required": ["media_item_id"]
                    }
                ),
                Tool(
                    name="ha_find_media_items",
                    description=(
                        "Search the Home Assistant media library. Returns media items including "
                        "their media IDs, which can be passed to ha_stream_media as media_item_id."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (camera, movie title, song name, etc.)"
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

            if self.init_error or not self.ha_client:
                msg = self.init_error or "Home Assistant clients not initialized"
                return [types.TextContent(type="text", text=f"Home Assistant unavailable: {msg}")]

            try:
                # Basic Home Assistant API Operations
                if name == "ha_get_domain_entity_states":
                    dmn = arguments.get("domain")
                    if 'media_player' in dmn:
                        result = await self._get_media_player_statuses()
                        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
                    result = await self._get_domain_entity_states(arguments.get("domain"))
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "ha_get_domain_services":
                    result = await self._get_domain_services(arguments.get("domain"))
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "ha_execute_service":
                    result = await self._execute_service(
                        arguments.get("entity_id"),
                        arguments.get("service"),
                        arguments.get("service_data") or {},
                    )
                    return [types.TextContent(type="text", text=result)]
                
                # Device & Automation Control Operations
                elif name == "ha_control_light":
                    result = await self._control_light(
                        arguments.get("entity_id"),
                        arguments.get("state"),
                        arguments.get("brightness_pct"),
                        arguments.get("color_name"),
                        arguments.get("color_temp_kelvin"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_control_climate":
                    result = await self._control_climate(
                        arguments.get("entity_id"),
                        arguments.get("temperature"),
                        arguments.get("hvac_mode"),
                        arguments.get("fan_mode"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_activate_scene":
                    result = await self._activate_scene(arguments.get("scene_entity"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_trigger_script":
                    result = await self._trigger_script(
                        arguments.get("script_entity"),
                        arguments.get("variables") or {},
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_trigger_automation":
                    result = await self._trigger_automation(arguments.get("automation_entity"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_toggle_automation":
                    result = await self._toggle_automation(
                        arguments.get("entity_id"),
                        bool(arguments.get("enabled")),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_send_notification":
                    result = await self._send_notification(
                        arguments.get("service"),
                        arguments.get("message"),
                        arguments.get("title"),
                        arguments.get("target"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_list_areas":
                    result = await self._list_areas()
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_get_entities_in_area":
                    result = await self._get_entities_in_area(
                        arguments.get("area"),
                        arguments.get("domains"),
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_get_presence":
                    result = await self._get_presence()
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                # Media Player Control Operations
                elif name == "ha_control_media_player":
                    result = await self._control_media_player(
                        arguments.get("action"),
                        arguments.get("device"),
                        arguments.get("value")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_stream_media":
                    result = await self._stream_media(
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
            result = await self.ha_client.get_domain_entity_states(domain)
            return f"Entity states for domain '{domain}':\n{result}"
        except Exception as e:
            return f"Error getting entity states for domain '{domain}': {str(e)}"

    async def _get_domain_services(self, domain: str) -> str:
        """Get services for a domain"""
        try:
            result = await self.ha_client.get_domain_services(domain)
            return f"Services for domain '{domain}':\n{result}"
        except Exception as e:
            return f"Error getting services for domain '{domain}': {str(e)}"

    async def _execute_service(self, entity_id: str, service: str, service_data: Optional[Dict[str, Any]] = None) -> str:
        """Execute a service on an entity, optionally with service_data parameters."""
        service_data = service_data or {}
        try:
            result = await self.ha_client.execute_service(entity_id, service, **service_data)
            return f"Service '{service}' executed on '{entity_id}': {result}"
        except Exception as e:
            return f"Error executing service '{service}' on '{entity_id}': {str(e)}"

    # Device & Automation Control Methods
    async def _control_light(
        self,
        entity_id: str,
        state: str,
        brightness_pct: Optional[int] = None,
        color_name: Optional[str] = None,
        color_temp_kelvin: Optional[int] = None,
    ) -> str:
        state = (state or "").lower()
        if state not in ("on", "off", "toggle"):
            return f"Error: invalid state '{state}' (expected on/off/toggle)"
        service = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[state]
        extras: Dict[str, Any] = {}
        if state == "on":
            if brightness_pct is not None:
                extras["brightness_pct"] = brightness_pct
            if color_name is not None:
                extras["color_name"] = color_name
            if color_temp_kelvin is not None:
                extras["color_temp_kelvin"] = color_temp_kelvin
        try:
            return await self.ha_client.execute_service(entity_id, service, **extras)
        except Exception as e:
            return f"Error controlling light '{entity_id}': {e}"

    async def _control_climate(
        self,
        entity_id: str,
        temperature: Optional[float] = None,
        hvac_mode: Optional[str] = None,
        fan_mode: Optional[str] = None,
    ) -> str:
        if temperature is None and hvac_mode is None and fan_mode is None:
            return "Error: ha_control_climate requires at least one of temperature, hvac_mode, fan_mode"
        results: List[str] = []
        try:
            if hvac_mode is not None:
                results.append(
                    await self.ha_client.execute_service(entity_id, "set_hvac_mode", hvac_mode=hvac_mode)
                )
            if temperature is not None:
                results.append(
                    await self.ha_client.execute_service(entity_id, "set_temperature", temperature=temperature)
                )
            if fan_mode is not None:
                results.append(
                    await self.ha_client.execute_service(entity_id, "set_fan_mode", fan_mode=fan_mode)
                )
            return "; ".join(results)
        except Exception as e:
            return f"Error controlling climate '{entity_id}': {e}"

    async def _activate_scene(self, scene_entity: str) -> str:
        try:
            return await self.ha_client.execute_service(scene_entity, "turn_on")
        except Exception as e:
            return f"Error activating scene '{scene_entity}': {e}"

    async def _trigger_script(self, script_entity: str, variables: Dict[str, Any]) -> str:
        try:
            return await self.ha_client.execute_service(script_entity, "turn_on", **variables)
        except Exception as e:
            return f"Error triggering script '{script_entity}': {e}"

    async def _trigger_automation(self, automation_entity: str) -> str:
        try:
            return await self.ha_client.execute_service(automation_entity, "trigger")
        except Exception as e:
            return f"Error triggering automation '{automation_entity}': {e}"

    async def _toggle_automation(self, entity_id: str, enabled: bool) -> str:
        service = "turn_on" if enabled else "turn_off"
        try:
            return await self.ha_client.execute_service(entity_id, service)
        except Exception as e:
            verb = "enable" if enabled else "disable"
            return f"Error trying to {verb} automation '{entity_id}': {e}"

    async def _send_notification(
        self,
        service: str,
        message: str,
        title: Optional[str] = None,
        target: Optional[Union[str, List[str]]] = None,
    ) -> str:
        if not service or not message:
            return "Error: ha_send_notification requires 'service' and 'message'"
        data: Dict[str, Any] = {"message": message}
        if title is not None:
            data["title"] = title
        if target is not None:
            data["target"] = target
        try:
            return await self.ha_client.execute_service(
                entity_id=None, service=service, domain="notify", **data
            )
        except Exception as e:
            return f"Error sending notification via 'notify.{service}': {e}"

    # Registry & Presence Methods
    async def _ws_registry(self, registry: str) -> List[Dict[str, Any]]:
        """Call config/<registry>_registry/list and return the result list."""
        if not self.media_controller:
            raise RuntimeError("Media controller not available for WS registry access")
        resp = await self.media_controller.library_manager.send_ws_command(
            {"type": f"config/{registry}_registry/list"}
        )
        if not resp.get("success", True):
            raise RuntimeError(f"WS registry {registry} call failed: {resp.get('error')}")
        return resp.get("result") or []

    async def _list_areas(self) -> List[Dict[str, Any]]:
        try:
            areas = await self._ws_registry("area")
        except Exception as e:
            return [{"error": str(e)}]
        return [
            {
                "area_id": a.get("area_id"),
                "name": a.get("name"),
                "aliases": a.get("aliases") or [],
            }
            for a in areas
        ]

    async def _get_entities_in_area(
        self,
        area: str,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not area:
            return {"error": "'area' is required"}
        try:
            areas = await self._ws_registry("area")
            entities = await self._ws_registry("entity")
            devices = await self._ws_registry("device")
        except Exception as e:
            return {"error": str(e)}

        needle = area.strip().lower()
        area_id = None
        for a in areas:
            if a.get("area_id") == area or a.get("area_id", "").lower() == needle:
                area_id = a.get("area_id"); break
            if (a.get("name") or "").lower() == needle:
                area_id = a.get("area_id"); break
            if needle in {(al or "").lower() for al in (a.get("aliases") or [])}:
                area_id = a.get("area_id"); break
        if area_id is None:
            return {"error": f"No area matching '{area}'", "known_areas": [a.get("name") for a in areas]}

        device_area = {d["id"]: d.get("area_id") for d in devices if d.get("id")}
        domain_filter = set(d.lower() for d in domains) if domains else None

        grouped: Dict[str, List[str]] = {}
        for ent in entities:
            if ent.get("disabled_by") or ent.get("hidden_by"):
                continue
            eid = ent.get("entity_id")
            if not eid:
                continue
            ent_area = ent.get("area_id") or device_area.get(ent.get("device_id"))
            if ent_area != area_id:
                continue
            dom = eid.split(".", 1)[0]
            if domain_filter and dom not in domain_filter:
                continue
            grouped.setdefault(dom, []).append(eid)

        for dom in grouped:
            grouped[dom].sort()
        return {"area_id": area_id, "entities_by_domain": grouped, "total": sum(len(v) for v in grouped.values())}

    async def _get_presence(self) -> Dict[str, Any]:
        try:
            raw = await self.ha_client._get("/api/states")
        except Exception as e:
            return {"error": str(e)}
        people: List[Dict[str, Any]] = []
        trackers: List[Dict[str, Any]] = []
        for s in raw:
            eid = s.get("entity_id", "")
            attrs = s.get("attributes") or {}
            summary = {
                "entity_id": eid,
                "state": s.get("state"),
                "friendly_name": attrs.get("friendly_name"),
            }
            if eid.startswith("person."):
                people.append(summary)
            elif eid.startswith("device_tracker."):
                trackers.append(summary)
        return {"persons": people, "device_trackers": trackers}

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
    
    async def _stream_media(self, media_item_id: str, playback_device_id: Optional[str] = None) -> Dict[str, Any]:
        """Play media item"""
        if not self.media_controller:
            return {"success": False, "error": "Media controller not available"}
        
        try:
            result = await self.media_controller.stream_media(media_item_id=media_item_id, playback_device_id=playback_device_id)
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