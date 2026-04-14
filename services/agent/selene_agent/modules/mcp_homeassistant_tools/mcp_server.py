#!/usr/bin/env python3
"""
Home Assistant MCP Server for HavenCore
Provides unified Home Assistant tools via MCP including device control, media management, and automation
"""

import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

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


class EntityNotFoundError(Exception):
    """Raised when a service call references an entity that does not exist in HA.

    HA's POST /api/services endpoint returns HTTP 200 with an empty body for
    non-existent entities — indistinguishable from a legitimate no-op (e.g.
    turn_off on an already-off light). We detect the missing-entity case with a
    pre-flight GET /api/states/<entity_id> and surface it explicitly so tool
    responses can't silently falsely claim success.
    """

    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        super().__init__(f"Entity '{entity_id}' does not exist in Home Assistant")


def _format_entity_not_found(err: "EntityNotFoundError", device_label: str = "entity") -> str:
    return (
        f"FAILED: {device_label} '{err.entity_id}' does not exist in Home Assistant. "
        "No action was taken. Call ha_get_domain_entity_states or ha_get_entities_in_area "
        "to look up the correct entity_id, then retry. Do not guess entity names."
    )


# Per-domain attribute projection. friendly_name is always included; anything
# listed here is surfaced when HA reports it. Kept deliberately tight so the
# LLM sees signal, not noise.
DOMAIN_ATTRS: Dict[str, tuple] = {
    "light":        ("brightness", "color_mode", "rgb_color", "hs_color",
                     "xy_color", "color_temp_kelvin", "effect",
                     "supported_color_modes"),
    "media_player": ("source", "volume_level", "is_volume_muted",
                     "media_title", "media_artist", "media_album_name",
                     "app_id", "source_list"),
    "climate":      ("current_temperature", "temperature",
                     "target_temp_high", "target_temp_low",
                     "hvac_action", "hvac_modes", "fan_mode",
                     "current_humidity", "humidity", "preset_mode"),
    "cover":        ("current_position", "current_tilt_position", "device_class"),
    "fan":          ("percentage", "preset_mode", "oscillating", "direction"),
    "sensor":       ("unit_of_measurement", "device_class"),
    "binary_sensor":("device_class",),
    "lock":         ("device_class",),
    "vacuum":       ("battery_level", "status", "fan_speed"),
    "water_heater": ("current_temperature", "temperature", "operation_mode"),
    "humidifier":   ("current_humidity", "humidity", "mode"),
    "weather":      ("temperature", "humidity", "pressure",
                     "wind_speed", "wind_bearing"),
    "person":       (),
    "device_tracker":(),
}


def _project_attrs(domain: str, raw_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the subset of HA attributes worth returning for the given domain.

    Always includes friendly_name. Drops keys whose value is None so the
    payload stays compact when HA doesn't report a field.
    """
    keys = ("friendly_name",) + DOMAIN_ATTRS.get(domain, ())
    out: Dict[str, Any] = {}
    for k in keys:
        v = raw_attrs.get(k)
        if v is not None:
            out[k] = v
    return out


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

    async def _post_text(self, path: str, payload: Dict[str, Any]) -> str:
        async with aiohttp.ClientSession(timeout=self._timeout, headers=self._headers) as session:
            async with session.post(f"{self._base}{path}", json=payload) as resp:
                resp.raise_for_status()
                return await resp.text()

    async def _ws_call(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Open a WS connection to HA, auth, send one command, return the result frame.

        HA's registry endpoints (config/*_registry/list) are WS-only, so we need this
        for area/entity/device lookups even though the rest of the client uses REST.
        """
        if TEST_MODE:
            return {"success": True, "result": []}
        if not HA_WS_URL:
            raise RuntimeError("HA_WS_URL not configured")
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.ws_connect(HA_WS_URL) as ws:
                first = await ws.receive_json()
                if first.get("type") != "auth_required":
                    raise RuntimeError(f"Unexpected HA WS greeting: {first}")
                await ws.send_json({"type": "auth", "access_token": HAOS_TOKEN})
                auth = await ws.receive_json()
                if auth.get("type") != "auth_ok":
                    raise RuntimeError(f"HA WS auth failed: {auth.get('message') or auth}")
                msg_id = 1
                await ws.send_json({"id": msg_id, **command})
                while True:
                    frame = await ws.receive_json()
                    if frame.get("id") == msg_id:
                        return frame

    async def get_domain_entity_states(self, domain: str) -> str:
        if TEST_MODE:
            return json.dumps({
                f"{domain}.entity1": {"state": "on", "attributes": {"friendly_name": "Entity 1"}},
                f"{domain}.entity2": {"state": "off", "attributes": {"friendly_name": "Entity 2"}},
                f"{domain}.entity3": {"state": "unavailable", "attributes": {"friendly_name": "Entity 3"}},
            })
        states = await self._get("/api/states")
        prefix = f"{domain}."
        filtered = {
            s["entity_id"]: {
                "state": s.get("state"),
                "attributes": _project_attrs(domain, s.get("attributes") or {}),
            }
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

    async def _entity_exists(self, entity_id: str) -> bool:
        """Return True if the entity exists, False if HA returns 404 for it."""
        try:
            await self._get(f"/api/states/{entity_id}")
            return True
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return False
            raise

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
        if entity_id and not await self._entity_exists(entity_id):
            raise EntityNotFoundError(entity_id)
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
                    description=(
                        "Get the current states of all entities in a Home Assistant domain "
                        "(e.g. light, switch, climate, cover, media_player, sensor). Returns a "
                        "JSON object keyed by entity_id with shape "
                        "{'state': <str>, 'attributes': {...}}. Attributes are curated per "
                        "domain: lights include brightness / rgb_color / hs_color / "
                        "color_temp_kelvin / color_mode; climate includes current_temperature "
                        "/ temperature / hvac_action; covers include current_position; etc. "
                        "The attribute values in the response round-trip back into the matching "
                        "ha_control_* tool (e.g. pass an rgb_color from here straight into "
                        "ha_control_light)."
                    ),
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
                        "service_data={'brightness_pct': 50, 'color_name': 'blue'} on a light entity. "
                        "Before calling: confirm the exact entity_id via ha_get_entities_in_area or "
                        "ha_get_domain_entity_states — do not guess."
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
                        "Turn a light on/off/toggle with optional brightness and color. "
                        "Color can be specified as rgb_color, hs_color, hex_color, "
                        "color_temp_kelvin, or color_name — specify at most ONE of these. "
                        "The rgb_color / hs_color / color_temp_kelvin values returned by "
                        "ha_get_domain_entity_states and ha_get_entity_history can be passed "
                        "back here unchanged to reproduce a prior color. turn_off and toggle "
                        "ignore brightness/color. "
                        "Before calling: confirm the exact entity_id via ha_get_entities_in_area or "
                        "ha_get_domain_entity_states — do not guess."
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
                            },
                            "rgb_color": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 0, "maximum": 255},
                                "minItems": 3,
                                "maxItems": 3,
                                "description": "Color as [R,G,B] integers 0-255 (on only). Matches the rgb_color shape returned by ha_get_domain_entity_states."
                            },
                            "hs_color": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                                "description": "Color as [hue 0-360, saturation 0-100] (on only). Matches the hs_color shape returned by ha_get_domain_entity_states."
                            },
                            "hex_color": {
                                "type": "string",
                                "description": "Color as a hex string like '#FF8040' or 'FF8040' (on only). Converted to rgb_color before calling HA."
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
                        "HA service call. "
                        "Before calling: confirm the exact entity_id via ha_get_entities_in_area or "
                        "ha_get_domain_entity_states — do not guess."
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
                        "direct area assignment. Results are grouped by domain. By default "
                        "returns bare entity_id strings (cheap directory lookup); set "
                        "include_state=true to also attach each entity's current state and "
                        "curated attributes (one extra /api/states fetch — prefer the default "
                        "for large areas)."
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
                            },
                            "include_state": {
                                "type": "boolean",
                                "default": False,
                                "description": "When true, replace each entity_id string with {entity_id, state, attributes}. Attributes follow the same curated-per-domain shape as ha_get_domain_entity_states."
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
                ),
                Tool(
                    name="ha_set_timer",
                    description=(
                        "Start a Home Assistant timer helper (timer.start). Requires a timer.* "
                        "entity configured in Home Assistant — discover available timers via "
                        "ha_get_domain_entity_states with domain='timer'. Duration uses HH:MM:SS "
                        "format (e.g. '0:05:00' = 5 minutes); omit to use the timer's configured "
                        "default duration."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Timer entity ID (e.g. 'timer.kitchen')"
                            },
                            "duration": {
                                "type": "string",
                                "description": "Duration in HH:MM:SS (e.g. '0:10:00' for 10 minutes). Optional."
                            }
                        },
                        "required": ["entity_id"]
                    }
                ),
                Tool(
                    name="ha_cancel_timer",
                    description="Cancel a running Home Assistant timer (timer.cancel).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Timer entity ID to cancel"
                            }
                        },
                        "required": ["entity_id"]
                    }
                ),
                Tool(
                    name="ha_evaluate_template",
                    description=(
                        "Evaluate a Home Assistant Jinja2 template server-side and return the "
                        "rendered text. Escape hatch for compound questions like "
                        "\"{{ is_state('binary_sensor.back_door','on') and is_state('person.matt','home') }}\". "
                        "See https://www.home-assistant.io/docs/configuration/templating/ for syntax."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template": {
                                "type": "string",
                                "description": "Jinja2 template string to render"
                            }
                        },
                        "required": ["template"]
                    }
                ),
                Tool(
                    name="ha_get_entity_history",
                    description=(
                        "Get recent state history for an entity over the last N hours. Returns a "
                        "list of {state, last_changed, attributes?} points, sampled if very dense. "
                        "Attributes are curated per domain (same shape as ha_get_domain_entity_states) "
                        "so you can see e.g. how a light's brightness or rgb_color changed over "
                        "time — useful for questions like \"what color was the lamp before?\" or "
                        "\"when did the thermostat setpoint last change?\"."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_id": {
                                "type": "string",
                                "description": "Entity ID to fetch history for"
                            },
                            "hours": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 168,
                                "default": 24,
                                "description": "Look-back window in hours (default 24, max 168 = 1 week)"
                            }
                        },
                        "required": ["entity_id"]
                    }
                ),
                Tool(
                    name="ha_get_calendar_events",
                    description=(
                        "Get upcoming events from a Home Assistant calendar entity over the next "
                        "N days. Discover calendar entities via ha_get_domain_entity_states with "
                        "domain='calendar'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "calendar_entity": {
                                "type": "string",
                                "description": "Calendar entity ID (e.g. 'calendar.family')"
                            },
                            "days": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 31,
                                "default": 7,
                                "description": "Look-ahead window in days (default 7, max 31)"
                            }
                        },
                        "required": ["calendar_entity"]
                    }
                )
            ])

            # Media Player Control Tools
            tools.extend([
                Tool(
                    name="ha_control_media_player",
                    description=(
                        "Control a media player: playback (play/pause/stop/toggle/next/previous/seek), "
                        "volume, power, or input source. The 'value' field is required for "
                        "volume_set, seek, and select_source; ignored otherwise.\n"
                        "\n"
                        "Playback semantics (IMPORTANT — play and pause are NOT toggles):\n"
                        "- 'play'   : start/resume playback. Use only when the device is paused or stopped. "
                        "Calling 'play' on an already-playing device is a no-op.\n"
                        "- 'pause'  : pause playback. Use only when the device is currently playing. "
                        "Calling 'pause' on an already-paused device is a no-op.\n"
                        "- 'toggle' : switch between play and pause based on current state. Prefer this "
                        "for user requests like 'unpause', 'resume', 'pause/play it' or whenever the "
                        "current playback state is unknown or ambiguous.\n"
                        "When in doubt between play/pause, use 'toggle'.\n"
                        "\n"
                        "Before calling: confirm the exact entity_id via ha_get_entities_in_area or "
                        "ha_get_domain_entity_states — do not guess."
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
                                "description": (
                                    "Action to perform on the media player. Note: 'play' and 'pause' "
                                    "are directional (not toggles) — use 'toggle' for 'unpause'/'resume' "
                                    "or when the current state is unknown."
                                )
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
                        arguments.get("rgb_color"),
                        arguments.get("hs_color"),
                        arguments.get("hex_color"),
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
                        bool(arguments.get("include_state", False)),
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_get_presence":
                    result = await self._get_presence()
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_set_timer":
                    result = await self._set_timer(
                        arguments.get("entity_id"),
                        arguments.get("duration"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_cancel_timer":
                    result = await self._cancel_timer(arguments.get("entity_id"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_evaluate_template":
                    result = await self._evaluate_template(arguments.get("template"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "ha_get_entity_history":
                    result = await self._get_entity_history(
                        arguments.get("entity_id"),
                        arguments.get("hours", 24),
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                elif name == "ha_get_calendar_events":
                    result = await self._get_calendar_events(
                        arguments.get("calendar_entity"),
                        arguments.get("days", 7),
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

                # Media Player Control Operations
                elif name == "ha_control_media_player":
                    result = await self._control_media_player(
                        arguments.get("action"),
                        arguments.get("device"),
                        arguments.get("value")
                    )
                    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

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
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "entity")
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
        rgb_color: Optional[List[int]] = None,
        hs_color: Optional[List[float]] = None,
        hex_color: Optional[str] = None,
    ) -> str:
        state = (state or "").lower()
        if state not in ("on", "off", "toggle"):
            return f"Error: invalid state '{state}' (expected on/off/toggle)"
        service = {"on": "turn_on", "off": "turn_off", "toggle": "toggle"}[state]
        extras: Dict[str, Any] = {}
        if state == "on":
            color_sources = {
                "rgb_color": rgb_color,
                "hs_color": hs_color,
                "hex_color": hex_color,
                "color_temp_kelvin": color_temp_kelvin,
                "color_name": color_name,
            }
            provided = [k for k, v in color_sources.items() if v is not None]
            if len(provided) > 1:
                return (
                    "Error: ha_control_light accepts only one of rgb_color, hs_color, "
                    f"hex_color, color_temp_kelvin, color_name (got {provided})"
                )

            if brightness_pct is not None:
                extras["brightness_pct"] = brightness_pct

            if hex_color is not None:
                h = hex_color.strip().lstrip("#")
                if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
                    return f"Error: hex_color must be 6 hex chars (got '{hex_color}')"
                extras["rgb_color"] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)]
            elif rgb_color is not None:
                if not (isinstance(rgb_color, list) and len(rgb_color) == 3):
                    return f"Error: rgb_color must be [R,G,B] (got {rgb_color})"
                extras["rgb_color"] = [int(c) for c in rgb_color]
            elif hs_color is not None:
                if not (isinstance(hs_color, list) and len(hs_color) == 2):
                    return f"Error: hs_color must be [hue,sat] (got {hs_color})"
                extras["hs_color"] = [float(c) for c in hs_color]
            elif color_temp_kelvin is not None:
                extras["color_temp_kelvin"] = color_temp_kelvin
            elif color_name is not None:
                extras["color_name"] = color_name
        try:
            return await self.ha_client.execute_service(entity_id, service, **extras)
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Light")
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
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Climate")
        except Exception as e:
            return f"Error controlling climate '{entity_id}': {e}"

    async def _activate_scene(self, scene_entity: str) -> str:
        try:
            return await self.ha_client.execute_service(scene_entity, "turn_on")
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Scene")
        except Exception as e:
            return f"Error activating scene '{scene_entity}': {e}"

    async def _trigger_script(self, script_entity: str, variables: Dict[str, Any]) -> str:
        try:
            return await self.ha_client.execute_service(script_entity, "turn_on", **variables)
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Script")
        except Exception as e:
            return f"Error triggering script '{script_entity}': {e}"

    async def _trigger_automation(self, automation_entity: str) -> str:
        try:
            return await self.ha_client.execute_service(automation_entity, "trigger")
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Automation")
        except Exception as e:
            return f"Error triggering automation '{automation_entity}': {e}"

    async def _toggle_automation(self, entity_id: str, enabled: bool) -> str:
        service = "turn_on" if enabled else "turn_off"
        try:
            return await self.ha_client.execute_service(entity_id, service)
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Automation")
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
        resp = await self.ha_client._ws_call({"type": f"config/{registry}_registry/list"})
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
        include_state: bool = False,
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

        state_by_id: Dict[str, Dict[str, Any]] = {}
        if include_state:
            try:
                raw_states = await self.ha_client._get("/api/states")
                state_by_id = {s.get("entity_id"): s for s in (raw_states or []) if s.get("entity_id")}
            except Exception as e:
                return {"error": f"failed to fetch states: {e}"}

        grouped: Dict[str, List[Any]] = {}
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
            if include_state:
                s = state_by_id.get(eid) or {}
                grouped.setdefault(dom, []).append({
                    "entity_id": eid,
                    "state": s.get("state"),
                    "attributes": _project_attrs(dom, s.get("attributes") or {}),
                })
            else:
                grouped.setdefault(dom, []).append(eid)

        for dom in grouped:
            if include_state:
                grouped[dom].sort(key=lambda e: e["entity_id"])
            else:
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

    # Timer / Template / History / Calendar Methods
    async def _set_timer(self, entity_id: Optional[str], duration: Optional[str] = None) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        extras: Dict[str, Any] = {}
        if duration:
            extras["duration"] = duration
        try:
            return await self.ha_client.execute_service(entity_id, "start", **extras)
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Timer")
        except Exception as e:
            return f"Error starting timer '{entity_id}': {e}"

    async def _cancel_timer(self, entity_id: Optional[str]) -> str:
        if not entity_id:
            return "Error: entity_id is required"
        try:
            return await self.ha_client.execute_service(entity_id, "cancel")
        except EntityNotFoundError as e:
            return _format_entity_not_found(e, "Timer")
        except Exception as e:
            return f"Error cancelling timer '{entity_id}': {e}"

    async def _evaluate_template(self, template: Optional[str]) -> str:
        if not template:
            return "Error: template is required"
        if TEST_MODE:
            return f"Mock template render: {template}"
        try:
            return await self.ha_client._post_text("/api/template", {"template": template})
        except Exception as e:
            return f"Error evaluating template: {e}"

    async def _get_entity_history(
        self,
        entity_id: Optional[str],
        hours: int = 24,
    ) -> Dict[str, Any]:
        if not entity_id:
            return {"error": "entity_id is required"}
        try:
            hours = int(hours)
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(hours, 168))
        start = (datetime.now(timezone.utc) - timedelta(hours=hours)).replace(microsecond=0).isoformat()
        path = (
            f"/api/history/period/{quote(start, safe='')}"
            f"?filter_entity_id={quote(entity_id)}"
        )
        try:
            data = await self.ha_client._get(path)
        except Exception as e:
            return {"error": str(e)}
        domain = entity_id.split(".", 1)[0]
        points: List[Dict[str, Any]] = []
        for series in data or []:
            for p in series or []:
                point: Dict[str, Any] = {
                    "state": p.get("state"),
                    "last_changed": p.get("last_changed") or p.get("last_updated"),
                }
                attrs = _project_attrs(domain, p.get("attributes") or {})
                if attrs:
                    point["attributes"] = attrs
                points.append(point)
        MAX_POINTS = 200
        result: Dict[str, Any] = {
            "entity_id": entity_id,
            "hours": hours,
            "total_points": len(points),
        }
        if len(points) > MAX_POINTS:
            step = max(1, len(points) // MAX_POINTS)
            result["sampled"] = True
            result["points"] = points[::step][:MAX_POINTS]
        else:
            result["points"] = points
        return result

    async def _get_calendar_events(
        self,
        calendar_entity: Optional[str],
        days: int = 7,
    ) -> Dict[str, Any]:
        if not calendar_entity:
            return {"error": "calendar_entity is required"}
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = 7
        days = max(1, min(days, 31))
        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = now.isoformat()
        end = (now + timedelta(days=days)).isoformat()
        path = (
            f"/api/calendars/{quote(calendar_entity)}"
            f"?start={quote(start, safe='')}&end={quote(end, safe='')}"
        )
        try:
            data = await self.ha_client._get(path)
        except Exception as e:
            return {"error": str(e)}

        def _extract_when(val: Any) -> Any:
            if isinstance(val, dict):
                return val.get("dateTime") or val.get("date") or val
            return val

        events: List[Dict[str, Any]] = []
        for ev in data or []:
            events.append({
                "start": _extract_when(ev.get("start")),
                "end": _extract_when(ev.get("end")),
                "summary": ev.get("summary"),
                "description": ev.get("description"),
                "location": ev.get("location"),
            })
        return {
            "calendar_entity": calendar_entity,
            "days": days,
            "events": events,
        }

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