#!/usr/bin/env python3
"""Music Assistant MCP Server for HavenCore.

Exposes cross-provider search, player enumeration, and queue-aware playback
on Music Assistant — the audio-only counterpart to the Plex module, targeting
Chromecasts / Google Homes / ESP32 satellites.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool
import mcp.types as types

from selene_agent.utils import config as agent_config
from selene_agent.utils.logger import get_logger

from .mass_client import MassAgent

logger = get_logger("loki")

MASS_URL = getattr(agent_config, "MASS_URL", None)
MASS_TOKEN = getattr(agent_config, "MASS_TOKEN", None)

TEST_MODE = not MASS_URL or not MASS_TOKEN


class MusicAssistantMCPServer:
    def __init__(self):
        self.server: Server = Server("havencore-music-assistant")
        self.agent: Optional[MassAgent] = None
        self.init_error: Optional[str] = None
        self._setup_handlers()

    async def initialize(self) -> None:
        if TEST_MODE:
            self.init_error = "MASS_URL / MASS_TOKEN not configured"
            logger.warning("Music Assistant MCP running without credentials — tools will return an error")
            return
        try:
            self.agent = MassAgent(MASS_URL, MASS_TOKEN)
            await self.agent.connect()
            logger.info(f"Music Assistant MCP initialized (url={MASS_URL})")
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to init Music Assistant agent: {e}")
            self.agent = None

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="mass_search",
                    description=(
                        "Search the Music Assistant library across all connected providers "
                        "(Plex, Spotify, etc.) for tracks/albums/artists/playlists/radio. "
                        "Returns rows with `uri` (opaque — pass to mass_play_media), `name`, "
                        "`artist`, `album`, `media_type`, and `providers`.\n\n"
                        "Query tip: MA's search is title-biased. Prefer a single short query "
                        "(a title OR an artist, not both in one string). The server will "
                        "auto-fallback if a typed search misses, but the cleanest result comes "
                        "from concise queries like 'The Better Life' or '3 Doors Down'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search text (title, artist, keywords)."},
                            "media_type": {
                                "type": "string",
                                "enum": ["track", "album", "artist", "playlist", "radio"],
                                "description": "Optional filter for result type.",
                            },
                            "limit": {"type": "integer", "description": "Max items per bucket (default 5).", "default": 5},
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="mass_list_players",
                    description=(
                        "Enumerate Music Assistant players (speakers). Returns `player_id`, "
                        "`display_name`, `available`, `powered`, `state`, `volume_level`, and "
                        "`current_item`. Hidden players are excluded unless `include_hidden=true`. "
                        "Use the `display_name` values as `player_name` inputs to other tools."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_hidden": {
                                "type": "boolean",
                                "description": "Include players that MA has flagged hide_in_ui (e.g. the web player). Default false.",
                                "default": False,
                            },
                        },
                    },
                ),
                Tool(
                    name="mass_play_media",
                    description=(
                        "Play a media item on a speaker. `uri` comes from a prior mass_search result. "
                        "`player_name` is a speaker display_name from mass_list_players (partial match accepted). "
                        "`mode` controls queue behavior: 'replace' (default — clear queue, play now), "
                        "'next' (insert after current), 'add' (append to end)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "uri": {"type": "string", "description": "MA URI from a prior search result."},
                            "player_name": {"type": "string", "description": "Speaker display_name — partial match accepted."},
                            "mode": {
                                "type": "string",
                                "enum": ["replace", "next", "add"],
                                "description": "Queue behavior. Default 'replace'.",
                                "default": "replace",
                            },
                        },
                        "required": ["uri", "player_name"],
                    },
                ),
                Tool(
                    name="mass_get_queue",
                    description=(
                        "Return what's playing and what's up next on a speaker. Powers "
                        "'what's playing in the living room?' and 'what's next?' questions."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string", "description": "Speaker display_name."},
                            "item_limit": {"type": "integer", "description": "Upcoming items to include. Default 5.", "default": 5},
                        },
                        "required": ["player_name"],
                    },
                ),
                Tool(
                    name="mass_queue_clear",
                    description="Empty the queue on a speaker and stop playback.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string", "description": "Speaker display_name."},
                        },
                        "required": ["player_name"],
                    },
                ),
                Tool(
                    name="mass_play_announcement",
                    description=(
                        "Play a short audio announcement URL on a speaker. Ducks any "
                        "currently playing track on Music Assistant's side and resumes it "
                        "when the announcement finishes. Used by the autonomy engine's "
                        "`speak` delivery channel to play TTS on ESP32 satellites / "
                        "Chromecasts / Google Homes. `player_name` is a display_name from "
                        "mass_list_players. `volume` is 0.0-1.0 (or 0-100 percent)."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string", "description": "Speaker display_name."},
                            "url": {
                                "type": "string",
                                "description": "HTTP(S) URL to the audio clip. Must be reachable from the MA host.",
                            },
                            "volume": {
                                "type": "number",
                                "description": "Optional volume (0.0-1.0 float or 0-100 int). Defaults to MA's configured announcement level.",
                            },
                            "pre_announce": {
                                "type": "boolean",
                                "description": "Optional — play MA's chime before the clip. Defaults to MA's player setting.",
                            },
                        },
                        "required": ["player_name", "url"],
                    },
                ),
                Tool(
                    name="mass_playback_control",
                    description=(
                        "Queue-level actions beyond basic pause/resume (which stay on ha_control_media_player). "
                        "Supported: 'shuffle_on', 'shuffle_off', 'repeat_off', 'repeat_one', 'repeat_all'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "player_name": {"type": "string", "description": "Speaker display_name."},
                            "action": {
                                "type": "string",
                                "enum": [
                                    "shuffle_on", "shuffle_off",
                                    "repeat_off", "repeat_one", "repeat_all",
                                ],
                            },
                        },
                        "required": ["player_name", "action"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            result = await self._dispatch(name, arguments or {})
            return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    async def _dispatch(self, name: str, args: Dict[str, Any]) -> Any:
        if self.agent is None:
            return {
                "error": self.init_error or "Music Assistant agent not initialized",
                "hint": "Set MASS_URL and MASS_TOKEN in .env and restart the agent service.",
            }

        try:
            if name == "mass_search":
                return await self.agent.search(
                    query=args["query"],
                    media_type=args.get("media_type"),
                    limit=int(args.get("limit", 5)),
                )
            if name == "mass_list_players":
                return self.agent.list_players(include_hidden=bool(args.get("include_hidden", False)))
            if name == "mass_play_media":
                return await self.agent.play_media(
                    uri=str(args["uri"]),
                    player_name=str(args["player_name"]),
                    mode=str(args.get("mode", "replace")),
                )
            if name == "mass_get_queue":
                return await self.agent.get_queue(
                    player_name=str(args["player_name"]),
                    item_limit=int(args.get("item_limit", 5)),
                )
            if name == "mass_queue_clear":
                return await self.agent.clear_queue(player_name=str(args["player_name"]))
            if name == "mass_play_announcement":
                volume_arg = args.get("volume")
                pre_arg = args.get("pre_announce")
                return await self.agent.play_announcement(
                    player_name=str(args["player_name"]),
                    url=str(args["url"]),
                    volume=float(volume_arg) if volume_arg is not None else None,
                    pre_announce=bool(pre_arg) if pre_arg is not None else None,
                )
            if name == "mass_playback_control":
                return await self.agent.playback_control(
                    player_name=str(args["player_name"]),
                    action=str(args["action"]),
                )
        except Exception as e:
            logger.error(f"mass tool {name} failed: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

        return {"error": f"unknown tool {name!r}"}


async def main() -> None:
    server_instance = MusicAssistantMCPServer()
    await server_instance.initialize()

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="havencore-music-assistant",
                    server_version="0.1.0",
                    capabilities=server_instance.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        if server_instance.agent is not None:
            await server_instance.agent.disconnect()
