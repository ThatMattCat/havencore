#!/usr/bin/env python3
"""Plex MCP Server for HavenCore.

Exposes library search, recent/on-deck listings, client discovery, and
cloud-relay playback on Plex clients (with optional HA wake+launch fallback).
"""

import json
from typing import Any, Dict, List, Optional

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool
import mcp.types as types

from selene_agent.utils import config as agent_config
from selene_agent.utils.logger import get_logger

from .plex_client import PlexAgent

logger = get_logger("loki")

PLEX_URL = getattr(agent_config, "PLEX_URL", None)
PLEX_TOKEN = getattr(agent_config, "PLEX_TOKEN", None)
HAOS_URL = getattr(agent_config, "HAOS_URL", None)
HAOS_TOKEN = getattr(agent_config, "HAOS_TOKEN", None)
PLEX_CLIENT_HA_MAP_RAW = getattr(agent_config, "PLEX_CLIENT_HA_MAP", "")

TEST_MODE = (
    not PLEX_URL
    or not PLEX_TOKEN
    or PLEX_TOKEN == "NO_PLEX_TOKEN_CONFIGURED"
)


def _parse_ha_map(raw: str) -> Dict[str, Dict[str, Any]]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning(f"Could not parse PLEX_CLIENT_HA_MAP as JSON: {e}")
        return {}


class PlexMCPServer:
    def __init__(self):
        self.server: Server = Server("havencore-plex")
        self.agent: Optional[PlexAgent] = None
        self.init_error: Optional[str] = None
        self._setup_handlers()

    def initialize(self):
        if TEST_MODE:
            self.init_error = "PLEX_URL / PLEX_TOKEN not configured"
            logger.warning("Plex MCP running without credentials — tools will return an error")
            return
        try:
            ha_map = _parse_ha_map(PLEX_CLIENT_HA_MAP_RAW)
            self.agent = PlexAgent(
                plex_url=PLEX_URL,
                plex_token=PLEX_TOKEN,
                ha_url=HAOS_URL,
                ha_token=HAOS_TOKEN,
                client_ha_map=ha_map,
            )
            logger.info(
                f"Plex MCP initialized (url={PLEX_URL}, "
                f"ha_map_keys={list(ha_map.keys())})"
            )
        except Exception as e:
            self.init_error = f"{type(e).__name__}: {e}"
            logger.error(f"Failed to init Plex agent: {e}")

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="plex_search",
                    description=(
                        "Search the Plex library for movies, shows, episodes, or music. "
                        "Returns up to `limit` items with rating_key (use with plex_play), title, year, type, and summary. "
                        "Use this before plex_play to find the exact item to play."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search text — title, keywords, or partial match."},
                            "media_type": {
                                "type": "string",
                                "description": "Optional filter: 'movie', 'show', 'episode', 'track', 'album', 'artist'.",
                            },
                            "limit": {"type": "integer", "description": "Max results (default 10).", "default": 10},
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="plex_list_recent",
                    description="List recently added media across Plex libraries. Use for 'what's new?' queries.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "media_type": {
                                "type": "string",
                                "description": "Optional: 'movie', 'show', or 'music'.",
                            },
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                ),
                Tool(
                    name="plex_list_on_deck",
                    description="List 'on deck' items — continue-watching queue across the Plex library.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                ),
                Tool(
                    name="plex_list_clients",
                    description=(
                        "List Plex player-capable devices the user can play media on. "
                        "Use to discover valid `client_name` values for plex_play, or when the user asks where they can play something."
                    ),
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="plex_play",
                    description=(
                        "Play a specific Plex item on a specific client. "
                        "Pass `rating_key` from a prior plex_search / plex_list_recent / plex_list_on_deck result, "
                        "and `client_name` from plex_list_clients (or the user's spoken name — partial matches accepted). "
                        "If a Home Assistant mapping is configured for the client, the TV is woken and the Plex app is launched first as needed."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "rating_key": {"type": "string", "description": "Plex rating_key of the item to play."},
                            "client_name": {"type": "string", "description": "Plex client/device name — partial match is fine."},
                        },
                        "required": ["rating_key", "client_name"],
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
                "error": self.init_error or "Plex agent not initialized",
                "hint": "Set PLEX_URL and PLEX_TOKEN in .env and restart the agent service.",
            }

        try:
            if name == "plex_search":
                return await self.agent.search(
                    query=args["query"],
                    media_type=args.get("media_type"),
                    limit=int(args.get("limit", 10)),
                )
            if name == "plex_list_recent":
                return await self.agent.list_recent(
                    media_type=args.get("media_type"),
                    limit=int(args.get("limit", 10)),
                )
            if name == "plex_list_on_deck":
                return await self.agent.list_on_deck(limit=int(args.get("limit", 10)))
            if name == "plex_list_clients":
                return await self.agent.list_clients()
            if name == "plex_play":
                return await self.agent.play(
                    rating_key=str(args["rating_key"]),
                    client_name=args["client_name"],
                )
        except Exception as e:
            logger.error(f"plex tool {name} failed: {e}")
            return {"error": f"{type(e).__name__}: {e}"}

        return {"error": f"unknown tool {name!r}"}


async def main():
    server_instance = PlexMCPServer()
    server_instance.initialize()

    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="havencore-plex",
                server_version="0.1.0",
                capabilities=server_instance.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
