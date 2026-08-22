#!/usr/bin/env python3
"""Plex MCP Server for HavenCore.

Exposes library search, recent/on-deck listings, client discovery, and
cloud-relay playback on Plex clients (with optional HA wake+launch fallback).

MCP surface is the mcp 2.0 ``MCPServer`` decorator API: the tools are typed
functions registered in ``PlexMCPServer._build_mcp`` (see
``mcp_reminder_tools/mcp_server.py`` for the pattern). The ``PlexAgent``
client layer (blocking ``PlexServer`` behind thread offloads, MyPlex
cloud-relay playback, HA wake+launch) and the synchronous ``initialize()``
the HTTP host / stdio main call before serving are unchanged.
"""

import json
from typing import Annotated, Any, Dict, Optional

from pydantic import Field

from mcp.server import MCPServer

from selene_agent.modules._mcp_params import NULL_OK
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
        self.agent: Optional[PlexAgent] = None
        self.init_error: Optional[str] = None
        self.mcp = self._build_mcp()

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

    def _build_mcp(self) -> MCPServer:
        """Register the decorated tool surface.

        structured_output=False on every tool: results stay a single
        TextContent JSON string, byte-identical to the pre-MCPServer wire
        format (no outputSchema in tools/list, no structuredContent).
        """
        mcp = MCPServer("havencore-plex", version="1.0.0")

        @mcp.tool(
            name="plex_search",
            description=(
                "Search the Plex library for movies, shows, episodes, or music. "
                "Returns up to `limit` items with rating_key (use with plex_play), title, year, type, and summary. "
                "Use this before plex_play to find the exact item to play."
            ),
            structured_output=False,
        )
        async def plex_search(
            query: Annotated[str, Field(
                description="Search text — title, keywords, or partial match.",
            )],
            media_type: Annotated[str, NULL_OK, Field(
                description="Optional filter: 'movie', 'show', 'episode', 'track', 'album', 'artist'.",
            )] = None,
            limit: Annotated[int, NULL_OK, Field(
                description="Max results (default 10).",
            )] = 10,
        ) -> str:
            return await self._call("plex_search", {
                "query": query,
                "media_type": media_type,
                "limit": limit,
            })

        @mcp.tool(
            name="plex_list_recent",
            description="List recently added media across Plex libraries. Use for 'what's new?' queries.",
            structured_output=False,
        )
        async def plex_list_recent(
            media_type: Annotated[str, NULL_OK, Field(
                description="Optional: 'movie', 'show', or 'music'.",
            )] = None,
            limit: Annotated[int, NULL_OK] = 10,
        ) -> str:
            return await self._call("plex_list_recent", {
                "media_type": media_type,
                "limit": limit,
            })

        @mcp.tool(
            name="plex_list_on_deck",
            description="List 'on deck' items — continue-watching queue across the Plex library.",
            structured_output=False,
        )
        async def plex_list_on_deck(
            limit: Annotated[int, NULL_OK] = 10,
        ) -> str:
            return await self._call("plex_list_on_deck", {"limit": limit})

        @mcp.tool(
            name="plex_list_clients",
            description=(
                "List Plex player-capable devices the user can play media on. "
                "Use to discover valid `client_name` values for plex_play, or when the user asks where they can play something."
            ),
            structured_output=False,
        )
        async def plex_list_clients() -> str:
            return await self._call("plex_list_clients", {})

        @mcp.tool(
            name="plex_play",
            description=(
                "Play a specific Plex item on a specific client. "
                "Pass `rating_key` from a prior plex_search / plex_list_recent / plex_list_on_deck result, "
                "and `client_name` from plex_list_clients (or the user's spoken name — partial matches accepted). "
                "If a Home Assistant mapping is configured for the client, the TV is woken and the Plex app is launched first as needed."
            ),
            structured_output=False,
        )
        async def plex_play(
            rating_key: Annotated[str, Field(
                description="Plex rating_key of the item to play.",
            )],
            client_name: Annotated[str, Field(
                description="Plex client/device name — partial match is fine.",
            )],
        ) -> str:
            return await self._call("plex_play", {
                "rating_key": rating_key,
                "client_name": client_name,
            })

        return mcp

    async def _call(self, name: str, args: Dict[str, Any]) -> str:
        """Run one tool with the old call_tool handler's exact contract:
        the dispatched result (success or error dict) as one compact-JSON
        text block in ordinary (non-``isError``) content."""
        result = await self._dispatch(name, args)
        return json.dumps(result, default=str)

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


def main():
    """Stdio entry point (``python -m selene_agent.modules.mcp_plex_tools``)."""
    server_instance = PlexMCPServer()
    server_instance.initialize()
    server_instance.mcp.run("stdio")


if __name__ == "__main__":
    main()
