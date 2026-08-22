#!/usr/bin/env python3
"""Music Assistant MCP Server for HavenCore.

Exposes cross-provider search, player enumeration, and queue-aware playback
on Music Assistant — the audio-only counterpart to the Plex module, targeting
Chromecasts / Google Homes / ESP32 satellites.

MCP surface is the mcp 2.0 ``MCPServer`` decorator API: the tools are typed
functions registered in ``MusicAssistantMCPServer._build_mcp`` (see
``mcp_reminder_tools/mcp_server.py`` for the pattern). The ``MassAgent``
client layer (long-lived Music Assistant WebSocket opened by the async
``initialize()``, closed by ``disconnect()``) is unchanged; the HTTP host
loader and the stdio ``main()`` both keep that init/cleanup lifecycle
around serving.
"""

import json
from typing import Annotated, Any, Dict, Literal, Optional

import anyio
from pydantic import Field

from mcp.server import MCPServer

from selene_agent.modules._mcp_params import NULL_OK
from selene_agent.utils import config as agent_config
from selene_agent.utils.logger import get_logger

from .mass_client import MassAgent

logger = get_logger("loki")

MASS_URL = getattr(agent_config, "MASS_URL", None)
MASS_TOKEN = getattr(agent_config, "MASS_TOKEN", None)

TEST_MODE = not MASS_URL or not MASS_TOKEN


class MusicAssistantMCPServer:
    def __init__(self):
        self.agent: Optional[MassAgent] = None
        self.init_error: Optional[str] = None
        self.mcp = self._build_mcp()

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

    def _build_mcp(self) -> MCPServer:
        """Register the decorated tool surface.

        structured_output=False on every tool: results stay a single
        TextContent JSON string, byte-identical to the pre-MCPServer wire
        format (no outputSchema in tools/list, no structuredContent).
        """
        mcp = MCPServer("havencore-music-assistant", version="1.0.0")

        @mcp.tool(
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
            structured_output=False,
        )
        async def mass_search(
            query: Annotated[str, Field(
                description="Search text (title, artist, keywords).",
            )],
            media_type: Annotated[Literal["track", "album", "artist", "playlist", "radio"], NULL_OK, Field(
                description="Optional filter for result type.",
            )] = None,
            limit: Annotated[int, NULL_OK, Field(
                description="Max items per bucket (default 5).",
            )] = 5,
        ) -> str:
            return await self._call("mass_search", {
                "query": query,
                "media_type": media_type,
                "limit": limit,
            })

        @mcp.tool(
            name="mass_list_players",
            description=(
                "Enumerate Music Assistant players (speakers). Returns `player_id`, "
                "`display_name`, `available`, `powered`, `state`, `volume_level`, and "
                "`current_item`. Hidden players are excluded unless `include_hidden=true`. "
                "Use the `display_name` values as `player_name` inputs to other tools."
            ),
            structured_output=False,
        )
        async def mass_list_players(
            include_hidden: Annotated[bool, NULL_OK, Field(
                description="Include players that MA has flagged hide_in_ui (e.g. the web player). Default false.",
            )] = False,
        ) -> str:
            return await self._call("mass_list_players", {
                "include_hidden": include_hidden,
            })

        @mcp.tool(
            name="mass_play_media",
            description=(
                "Play a media item on a speaker. `uri` comes from a prior mass_search result. "
                "`player_name` is a speaker display_name from mass_list_players (partial match accepted). "
                "`mode` controls queue behavior: 'replace' (default — clear queue, play now), "
                "'next' (insert after current), 'add' (append to end)."
            ),
            structured_output=False,
        )
        async def mass_play_media(
            uri: Annotated[str, Field(
                description="MA URI from a prior search result.",
            )],
            player_name: Annotated[str, Field(
                description="Speaker display_name — partial match accepted.",
            )],
            mode: Annotated[Literal["replace", "next", "add"], NULL_OK, Field(
                description="Queue behavior. Default 'replace'.",
            )] = "replace",
        ) -> str:
            return await self._call("mass_play_media", {
                "uri": uri,
                "player_name": player_name,
                "mode": mode,
            })

        @mcp.tool(
            name="mass_get_queue",
            description=(
                "Return what's playing and what's up next on a speaker. Powers "
                "'what's playing in the living room?' and 'what's next?' questions."
            ),
            structured_output=False,
        )
        async def mass_get_queue(
            player_name: Annotated[str, Field(
                description="Speaker display_name.",
            )],
            item_limit: Annotated[int, NULL_OK, Field(
                description="Upcoming items to include. Default 5.",
            )] = 5,
        ) -> str:
            return await self._call("mass_get_queue", {
                "player_name": player_name,
                "item_limit": item_limit,
            })

        @mcp.tool(
            name="mass_queue_clear",
            description="Empty the queue on a speaker and stop playback.",
            structured_output=False,
        )
        async def mass_queue_clear(
            player_name: Annotated[str, Field(
                description="Speaker display_name.",
            )],
        ) -> str:
            return await self._call("mass_queue_clear", {"player_name": player_name})

        @mcp.tool(
            name="mass_play_announcement",
            description=(
                "Play a short audio announcement URL on a speaker. Ducks any "
                "currently playing track on Music Assistant's side and resumes it "
                "when the announcement finishes. Used by the autonomy engine's "
                "`speak` delivery channel to play TTS on ESP32 satellites / "
                "Chromecasts / Google Homes. `player_name` is a display_name from "
                "mass_list_players. `volume` is 0.0-1.0 (or 0-100 percent)."
            ),
            structured_output=False,
        )
        async def mass_play_announcement(
            player_name: Annotated[str, Field(
                description="Speaker display_name.",
            )],
            url: Annotated[str, Field(
                description="HTTP(S) URL to the audio clip. Must be reachable from the MA host.",
            )],
            volume: Annotated[float, NULL_OK, Field(
                description="Optional volume (0.0-1.0 float or 0-100 int). Defaults to MA's configured announcement level.",
            )] = None,
            pre_announce: Annotated[bool, NULL_OK, Field(
                description="Optional — play MA's chime before the clip. Defaults to MA's player setting.",
            )] = None,
        ) -> str:
            return await self._call("mass_play_announcement", {
                "player_name": player_name,
                "url": url,
                "volume": volume,
                "pre_announce": pre_announce,
            })

        @mcp.tool(
            name="mass_playback_control",
            description=(
                "Queue-level actions beyond basic pause/resume (which stay on ha_control_media_player). "
                "Supported: 'shuffle_on', 'shuffle_off', 'repeat_off', 'repeat_one', 'repeat_all'."
            ),
            structured_output=False,
        )
        async def mass_playback_control(
            player_name: Annotated[str, Field(
                description="Speaker display_name.",
            )],
            action: Literal[
                "shuffle_on", "shuffle_off",
                "repeat_off", "repeat_one", "repeat_all",
            ],
        ) -> str:
            return await self._call("mass_playback_control", {
                "player_name": player_name,
                "action": action,
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


async def _stdio_main() -> None:
    """Connect to Music Assistant, serve stdio, disconnect on exit.

    Same lifecycle as the pre-MCPServer async main: the WebSocket must be
    opened and closed on the event loop that serves the session, so the
    synchronous ``main()`` runs this wrapper instead of ``mcp.run("stdio")``
    (which could not await ``initialize()`` / ``disconnect()``).
    """
    server_instance = MusicAssistantMCPServer()
    await server_instance.initialize()
    try:
        await server_instance.mcp.run_stdio_async()
    finally:
        if server_instance.agent is not None:
            await server_instance.agent.disconnect()


def main() -> None:
    """Stdio entry point (``python -m selene_agent.modules.mcp_music_assistant_tools``)."""
    # Same event-loop runner MCPServer.run("stdio") uses internally.
    anyio.run(_stdio_main)


if __name__ == "__main__":
    main()
