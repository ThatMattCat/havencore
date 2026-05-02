#!/usr/bin/env python3
"""MCP server: device-side actions for HavenCore.

Tools in this module do not perform their action server-side — they exist so
the LLM has a discoverable capability to call, and so the orchestrator has a
``tool_call`` to attach a ``device_action`` event to. The companion app on the
session's device receives that event over ``/ws/chat`` and fires the matching
platform intent (e.g. ``AlarmClock.ACTION_SET_ALARM``).

To add a new device action: declare a Tool here, add a handler that returns a
structured status string, and add the tool name to
``orchestrator.DEVICE_ACTION_TOOLS``.
"""
import asyncio
import json
from typing import Any, Dict, List

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.types import Tool

from selene_agent.utils.logger import get_logger

logger = get_logger('loki')


class DeviceActionToolsServer:
    """MCP server exposing device-side action tools."""

    def __init__(self):
        self.server = Server("havencore-device-action-tools")
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            tools = [
                Tool(
                    name="set_alarm",
                    description=(
                        "Schedule an alarm on the user's phone via the device's "
                        "alarm clock app. Use this when the user asks to set an alarm, "
                        "schedule a wake-up, or be reminded at a specific time of day. "
                        "Hour and minute are local to the device. For one-off alarms "
                        "omit days_of_week; for repeating alarms supply 1=Sunday, "
                        "2=Monday, ..., 7=Saturday."
                    ),
                    inputSchema={
                        "type": "object",
                        "required": ["hour", "minute"],
                        "properties": {
                            "hour":   {"type": "integer", "minimum": 0, "maximum": 23},
                            "minute": {"type": "integer", "minimum": 0, "maximum": 59},
                            "label":  {"type": "string", "description": "Optional label."},
                            "days_of_week": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 1, "maximum": 7},
                                "description": "Repeating days. 1=Sun..7=Sat."
                            }
                        }
                    },
                ),
            ]
            logger.info(f"Listing {len(tools)} device-action tools")
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.BaseModel]:
            logger.info(f"Device-action tool called: {name}")
            try:
                if name == "set_alarm":
                    result = await self.set_alarm(arguments)
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                return [types.TextContent(type="text", text=json.dumps(result))]
            except Exception as e:
                logger.exception(f"device-action tool {name} failed")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": str(e)})
                )]

    async def set_alarm(self, args: Dict[str, Any]) -> Dict[str, Any]:
        hour = args.get("hour")
        minute = args.get("minute")
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            return {"status": "error", "error": "hour must be an integer 0..23"}
        if not isinstance(minute, int) or not (0 <= minute <= 59):
            return {"status": "error", "error": "minute must be an integer 0..59"}

        label = args.get("label")
        days_of_week = args.get("days_of_week") or []
        if not isinstance(days_of_week, list) or not all(
            isinstance(d, int) and 1 <= d <= 7 for d in days_of_week
        ):
            return {
                "status": "error",
                "error": "days_of_week must be a list of integers 1..7",
            }

        return {
            "status": "scheduled",
            "hour": hour,
            "minute": minute,
            "label": label,
            "days_of_week": days_of_week,
        }

    async def run(self):
        logger.info("Starting HavenCore Device Action Tools MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Device-action MCP server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="HavenCore Device Action Tools MCP Server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    server = DeviceActionToolsServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
