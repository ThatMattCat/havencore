#!/usr/bin/env python3
"""
Home Assistant MCP Server for HavenCore - Simple Test Version
Basic version for testing the MCP server functionality
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomeAssistantMCPServerSimple:
    """Simple MCP server for testing Home Assistant tools"""
    
    def __init__(self):
        self.server = Server("havencore-homeassistant-simple")
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available Home Assistant tools"""
            tools = [
                Tool(
                    name="ha_test_tool",
                    description="Simple test tool to verify Home Assistant MCP server is working",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Test message to echo back"
                            }
                        },
                        "required": ["message"]
                    }
                ),
                Tool(
                    name="ha_get_domain_entity_states",
                    description="Get the current states of all entities in a Home Assistant domain",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "The Home Assistant domain (e.g., 'light', 'switch')"
                            }
                        },
                        "required": ["domain"]
                    }
                )
            ]
            
            logger.info(f"Listing {len(tools)} Home Assistant tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.BaseModel]:
            """Execute a Home Assistant tool"""
            logger.info(f"Home Assistant tool called: {name} with args: {arguments}")
            
            try:
                if name == "ha_test_tool":
                    message = arguments.get("message", "No message provided")
                    result = f"Home Assistant MCP Server received: {message}"
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "ha_get_domain_entity_states":
                    domain = arguments.get("domain", "unknown")
                    # Simulate getting entity states
                    result = f"Mock entity states for domain '{domain}': entity1: on, entity2: off"
                    return [types.TextContent(type="text", text=result)]
                
                else:
                    return [types.TextContent(type="text", text=f"Unknown Home Assistant tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing Home Assistant tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting HavenCore Home Assistant MCP Server (Simple)...")
        
        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Home Assistant MCP Server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="HavenCore Home Assistant MCP Server (Simple)",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    )
                )
            )


async def main():
    """Main entry point"""
    server = HomeAssistantMCPServerSimple()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())