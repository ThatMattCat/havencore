#!/usr/bin/env python3
"""
Simple MCP Server for HavenCore
Provides general tools like weather and web search via MCP
"""

import os
import json
import asyncio
from asyncio import Future
import logging
from typing import Any, Dict, List, Optional
import paho.mqtt.client as mqtt

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent, CallToolResult
from mcp.server.models import InitializationOptions

# TODO: Fix logger here
logger = logging.getLogger(__name__)

HAOS_URL = os.getenv("HAOS_URL", "NO_HAOS_URL_SET")
HAOS_TOKEN = os.getenv("HAOS_TOKEN", "NO_HAOS_TOKEN_SET")
MQTT_BROKER = os.getenv("MQTT_BROKER", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))

class HACamSnapper:
    def __init__(self, ha_url, ha_token, mqtt_broker: str = "mosquitto", mqtt_port: int = 1883):
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.snapshot_urls = []
        
        # Future for waiting on MQTT responses
        self._snapshot_future: Optional[Future] = None
        
        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Connect and start loop
        self.mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        self.mqtt_client.loop_start()  # This is correct for threaded operation
        
        self.logger = logging.getLogger(__name__)
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            # Subscribe to snapshot notifications
            client.subscribe("home/cameras/snapshots")
            client.subscribe("home/cameras/cleanup/status")
            self.logger.info("MQTT connected and subscribed")
        else:
            self.logger.error(f"MQTT connection failed with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self.logger.warning(f"Unexpected MQTT disconnection. Code: {rc}")
            # You might want to implement reconnection logic here
    
    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == "home/cameras/snapshots":
                self.snapshot_urls = payload.get('urls', [])
                self.logger.info(f"Received snapshot URLs: {self.snapshot_urls}")
                
                # Resolve the future if we're waiting for snapshots
                if self._snapshot_future and not self._snapshot_future.done():
                    self._snapshot_future.set_result(self.snapshot_urls)
                    
        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
            if self._snapshot_future and not self._snapshot_future.done():
                self._snapshot_future.set_exception(e)
    
    async def get_camera_snapshots(self, timeout: float = 10.0) -> dict:
        """
        Trigger snapshot capture from all cameras (async version)
        
        Args:
            timeout: Maximum time to wait for MQTT response (seconds)
        """
        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json"
        }
        
        # Clear previous URLs
        self.snapshot_urls = []
        
        # Create a future to wait for MQTT response
        loop = asyncio.get_event_loop()
        self._snapshot_future = loop.create_future()
        
        try:
            # Make the HTTP request async
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ha_url}/services/script/capture_all_cameras",
                    headers=headers,
                    json={}
                ) as response:
                    if response.status != 200:
                        return {
                            "success": False,
                            "error": f"Failed to trigger snapshots: {response.status}"
                        }
            
            self.logger.info("Snapshot capture triggered, waiting for MQTT response...")
            
            # Wait for MQTT message with timeout
            try:
                urls = await asyncio.wait_for(self._snapshot_future, timeout=timeout)
                return {
                    "success": True,
                    "message": "Snapshots captured",
                    "urls": urls
                }
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout waiting for snapshot URLs after {timeout}s")
                return {
                    "success": False,
                    "error": f"Timeout waiting for snapshot URLs after {timeout} seconds",
                    "partial_urls": self.snapshot_urls  # Return any URLs we did get
                }
                
        except Exception as e:
            self.logger.error(f"Error triggering snapshot capture: {e}")
            return {
                "success": False,
                "error": f"Failed to trigger snapshots: {e}"
            }
        finally:
            self._snapshot_future = None
    
    def cleanup(self):
        """Clean up MQTT connection"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


class MQTTServer:
    """MCP server providing general utility tools"""
    
    def __init__(self):
        self.server = Server("havencore-general-tools")
        self.snapshotter = HACamSnapper(
            ha_url=HAOS_URL,
            ha_token=HAOS_TOKEN,
            mqtt_broker=MQTT_BROKER,
            mqtt_port=MQTT_PORT
        )   
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            tools = []

            if self.snapshotter.mqtt_client.is_connected():
                tools.append(Tool(
                    name="get_camera_snapshots",
                    description="Capture a snapshot from all cameras and return a text description of the images",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ))
            
            logger.info(f"Listing {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.BaseModel]:
            """Execute a tool"""
            logger.info(f"Tool called: {name} with args: {arguments if arguments else '{}'}")
            
            try:
                if name == "get_camera_snapshots":
                    result = await self.snapshotter.get_camera_snapshots()
                    result = json.dumps(result, indent=2)
                    return [types.TextContent(type="text", text=result)]
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting HavenCore General Tools MCP Server...")
        
        # Run the stdio server
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="HavenCore General Tools MCP Server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    )
                )
            )

async def main():
    """Main entry point"""
    server = MQTTServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())