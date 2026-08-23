#!/usr/bin/env python3
"""
Simple MCP Server for HavenCore
Provides MQTT camera-snapshot tools via MCP.

MCP surface is the mcp 2.0 ``MCPServer`` decorator API: the tool is a typed
function registered in ``MQTTServer._build_mcp`` (see
``mcp_reminder_tools/mcp_server.py`` for the pattern). ``HACamSnapper`` — the
paho-MQTT + Home Assistant client layer — is unchanged, and is also imported
lazily by ``mcp_vision_tools`` (keep the import path stable).
"""

import os
import json
import asyncio
from asyncio import Future
from typing import Optional
import paho.mqtt.client as mqtt

from mcp.server import MCPServer

from selene_agent.utils.logger import get_logger

logger = get_logger('loki')

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
        
        # Future for waiting on MQTT responses (+ the loop that owns it, so the
        # paho network thread can resolve it thread-safely).
        self._snapshot_future: Optional[Future] = None
        self._snapshot_loop = None
        
        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Connect and start loop
        self.mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        self.mqtt_client.loop_start()  # This is correct for threaded operation
        
        self.logger = logger
    
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
                
                # Resolve the future if we're waiting for snapshots. This
                # callback runs on paho's network thread, so the asyncio Future
                # must be resolved via the loop's thread-safe scheduler — a bare
                # set_result() does not wake a loop parked in its selector, and
                # the waiter would stall until the wait_for timeout expired.
                self._resolve_snapshot_future(result=self.snapshot_urls)

        except Exception as e:
            self.logger.error(f"Error processing MQTT message: {e}")
            self._resolve_snapshot_future(exc=e)

    def _resolve_snapshot_future(self, result=None, exc=None):
        loop = self._snapshot_loop
        fut = self._snapshot_future
        if not fut or not loop:
            return

        def _set():
            if fut.done():
                return
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)

        try:
            loop.call_soon_threadsafe(_set)
        except RuntimeError:
            # Loop already closed — nothing waiting.
            pass
    
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
        
        # Create a future to wait for MQTT response. Stash the running loop so
        # the paho network-thread callback can resolve the future thread-safely.
        loop = asyncio.get_event_loop()
        self._snapshot_loop = loop
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
            self._snapshot_loop = None

    def cleanup(self):
        """Clean up MQTT connection"""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


class MQTTServer:
    """MCP server providing MQTT camera-snapshot tools.

    ``self.mcp`` is the configured ``MCPServer``; the decorated closure in
    ``_build_mcp`` is the MCP surface and delegates to ``HACamSnapper``.
    """

    def __init__(self):
        self.snapshotter = HACamSnapper(
            ha_url=HAOS_URL,
            ha_token=HAOS_TOKEN,
            mqtt_broker=MQTT_BROKER,
            mqtt_port=MQTT_PORT
        )
        self.mcp = self._build_mcp()

    def _build_mcp(self) -> MCPServer:
        mcp = MCPServer("havencore-mqtt-tools", version="1.0.0")

        # structured_output=False: the result stays a single TextContent JSON
        # string (no outputSchema, no structuredContent) — the pre-MCPServer
        # wire format.

        @mcp.tool(
            name="get_camera_snapshots",
            description="Capture a snapshot from all cameras and return a text description of the images",
            structured_output=False,
        )
        async def get_camera_snapshots() -> str:
            return await self._dispatch("get_camera_snapshots")

        return mcp

    async def _dispatch(self, name: str) -> str:
        """Run the snapshot tool with the old call_tool handler's contract.

        Success is the impl dict as indented JSON; an unexpected exception
        becomes ``Error: <e>`` text in ordinary (non-``isError``) content —
        identical to the hand-dispatch era. The tool used to disappear from
        tools/list while MQTT was down; the decorator surface is static, so a
        broker outage is now reported as an ordinary error result instead
        (server-side fallback over a dynamic tool list).
        """
        logger.info(f"Tool called: {name} with args: {{}}")
        try:
            if not self.snapshotter.mqtt_client.is_connected():
                result = {
                    "success": False,
                    "error": "MQTT not connected; camera snapshots are unavailable right now",
                }
            else:
                result = await self.snapshotter.get_camera_snapshots()
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {str(e)}"


def main():
    """Stdio entry point (``python -m selene_agent.modules.mcp_mqtt_tools``)."""
    logger.info("Starting HavenCore MQTT Tools MCP Server (stdio)...")
    MQTTServer().mcp.run("stdio")


if __name__ == "__main__":
    main()