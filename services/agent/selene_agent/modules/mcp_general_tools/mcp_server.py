#!/usr/bin/env python3
"""
Simple MCP Server for HavenCore
Provides general tools like weather and web search via MCP
"""

import os
import sys
import json
import asyncio
import aiohttp
import logging
from typing import Any, Dict, List, Optional, Union
import requests
from datetime import datetime
import pytz

import base64

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent, CallToolResult
from mcp.server.models import InitializationOptions

from .comfyui_tools import SimpleComfyUI
from .wiki_tools import query_wikipedia
from selene_agent.utils.logger import get_logger

logger = get_logger('loki')

# Get configuration from environment
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
TIMEZONE = os.getenv("TIMEZONE")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")

SIGNAL_API_URL = os.environ.get('SIGNAL_API_URL', 'http://signal-api:8080').rstrip('/')
SIGNAL_PHONE_NUMBER = os.environ.get('SIGNAL_PHONE_NUMBER', '').strip()
SIGNAL_DEFAULT_RECIPIENT = (os.environ.get('SIGNAL_DEFAULT_RECIPIENT', '').strip()
                            or SIGNAL_PHONE_NUMBER)
SIGNAL_MAX_ATTACHMENT_BYTES = 95 * 1024 * 1024  # Signal's practical upload cap is ~100 MB

class GeneralToolsServer:
    """MCP server providing general utility tools"""
    
    def __init__(self):
        self.server = Server("havencore-general-tools")
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            tools = []

            tools.append(Tool(
                name="generate_image",
                description="Generate an image from a text prompt and return the filepath and URL link to the image.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt to generate an image from, written as tags. eg: mountain, snow, realistic"
                        }
                    },
                    "required": ["prompt"]
                }
            ))

            if SIGNAL_PHONE_NUMBER and SIGNAL_DEFAULT_RECIPIENT:
                tools.append(Tool(
                    name="send_signal_message",
                    description="Send a Signal message (text, optionally with images or short videos) to the homeowner.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Message text. Plain text only."
                            },
                            "attachments": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "URL (auto-downloaded) or local file path of an image or short video to attach."
                                }
                            }
                        },
                        "required": ["message"]
                    }
                ))

            tools.append(Tool(
                name="query_multimodal_api",
                description="Send an image (and optional text prompt) to the vision LLM for analysis. Use for camera snapshots, photos, screenshots, etc.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text prompt describing what to analyze in the image (e.g. 'describe what you see', 'is anyone in this image?')."
                        },
                        "image_url": {
                            "type": "string",
                            "description": "HTTP(S) URL to an image. Common formats supported (PNG, JPEG, WebP). The vision service fetches the URL itself."
                        }
                    },
                    "required": ["image_url"],
                    "additionalProperties": False
                }
            ))

            if WOLFRAM_ALPHA_API_KEY:
                tools.append(Tool(
                    name="wolfram_alpha",
                    description="Query Wolfram Alpha for answers to factual questions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Question to ask Wolfram Alpha"
                            }
                        },
                        "required": ["query"]
                    }
                ))

            if WEATHER_API_KEY:
                tools.append(Tool(
                    name="get_weather_forecast",
                    description="Get weather forecast and astronomy data for a location",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name, ZIP code, or coordinates"
                            },
                            "date": {
                                "type": "string",
                                "description": "Date in YYYY-MM-DD format (optional)"
                            }
                        },
                        "required": ["location"]
                    }
                ))
            
            if BRAVE_API_KEY:
                tools.append(Tool(
                    name="brave_search",
                    description="Retrieve a list of relevant websites using Brave Search API",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of results (default: 4)",
                                "default": 4
                            }
                        },
                        "required": ["query"]
                    }
                ))
            
            tools.append(Tool(
                name="search_wikipedia",
                description="Search Wikipedia for information about a topic and return a summary.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "search_string": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "sentences": {
                            "type": "integer",
                            "description": "Number of sentences to return"
                        }
                    },
                    "required": ["search_string"]
                }
            ))
            
            logger.info(f"Listing {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.BaseModel]:
            """Execute a tool"""
            logger.info(f"Tool called: {name} with args: {arguments}")
            
            try:
                if name == "wolfram_alpha":
                    result = await self.wolfram_alpha(arguments.get("query"))
                    return [types.TextContent(type="text", text=result)]
                
                elif name == "generate_image":
                    async with SimpleComfyUI("text-to-image:8188") as comfy:
                        result = await comfy.text_to_image(
                            prompt=arguments.get("prompt"),
                            workflow_name="default"
                        )
                        return [types.TextContent(type="text", text=json.dumps(result))]
                elif name == "send_signal_message":
                    result = await self.send_signal_message(
                        message=arguments.get("message"),
                        attachments=arguments.get("attachments")
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "query_multimodal_api":
                    result = await self.query_multimodal_ai(
                        text=arguments.get("text"),
                        image_url=arguments.get("image_url"),
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "get_weather_forecast":
                    result = await self.get_weather_forecast(
                        arguments.get("location"),
                        arguments.get("date")
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "brave_search":
                    result = await self.brave_search(
                        arguments.get("query"),
                        arguments.get("count", 4)
                    )
                    return [types.TextContent(type="text", text=result)]
    
                elif name == "search_wikipedia":
                    result = await query_wikipedia(arguments.get("search_string"), arguments.get("sentences", 7))
                    return [types.TextContent(type="text", text=result)]

                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    async def send_signal_message(
        self,
        message: str,
        attachments: Optional[Union[str, List[str]]] = None,
        to: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Send a Signal message via signal-cli-rest-api.

        Attachments may be URLs (auto-downloaded) or local file paths; each is
        base64-encoded and passed to ``POST /v2/send``. Recipient defaults to
        ``SIGNAL_DEFAULT_RECIPIENT`` (which itself defaults to
        ``SIGNAL_PHONE_NUMBER`` — i.e. Note to Self).
        """
        sender = SIGNAL_PHONE_NUMBER
        if not sender:
            return '{"success": false, "error": "SIGNAL_PHONE_NUMBER is not configured"}'

        if not to:
            to = SIGNAL_DEFAULT_RECIPIENT
        if not to:
            return '{"success": false, "error": "No recipient specified and no SIGNAL_DEFAULT_RECIPIENT set"}'

        recipients = [to] if isinstance(to, str) else list(to)

        attachment_errors: List[str] = []
        base64_attachments: List[str] = []

        if attachments:
            if isinstance(attachments, str):
                attachments = [attachments]

            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for idx, attachment in enumerate(attachments):
                    try:
                        if not isinstance(attachment, str):
                            raise ValueError(f"Invalid attachment type: {type(attachment)}")

                        if attachment.startswith(('http://', 'https://')):
                            file_content, _ = await self.download_file_safe(attachment, session)
                        else:
                            if not os.path.exists(attachment):
                                raise FileNotFoundError("Local file not found")
                            with open(attachment, 'rb') as f:
                                file_content = f.read()

                        if not file_content:
                            raise ValueError("Attachment is empty")
                        if len(file_content) > SIGNAL_MAX_ATTACHMENT_BYTES:
                            raise ValueError(
                                f"Attachment exceeds {SIGNAL_MAX_ATTACHMENT_BYTES // (1024*1024)} MB cap"
                            )

                        base64_attachments.append(base64.b64encode(file_content).decode('ascii'))
                    except Exception as e:
                        ref = attachment if isinstance(attachment, str) else f"attachment_{idx}"
                        attachment_errors.append(f"{ref}: {str(e)}")

            if attachments and len(attachment_errors) == len(attachments):
                return (f'{{"success": false, "error": "All attachments failed", '
                        f'"details": {json.dumps(attachment_errors)}}}')

        payload: Dict[str, Any] = {
            "message": message or "",
            "number": sender,
            "recipients": recipients,
        }
        if base64_attachments:
            payload["base64_attachments"] = base64_attachments

        url = f"{SIGNAL_API_URL}/v2/send"
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(url, json=payload) as response:
                    body = await response.text()
                    if response.status >= 400:
                        return (f'{{"success": false, "error": "Signal API returned {response.status}", '
                                f'"detail": {json.dumps(body[:500])}}}')
        except asyncio.TimeoutError:
            return '{"success": false, "error": "Signal API send timeout after 60 seconds"}'
        except aiohttp.ClientError as e:
            return f'{{"success": false, "error": "Signal API connection error: {str(e)}"}}'

        if attachment_errors:
            return (f'{{"success": true, "message": "Signal message sent to {", ".join(recipients)}", '
                    f'"warnings": {json.dumps(attachment_errors)}}}')
        return f'{{"success": true, "message": "Signal message sent to {", ".join(recipients)}"}}'


    async def download_file_safe(self, url: str, session: aiohttp.ClientSession) -> tuple:
        """
        Safely download a file from URL with proper error handling.
        
        Args:
            url: URL to download from
            session: aiohttp session to use
            
        Returns:
            tuple: (file_content_bytes, filename)
            
        Raises:
            Various exceptions with descriptive messages
        """
        try:
            async with session.get(url) as response:
                # Check status
                response.raise_for_status()
                
                # Get filename from URL or headers
                filename = None
                if 'content-disposition' in response.headers:
                    cd = response.headers['content-disposition']
                    import re
                    fname_match = re.search(r'filename="?([^"]+)"?', cd)
                    if fname_match:
                        filename = fname_match.group(1)
                
                if not filename:
                    # Extract from URL
                    from urllib.parse import urlparse
                    path = urlparse(url).path
                    filename = os.path.basename(path) if path else 'attachment'
                    if not filename or filename == '/':
                        filename = 'attachment'
                
                # Read content with size limit (e.g., 50MB)
                max_size = 50 * 1024 * 1024  # 50MB
                content = b''
                bytes_read = 0
                
                async for chunk in response.content.iter_chunked(8192):
                    bytes_read += len(chunk)
                    if bytes_read > max_size:
                        raise ValueError(f"File too large: exceeds {max_size/1024/1024:.1f}MB limit")
                    content += chunk
                
                if not content:
                    raise ValueError("Downloaded file is empty")
                    
                return content, filename
                
        except aiohttp.ClientError as e:
            raise ValueError(f"HTTP error downloading {url}: {str(e)}")
        except asyncio.TimeoutError:
            raise ValueError(f"Timeout downloading {url}")
        except Exception as e:
            raise ValueError(f"Failed to download {url}: {str(e)}")

    async def get_weather_forecast(self, location: str, date: Optional[str] = None) -> str:
        """Get weather forecast from weatherapi.com"""
        
        if not WEATHER_API_KEY:
            return "Weather API key not configured"
        
        try:
            base_url = "https://api.weatherapi.com/v1"
            
            if date:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
                local_tz = pytz.timezone(TIMEZONE)
                now_local = datetime.now(local_tz)
                today = now_local.date()
                days_ahead = (target_date - today).days
                
                if 0 <= days_ahead <= 14:
                    url = f"{base_url}/forecast.json"
                    params = {
                        "key": WEATHER_API_KEY,
                        "q": location,
                        "days": min(days_ahead + 1, 14),
                        "dt": date
                    }
                elif 14 < days_ahead <= 365:
                    url = f"{base_url}/future.json"
                    params = {"key": WEATHER_API_KEY, "q": location, "dt": date}
                else:
                    return "Date must be between today and 365 days in the future"
            else:
                url = f"{base_url}/forecast.json"
                params = {"key": WEATHER_API_KEY, "q": location, "days": 1}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Format the response
            loc = data.get("location", {})
            location_info = f"{loc.get('name', 'Unknown')}, {loc.get('region', '')}, {loc.get('country', '')}"
            
            forecast_days = data.get("forecast", {}).get("forecastday", [])
            if not forecast_days:
                return f"No forecast data available for {location_info}"
            
            day_data = forecast_days[0]
            day = day_data.get("day", {})
            astro = day_data.get("astro", {})
            
            response_text = f"""Weather forecast for {location_info}
Date: {day_data.get('date', 'Unknown')}

Temperature:
  High: {day.get('maxtemp_f', 'N/A')}°F ({day.get('maxtemp_c', 'N/A')}°C)
  Low: {day.get('mintemp_f', 'N/A')}°F ({day.get('mintemp_c', 'N/A')}°C)

Conditions: {day.get('condition', {}).get('text', 'Unknown')}
Precipitation: {day.get('totalprecip_in', 0)} in
Humidity: {day.get('avghumidity', 'N/A')}%
Wind: {day.get('maxwind_mph', 'N/A')} mph

Astronomy:
  Sunrise: {astro.get('sunrise', 'N/A')}
  Sunset: {astro.get('sunset', 'N/A')}
  Moon Phase: {astro.get('moon_phase', 'N/A')}"""
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except Exception as e:
            return f"Error processing weather request: {str(e)}"
    
    async def brave_search(self, query: str, count: int = 4) -> str:
        """Search using Brave Search API"""
        
        if not BRAVE_API_KEY:
            return "Brave Search API key not configured"
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY
            }
            params = {"q": query, "count": count}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            results = response.json().get('web', {}).get('results', [])
            formatted_results = []
            
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. {result['title']}\n   {result['url']}\n   {result.get('description', '')}"
                )
            
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
            
        except Exception as e:
            return f"Error searching: {str(e)}"
    
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

    async def wolfram_alpha(
        self,
        query: str,
        max_chars: Optional[int] = 1000,
        timeout: Optional[int] = 30
    ) -> str:
        """ Query the WolframAlpha LLM API."""
        
        url = "https://www.wolframalpha.com/api/v1/llm-api"
        
        params = {
            "input": query,
            "appid": WOLFRAM_ALPHA_API_KEY,
            "maxchars": max_chars
        }
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error occurred: {e}"
            if response.status_code == 403:
                error_msg = "Invalid API key or unauthorized access"
            elif response.status_code == 400:
                error_msg = "Bad request - check your query format"
            raise ValueError(error_msg)
        except requests.exceptions.ConnectionError:
            raise ValueError("Failed to connect to WolframAlpha API")
        except requests.exceptions.Timeout:
            raise ValueError(f"Request timed out after {timeout} seconds") 
        except requests.exceptions.RequestException as e:
            raise ValueError(f"An error occurred: {e}")
        return response.text

    async def query_multimodal_ai(
        self,
        text: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> str:
        """Query the vision model via the agent's /api/vision/ask_url chokepoint."""
        if not (text or image_url):
            raise ValueError("At least one of text or image_url must be provided")

        payload = {"text": text, "image_url": image_url}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://agent:6002/api/vision/ask_url",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=180),
            ) as response:
                data = await response.json()
                if response.status >= 400:
                    detail = data.get("detail") if isinstance(data, dict) else str(data)
                    raise ValueError(f"Vision API error ({response.status}): {detail}")
                try:
                    return data["response"]
                except (KeyError, TypeError) as e:
                    raise ValueError(f"Unexpected response structure: {e}") from e


async def main():
    """Main entry point"""
    server = GeneralToolsServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())