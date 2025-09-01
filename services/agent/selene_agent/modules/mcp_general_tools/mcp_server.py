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

import asyncio
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import aiosmtplib

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent, CallToolResult
from mcp.server.models import InitializationOptions

from .wiki_tools import query_wikipedia
# import shared.scripts.logger as logger_module
# import selene_agent.config as config

# TODO: Fix logger here
#logger = logger_module.get_logger('loki')
logger = logging.getLogger(__name__)

# Get configuration from environment
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
TIMEZONE = os.getenv("TIMEZONE")
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")

SENDER_EMAIL = os.environ.get('GMAIL_ADDRESS')
EMAIL_APP_PASSWORD = os.environ.get('GMAIL_APP_PASSWORD')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
DEFAULT_RECIPIENT = os.environ.get('DEFAULT_RECIPIENT')

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

            if EMAIL_APP_PASSWORD and SENDER_EMAIL:
                tools.append(Tool(
                    name="send_email",
                    description="Send an email to the homeowner. HTML supported.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            # "to": {
                            #     "type": "string",
                            #     "description": "Recipient email address"
                            # },
                            "subject": {
                                "type": "string",
                                "description": "Email subject"
                            },
                            "body": {
                                "type": "string",
                                "description": "Email body with HTML support"
                            }
                        },
                        "required": ["subject", "body"]
                    }
                ))

            tools.append(Tool(
                name="query_multimodal_api",
                description="Query a multimodal AI API with any combination of text, image, audio, and/or video inputs to get AI-generated responses",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text content to send to the AI for processing"
                        },
                        "image_url": {
                            "type": "string",
                            "description": "URL or file path (file://) to an image for visual analysis. Supports common image formats like PNG, JPEG, etc."
                        },
                        "audio_url": {
                            "type": "string",
                            "description": "URL or file path (file://) to an audio file for audio analysis. Supports common audio formats like WAV, MP3, etc."
                        },
                        "video_url": {
                            "type": "string",
                            "description": "URL or file path (file://) to a video file for video analysis. Supports common video formats like MP4, AVI, etc."
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                    "anyOf": [
                        {"required": ["text"]},
                        {"required": ["image_url"]},
                        {"required": ["audio_url"]},
                        {"required": ["video_url"]}
                    ]
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
                    description="Search the web using Brave Search API",
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
                name="calculate",
                description="Perform basic mathematical calculations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
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
                
                elif name == "send_email":
                    result = await self.send_email(
                        # to=arguments.get("to"),
                        subject=arguments.get("subject"),
                        body=arguments.get("body")
                    )
                    return [types.TextContent(type="text", text=result)]

                elif name == "query_multimodal_api":
                    result = await self.query_multimodal_ai(
                        text=arguments.get("text"),
                        image_url=arguments.get("image_url"),
                        audio_url=arguments.get("audio_url"),
                        video_url=arguments.get("video_url")
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
                
                elif name == "calculate":
                    result = await self.calculate(arguments.get("expression"))
                    return [types.TextContent(type="text", text=result)]

                elif name == "echo":
                    message = arguments.get("message", "")
                    return [types.TextContent(type="text", text=f"Echo: {message}")]
                
                elif name == "search_wikipedia":
                    result = await query_wikipedia(arguments.get("search_string"), arguments.get("sentences", 7))
                    return [types.TextContent(type="text", text=result)]

                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]
            
    async def send_email(self,
        subject: str, 
        body: str,
        to: Optional[Union[str, List[str]]] = None, 
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        html: bool = True
    ) -> str:
        """
        Send an email using Gmail SMTP asynchronously.
        
        Environment variables required:
        - GMAIL_ADDRESS: Your Gmail address
        - GMAIL_APP_PASSWORD: Your Gmail app-specific password
        - SMTP_SERVER: SMTP server (default: smtp.gmail.com)
        - SMTP_PORT: SMTP port (default: 587)
        
        Returns:
            str: Success status and message
        """
        # Get config from environment
        # sender_email = os.environ.get('GMAIL_ADDRESS')
        # password = os.environ.get('GMAIL_APP_PASSWORD')
        # smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        # smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        # default_recipient = os.environ.get('DEFAULT_RECIPIENT')

        if not SENDER_EMAIL or not EMAIL_APP_PASSWORD:
            return '{"success": False,"error": "Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD environment variables"}'
        if not to:
            to = DEFAULT_RECIPIENT

        # Normalize recipients to lists
        to_list = [to] if isinstance(to, str) else to
        cc_list = [] if cc is None else ([cc] if isinstance(cc, str) else cc)
        bcc_list = [] if bcc is None else ([bcc] if isinstance(bcc, str) else bcc)
        
        try:
            # Create message
            msg = MIMEMultipart('alternative') if html else MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = ', '.join(to_list)
            msg['Subject'] = subject
            
            if cc_list:
                msg['Cc'] = ', '.join(cc_list)
            
            # Add body
            mime_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, mime_type))
            
            # All recipients for sending
            all_recipients = to_list + cc_list + bcc_list
            
            # Send email asynchronously
            await aiosmtplib.send(
                msg,
                sender=SENDER_EMAIL,
                recipients=all_recipients,
                hostname=SMTP_SERVER,
                port=SMTP_PORT,
                start_tls=True,
                username=SENDER_EMAIL,
                password=EMAIL_APP_PASSWORD,
            )
            return_message = f"Email sent successfully to {', '.join(all_recipients)}"
            return f'{{"success": True,"message": "{return_message}"}}'

        except Exception as e:
            return f'{{"success": false, "error": "{str(e)}"}}'

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
    
    async def calculate(self, expression: str) -> str:
        """Safely evaluate mathematical expressions"""
        try:
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len
            }
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
            
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"
    
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
        audio_url: Optional[str] = None,
        video_url: Optional[str] = None
    ) -> str:
        """
        Query a multimodal AI API with text, image, audio, and/or video inputs.
        
        Args:
            text: The text content to send to the AI
            image_url: URL or file path (file://) to an image
            audio_url: URL or file path (file://) to an audio file
            video_url: URL or file path (file://) to a video file
            system_prompt: System message to set AI behavior
            
        Returns:
            Dict containing the API response
            
        Raises:
            ValueError: If no content is provided
            aiohttp.ClientError: If the API request fails
        """
        # Validate that at least one content type is provided
        if not any([text, image_url, audio_url, video_url]):
            raise ValueError("At least one of text, image_url, audio_url, or video_url must be provided")
        
        # Build the content array for the user message
        content = []
        system_prompt = "You are a helpful assistant."
        
        if text:
            content.append({"type": "text", "text": text})
        
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        if audio_url:
            content.append({
                "type": "audio_url", 
                "audio_url": {"url": audio_url}
            })
        
        if video_url:
            content.append({
                "type": "video_url",
                "video_url": {"url": video_url}
            })
        # Construct the request payload
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        }
        # logger.info(f"Multimodal API payload: {json.dumps(payload, indent=2)}")
        
        # Make the async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://nginx/iav/api",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    response.raise_for_status()
                data = await response.json()
                
                # Extract content with error handling
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as e:
                    # Handle missing keys or empty choices array
                    raise ValueError(f"Unexpected response structure: {e}") from e


async def main():
    """Main entry point"""
    server = GeneralToolsServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())