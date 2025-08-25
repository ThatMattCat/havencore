#!/usr/bin/env python3
"""
Simple MCP Server for HavenCore
Provides general tools like weather and web search via MCP
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
import requests
from datetime import datetime
import pytz

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.types import Tool, TextContent, CallToolResult
from mcp.server.models import InitializationOptions

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import shared.scripts.logger as logger_module
import config

# TODO: Fix logger here
#logger = logger_module.get_logger('loki')
logger = logging.getLogger(__name__)

# Get configuration from environment
WEATHER_API_KEY = config.WEATHER_API_KEY
BRAVE_API_KEY = config.BRAVE_SEARCH_API_KEY
TIMEZONE = config.TIMEZONE


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
                name="echo",
                description="Echo back the input (for testing)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo"
                        }
                    },
                    "required": ["message"]
                }
            ))
            
            logger.info(f"Listing {len(tools)} tools")
            return tools
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> list[types.BaseModel]:
            """Execute a tool"""
            logger.info(f"Tool called: {name} with args: {arguments}")
            
            try:
                if name == "get_weather_forecast":
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

                
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(type="text", text=f"Error: {str(e)}")]

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


async def main():
    """Main entry point"""
    server = GeneralToolsServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())