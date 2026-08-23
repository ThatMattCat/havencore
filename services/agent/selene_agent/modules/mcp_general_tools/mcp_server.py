#!/usr/bin/env python3
"""
Simple MCP Server for HavenCore
Provides general tools like weather and web search via MCP
"""

import os
import json
import asyncio
import aiohttp
import logging
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Optional, Union
import requests
from datetime import datetime
import pytz

import base64

from pydantic import Field

from mcp.server import MCPServer

from selene_agent.modules._mcp_params import NULL_OK
from . import fetch_tools
from .comfyui_tools import SimpleComfyUI
from .wiki_tools import query_wikipedia
from selene_agent.utils.logger import get_logger

logger = get_logger('loki')

# Get configuration from environment
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
# Nothing in the deployment sets TIMEZONE — compose/.env define CURRENT_TIMEZONE
# (via shared config) and TZ. Reading the wrong name left this None, and
# pytz.timezone(None) raised on every dated forecast query.
TIMEZONE = os.getenv("CURRENT_TIMEZONE") or os.getenv("TZ") or "UTC"
WOLFRAM_ALPHA_API_KEY = os.getenv("WOLFRAM_ALPHA_API_KEY")

SIGNAL_API_URL = os.environ.get('SIGNAL_API_URL', 'http://signal-api:8080').rstrip('/')
SIGNAL_PHONE_NUMBER = os.environ.get('SIGNAL_PHONE_NUMBER', '').strip()
SIGNAL_DEFAULT_RECIPIENT = (os.environ.get('SIGNAL_DEFAULT_RECIPIENT', '').strip()
                            or SIGNAL_PHONE_NUMBER)
SIGNAL_MAX_ATTACHMENT_BYTES = 95 * 1024 * 1024  # Signal's practical upload cap is ~100 MB

class GeneralToolsServer:
    """MCP server providing general utility tools.

    ``self.mcp`` is the configured mcp 2.0 ``MCPServer``; the decorated
    closures in ``_build_mcp`` are the MCP surface and delegate to the
    existing async impl methods (unchanged from the hand-dispatch era).
    Registration is env-gated exactly like the old list_tools handler:
    tools whose backing API key/number is unset are never registered.
    """

    def __init__(self):
        self.mcp = self._build_mcp()

    def _build_mcp(self) -> MCPServer:
        """Register the decorated tool surface.

        structured_output=False on every tool: results stay a single
        TextContent string, byte-identical to the pre-MCPServer wire format
        (no outputSchema in tools/list, no structuredContent).
        """
        mcp = MCPServer("havencore-general-tools", version="1.0.0")

        @mcp.tool(
            name="generate_image",
            description="Generate an image from a text prompt and return the filepath and URL link to the image.",
            structured_output=False,
        )
        async def generate_image(
            prompt: Annotated[str, Field(description=(
                "The text prompt to generate an image from, written as tags. "
                "eg: mountain, snow, realistic"
            ))],
        ) -> str:
            async def run() -> str:
                async with SimpleComfyUI("text-to-image:8188") as comfy:
                    result = await comfy.text_to_image(
                        prompt=prompt,
                        workflow_name="default"
                    )
                    return json.dumps(result)
            return await self._call("generate_image", {"prompt": prompt}, run)

        if SIGNAL_PHONE_NUMBER and SIGNAL_DEFAULT_RECIPIENT:
            @mcp.tool(
                name="send_signal_message",
                description="Send a Signal message (text, optionally with images or short videos) to the homeowner.",
                structured_output=False,
            )
            async def send_signal_message(
                message: Annotated[str, Field(description="Message text. Plain text only.")],
                attachments: Annotated[List[Annotated[str, Field(description=(
                    "URL (auto-downloaded) or local file path of an image or "
                    "short video to attach."
                ))]], NULL_OK] = None,
            ) -> str:
                return await self._call(
                    "send_signal_message",
                    {"message": message, "attachments": attachments},
                    lambda: self.send_signal_message(
                        message=message,
                        attachments=attachments,
                    ),
                )

        @mcp.tool(
            name="query_multimodal_api",
            description="Send an image (and optional text prompt) to the vision LLM for analysis. Use for camera snapshots, photos, screenshots, etc.",
            structured_output=False,
        )
        async def query_multimodal_api(
            image_url: Annotated[str, Field(description=(
                "HTTP(S) URL to an image. Common formats supported (PNG, JPEG, "
                "WebP). The vision service fetches the URL itself."
            ))],
            text: Annotated[str, NULL_OK, Field(description=(
                "The text prompt describing what to analyze in the image (e.g. "
                "'describe what you see', 'is anyone in this image?')."
            ))] = None,
        ) -> str:
            return await self._call(
                "query_multimodal_api",
                {"text": text, "image_url": image_url},
                lambda: self.query_multimodal_ai(text=text, image_url=image_url),
            )

        # The baseline schema forbids extra args on this tool. pydantic's
        # generated argument model *ignores* extras rather than forbidding
        # them, so pin the advertised schema back to the baseline shape (the
        # old hand-written inputSchema carried it; advisory either way — the
        # low-level server never validated arguments against it).
        mcp._tool_manager.get_tool("query_multimodal_api").parameters["additionalProperties"] = False

        if WOLFRAM_ALPHA_API_KEY:
            @mcp.tool(
                name="wolfram_alpha",
                description="Query Wolfram Alpha for answers to factual questions",
                structured_output=False,
            )
            async def wolfram_alpha(
                query: Annotated[str, Field(description="Question to ask Wolfram Alpha")],
            ) -> str:
                return await self._call(
                    "wolfram_alpha",
                    {"query": query},
                    lambda: self.wolfram_alpha(query),
                )

        if WEATHER_API_KEY:
            @mcp.tool(
                name="get_weather_forecast",
                description="Get weather forecast and astronomy data for a location",
                structured_output=False,
            )
            async def get_weather_forecast(
                location: Annotated[str, Field(description="City name, ZIP code, or coordinates")],
                date: Annotated[str, NULL_OK, Field(description="Date in YYYY-MM-DD format (optional)")] = None,
            ) -> str:
                return await self._call(
                    "get_weather_forecast",
                    {"location": location, "date": date},
                    lambda: self.get_weather_forecast(location, date),
                )

        if BRAVE_API_KEY:
            @mcp.tool(
                name="brave_search",
                description="Retrieve a list of relevant websites using Brave Search API",
                structured_output=False,
            )
            async def brave_search(
                query: Annotated[str, Field(description="Search query")],
                count: Annotated[int, NULL_OK, Field(description="Number of results (default: 4)")] = 4,
            ) -> str:
                return await self._call(
                    "brave_search",
                    {"query": query, "count": count},
                    lambda: self.brave_search(query, count if count is not None else 4),
                )

        @mcp.tool(
            name="search_wikipedia",
            description="Search Wikipedia for information about a topic and return a summary.",
            structured_output=False,
        )
        async def search_wikipedia(
            search_string: Annotated[str, Field(description="Search query")],
            sentences: Annotated[int, NULL_OK, Field(description="Number of sentences to return")] = None,
        ) -> str:
            return await self._call(
                "search_wikipedia",
                {"search_string": search_string, "sentences": sentences},
                lambda: query_wikipedia(
                    search_string, sentences if sentences is not None else 7
                ),
            )

        @mcp.tool(
            name="fetch_webpage",
            description=(
                "Fetch a webpage and return its readable content as markdown. Use for reading "
                "articles, documentation, search results, or any page the user asks about. "
                "HTML is converted to markdown; long documents are truncated — call again with "
                "the suggested start_index to read further."
            ),
            structured_output=False,
        )
        async def fetch_webpage(
            url: Annotated[str, Field(description=(
                "The URL to fetch. Only http:// and https:// URLs are supported."
            ))],
            max_length: Annotated[int, Field(ge=1000, le=50000, description=(
                "Maximum number of characters to return (default: 10000)."
            )), NULL_OK] = 10000,
            start_index: Annotated[int, Field(ge=0, description=(
                "Character offset to start reading from (default: 0). A truncated "
                "response says which start_index to pass to continue."
            )), NULL_OK] = 0,
        ) -> str:
            return await self._call(
                "fetch_webpage",
                {"url": url, "max_length": max_length, "start_index": start_index},
                lambda: fetch_tools.fetch_webpage(
                    url,
                    max_length if max_length is not None else fetch_tools.DEFAULT_MAX_LENGTH,
                    start_index if start_index is not None else 0,
                ),
            )

        logger.info(f"Registered {len(mcp._tool_manager.list_tools())} tools")
        return mcp

    async def _call(
        self,
        name: str,
        args: Dict[str, Any],
        thunk: Callable[[], Awaitable[str]],
    ) -> str:
        """Run one tool body with the old call_tool handler's exact contract.

        Same entry log line, plain-text result, and any exception becomes an
        ``Error: ...`` text payload in ordinary (non-``isError``) content, so
        the agent-visible text stays identical to the hand-dispatch era.
        """
        logger.info(f"Tool called: {name} with args: {args}")
        try:
            return await thunk()
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {str(e)}"

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
                
                # Read content with size limit — align with the Signal cap so a
                # URL attachment isn't rejected for a size a local-path file of
                # the same bytes would pass (this helper only feeds Signal sends).
                max_size = SIGNAL_MAX_ATTACHMENT_BYTES
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
                try:
                    local_tz = pytz.timezone(TIMEZONE)
                except pytz.UnknownTimeZoneError:
                    logger.warning(f"Unknown timezone {TIMEZONE!r}, falling back to UTC")
                    local_tz = pytz.UTC
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


def main():
    """Stdio entry point (``python -m selene_agent.modules.mcp_general_tools``)."""
    logger.info("Starting HavenCore General Tools MCP Server...")
    GeneralToolsServer().mcp.run("stdio")


if __name__ == "__main__":
    main()