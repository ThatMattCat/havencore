"""
Vision Tools MCP Server — purpose-built tools on top of the vllm-vision service.

The general-purpose `query_multimodal_api` (in `mcp_general_tools`) already routes
through the agent's `/api/vision/ask_url` chokepoint. This module layers
higher-leverage, well-described tools on top so the LLM doesn't have to
assemble image URLs and prompts manually.

All single-image tools route through the same `/api/vision/ask_url` chokepoint
so logging, auth, and the served-model-name stay in one place. The one
exception is `compare_snapshots`, which needs two images in a single message —
the chokepoint's body schema is single-image only, so this tool talks to
vllm-vision's OpenAI-compat endpoint directly. Both code paths share a single
`_post_json` helper for HTTP plumbing.

Tools exposed:
  describe_image, describe_camera_snapshot, compare_snapshots,
  identify_object, read_text_in_image.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

import aiohttp

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from selene_agent.utils.logger import get_logger
from selene_agent.utils import config

logger = get_logger('loki')


ASK_URL_ENDPOINT = os.getenv(
    "VISION_ASK_URL_ENDPOINT", "http://agent:6002/api/vision/ask_url"
)
HTTP_TIMEOUT_SEC = float(os.getenv("VISION_HTTP_TIMEOUT_SEC", "180"))

DEFAULT_DESCRIBE_PROMPT = (
    "Briefly describe what is visible in this image. Note people (clothing, "
    "posture, what they're holding), animals, vehicles, packages, and anything "
    "that looks unusual. 2-3 sentences. No speculation about intent."
)

DEFAULT_IDENTIFY_PROMPT = (
    "Identify the primary subject of this image. Give a concise name and a "
    "one-sentence description (material, model, species — whatever is most "
    "salient). If you cannot identify it confidently, say so and offer the "
    "closest plausible match."
)

DEFAULT_OCR_PROMPT = (
    "Transcribe all visible text in this image. Preserve line breaks and "
    "rough layout where it carries meaning (receipts, forms, tables, code). "
    "If a region is illegible, mark it [illegible]. Do not paraphrase."
)

DEFAULT_COMPARE_PROMPT = (
    "These are two camera frames from the same scene, A then B. Describe what "
    "changed between them: people coming or going, objects added or removed, "
    "lighting or weather shifts. Be specific. If nothing meaningful changed, "
    "say so plainly."
)


class VisionMCPServer:
    def __init__(self):
        self.server = Server("havencore-vision-tools")
        self._snapshotter = None  # lazy — see _get_snapshotter
        self._setup_handlers()

    # --- HTTP helpers --------------------------------------------------

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Single chokepoint for outbound HTTP. Raises ValueError on non-2xx."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SEC),
            ) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    text = await resp.text()
                    if resp.status >= 400:
                        raise ValueError(
                            f"vision API error ({resp.status}): {text[:300]}"
                        )
                    raise ValueError(f"vision API returned non-JSON: {text[:300]}")
                if resp.status >= 400:
                    detail = (
                        data.get("detail")
                        if isinstance(data, dict)
                        else str(data)[:300]
                    )
                    raise ValueError(f"vision API error ({resp.status}): {detail}")
                return data

    async def _ask_url(
        self,
        text: str,
        image_url: Optional[str] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Route a single-image (or text-only) request through the chokepoint."""
        payload: Dict[str, Any] = {
            "text": text,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if image_url:
            payload["image_url"] = image_url
        data = await self._post_json(ASK_URL_ENDPOINT, payload)
        try:
            return data["response"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"unexpected ask_url response shape: {e}")

    async def _vllm_chat_multi_image(
        self,
        prompt: str,
        image_urls: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Call vllm-vision directly with a multi-image message. Used only
        for compare_snapshots; the ask_url chokepoint is single-image."""
        if not config.VISION_API_BASE:
            raise ValueError("VISION_API_BASE is not configured")
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {
            "model": config.VISION_SERVED_NAME,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        data = await self._post_json(
            f"{config.VISION_API_BASE.rstrip('/')}/chat/completions", payload
        )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected vllm-vision response shape: {e}")

    # --- snapshot resolution -------------------------------------------

    def _get_snapshotter(self):
        """Lazy-instantiate an HACamSnapper. Reusing the class avoids
        duplicating the HA-script + MQTT-await dance. The MQTT subscription
        races harmlessly with mcp_mqtt_tools' own subscriber — each module
        only consumes the URLs it triggered."""
        if self._snapshotter is None:
            from selene_agent.modules.mcp_mqtt_tools.mcp_server import HACamSnapper
            self._snapshotter = HACamSnapper(
                ha_url=os.getenv("HAOS_URL", "NO_HAOS_URL_SET"),
                ha_token=os.getenv("HAOS_TOKEN", "NO_HAOS_TOKEN_SET"),
                mqtt_broker=os.getenv("MQTT_BROKER", "mosquitto"),
                mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
            )
        return self._snapshotter

    @staticmethod
    def _match_camera_url(camera_name: str, urls: List[str]) -> Optional[str]:
        """Pick the URL that best matches camera_name. Server-side fallback
        so the LLM can pass 'backyard' and we'll find a URL containing it.

        Match stages: case-insensitive substring, then loose token match
        (split camera_name on space/underscore/dash and require all tokens
        to appear in the URL). Returns None if nothing matches."""
        if not urls:
            return None
        if not camera_name:
            return urls[0]
        q = camera_name.lower().strip()
        for u in urls:
            if q in u.lower():
                return u
        tokens = [t for t in q.replace("-", " ").replace("_", " ").split() if t]
        if tokens:
            for u in urls:
                lu = u.lower()
                if all(t in lu for t in tokens):
                    return u
        return None

    # --- tool implementations ------------------------------------------

    async def _describe_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        image_url = args.get("image_url")
        if not image_url:
            return {"error": "image_url is required"}
        prompt = (args.get("prompt") or "").strip() or DEFAULT_DESCRIBE_PROMPT
        description = await self._ask_url(prompt, image_url, max_tokens=512)
        return {
            "image_url": image_url,
            "prompt": prompt,
            "description": description,
        }

    async def _describe_camera_snapshot(self, args: Dict[str, Any]) -> Dict[str, Any]:
        camera_name = (args.get("camera_name") or "").strip()
        if not camera_name:
            return {"error": "camera_name is required"}
        prompt = (args.get("prompt") or "").strip() or DEFAULT_DESCRIBE_PROMPT

        try:
            snap = await self._get_snapshotter().get_camera_snapshots()
        except Exception as e:
            return {"error": f"snapshot capture failed: {e}"}
        if not snap.get("success"):
            return {
                "error": snap.get("error", "snapshot capture failed"),
                "camera_name": camera_name,
            }
        urls = snap.get("urls") or []
        if not urls:
            return {
                "error": "no snapshot URLs returned",
                "camera_name": camera_name,
            }

        chosen = self._match_camera_url(camera_name, urls)
        if not chosen:
            return {
                "error": f"no snapshot URL matched '{camera_name}'",
                "camera_name": camera_name,
                "available_urls": urls,
            }

        description = await self._ask_url(prompt, chosen, max_tokens=512)
        return {
            "camera_name": camera_name,
            "image_url": chosen,
            "prompt": prompt,
            "description": description,
        }

    async def _compare_snapshots(self, args: Dict[str, Any]) -> Dict[str, Any]:
        a = args.get("image_url_a")
        b = args.get("image_url_b")
        if not (a and b):
            return {"error": "image_url_a and image_url_b are both required"}
        focus = (args.get("focus") or "").strip()
        prompt = DEFAULT_COMPARE_PROMPT
        if focus:
            prompt = f"{prompt}\n\nFocus specifically on: {focus}"
        comparison = await self._vllm_chat_multi_image(
            prompt, [a, b], max_tokens=512
        )
        return {
            "image_url_a": a,
            "image_url_b": b,
            "focus": focus or None,
            "comparison": comparison,
        }

    async def _identify_object(self, args: Dict[str, Any]) -> Dict[str, Any]:
        image_url = args.get("image_url")
        if not image_url:
            return {"error": "image_url is required"}
        hint = (args.get("hint") or "").strip()
        prompt = DEFAULT_IDENTIFY_PROMPT
        if hint:
            prompt = f"{prompt}\n\nHint from the user about what this might be: {hint}"
        identification = await self._ask_url(prompt, image_url, max_tokens=300)
        return {
            "image_url": image_url,
            "hint": hint or None,
            "identification": identification,
        }

    async def _read_text_in_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        image_url = args.get("image_url")
        if not image_url:
            return {"error": "image_url is required"}
        text = await self._ask_url(
            DEFAULT_OCR_PROMPT, image_url, max_tokens=1024, temperature=0.1
        )
        return {"image_url": image_url, "text": text}

    # --- MCP wiring ----------------------------------------------------

    def _setup_handlers(self):

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="describe_image",
                    description=(
                        "General-purpose vision: describe an image at an HTTP(S) URL. "
                        "Use for camera frames, photos, screenshots, anything a user "
                        "or another tool produces a URL for. Pass a `prompt` to focus "
                        "the description; omit it for a generic 'what's in this scene' "
                        "answer."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {
                                "type": "string",
                                "description": "HTTP(S) URL of the image to analyze.",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Optional question or instruction for the vision model.",
                            },
                        },
                        "required": ["image_url"],
                    },
                ),
                Tool(
                    name="describe_camera_snapshot",
                    description=(
                        "One-shot 'what's happening on the {camera} camera?' — captures a "
                        "fresh frame from the named Home Assistant camera and runs vision "
                        "on it. Replaces the two-step 'snapshot then describe' chain. "
                        "`camera_name` is fuzzy-matched against snapshot URLs, so "
                        "'backyard' or 'front door' both work."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "camera_name": {
                                "type": "string",
                                "description": "Camera name or alias (e.g. 'backyard', 'front_door').",
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Optional focus for the description.",
                            },
                        },
                        "required": ["camera_name"],
                    },
                ),
                Tool(
                    name="compare_snapshots",
                    description=(
                        "Compare two images side-by-side and describe what changed. "
                        "Useful for 'did the package get picked up?', 'did anyone enter "
                        "the room?', or any A/B before/after question. Both images are "
                        "sent to the vision model in one call. Optional `focus` narrows "
                        "the comparison."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url_a": {"type": "string", "description": "First (earlier) image URL."},
                            "image_url_b": {"type": "string", "description": "Second (later) image URL."},
                            "focus": {
                                "type": "string",
                                "description": "Optional aspect to focus on (e.g. 'people', 'the porch').",
                            },
                        },
                        "required": ["image_url_a", "image_url_b"],
                    },
                ),
                Tool(
                    name="identify_object",
                    description=(
                        "Identify the primary subject of an image — plant, bug, "
                        "appliance, error code, gadget, etc. Returns a concise name "
                        "and one-sentence description. Pass `hint` to narrow the "
                        "domain ('plant', 'bug', 'appliance brand')."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string"},
                            "hint": {
                                "type": "string",
                                "description": "Optional category hint to narrow identification.",
                            },
                        },
                        "required": ["image_url"],
                    },
                ),
                Tool(
                    name="read_text_in_image",
                    description=(
                        "Transcribe all visible text in an image (OCR-flavored). "
                        "For receipts, mail, error screenshots, whiteboards, code on "
                        "screens. Preserves rough layout where it matters."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_url": {"type": "string"},
                        },
                        "required": ["image_url"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"vision tool called: {name}")
            try:
                if name == "describe_image":
                    result = await self._describe_image(arguments)
                elif name == "describe_camera_snapshot":
                    result = await self._describe_camera_snapshot(arguments)
                elif name == "compare_snapshots":
                    result = await self._compare_snapshots(arguments)
                elif name == "identify_object":
                    result = await self._identify_object(arguments)
                elif name == "read_text_in_image":
                    result = await self._read_text_in_image(arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}
            except aiohttp.ClientError as e:
                logger.warning(f"vision tool {name} network error: {e}")
                result = {"error": f"vision service unreachable: {e}"}
            except ValueError as e:
                logger.warning(f"vision tool {name} failed: {e}")
                result = {"error": str(e)}
            except Exception as e:
                logger.exception(f"vision tool {name} crashed")
                result = {"error": str(e)}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def run(self):
        options = self.server.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, options, raise_exceptions=True)


async def main():
    logger.info("Starting Vision Tools MCP Server...")
    server = VisionMCPServer()
    await server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
