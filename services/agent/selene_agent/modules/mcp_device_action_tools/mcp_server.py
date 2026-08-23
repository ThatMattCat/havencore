#!/usr/bin/env python3
"""MCP server: device-side actions for HavenCore.

Tools in this module do not perform their action server-side — they exist so
the LLM has a discoverable capability to call, and so the orchestrator has a
``tool_call`` to attach a ``device_action`` event to. The companion app on the
session's device receives that event over ``/ws/chat`` and fires the matching
platform intent (e.g. ``AlarmClock.ACTION_SET_ALARM``).

To add a new device action: register a tool here, add a handler that returns a
structured status string, and add the tool name to
``orchestrator.DEVICE_ACTION_TOOLS``.

Camera tools (``take_photo`` and friends) are declared here so the LLM
discovers them, but their result is filled by the agent process via the
companion-app upload endpoint (``/api/companion/upload``), not by the
handler in this module. The orchestrator short-circuits these through
``COMPANION_UPLOAD_TOOLS``; the handler below is a fallback that returns a
benign error in case anything ever calls the MCP path directly.
"""
import json
import sys
from typing import Annotated, Any, Awaitable, Callable, Dict

from pydantic import Field

from mcp.server import MCPServer

from selene_agent.modules._mcp_params import NULL_OK
from selene_agent.utils.logger import get_logger

logger = get_logger('loki')


class DeviceActionToolsServer:
    """MCP server exposing device-side action tools.

    ``self.mcp`` is the configured mcp 2.0 ``MCPServer``; the decorated
    closures in ``_build_mcp`` are the MCP surface and delegate to the
    dict-shaped impl methods (unchanged from the hand-dispatch era).
    """

    def __init__(self):
        self.mcp = self._build_mcp()

    # --- MCP wiring ----------------------------------------------------

    def _build_mcp(self) -> MCPServer:
        """Register the decorated tool surface.

        structured_output=False on every tool: results stay a single
        TextContent JSON string, byte-identical to the pre-MCPServer wire
        format (no outputSchema in tools/list, no structuredContent).
        """
        mcp = MCPServer("havencore-device-action-tools", version="1.0.0")

        @mcp.tool(
            name="set_alarm",
            description=(
                "Schedule an alarm on the user's phone via the device's "
                "alarm clock app. Use this when the user asks to set an alarm, "
                "schedule a wake-up, or be reminded at a specific time of day. "
                "Hour and minute are local to the device. For one-off alarms "
                "omit days_of_week; for repeating alarms supply 1=Sunday, "
                "2=Monday, ..., 7=Saturday."
            ),
            structured_output=False,
        )
        async def set_alarm(
            hour: Annotated[int, Field(ge=0, le=23)],
            minute: Annotated[int, Field(ge=0, le=59)],
            label: Annotated[str, NULL_OK, Field(description="Optional label.")] = None,
            days_of_week: Annotated[
                list[Annotated[int, Field(ge=1, le=7)]],
                NULL_OK,
                Field(description="Repeating days. 1=Sun..7=Sat."),
            ] = None,
        ) -> str:
            return await self._dispatch("set_alarm", lambda: self.set_alarm({
                "hour": hour,
                "minute": minute,
                "label": label,
                "days_of_week": days_of_week,
            }))

        @mcp.tool(
            name="take_photo",
            description=(
                "Ask the user's companion-app phone to capture a photo "
                "with its camera and upload it back. Returns "
                "`{status: 'captured', image_url}` on success — the "
                "image_url can then be passed to vision tools (e.g. "
                "`query_multimodal_ai`) for analysis. Use when the user "
                "asks to take/snap/capture a picture WITHOUT a follow-up "
                "vision question. If the user is asking 'what is this?' "
                "or 'read this for me', prefer the more specific "
                "`identify_object_in_photo` / `read_text_from_image` "
                "tools — they capture and analyze in one step. "
                "Optionally pass a short `reason` describing why the "
                "photo is needed; the companion app may surface it. "
                "Times out after roughly 25s if no upload arrives."
            ),
            structured_output=False,
        )
        async def take_photo(
            reason: Annotated[str, NULL_OK, Field(description=(
                "Brief, user-facing rationale for the capture "
                "(e.g. 'to identify the plant')."
            ))] = None,
        ) -> str:
            return await self._dispatch(
                "take_photo",
                lambda: self.companion_camera_fallback("take_photo"),
            )

        @mcp.tool(
            name="identify_object_in_photo",
            description=(
                "Ask the user's phone to capture a photo of something in "
                "front of them, then identify the primary subject (plant, "
                "bug, appliance, gadget, error code, etc.). Returns the "
                "vision model's identification text along with the "
                "`image_url`. Use when the user asks 'what is this?', "
                "'what plant/bug/animal is this?', or similar one-shot "
                "identification questions about something physical near "
                "them. Pass `hint` to narrow the domain ('plant', "
                "'electronics'). Times out after roughly 25s if no "
                "upload arrives."
            ),
            structured_output=False,
        )
        async def identify_object_in_photo(
            hint: Annotated[str, NULL_OK, Field(description=(
                "Optional category hint to narrow "
                "identification (e.g. 'plant', 'bird', "
                "'appliance')."
            ))] = None,
        ) -> str:
            return await self._dispatch(
                "identify_object_in_photo",
                lambda: self.companion_camera_fallback("identify_object_in_photo"),
            )

        @mcp.tool(
            name="read_text_from_image",
            description=(
                "Ask the user's phone to capture a photo and transcribe "
                "all visible text in it. Returns the OCR text along with "
                "the `image_url`. Use when the user wants to read a "
                "receipt, label, sign, error screen, business card, "
                "page of a book, or any other text in front of them. "
                "Times out after roughly 25s if no upload arrives."
            ),
            structured_output=False,
        )
        async def read_text_from_image() -> str:
            return await self._dispatch(
                "read_text_from_image",
                lambda: self.companion_camera_fallback("read_text_from_image"),
            )

        @mcp.tool(
            name="who_is_in_view",
            description=(
                "Ask the user's phone to capture a photo and identify "
                "the person in it against the enrolled face gallery. "
                "Returns `{found: bool, name?, confidence?}` along with "
                "the `image_url`. Use when the user asks 'who is "
                "this?', 'do you know who that is?', 'is anyone in "
                "front of me?', or similar identity questions about a "
                "person physically near them. The face must already "
                "be enrolled in the face-recognition gallery for a "
                "match to land; unrecognized faces return "
                "`{found: false}` rather than an error. Times out "
                "after roughly 25s if no upload arrives."
            ),
            structured_output=False,
        )
        async def who_is_in_view() -> str:
            return await self._dispatch(
                "who_is_in_view",
                lambda: self.companion_camera_fallback("who_is_in_view"),
            )

        return mcp

    async def _dispatch(
        self,
        name: str,
        impl: Callable[[], Awaitable[Dict[str, Any]]],
    ) -> str:
        """Run one tool impl with the old call_tool handler's exact contract.

        Compact-JSON text out (this module never indented); an unexpected
        exception becomes a ``{"status": "error"}`` payload in ordinary
        (non-``isError``) content, so the agent-visible text stays identical
        to the hand-dispatch era.
        """
        logger.info(f"Device-action tool called: {name}")
        try:
            result = await impl()
            return json.dumps(result)
        except Exception as e:
            logger.exception(f"device-action tool {name} failed")
            return json.dumps({"status": "error", "error": str(e)})

    # --- tool implementations ------------------------------------------

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

    async def companion_camera_fallback(self, name: str) -> Dict[str, Any]:
        # Fallback only — the orchestrator routes camera tools through the
        # in-process companion-upload registry rather than the MCP path,
        # because the future + blob store both live in the agent process and
        # are unreachable from this MCP server process. Vision-chained tools
        # (identify_object_in_photo, read_text_from_image) additionally chain
        # to the vision pipeline server-side after the upload arrives;
        # who_is_in_view chains to the face-recognition service.
        return {
            "status": "error",
            "error": (
                f"{name} must be invoked through the orchestrator's "
                "companion-upload path; direct MCP invocation is not supported."
            ),
        }


def main():
    """Stdio entry point (``python -m selene_agent.modules.mcp_device_action_tools``)."""
    logger.info("Starting HavenCore Device Action Tools MCP Server (stdio)...")
    DeviceActionToolsServer().mcp.run("stdio")


if __name__ == "__main__":
    try:
        main()  # MCPServer.run("stdio") is synchronous (it owns the event loop)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
