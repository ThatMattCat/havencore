#!/usr/bin/env python3
"""MCP server: reminder scheduling for HavenCore.

Wraps the autonomy ``/autonomy/items`` REST API so the LLM can create, list,
and cancel one-shot or recurring reminders. The actual delivery at fire time
is handled by ``selene_agent/autonomy/handlers/reminder.py``, which dispatches
to Signal / Home Assistant push / speaker TTS notifiers.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import aiohttp

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.types import Tool

from selene_agent.utils.logger import get_logger

logger = get_logger('loki')

AGENT_API_BASE = os.environ.get("AGENT_API_BASE", "http://localhost:6002").rstrip("/")
# Autonomy CRUD lives on the /api-prefixed router (selene_agent.py mounts
# autonomy_router with prefix="/api"), so all endpoints are /api/autonomy/...
AUTONOMY_ITEMS_URL = f"{AGENT_API_BASE}/api/autonomy/items"
LOCAL_TZ_NAME = os.environ.get("CURRENT_TIMEZONE", "UTC") or "UTC"
DEFAULT_CHANNEL = "signal"
ALLOWED_CHANNELS = {"signal", "ha_push", "speaker"}


def _local_tz() -> ZoneInfo:
    try:
        return ZoneInfo(LOCAL_TZ_NAME)
    except Exception:
        return ZoneInfo("UTC")


def _one_shot_cron(fire_at_utc: datetime) -> str:
    """Format a UTC datetime as a single-fire 5-field cron expression in local time."""
    local_dt = fire_at_utc.astimezone(_local_tz())
    return f"{local_dt.minute} {local_dt.hour} {local_dt.day} {local_dt.month} *"


def _parse_iso(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        # Naive ISO timestamps are interpreted in the user's local tz.
        dt = dt.replace(tzinfo=_local_tz())
    return dt.astimezone(timezone.utc)


def _resolve_when(args: Dict[str, Any], *, now_utc: datetime | None = None) -> Tuple[str, bool]:
    """Resolve the time-spec from raw args into ``(cron_expr, one_shot)``.

    Exactly one of ``in_seconds``, ``at``, ``cron`` must be provided.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    keys_set = [k for k in ("in_seconds", "at", "cron") if args.get(k) is not None]
    if len(keys_set) != 1:
        raise ValueError(
            "must specify exactly one of in_seconds / at / cron; got: "
            + (", ".join(keys_set) if keys_set else "none")
        )

    key = keys_set[0]
    if key == "in_seconds":
        secs = int(args["in_seconds"])
        if secs <= 0:
            raise ValueError("in_seconds must be > 0")
        target = now_utc + timedelta(seconds=secs)
        # Cron fires on minute boundaries; round up so the firing is at-or-after the requested moment.
        if target.second or target.microsecond:
            target = target.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return _one_shot_cron(target), True

    if key == "at":
        target = _parse_iso(str(args["at"]))
        if target <= now_utc:
            raise ValueError("`at` must be in the future")
        return _one_shot_cron(target), True

    cron_expr = str(args["cron"]).strip()
    if not cron_expr:
        raise ValueError("`cron` cannot be empty")
    return cron_expr, False


class ReminderToolsServer:
    """MCP server exposing reminder scheduling tools."""

    def __init__(self):
        self.server = Server("havencore-reminder-tools")
        self.setup_handlers()

    def setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            tools = [
                Tool(
                    name="schedule_reminder",
                    description=(
                        "Schedule a reminder for the user. Use for any 'remind me to X' request. "
                        "Provide exactly one of `in_seconds` (relative delay; preferred for "
                        "phrases like 'in an hour'), `at` (ISO 8601 absolute time, e.g. "
                        "'2026-04-27T18:00:00'; naive times are interpreted in local tz), or "
                        "`cron` (5-field cron in local tz for recurring; '0 18 * * 0' = "
                        "Sundays at 6pm). Default channel is 'signal'."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short reminder text (e.g. 'Take the trash out'). Used as the notification title and, if `body` is omitted, the body."
                            },
                            "body": {
                                "type": "string",
                                "description": "Optional longer body text. Defaults to title."
                            },
                            "in_seconds": {
                                "type": "integer",
                                "description": "Fire once after this many seconds. Use for 'in N minutes/hours' style requests."
                            },
                            "at": {
                                "type": "string",
                                "description": "Fire once at this absolute time (ISO 8601). Naive timestamps are interpreted in the user's local timezone."
                            },
                            "cron": {
                                "type": "string",
                                "description": "Recurring 5-field cron (minute hour day-of-month month day-of-week) in local tz. Examples: '0 18 * * 0' = Sundays 6pm; '30 7 * * 1-5' = weekdays 7:30am."
                            },
                            "channel": {
                                "type": "string",
                                "enum": ["signal", "ha_push", "speaker"],
                                "description": "Delivery channel. signal = Signal message (default; cross-location). ha_push = phone push via Home Assistant. speaker = TTS announcement on a Music Assistant target."
                            },
                            "to": {
                                "type": "string",
                                "description": "Optional recipient override per channel: signal phone number, ha_push notify.<service>, or speaker MA device target."
                            }
                        },
                        "required": ["title"]
                    }
                ),
                Tool(
                    name="list_reminders",
                    description=(
                        "List the user's reminders. Returns id, title, next fire time, "
                        "cron expression, channel, and one_shot flag. Use this before "
                        "cancel_reminder to find the right id."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_disabled": {
                                "type": "boolean",
                                "description": "If true, also include disabled reminders (cancelled or already-fired one-shots). Default false."
                            }
                        }
                    }
                ),
                Tool(
                    name="cancel_reminder",
                    description="Cancel a reminder by id. Use list_reminders first to look up the id.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "The reminder id (UUID), as returned by list_reminders or schedule_reminder."
                            }
                        },
                        "required": ["id"]
                    }
                ),
            ]
            logger.info(f"Listing {len(tools)} reminder tools")
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.BaseModel]:
            logger.info(f"Reminder tool called: {name}")
            try:
                if name == "schedule_reminder":
                    result = await self.schedule_reminder(arguments)
                elif name == "list_reminders":
                    result = await self.list_reminders(arguments)
                elif name == "cancel_reminder":
                    result = await self.cancel_reminder(arguments)
                else:
                    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
                return [types.TextContent(type="text", text=json.dumps(result))]
            except Exception as e:
                logger.exception(f"reminder tool {name} failed")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "error": str(e)})
                )]

    async def schedule_reminder(self, args: Dict[str, Any]) -> Dict[str, Any]:
        title = (args.get("title") or "").strip()
        if not title:
            return {"status": "error", "error": "title is required"}

        body = (args.get("body") or "").strip() or title

        try:
            cron_expr, one_shot = _resolve_when(args)
        except ValueError as e:
            return {"status": "error", "error": str(e)}

        channel = (args.get("channel") or DEFAULT_CHANNEL).strip()
        if channel not in ALLOWED_CHANNELS:
            return {
                "status": "error",
                "error": f"channel must be one of {sorted(ALLOWED_CHANNELS)}; got {channel!r}",
            }

        cfg: Dict[str, Any] = {
            "title": title,
            "body": body,
            "channel": channel,
            "one_shot": one_shot,
        }
        to = (args.get("to") or "").strip()
        if to:
            cfg["to"] = to

        payload = {
            "kind": "reminder",
            "name": title[:120],
            "schedule_cron": cron_expr,
            "config": cfg,
            "autonomy_level": "notify",
            "enabled": True,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    AUTONOMY_ITEMS_URL,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "status": "error",
                            "error": f"autonomy api returned {resp.status}: {text[:300]}",
                        }
                    data = json.loads(text)
        except aiohttp.ClientError as e:
            return {"status": "error", "error": f"autonomy api unreachable: {e}"}

        item = data.get("item") or {}
        return {
            "status": "ok",
            "id": item.get("id"),
            "title": title,
            "channel": channel,
            "cron": cron_expr,
            "one_shot": one_shot,
            "next_fire_at": item.get("next_fire_at"),
        }

    async def list_reminders(self, args: Dict[str, Any]) -> Dict[str, Any]:
        include_disabled = bool(args.get("include_disabled"))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    AUTONOMY_ITEMS_URL,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        return {
                            "status": "error",
                            "error": f"autonomy api returned {resp.status}: {text[:300]}",
                        }
                    data = json.loads(text)
        except aiohttp.ClientError as e:
            return {"status": "error", "error": f"autonomy api unreachable: {e}"}

        items = data.get("items") or []
        out: List[Dict[str, Any]] = []
        for it in items:
            if it.get("kind") != "reminder":
                continue
            if not include_disabled and not it.get("enabled"):
                continue
            cfg = it.get("config") or {}
            out.append({
                "id": it.get("id"),
                "title": cfg.get("title") or it.get("name"),
                "channel": cfg.get("channel"),
                "one_shot": cfg.get("one_shot", False),
                "cron": it.get("schedule_cron"),
                "next_fire_at": it.get("next_fire_at"),
                "enabled": it.get("enabled", True),
            })
        return {"status": "ok", "count": len(out), "reminders": out}

    async def cancel_reminder(self, args: Dict[str, Any]) -> Dict[str, Any]:
        item_id = (args.get("id") or "").strip()
        if not item_id:
            return {"status": "error", "error": "id is required"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{AUTONOMY_ITEMS_URL}/{item_id}",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    text = await resp.text()
                    if resp.status == 404:
                        return {"status": "error", "error": "reminder not found", "id": item_id}
                    if resp.status != 200:
                        return {
                            "status": "error",
                            "error": f"autonomy api returned {resp.status}: {text[:300]}",
                        }
        except aiohttp.ClientError as e:
            return {"status": "error", "error": f"autonomy api unreachable: {e}"}

        return {"status": "ok", "deleted": True, "id": item_id}

    async def run(self):
        logger.info("Starting HavenCore Reminder Tools MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Reminder MCP server running on stdio")
            await self.server.run(
                read_stream,
                write_stream,
                initialization_options=InitializationOptions(
                    server_name="HavenCore Reminder Tools MCP Server",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    server = ReminderToolsServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
