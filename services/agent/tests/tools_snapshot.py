#!/usr/bin/env python3
"""Snapshot the tool schemas exposed by every in-tree MCP module.

Spawns each ``selene_agent.modules.mcp_*`` server over stdio, initializes an
MCP client session, calls list_tools, and writes each module's tools
(name, description, inputSchema) as pretty-printed sorted-key JSON to
``tests/fixtures/tool_schemas/<module>.tools.json``.

Usage (from the repo root, inside the agent container):

    docker compose exec -T agent python /app/tests/tools_snapshot.py [OUT_DIR]

OUT_DIR defaults to ``/app/tests/fixtures/tool_schemas``. The fixtures are the
migration baseline for the MCP SDK 1.x -> 2.x upgrade: re-run against a new
SDK and diff to prove schema parity.

Per-module failures are reported but do not abort the rest of the run.
"""
import asyncio
import json
import os
import sys
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODULES = [
    "mcp_device_action_tools",
    "mcp_face_tools",
    "mcp_general_tools",
    "mcp_github_tools",
    "mcp_homeassistant_tools",
    "mcp_mqtt_tools",
    "mcp_music_assistant_tools",
    "mcp_plex_tools",
    "mcp_qdrant_tools",
    "mcp_reminder_tools",
    "mcp_vision_tools",
]

# github's __init__ can block ~180s on a git fetch; leave generous headroom.
PER_MODULE_TIMEOUT = 240.0

DEFAULT_OUT_DIR = "/app/tests/fixtures/tool_schemas"


def _normalize(tools) -> list:
    """Reduce a list_tools response to a stable, diffable structure.

    The JSON key stays wire-format ``inputSchema`` so snapshots diff cleanly
    across SDK versions; mcp 2.x exposes the field as ``input_schema``.
    """
    out = []
    for tool in tools:
        out.append({
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        })
    out.sort(key=lambda t: t["name"])
    return out


async def snapshot_module(module: str, out_dir: str) -> tuple:
    """Spawn one module as a stdio server and dump its tool schemas.

    Returns (tool_count, None) on success or (0, error_string) on failure.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", f"selene_agent.modules.{module}"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            response = await session.list_tools()
            tools = _normalize(response.tools or [])

    path = os.path.join(out_dir, f"{module}.tools.json")
    with open(path, "w") as f:
        json.dump(tools, f, indent=2, sort_keys=True)
        f.write("\n")
    return len(tools), None


async def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    failures = {}
    for module in MODULES:
        print(f"--- {module}: spawning ...", flush=True)
        try:
            count, _ = await asyncio.wait_for(
                snapshot_module(module, out_dir), timeout=PER_MODULE_TIMEOUT
            )
            results[module] = count
            print(f"--- {module}: {count} tools", flush=True)
        except asyncio.TimeoutError:
            failures[module] = f"timed out after {PER_MODULE_TIMEOUT:.0f}s"
            print(f"--- {module}: TIMEOUT", flush=True)
        except Exception as e:
            failures[module] = f"{type(e).__name__}: {e}"
            print(f"--- {module}: FAILED: {failures[module]}", flush=True)
            traceback.print_exc()

    print("\n=== Snapshot summary ===")
    for module in MODULES:
        if module in results:
            print(f"  {module}: {results[module]} tools")
        else:
            print(f"  {module}: FAILED ({failures[module]})")
    print(f"Output dir: {out_dir}")
    print(f"OK: {len(results)}/{len(MODULES)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
