#!/usr/bin/env python3
"""Snapshot the tool schemas exposed by every in-tree MCP module.

Two modes:

stdio (default): spawns each ``selene_agent.modules.mcp_*`` server over
stdio, initializes an MCP client session, calls list_tools, and writes each
module's tools (name, description, inputSchema) as pretty-printed sorted-key
JSON to ``tests/fixtures/tool_schemas/<module>.tools.json``.

HTTP (``--http``): reads the ``MCP_SERVERS`` env var (the Streamable HTTP
form: entries with ``url`` + ``token_env``), connects to each mount on the
mcp-tools service with its bearer token, and writes the same files — the
server short names are mapped back to module names so the output diffs
directly against the stdio baseline.

Usage (from the repo root, inside the agent container):

    docker compose exec -T agent python /app/tests/tools_snapshot.py [OUT_DIR]
    docker compose exec -T agent python /app/tests/tools_snapshot.py --http [OUT_DIR]

OUT_DIR defaults to ``/app/tests/fixtures/tool_schemas``. The fixtures are the
migration baseline for the MCP SDK 1.x -> 2.x / stdio -> Streamable HTTP
migration: re-run against a new SDK or transport and diff to prove schema
parity.

Per-module failures are reported but do not abort the rest of the run.
"""
import asyncio
import json
import os
import sys
import traceback

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

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

# MCP_SERVERS short name -> module fixture name (HTTP mode).
SERVER_TO_MODULE = {
    "general_tools": "mcp_general_tools",
    "mcp_server_qdrant": "mcp_qdrant_tools",
    "homeassistant": "mcp_homeassistant_tools",
    "mqtt": "mcp_mqtt_tools",
    "plex": "mcp_plex_tools",
    "music_assistant": "mcp_music_assistant_tools",
    "github": "mcp_github_tools",
    "face": "mcp_face_tools",
    "vision": "mcp_vision_tools",
    "reminder": "mcp_reminder_tools",
    "device_action": "mcp_device_action_tools",
}

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


def _write(module: str, tools: list, out_dir: str) -> None:
    path = os.path.join(out_dir, f"{module}.tools.json")
    with open(path, "w") as f:
        json.dump(tools, f, indent=2, sort_keys=True)
        f.write("\n")


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

    _write(module, tools, out_dir)
    return len(tools), None


async def snapshot_http_server(module: str, url: str, token: str, out_dir: str) -> tuple:
    """List one Streamable HTTP mount's tools and dump its schemas."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with create_mcp_http_client(headers=headers) as http:
        async with streamable_http_client(url, http_client=http) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.list_tools()
                tools = _normalize(response.tools or [])

    _write(module, tools, out_dir)
    return len(tools), None


def _http_targets() -> list:
    """(module, url, token) triples from the MCP_SERVERS env var."""
    servers = json.loads(os.environ.get("MCP_SERVERS", "[]"))
    targets = []
    for entry in servers:
        url = entry.get("url")
        if not url or not entry.get("enabled", True):
            continue
        name = entry.get("name")
        module = SERVER_TO_MODULE.get(name)
        if module is None:
            print(f"!!! no module mapping for MCP server {name!r}; skipping")
            continue
        token = os.environ.get(entry.get("token_env") or "", "").strip()
        targets.append((module, url, token))
    return targets


async def main() -> int:
    args = [a for a in sys.argv[1:]]
    http_mode = "--http" in args
    if http_mode:
        args.remove("--http")
    out_dir = args[0] if args else DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if http_mode:
        jobs = [
            (module, lambda u=url, t=token, m=module: snapshot_http_server(m, u, t, out_dir))
            for module, url, token in _http_targets()
        ]
    else:
        jobs = [
            (module, lambda m=module: snapshot_module(m, out_dir))
            for module in MODULES
        ]

    expected = [module for module, _ in jobs]
    results = {}
    failures = {}
    for module, job in jobs:
        print(f"--- {module}: {'connecting' if http_mode else 'spawning'} ...", flush=True)
        try:
            count, _ = await asyncio.wait_for(job(), timeout=PER_MODULE_TIMEOUT)
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
    for module in expected:
        if module in results:
            print(f"  {module}: {results[module]} tools")
        else:
            print(f"  {module}: FAILED ({failures[module]})")
    print(f"Output dir: {out_dir}")
    print(f"OK: {len(results)}/{len(expected)}")
    if http_mode and len(expected) < len(MODULES):
        print(f"NOTE: only {len(expected)}/{len(MODULES)} modules present in MCP_SERVERS")
    return 1 if failures or not expected else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
