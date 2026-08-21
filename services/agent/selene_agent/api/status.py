"""
Status API router — system health, MCP status, tool listing.
"""

import asyncio
import functools

import requests as http_requests
from fastapi import APIRouter, Request

from selene_agent.utils import config
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()


@router.get("/status")
async def get_system_status(req: Request):
    """Comprehensive system status for the dashboard"""
    session_pool = getattr(req.app.state, "session_pool", None)
    mcp_mgr = req.app.state.mcp_manager

    status = {
        "agent": {
            "name": config.AGENT_NAME,
            "healthy": session_pool is not None,
            "sessions": session_pool.health() if session_pool else {"active_sessions": 0, "max_size": 0},
        },
        "mcp": mcp_mgr.get_server_status() if mcp_mgr else {"error": "not initialized"},
        "database": {
            "connected": (
                conversation_db.pool is not None
                and not conversation_db.pool._closed
            ) if conversation_db.pool else False,
        },
    }

    # Proxy vLLM model info — the sync request runs in a thread so a slow/
    # unreachable vLLM can't block the event loop (and every concurrent turn).
    def _probe_llm():
        # vLLM enforces --api-key on /v1/*; harmless extra header if it doesn't.
        resp = http_requests.get(
            f"{config.LLM_API_BASE.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=3,
        )
        resp.raise_for_status()
        return resp.json()

    try:
        models = await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(_probe_llm)
        )
        status["llm"] = {"healthy": True, "models": models}
    except Exception as e:
        status["llm"] = {"healthy": False, "error": str(e)}

    return status


@router.get("/tools")
async def get_tools(req: Request):
    """List all registered tools grouped by MCP server"""
    mcp_mgr = req.app.state.mcp_manager

    if not mcp_mgr:
        return {"error": "MCP not initialized"}

    tools_by_server = {}
    for tool_name, tool in mcp_mgr.mcp_tools.items():
        server = tool.server_name or "unknown"
        if server not in tools_by_server:
            tools_by_server[server] = []
        tools_by_server[server].append({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        })

    return {"tools_by_server": tools_by_server, "total": len(mcp_mgr.mcp_tools)}


@router.get("/mcp/status")
async def get_mcp_status(req: Request):
    """Get status of MCP connections (legacy endpoint moved here)"""
    mcp_mgr = req.app.state.mcp_manager

    if not mcp_mgr:
        return {"error": "MCP not enabled or not initialized"}

    return mcp_mgr.get_server_status()
