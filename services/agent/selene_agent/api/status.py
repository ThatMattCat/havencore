"""
Status API router — system health, MCP status, tool listing.
"""

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
    orchestrator = req.app.state.orchestrator
    mcp_mgr = req.app.state.mcp_manager

    status = {
        "agent": {
            "name": config.AGENT_NAME,
            "healthy": orchestrator is not None,
        },
        "mcp": mcp_mgr.get_server_status() if mcp_mgr else {"error": "not initialized"},
        "database": {
            "connected": (
                conversation_db.pool is not None
                and not conversation_db.pool._closed
            ) if conversation_db.pool else False,
        },
    }

    # Proxy vLLM model info
    try:
        resp = http_requests.get(f"{config.LLM_API_BASE.rstrip('/')}/models", timeout=3)
        resp.raise_for_status()
        status["llm"] = {"healthy": True, "models": resp.json()}
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
