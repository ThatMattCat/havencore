"""
Selene Agent — FastAPI application entry point.

Handles startup/shutdown lifecycle, OpenAI-compatible API endpoints,
and registers API routers for the dashboard.
"""

import json
import time
from typing import List, Optional
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import os
from pydantic import BaseModel
from openai import AsyncOpenAI
import uvicorn

from selene_agent.utils import config
from selene_agent.utils.mcp_client_manager import MCPClientManager, MCPServerConfig
from selene_agent.utils.conversation_db import conversation_db
from selene_agent.utils import logger as custom_logger
from selene_agent.orchestrator import AgentOrchestrator, EventType, collect_response

# API routers
from selene_agent.api.chat import router as chat_router, ws_router as chat_ws_router
from selene_agent.api.conversations import router as conversations_router
from selene_agent.api.status import router as status_router
from selene_agent.api.homeassistant import router as ha_router
from selene_agent.api.metrics import router as metrics_router
from selene_agent.api.tts import router as tts_router
from selene_agent.api.stt import router as stt_router, ws_router as stt_ws_router
from selene_agent.api.vision import router as vision_router
from selene_agent.api.comfy import router as comfy_router
from selene_agent.api.logs import ws_router as logs_ws_router
from selene_agent.utils import log_stream
from selene_agent.utils.metrics_db import metrics_db

logger = custom_logger.get_logger('loki')


# --- Pydantic models for OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "selene"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


# --- Startup helpers ---

def _load_mcp_server_configs(mcp_manager: MCPClientManager):
    """Load MCP server configurations from environment"""
    if hasattr(config, 'MCP_SERVERS'):
        try:
            servers_config = json.loads(config.MCP_SERVERS)
            for server_cfg in servers_config:
                mcp_config = MCPServerConfig(
                    name=server_cfg.get('name'),
                    command=server_cfg.get('command'),
                    args=server_cfg.get('args', []),
                    env=server_cfg.get('env', {}),
                    enabled=server_cfg.get('enabled', True)
                )
                mcp_manager.add_server(mcp_config)
                logger.info(f"Loaded MCP server config: {mcp_config.name}")
        except Exception as e:
            logger.warning(f"Could not parse MCP_SERVERS JSON: {e}")


def _detect_model(base_url: str, max_retries: int = 30, retry_interval: int = 30) -> str:
    """Auto-detect the loaded model from the API, retrying until the LLM backend is ready."""
    for attempt in range(1, max_retries + 1):
        try:
            models_url = f"{base_url.rstrip('/')}/models"
            response = requests.get(models_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'data' in data and data['data']:
                return data['data'][0]['id']
            elif 'models' in data and data['models']:
                return data['models'][0]['name']
        except Exception as e:
            logger.debug(f"Model detection failed (attempt {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            logger.info(f"LLM backend not ready, retrying in {retry_interval}s (attempt {attempt}/{max_retries})")
            time.sleep(retry_interval)

    logger.error("Could not detect model after all retries — LLM backend may be down")
    return "llama"


# --- Application lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler for startup/shutdown"""
    logger.info("Starting Selene Agent")

    # Install in-process log ring buffer for the /ws/logs stream
    log_stream.install()

    # Initialize MCP
    mcp_manager = MCPClientManager()
    _load_mcp_server_configs(mcp_manager)

    # Initialize database
    try:
        await conversation_db.initialize()
        logger.info("Database connection initialized successfully")
        await metrics_db.ensure_schema()
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")

    # Initialize MCP connections
    try:
        await mcp_manager.initialize()
        logger.info("MCP manager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MCP manager: {e}")

    # Collect tools
    tools = []
    for tool in mcp_manager.get_all_mcp_tools():
        logger.info(f"Registering tool: {tool.name}")
        tools.append(tool.to_openai_format())

    if not tools:
        logger.warning("No tools registered.")

    # Detect model and create client
    api_base = config.LLM_API_BASE
    api_key = config.LLM_API_KEY or "dummy-key"
    model_name = _detect_model(api_base)
    logger.info(f"Using model: {model_name}")

    client = AsyncOpenAI(base_url=api_base, api_key=api_key)

    # Create orchestrator
    orchestrator = AgentOrchestrator(
        client=client,
        mcp_manager=mcp_manager,
        model_name=model_name,
        tools=tools,
    )
    await orchestrator.initialize()

    # Surface MCP failures in the system context
    if mcp_manager.failed_servers:
        failed_names = ", ".join(mcp_manager.failed_servers.keys())
        logger.warning(f"MCP servers failed to connect: {failed_names}")
        orchestrator.messages.append({
            "role": "system",
            "content": f"Note: The following tool servers failed to connect: {failed_names}. Some capabilities may be unavailable.",
        })

    # Store shared state on app for routers to access via request.app.state
    app.state.orchestrator = orchestrator
    app.state.mcp_manager = mcp_manager

    logger.info("Selene Agent initialized successfully")

    yield

    logger.info("Shutting down Selene Agent")
    if mcp_manager:
        await mcp_manager.cleanup()
    await conversation_db.close()


# --- FastAPI app ---

app = FastAPI(title="Selene Agent API", version="1.0.0", lifespan=lifespan)

# Mount static files directory for generated images
outputs_dir = "/app/selene_agent/outputs"
if os.path.exists(outputs_dir):
    app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")
    logger.info(f"Mounted static files from {outputs_dir} at /outputs")
else:
    logger.warning(f"Outputs directory {outputs_dir} does not exist")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(chat_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(status_router, prefix="/api")
app.include_router(ha_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")
app.include_router(tts_router, prefix="/api")
app.include_router(stt_router, prefix="/api")
app.include_router(vision_router, prefix="/api")
app.include_router(comfy_router, prefix="/api")
app.include_router(chat_ws_router, prefix="/ws")
app.include_router(stt_ws_router, prefix="/ws")
app.include_router(logs_ws_router, prefix="/ws")


# --- OpenAI-compatible endpoints (kept on /v1/* for voice pipeline backward compat) ---

async def _stream_sse(orchestrator: AgentOrchestrator, user_content: str, model: str):
    """Generate SSE events in OpenAI streaming format"""
    completion_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    async for event in orchestrator.run(user_content):
        if event.type == EventType.DONE:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": event.data.get("content", "")},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            finish_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(finish_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        elif event.type == EventType.ERROR:
            error_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": event.data.get("error", "")},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint (supports stream=true)"""
    orchestrator: AgentOrchestrator = app.state.orchestrator

    try:
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        user_content = None
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                user_content = msg.content
                break

        if not user_content:
            raise HTTPException(status_code=400, detail="No user message found in request")

        logger.info(f"API Query: {user_content}")

        if request.stream:
            return StreamingResponse(
                _stream_sse(orchestrator, user_content, request.model),
                media_type="text/event-stream",
            )

        response_content = await collect_response(orchestrator, user_content)

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=len(str(request.messages)),
                completion_tokens=len(response_content),
                total_tokens=len(str(request.messages)) + len(response_content)
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "selene",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "selene-agent",
        }]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "selene"}


# Legacy endpoint — kept for backward compatibility, delegates to status router
@app.get("/mcp/status")
async def get_mcp_status():
    """Get status of MCP connections (legacy — use /api/mcp/status)"""
    mcp_mgr = app.state.mcp_manager
    if not mcp_mgr:
        return {"error": "MCP not enabled or not initialized"}
    return mcp_mgr.get_server_status()


# --- Mount SvelteKit static frontend (MUST be after all route definitions) ---

_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if not os.path.isdir(_static_dir):
    _static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")
if not os.path.isdir(_static_dir):
    _static_dir = "/app/static"
if os.path.isdir(_static_dir):
    from fastapi.responses import FileResponse

    # Serve static assets (JS, CSS, etc.) under /_app/
    app.mount("/_app", StaticFiles(directory=os.path.join(_static_dir, "_app")), name="frontend_assets")

    # SPA catch-all: any non-API route serves index.html for client-side routing
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Serve actual files if they exist (e.g. favicon.ico)
        file_path = os.path.join(_static_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_static_dir, "index.html"))

    logger.info(f"Mounted SvelteKit frontend from {_static_dir}")


# --- Entry point ---

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run Selene Agent with auto-detected model')
    parser.add_argument('--api-base', type=str, default=None, help='Override API base URL')
    parser.add_argument('--api-key', type=str, default=None, help='Override API key')

    args = parser.parse_args()
    if args.api_base:
        config.LLM_API_BASE = args.api_base
    if args.api_key:
        config.LLM_API_KEY = args.api_key

    uvicorn.run(app, host="0.0.0.0", port=6002, log_level="info")


if __name__ == "__main__":
    main()
