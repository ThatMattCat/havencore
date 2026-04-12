"""
Chat API router — non-streaming chat and WebSocket streaming with tool visibility.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from typing import List

from selene_agent.orchestrator import AgentOrchestrator, EventType
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.metrics_db import metrics_db

logger = custom_logger.get_logger('loki')

router = APIRouter()
ws_router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    events: List[dict]


@router.post("/chat")
async def chat(request: ChatRequest, req: Request):
    """Non-streaming chat endpoint that returns the full response plus tool events"""
    orchestrator: AgentOrchestrator = req.app.state.orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    logger.info(f"Chat API query: {request.message}")

    events = []
    final_content = ""

    async for event in orchestrator.run(request.message):
        events.append({"type": event.type.value, **event.data})
        if event.type == EventType.METRIC:
            await metrics_db.record_turn(orchestrator.session_id, event.data)
        elif event.type == EventType.DONE:
            final_content = event.data.get("content", "")
        elif event.type == EventType.ERROR:
            final_content = event.data.get("error", "ERROR: Unknown error")

    return ChatResponse(response=final_content, events=events)


@ws_router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with tool visibility.

    Connect: ws://host:port/ws/chat
    Send: {"message": "your question"}
    Receive: {"type": "thinking|tool_call|tool_result|done|error", ...data}
    """
    orchestrator: AgentOrchestrator = websocket.app.state.orchestrator

    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                await websocket.send_json({"type": "error", "error": "No message provided"})
                continue

            if not orchestrator:
                await websocket.send_json({"type": "error", "error": "Agent not initialized"})
                continue

            async for event in orchestrator.run(user_message):
                await websocket.send_json({
                    "type": event.type.value,
                    **event.data,
                })
                if event.type == EventType.METRIC:
                    await metrics_db.record_turn(orchestrator.session_id, event.data)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
