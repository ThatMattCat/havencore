"""
Chat API router — non-streaming chat and WebSocket streaming with tool visibility.

Session identity:
- REST `/api/chat` accepts an `X-Session-Id` request header. Missing/unknown →
  pool mints a new session_id. The active session_id is always echoed back as
  an `X-Session-Id` response header.
- WS `/ws/chat` clients MAY send `{"type": "session", "session_id": "..."}` as
  the first frame to request a specific session. The server responds with
  `{"type": "session", "session_id": "..."}` before any `thinking` event.
  Clients that send `{"message": "..."}` as the first frame get a minted
  session.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Response, Header
from pydantic import BaseModel
from typing import List, Optional

from selene_agent.orchestrator import EventType
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.metrics_db import metrics_db
from selene_agent.utils.session_pool import SessionOrchestratorPool

logger = custom_logger.get_logger('loki')

router = APIRouter()
ws_router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    events: List[dict]
    session_id: str


@router.post("/chat")
async def chat(
    request: ChatRequest,
    req: Request,
    response: Response,
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-Id"),
):
    """Non-streaming chat endpoint that returns the full response plus tool events."""
    pool: SessionOrchestratorPool = req.app.state.session_pool

    if not pool:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    orchestrator = await pool.get_or_create(x_session_id)
    session_id = orchestrator.session_id
    response.headers["X-Session-Id"] = session_id

    logger.info(f"Chat API query (session={session_id}): {request.message}")

    events = []
    final_content = ""

    lock = pool.lock_for(session_id)
    async with lock:
        async for event in orchestrator.run(request.message):
            events.append({"type": event.type.value, **event.data})
            if event.type == EventType.METRIC:
                await metrics_db.record_turn(orchestrator.session_id, event.data)
            elif event.type == EventType.DONE:
                final_content = event.data.get("content", "")
            elif event.type == EventType.ERROR:
                final_content = event.data.get("error", "ERROR: Unknown error")

    return ChatResponse(response=final_content, events=events, session_id=session_id)


@ws_router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with tool visibility.

    Connect: ws://host:port/ws/chat
    Protocol:
      Client → {"type": "session", "session_id": "..."}  (optional first frame)
      Server → {"type": "session", "session_id": "..."}  (always sent once, before first turn)
      Client → {"message": "your question"}
      Server → {"type": "thinking|tool_call|tool_result|metric|done|error", ...}
    """
    pool: SessionOrchestratorPool = websocket.app.state.session_pool

    await websocket.accept()

    if not pool:
        try:
            await websocket.send_json({"type": "error", "error": "Agent not initialized"})
        finally:
            await websocket.close()
        return

    session_id: Optional[str] = None
    orchestrator = None
    session_announced = False

    async def _ensure_session(requested_sid: Optional[str]):
        """Bind this WS connection to a session (hydrating or minting as needed).

        Returns (orchestrator, session_id). Announces the session_id to the
        client exactly once via a `{"type": "session"}` frame.
        """
        nonlocal orchestrator, session_id, session_announced
        if orchestrator is None:
            orchestrator = await pool.get_or_create(requested_sid)
            session_id = orchestrator.session_id
        if not session_announced:
            await websocket.send_json({"type": "session", "session_id": session_id})
            session_announced = True
        return orchestrator, session_id

    try:
        while True:
            data = await websocket.receive_json()

            # Session-bind frame: optional, must be the first frame if used.
            if data.get("type") == "session":
                await _ensure_session(data.get("session_id"))
                continue

            user_message = data.get("message", "")
            if not user_message:
                await websocket.send_json({"type": "error", "error": "No message provided"})
                continue

            orch, sid = await _ensure_session(None)

            lock = pool.lock_for(sid)
            async with lock:
                async for event in orch.run(user_message):
                    await websocket.send_json({
                        "type": event.type.value,
                        **event.data,
                    })
                    if event.type == EventType.METRIC:
                        await metrics_db.record_turn(orch.session_id, event.data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (session={session_id})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
