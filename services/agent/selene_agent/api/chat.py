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

Per-session idle timeout:
- REST `/api/chat` accepts an optional `X-Idle-Timeout: <seconds>` header. The
  value sticks to the session and governs when the idle sweep summarizes and
  resets it. Bad values are log-and-ignored; out-of-range values are clamped.
- WS `/ws/chat` accepts an optional `idle_timeout` field on any
  `{"type": "session", ...}` frame (first frame or mid-stream) to apply or
  update the override.

Device-name attribution:
- REST `/api/chat` accepts an optional `X-Device-Name` header (e.g. "Kitchen
  Speaker") that labels the satellite/client driving the session. The label
  rides with every flush to `conversation_histories.metadata.device_name` and
  is denormalized onto each `turn_metrics` row for the dashboard's history
  and metrics views.
- WS `/ws/chat` accepts an optional `device_name` field on any
  `{"type": "session", ...}` frame (first frame or mid-stream) with the same
  semantics. Empty/whitespace values are ignored so a frame omitting the field
  doesn't clobber a previously set name.
- Note: `/v1/chat/completions` is stateless and unattributed; device-name
  attribution applies to `/api/chat` and `/ws/chat` only.
"""

import asyncio
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Request, Response, Header
from pydantic import BaseModel

from selene_agent.orchestrator import AgentOrchestrator, EventType
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.metrics_db import metrics_db
from selene_agent.utils.session_pool import SessionOrchestratorPool

logger = custom_logger.get_logger('loki')

router = APIRouter()
ws_router = APIRouter()


def _apply_idle_timeout_override(orch: AgentOrchestrator, raw: Any) -> None:
    """Parse and apply an idle-timeout override onto the orchestrator.

    Bad inputs (None, empty, non-numeric) are ignored. The sentinel value -1
    means "never auto-summarize" and is stored as-is. Other out-of-range
    values are clamped to [CONVERSATION_TIMEOUT_MIN, CONVERSATION_TIMEOUT_MAX].
    """
    if raw is None or raw == "":
        return
    try:
        v = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Ignoring invalid idle-timeout override (session={orch.session_id}): {raw!r}"
        )
        return
    if v == -1:
        orch.idle_timeout_override = -1
        return
    lo, hi = config.CONVERSATION_TIMEOUT_MIN, config.CONVERSATION_TIMEOUT_MAX
    if v < lo:
        logger.warning(
            f"Clamping idle-timeout override {v} to minimum {lo} (session={orch.session_id})"
        )
        v = lo
    elif v > hi:
        logger.warning(
            f"Clamping idle-timeout override {v} to maximum {hi} (session={orch.session_id})"
        )
        v = hi
    orch.idle_timeout_override = v


DEVICE_NAME_MAX_LEN = 64


def _apply_device_name(orch: AgentOrchestrator, raw: Any) -> None:
    """Parse and apply a device_name onto the orchestrator.

    None / empty / whitespace-only are no-ops (so a frame that omits the field
    doesn't clobber a previously set name). Non-string values are log-and-ignored.
    Strings are trimmed, ASCII control chars stripped, and truncated to
    DEVICE_NAME_MAX_LEN with a warning. Unicode and emoji are allowed — this is
    a UI label, not spoken output.
    """
    if raw is None:
        return
    if not isinstance(raw, str):
        logger.warning(
            f"Ignoring non-string device name (session={orch.session_id}): {raw!r}"
        )
        return
    s = raw.strip()
    if not s:
        return
    s = "".join(c for c in s if ord(c) >= 0x20 and c != "\x7f")
    if not s:
        return
    if len(s) > DEVICE_NAME_MAX_LEN:
        logger.warning(
            f"Truncating device name to {DEVICE_NAME_MAX_LEN} chars (session={orch.session_id})"
        )
        s = s[:DEVICE_NAME_MAX_LEN]
    orch.device_name = s


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
    x_idle_timeout: Optional[str] = Header(default=None, alias="X-Idle-Timeout"),
    x_device_name: Optional[str] = Header(default=None, alias="X-Device-Name"),
):
    """Non-streaming chat endpoint that returns the full response plus tool events."""
    pool: SessionOrchestratorPool = req.app.state.session_pool

    if not pool:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    orchestrator = await pool.get_or_create(x_session_id)
    _apply_idle_timeout_override(orchestrator, x_idle_timeout)
    _apply_device_name(orchestrator, x_device_name)
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
                await metrics_db.record_turn(
                    orchestrator.session_id,
                    event.data,
                    device_name=orchestrator.device_name,
                )
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
      Client → {"type": "session", "session_id": "...", "idle_timeout": 90,
                "device_name": "Kitchen Speaker"}
               (optional first frame; all fields optional. A later
                `{"type": "session", ...}` mid-stream may update `idle_timeout`
                or `device_name` on the active session. `session_id` is honored
                only on the first frame.)
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
    notif_queue: Optional[asyncio.Queue] = None

    async def _ensure_session(requested_sid: Optional[str]):
        """Bind this WS connection to a session (hydrating or minting as needed).

        Returns (orchestrator, session_id). Announces the session_id to the
        client exactly once via a `{"type": "session"}` frame. Also subscribes
        this connection to the pool's per-session notification channel on the
        first bind, so out-of-band events (e.g. idle-sweep summary_reset) can
        reach the client between turns.
        """
        nonlocal orchestrator, session_id, session_announced, notif_queue
        if orchestrator is None:
            orchestrator = await pool.get_or_create(requested_sid)
            session_id = orchestrator.session_id
            notif_queue = pool.subscribe(session_id)
        if not session_announced:
            await websocket.send_json({"type": "session", "session_id": session_id})
            session_announced = True
        return orchestrator, session_id

    recv_task: Optional[asyncio.Task] = None
    notif_task: Optional[asyncio.Task] = None
    try:
        recv_task = asyncio.create_task(websocket.receive_json())

        while True:
            # Wait on either an inbound client frame or an out-of-band pool
            # notification for our session. Notification queue is only
            # subscribed once a session is bound, so race it in only then.
            wait_set = {recv_task}
            if notif_queue is not None:
                if notif_task is None or notif_task.done():
                    notif_task = asyncio.create_task(notif_queue.get())
                wait_set.add(notif_task)

            done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            if notif_task is not None and notif_task in done:
                event = notif_task.result()
                try:
                    await websocket.send_json(event)
                except Exception as e:
                    logger.warning(f"Failed to forward notif to WS: {e}")
                notif_task = None

            if recv_task in done:
                data = recv_task.result()
                recv_task = asyncio.create_task(websocket.receive_json())

                # Session-bind frame: may be sent as the first frame or mid-stream.
                # `session_id` is honored only on the first frame (once the session is
                # announced, re-binding to a different session is not supported — open
                # a new WS). `idle_timeout` and `device_name` apply both on first
                # frame and mid-stream.
                if data.get("type") == "session":
                    orch, _ = await _ensure_session(data.get("session_id"))
                    _apply_idle_timeout_override(orch, data.get("idle_timeout"))
                    _apply_device_name(orch, data.get("device_name"))
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
                            await metrics_db.record_turn(
                                orch.session_id,
                                event.data,
                                device_name=orch.device_name,
                            )

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (session={session_id})")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        if session_id and notif_queue is not None:
            pool.unsubscribe(session_id, notif_queue)
        for t in (recv_task, notif_task):
            if t is not None and not t.done():
                t.cancel()
