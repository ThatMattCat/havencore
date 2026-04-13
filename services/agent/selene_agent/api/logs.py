"""Live log streaming via WebSocket."""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from selene_agent.utils import log_stream

ws_router = APIRouter()


@ws_router.websocket("/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    handler = log_stream.get_handler()
    if handler is None:
        await websocket.send_json({"type": "error", "error": "log stream unavailable"})
        await websocket.close()
        return

    for entry in handler.snapshot():
        await websocket.send_json({"type": "log", **entry})

    queue = handler.subscribe()
    try:
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json({"type": "log", **entry})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        handler.unsubscribe(queue)
