"""STT proxy — HTTP file transcription + bidirectional WS for live mic streaming."""
import asyncio
import json

import aiohttp
import websockets
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()
ws_router = APIRouter()

STT_HTTP_BASE = "http://speech-to-text:6001"
STT_WS_URL = "ws://speech-to-text:6000"


@router.post("/stt/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None),
    response_format: str = Form("json"),
):
    try:
        content = await file.read()
        form = aiohttp.FormData()
        form.add_field(
            "file", content,
            filename=file.filename or "upload.wav",
            content_type=file.content_type or "application/octet-stream",
        )
        if language:
            form.add_field("language", language)
        form.add_field("response_format", response_format)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{STT_HTTP_BASE}/v1/audio/transcriptions",
                data=form,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                ctype = resp.headers.get("Content-Type", "application/json")
                body = await resp.read()
                if resp.status >= 400:
                    raise HTTPException(status_code=resp.status, detail=body.decode(errors="replace")[:500])
                if "json" in ctype:
                    return JSONResponse(content=json.loads(body))
                return JSONResponse(content={"text": body.decode(errors="replace")})
    except aiohttp.ClientError as e:
        logger.error(f"STT proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"STT service unreachable: {e}")


@router.get("/stt/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{STT_HTTP_BASE}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return JSONResponse(
                    status_code=resp.status,
                    content=await resp.json() if "json" in resp.headers.get("Content-Type", "") else {"status": resp.status},
                )
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})


@ws_router.websocket("/stt/stream")
async def stt_stream(websocket: WebSocket):
    """Bidirectional WS proxy to the STT streaming service.

    Browser sends: binary PCM chunks + JSON control messages.
    Browser receives: JSON transcription responses from upstream.
    """
    await websocket.accept()

    try:
        upstream = await websockets.connect(STT_WS_URL + "/", max_size=None)
    except Exception as e:
        logger.error(f"STT upstream connect failed: {e}")
        await websocket.send_json({"type": "error", "error": f"upstream unreachable: {e}"})
        await websocket.close()
        return

    async def pump_client_to_upstream():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"] is not None:
                    await upstream.send(msg["bytes"])
                elif "text" in msg and msg["text"] is not None:
                    await upstream.send(msg["text"])
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"STT client->upstream error: {e}")
        finally:
            try:
                await upstream.close()
            except Exception:
                pass

    async def pump_upstream_to_client():
        try:
            async for message in upstream:
                if isinstance(message, (bytes, bytearray)):
                    await websocket.send_bytes(bytes(message))
                else:
                    await websocket.send_text(message)
        except Exception as e:
            logger.error(f"STT upstream->client error: {e}")
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    await asyncio.gather(
        pump_client_to_upstream(),
        pump_upstream_to_client(),
        return_exceptions=True,
    )
