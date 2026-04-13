"""STT proxy — HTTP file transcription."""
import json

import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()

STT_HTTP_BASE = "http://speech-to-text:6001"


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
