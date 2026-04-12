"""Vision / IAV proxy — wraps the iav-to-text vLLM chat-completion endpoint."""
import base64
import time

import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()

IAV_BASE = "http://iav-to-text:8100"


@router.post("/vision/ask")
async def ask(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    data = await image.read()
    mime = image.content_type or "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{IAV_BASE}/v1/chat/completions",
                json=body,
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                payload = await resp.json()
                if resp.status >= 400:
                    raise HTTPException(status_code=resp.status, detail=str(payload)[:500])
    except aiohttp.ClientError as e:
        logger.error(f"Vision proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"Vision service unreachable: {e}")

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=502, detail="Unexpected vLLM response shape")

    usage = payload.get("usage", {})
    return {"response": content, "latency_ms": latency_ms, "usage": usage}


@router.get("/vision/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{IAV_BASE}/v1/models",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status >= 400:
                    return JSONResponse(status_code=resp.status, content={"status": "unhealthy"})
                data = await resp.json()
                return {"status": "healthy", "models": data}
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})
