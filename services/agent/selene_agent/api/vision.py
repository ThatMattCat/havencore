"""Vision proxy — wraps the vllm-vision OpenAI-compat chat-completion endpoint."""
import base64
import time
from typing import Any, Optional

import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()


_IMAGE_MIME_PREFIX = "image/"
_VIDEO_MIME_PREFIX = "video/"


async def _call_vision(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, int, dict]:
    """POST a chat-completions request to vllm-vision and return (content, latency_ms, usage)."""
    body = {
        "model": config.VISION_SERVED_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{config.VISION_API_BASE}/chat/completions",
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

    return content, latency_ms, payload.get("usage", {})


@router.post("/vision/ask")
async def ask(
    prompt: str = Form(...),
    file: UploadFile = File(None),
    image: UploadFile = File(None),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
):
    """Multipart form: file upload + prompt. Used by the dashboard playground.

    Accepts both `file` (preferred — image OR video) and `image` (legacy
    image-only field name retained so older callers keep working). The MIME
    type on the upload picks the content-part shape: `image/*` -> image_url,
    `video/*` -> video_url. vllm-vision handles both via the OpenAI-compat
    multimodal schema.
    """
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="file is required")

    data = await upload.read()
    mime = (upload.content_type or "").lower() or "image/png"

    if mime.startswith(_IMAGE_MIME_PREFIX):
        part_type = "image_url"
        url_key = "image_url"
    elif mime.startswith(_VIDEO_MIME_PREFIX):
        part_type = "video_url"
        url_key = "video_url"
    else:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported media type '{mime}'; expected image/* or video/*",
        )

    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": part_type, url_key: {"url": data_url}},
            ],
        }
    ]
    content, latency_ms, usage = await _call_vision(
        messages, max_tokens=max_tokens, temperature=temperature
    )
    return {
        "response": content,
        "latency_ms": latency_ms,
        "usage": usage,
        "model": config.VISION_SERVED_NAME,
        "media_type": part_type,
    }


class VisionAskUrlRequest(BaseModel):
    text: Optional[str] = None
    image_url: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7


@router.post("/vision/ask_url")
async def ask_url(req: VisionAskUrlRequest):
    """JSON body with text + image_url. Used by the query_multimodal_api MCP tool.

    vllm-vision fetches image_url itself (http(s):// or data: URLs).
    """
    if not (req.text or req.image_url):
        raise HTTPException(status_code=400, detail="text or image_url is required")

    content_parts: list[dict[str, Any]] = []
    if req.text:
        content_parts.append({"type": "text", "text": req.text})
    if req.image_url:
        content_parts.append({"type": "image_url", "image_url": {"url": req.image_url}})

    messages = [{"role": "user", "content": content_parts}]
    content, latency_ms, usage = await _call_vision(
        messages, max_tokens=req.max_tokens, temperature=req.temperature
    )
    return {
        "response": content,
        "latency_ms": latency_ms,
        "usage": usage,
        "model": config.VISION_SERVED_NAME,
    }


@router.get("/vision/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{config.VISION_API_BASE}/models",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status >= 400:
                    return JSONResponse(status_code=resp.status, content={"status": "unhealthy"})
                data = await resp.json()
                return {"status": "healthy", "models": data}
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})
