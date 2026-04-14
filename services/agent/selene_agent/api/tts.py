"""TTS proxy — exposes the text-to-speech service to the dashboard."""
import json
from typing import Optional

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()

TTS_BASE = "http://text-to-speech:6005"

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


class SpeakRequest(BaseModel):
    text: str
    voice: Optional[str] = "af_heart"
    format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    model: Optional[str] = "tts-1"


@router.post("/tts/speak")
async def speak(payload: SpeakRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    body = {
        "input": payload.text,
        "model": payload.model or "tts-1",
        "voice": payload.voice or "af_heart",
        "response_format": payload.format or "mp3",
        "speed": payload.speed or 1.0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{TTS_BASE}/v1/audio/speech",
                json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    detail = data.decode(errors="replace")[:500]
                    raise HTTPException(status_code=resp.status, detail=detail)
                content_type = CONTENT_TYPES.get(
                    payload.format or "mp3", resp.headers.get("Content-Type", "audio/mpeg")
                )
                return Response(content=data, media_type=content_type)
    except aiohttp.ClientError as e:
        logger.error(f"TTS proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS service unreachable: {e}")


@router.get("/tts/voices")
async def voices():
    # The Kokoro deployment exposes one native voice; the other names are
    # OpenAI-compat aliases mapped server-side to the same voice.
    native = [{"id": "af_heart", "label": "af_heart (Kokoro)"}]
    aliases = [
        {"id": name, "label": f"{name} (alias)"}
        for name in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    ]
    formats = ["mp3", "wav", "opus", "aac", "flac", "pcm"]
    return {"voices": native + aliases, "formats": formats}


class AnnounceRequest(BaseModel):
    text: str
    device: str
    voice: Optional[str] = None
    volume: Optional[float] = None


@router.post("/tts/announce")
async def announce(payload: AnnounceRequest, req: Request):
    """Render ``text`` with TTS and play it as an MA announcement on
    ``device``. One-off convenience endpoint used by the dashboard; reuses
    ``SpeakerNotifier`` so behavior matches autonomy speak-tier delivery.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    if not payload.device.strip():
        raise HTTPException(status_code=400, detail="device is required")

    mcp_mgr = getattr(req.app.state, "mcp_manager", None)
    if mcp_mgr is None:
        raise HTTPException(status_code=503, detail="MCP manager not initialized")

    from selene_agent.autonomy.notifiers import SpeakerNotifier

    notifier = SpeakerNotifier(
        mcp_mgr,
        device=payload.device,
        voice=payload.voice or "",
        volume=payload.volume,
    )
    ok = await notifier.send(title="", body=payload.text)
    if not ok:
        raise HTTPException(status_code=502, detail="announcement failed; see logs")
    return {"ok": True, "device": payload.device}


@router.get("/tts/players")
async def players(req: Request):
    """Proxy ``mass_list_players`` so the dashboard can populate a dropdown."""
    mcp_mgr = getattr(req.app.state, "mcp_manager", None)
    if mcp_mgr is None:
        raise HTTPException(status_code=503, detail="MCP manager not initialized")
    try:
        result = await mcp_mgr.execute_tool("mass_list_players", {})
    except Exception as e:
        logger.warning(f"tts/players: mass_list_players failed: {e}")
        return {"players": [], "error": str(e)}

    # MCP execute_tool returns a JSON-serialized string; older paths may yield
    # a list/dict directly. Normalize all three.
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"tts/players: non-JSON result: {result[:200]!r}")
            return {"players": []}

    raw: list
    if isinstance(result, list):
        raw = result
    elif isinstance(result, dict):
        raw = result.get("players") or []
    else:
        raw = []

    # MA players expose ``display_name`` — the dashboard and
    # ``mass_play_announcement`` both key off a plain ``name`` field.
    players_out = []
    for p in raw:
        if isinstance(p, str):
            players_out.append({"name": p})
            continue
        if not isinstance(p, dict):
            continue
        name = p.get("name") or p.get("display_name") or p.get("player_id")
        if not name:
            continue
        players_out.append({
            "name": name,
            "player_id": p.get("player_id"),
            "available": p.get("available"),
            "powered": p.get("powered"),
        })
    return {"players": players_out}


@router.get("/tts/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{TTS_BASE}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return JSONResponse(
                    status_code=resp.status,
                    content=await resp.json() if resp.content_type.endswith("json") else {"status": resp.status},
                )
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})
