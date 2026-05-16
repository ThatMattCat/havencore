"""TTS proxy — exposes the active text-to-speech service to the dashboard.

Engine selection follows ``shared_config.TTS_PROVIDER`` (v1=Kokoro,
v2=Chatterbox-Turbo). Both upstreams expose the same /v1/audio/speech
surface + X-Visemes header, so this proxy is engine-agnostic.

Also exposes voice-management endpoints (upload / delete / set default)
that proxy through to the active engine. The runtime-override default
voice persists in ``agent_state.tts_default_voice`` and is applied here so
all callers (dashboard playground, autonomy speak path, companion app)
share one source of truth without each needing to look it up themselves.
"""
import json
from typing import Optional

import aiohttp
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from selene_agent.utils import agent_state, config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()


def _tts_base() -> str:
    """Resolve at call time so a swap via /api/system doesn't need a restart."""
    return config.TTS_BASE_URL


def _engine_label() -> str:
    return "Chatterbox-Turbo" if config.TTS_PROVIDER == "v2" else "Kokoro"

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
    voice: Optional[str] = None
    format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    model: Optional[str] = "tts-1"
    # Bypass the runtime-default override and use ``voice`` as-is. Only the
    # /playgrounds/tts page should set this — every other surface (chat
    # dashboard, companion app, satellites, autonomy speaker) is expected
    # to follow the assistant's current voice, which the override controls.
    force_voice: Optional[bool] = False


@router.post("/tts/speak")
async def speak(payload: SpeakRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    body: dict = {
        "input": payload.text,
        "model": payload.model or "tts-1",
        "response_format": payload.format or "mp3",
        "speed": payload.speed or 1.0,
    }
    # Resolution: runtime override wins unconditionally (one assistant voice
    # everywhere), unless the caller passes ``force_voice: true`` — which
    # only the voice-testing playground does. Falls through to caller's
    # ``voice``, then to the engine's configured default.
    override = await agent_state.get_default_voice()
    if payload.force_voice and payload.voice:
        voice = payload.voice
    else:
        voice = override or payload.voice
    if voice:
        body["voice"] = voice

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_tts_base()}/v1/audio/speech",
                json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    detail = data.decode(errors="replace")[:500]
                    raise HTTPException(status_code=resp.status, detail=detail)
                # Trust the upstream's Content-Type — libsndfile may fall back
                # to WAV when the requested format (e.g. mp3) isn't encodable,
                # and the header is the source of truth for the actual bytes.
                content_type = resp.headers.get("Content-Type") or CONTENT_TYPES.get(
                    payload.format or "mp3", "audio/mpeg"
                )
                # Forward the Rhubarb viseme timeline (base64 JSON) when the
                # TTS service produced one, so the companion app's Live2D
                # overlay can lip-sync against it. Header name follows the
                # contract documented in services/text-to-speech/main.py.
                forward_headers: dict[str, str] = {}
                visemes = resp.headers.get("X-Visemes")
                if visemes:
                    forward_headers["X-Visemes"] = visemes
                    forward_headers["Access-Control-Expose-Headers"] = "X-Visemes"
                return Response(
                    content=data,
                    media_type=content_type,
                    headers=forward_headers,
                )
    except aiohttp.ClientError as e:
        logger.error(f"TTS proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS service unreachable: {e}")


@router.get("/tts/voices")
async def voices():
    formats = ["mp3", "wav", "opus", "aac", "flac", "pcm"]
    native_ids: list[str] = []
    alias_ids: list[str] = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    # Engine-appropriate fallback if the upstream is unreachable. Kept in
    # sync with each service's own DEFAULT_VOICE.
    fallback_default = "Olivia" if config.TTS_PROVIDER == "v2" else "af_heart"
    default_voice = fallback_default

    user_ids: list[str] = []
    bundled_ids: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_tts_base()}/v1/voices",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status < 400:
                    data = await resp.json()
                    native_ids = list(data.get("native") or [])
                    alias_ids = list(data.get("aliases") or alias_ids)
                    default_voice = data.get("default") or default_voice
                    user_ids = list(data.get("user") or [])
                    bundled_ids = list(data.get("bundled") or [])
    except Exception as e:
        logger.warning(f"TTS /v1/voices unreachable, using fallback list: {e}")

    if not native_ids:
        native_ids = [fallback_default]

    # Runtime override wins over the engine's reported default. Falls back
    # when the override points at a voice the engine doesn't recognize.
    override = await agent_state.get_default_voice()
    if override and override in native_ids:
        default_voice = override

    # Put the default first so the dashboard dropdown selects it by default.
    if default_voice in native_ids:
        native_ids = [default_voice] + [v for v in native_ids if v != default_voice]

    label = _engine_label()
    user_set = set(user_ids)
    native = [
        {
            "id": v,
            "label": f"{v} ({label})",
            "kind": "user" if v in user_set else "bundled",
            "deletable": v in user_set,
        }
        for v in native_ids
    ]
    aliases = [{"id": v, "label": f"{v} (OpenAI alias → {default_voice})"} for v in alias_ids]
    return {
        "voices": native + aliases,
        "formats": formats,
        "default": default_voice,
        "default_override": override,
        "user_voices": user_ids,
        "bundled_voices": bundled_ids,
    }


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


class DefaultVoiceRequest(BaseModel):
    voice: Optional[str] = None  # None / "" clears the override


@router.post("/tts/voices/default")
async def set_default(payload: DefaultVoiceRequest):
    """Persist a runtime-override default voice.

    Pass ``{"voice": "Olivia"}`` to set, ``{"voice": null}`` (or empty
    string) to clear and fall back to the engine's configured default.
    """
    await agent_state.set_default_voice(payload.voice or None)
    return {"voice": payload.voice or None}


@router.post("/tts/voices/upload")
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    """Forward a multipart upload through to the active TTS engine.

    Only v2 (Chatterbox-Turbo) supports voice cloning. v1 (Kokoro) has
    fixed voices baked into the model, so this returns 501 there.
    """
    if config.TTS_PROVIDER != "v2":
        raise HTTPException(
            status_code=501,
            detail="Voice cloning is only supported by TTS v2 (Chatterbox-Turbo). "
                   "Set TTS_PROVIDER=v2 in .env to enable.",
        )
    raw = await file.read()
    form = aiohttp.FormData()
    form.add_field("name", name)
    form.add_field(
        "file", raw,
        filename=file.filename or f"{name}.wav",
        content_type=file.content_type or "audio/wav",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_tts_base()}/v1/voices/upload",
                data=form,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise HTTPException(
                        status_code=resp.status,
                        detail=body.get("detail") or str(body),
                    )
                return body
    except aiohttp.ClientError as e:
        logger.error(f"voice upload proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS service unreachable: {e}")


@router.delete("/tts/voices/{name}")
async def delete_voice(name: str):
    """Delete an uploaded reference clip on the active TTS engine."""
    if config.TTS_PROVIDER != "v2":
        raise HTTPException(status_code=501, detail="Voice management requires TTS v2.")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{_tts_base()}/v1/voices/{name}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.json()
                if resp.status >= 400:
                    raise HTTPException(
                        status_code=resp.status,
                        detail=body.get("detail") or str(body),
                    )
                # If this voice happened to be the runtime default, clear the
                # override so callers don't keep sending a now-missing name.
                override = await agent_state.get_default_voice()
                if override == name:
                    await agent_state.set_default_voice(None)
                    logger.info(
                        "Cleared default-voice override after deleting %r", name,
                    )
                return body
    except aiohttp.ClientError as e:
        logger.error(f"voice delete proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS service unreachable: {e}")


@router.get("/tts/health")
async def health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_tts_base()}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return JSONResponse(
                    status_code=resp.status,
                    content=await resp.json() if resp.content_type.endswith("json") else {"status": resp.status},
                )
    except Exception as e:
        return JSONResponse(status_code=502, content={"status": "unhealthy", "error": str(e)})
