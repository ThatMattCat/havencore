"""Public fetch endpoint for staged TTS audio blobs.

The ``SpeakerNotifier`` synthesises speech, parks the bytes in the
process-local ``AudioStore``, then hands Music Assistant a URL under this
router. MA pulls the bytes, we drop them. No auth beyond the random token —
the token IS the capability (single fetch, short TTL).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from selene_agent.services.audio_store import get_audio_store

router = APIRouter()


@router.get("/tts/audio/{token}.mp3")
async def fetch_audio(token: str):
    store = get_audio_store()
    result = await store.get(token)
    if result is None:
        raise HTTPException(status_code=404, detail="audio not found or expired")
    data, content_type = result
    return Response(content=data, media_type=content_type)
