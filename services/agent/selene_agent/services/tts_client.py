"""Async TTS client — thin wrapper around the text-to-speech service.

Targets the Kokoro TTS container directly (``text-to-speech:6005``), not
the agent's ``/api/tts/speak`` proxy, to keep the autonomy path free of
self-referential HTTP hops.
"""
from __future__ import annotations

from typing import Optional

import aiohttp

TTS_BASE_DEFAULT = "http://text-to-speech:6005"


class TTSClient:
    def __init__(
        self,
        base_url: str = TTS_BASE_DEFAULT,
        *,
        timeout_sec: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def synth(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> bytes:
        """Render ``text`` to audio bytes. Raises ``RuntimeError`` on failure."""
        if not text or not text.strip():
            raise ValueError("TTSClient.synth: empty text")
        body = {
            "input": text,
            "model": "tts-1",
            "voice": voice or "af_heart",
            "response_format": response_format or "mp3",
            "speed": speed or 1.0,
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                f"{self._base_url}/v1/audio/speech", json=body
            ) as resp:
                data = await resp.read()
                if resp.status >= 400:
                    detail = data.decode(errors="replace")[:500]
                    raise RuntimeError(
                        f"TTS service returned {resp.status}: {detail}"
                    )
                return data

    async def health_ok(self) -> bool:
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3)
            ) as session:
                async with session.get(f"{self._base_url}/health") as resp:
                    return resp.status == 200
        except Exception:
            return False


_default_format_content_types = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


def content_type_for(fmt: Optional[str]) -> str:
    return _default_format_content_types.get((fmt or "mp3").lower(), "audio/mpeg")
