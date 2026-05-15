"""Async TTS client — thin wrapper around the text-to-speech service.

Targets whichever engine ``TTS_PROVIDER`` selects (v1=Kokoro at 6005,
v2=Chatterbox-Turbo at 6015) directly, not the agent's ``/api/tts/speak``
proxy, to keep the autonomy path free of self-referential HTTP hops. Both
engines accept the same request shape and emit the same X-Visemes header,
so the only thing that changes is the base URL.
"""
from __future__ import annotations

from typing import Optional

import aiohttp

from selene_agent.utils import config

TTS_BASE_DEFAULT = config.TTS_BASE_URL


class TTSClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout_sec: float = 60.0,
    ):
        self._base_url = (base_url or config.TTS_BASE_URL).rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_sec)

    async def synth(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        response_format: str = "mp3",
        speed: float = 1.0,
    ) -> bytes:
        """Render ``text`` to audio bytes. Raises ``RuntimeError`` on failure.

        Voice resolution: explicit ``voice`` arg → runtime default override
        (set via the voice-management UI) → engine fallback. Forwarded
        as-is once resolved; either engine reduces unknown names to its
        own default with a warning log.
        """
        if not text or not text.strip():
            raise ValueError("TTSClient.synth: empty text")
        body: dict = {
            "input": text,
            "model": "tts-1",
            "response_format": response_format or "mp3",
            "speed": speed or 1.0,
        }
        # Lazy import — keeps this client usable from non-DB contexts (tests,
        # standalone scripts) where agent_state's pool isn't initialized.
        resolved = voice
        if not resolved:
            try:
                from selene_agent.utils import agent_state
                resolved = await agent_state.get_default_voice()
            except Exception:
                resolved = None
        if resolved:
            body["voice"] = resolved
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
