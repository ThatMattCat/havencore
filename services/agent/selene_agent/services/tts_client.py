"""Async TTS client — thin wrapper around the text-to-speech service.

Targets the active TTS engine (Kokoro or Chatterbox-Turbo, at
``config.TTS_BASE_URL``) directly, not the agent's ``/api/tts/speak`` proxy,
to keep the autonomy path free of self-referential HTTP hops. Uses the
buffered ``/v1/audio/speech`` endpoint, which both engines support.
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

        Voice resolution: the runtime default override (set via the voice-
        management UI) wins unconditionally — internal callers (autonomy
        speaker, etc.) all represent "the assistant" and should follow the
        configured voice, not a per-call argument. Falls back to the
        passed ``voice``, then to the engine's configured default. The
        playground bypass lives on the HTTP proxy (``force_voice`` field),
        not here.
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
        override: Optional[str] = None
        try:
            from selene_agent.utils import agent_state
            override = await agent_state.get_default_voice()
        except Exception:
            override = None
        resolved = override or voice
        if resolved:
            body["voice"] = resolved
        # max_field_size=65536 (default 8190) — the upstream's X-Visemes
        # header is base64-encoded Rhubarb JSON, which overflows the default
        # on ~30 s utterances. Bumped defensively here even though autonomy
        # speak-tier replies are usually short.
        async with aiohttp.ClientSession(
            timeout=self._timeout, max_field_size=65536,
        ) as session:
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
