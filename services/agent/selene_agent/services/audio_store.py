"""Token-keyed in-memory audio blob store.

Used by ``SpeakerNotifier`` to stage TTS-rendered audio behind a short-lived
random URL. Music Assistant fetches the URL; we serve the bytes once (or
until the TTL) and drop them.

No persistence — a process restart evicts everything. That's deliberate:
if MA couldn't fetch an announcement inside the TTL window, retrying on the
old URL shouldn't suddenly succeed later.
"""
from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class _Entry:
    data: bytes
    expires_at: float
    content_type: str


class AudioStore:
    """In-memory ``{token: (bytes, expires_at)}`` map with a size cap.

    Semantics:
    - ``put(data)`` -> returns a url-safe token (16 random bytes encoded).
    - ``get(token)`` -> returns ``(bytes, content_type)`` or ``None`` if the
      token is unknown, expired, or already consumed. Entries evict after
      first successful ``get`` (single-fetch) or when their TTL passes.
    - ``max_entries`` caps total staged blobs; oldest entries are evicted
      first when the cap is exceeded, so a stuck MA consumer can't fill the
      process heap.
    """

    def __init__(
        self,
        *,
        default_ttl_sec: int = 600,
        max_entries: int = 64,
    ):
        self._default_ttl_sec = default_ttl_sec
        self._max_entries = max_entries
        self._entries: Dict[str, _Entry] = {}
        self._lock = asyncio.Lock()

    async def put(
        self,
        data: bytes,
        *,
        ttl_sec: Optional[int] = None,
        content_type: str = "audio/mpeg",
    ) -> str:
        if not data:
            raise ValueError("AudioStore.put: empty data")
        ttl = int(ttl_sec if ttl_sec is not None else self._default_ttl_sec)
        token = secrets.token_urlsafe(16)
        entry = _Entry(
            data=data,
            expires_at=time.monotonic() + ttl,
            content_type=content_type,
        )
        async with self._lock:
            self._evict_expired_locked()
            self._entries[token] = entry
            if len(self._entries) > self._max_entries:
                self._evict_oldest_locked()
        return token

    async def get(self, token: str) -> Optional[Tuple[bytes, str]]:
        if not token:
            return None
        async with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._entries.pop(token, None)
                return None
            # Single-fetch semantics: pop on successful read.
            self._entries.pop(token, None)
            return entry.data, entry.content_type

    async def size(self) -> int:
        async with self._lock:
            self._evict_expired_locked()
            return len(self._entries)

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        expired = [t for t, e in self._entries.items() if e.expires_at <= now]
        for t in expired:
            self._entries.pop(t, None)

    def _evict_oldest_locked(self) -> None:
        # _entries preserves insertion order — popitem(last=False) evicts oldest.
        while len(self._entries) > self._max_entries:
            oldest = next(iter(self._entries))
            self._entries.pop(oldest, None)


# Process-global singleton — autonomy handlers and the tts_audio router both
# need to see the same store.
_store: Optional[AudioStore] = None


def get_audio_store() -> AudioStore:
    global _store
    if _store is None:
        from selene_agent.utils import config
        _store = AudioStore(
            default_ttl_sec=getattr(config, "AUTONOMY_TTS_AUDIO_TTL_SEC", 600),
        )
    return _store
