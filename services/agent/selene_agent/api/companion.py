"""Companion-app side-channel: image uploads from the phone.

Round-trip pattern for camera-style device-action tools (e.g. ``take_photo``):

1. The LLM invokes ``take_photo`` (a tool declared by ``mcp_device_action_tools``).
2. The orchestrator emits a ``device_action`` WS frame *before* running the
   tool, then registers an ``asyncio.Future`` keyed by ``tool_call_id`` via
   :func:`register_pending_upload` and awaits it.
3. The companion app receives the frame, captures a photo, and POSTs it to
   ``/api/companion/upload`` with the original ``tool_call_id``.
4. This module stashes the bytes in :class:`BlobStore`, mints a short-lived
   blob URL, and resolves the matching future. The orchestrator's await
   returns and the tool result (containing ``image_url``) flows back to the
   LLM.

The MCP server's handler for ``take_photo`` is a fallback no-op — the
orchestrator short-circuits this path because the future + blob registry
both live in this (agent) process, not the stdio MCP subprocess.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter()

# --- Pending upload registry ---------------------------------------------
#
# Mapping from ``tool_call_id`` -> ``asyncio.Future``. Populated by the
# orchestrator before it emits the ``device_action`` event and awaited
# until either the upload arrives or the per-tool timeout fires. Lives at
# module scope (not on app.state) so the orchestrator can register from
# anywhere without a request handle.

_pending_uploads: Dict[str, asyncio.Future] = {}


def register_pending_upload(tool_call_id: str) -> asyncio.Future:
    """Register a future to be resolved when the companion uploads a blob.

    Subsequent calls with the same ``tool_call_id`` replace the prior future
    (the previous one is cancelled to unblock its waiter). The orchestrator
    is expected to ``pop_pending_upload`` after its wait returns or times
    out so stale entries don't accumulate.
    """
    prev = _pending_uploads.get(tool_call_id)
    if prev is not None and not prev.done():
        prev.cancel()
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending_uploads[tool_call_id] = fut
    return fut


def pop_pending_upload(tool_call_id: str) -> Optional[asyncio.Future]:
    """Remove and return the future for ``tool_call_id`` if present."""
    return _pending_uploads.pop(tool_call_id, None)


# --- BlobStore -----------------------------------------------------------


@dataclass
class _Blob:
    data: bytes
    mime: str
    expires_at: float
    tool_call_id: str
    device_id: Optional[str]


class BlobStore:
    """In-memory TTL-bounded store for companion-uploaded blobs.

    Keyed by an opaque token. Vision/face pipelines fetch the blob via
    ``GET /api/companion/blob/{token}`` while it lives. The store enforces
    a per-blob TTL and a total byte cap (oldest entries evicted first).
    """

    def __init__(
        self,
        ttl_sec: int = 600,
        max_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.ttl_sec = ttl_sec
        self.max_bytes = max_bytes
        self._blobs: Dict[str, _Blob] = {}
        self._cleanup_task: Optional[asyncio.Task] = None

    def put(
        self,
        data: bytes,
        mime: str,
        tool_call_id: str,
        device_id: Optional[str],
    ) -> tuple[str, float]:
        """Stash ``data`` and return ``(token, expires_at_unix)``."""
        token = secrets.token_urlsafe(24)
        expires_at = time.time() + self.ttl_sec
        self._blobs[token] = _Blob(
            data=data,
            mime=mime,
            expires_at=expires_at,
            tool_call_id=tool_call_id,
            device_id=device_id,
        )
        self._enforce_byte_cap()
        return token, expires_at

    def get(self, token: str) -> Optional[_Blob]:
        blob = self._blobs.get(token)
        if blob is None:
            return None
        if blob.expires_at < time.time():
            self._blobs.pop(token, None)
            return None
        return blob

    def _enforce_byte_cap(self) -> None:
        total = sum(len(b.data) for b in self._blobs.values())
        if total <= self.max_bytes:
            return
        # Evict oldest (lowest expires_at) until we fit. Stable for ties.
        ordered = sorted(self._blobs.items(), key=lambda kv: kv[1].expires_at)
        for token, blob in ordered:
            if total <= self.max_bytes:
                break
            self._blobs.pop(token, None)
            total -= len(blob.data)
            logger.info(
                f"BlobStore: evicted {token[:8]}… ({len(blob.data)} bytes) to enforce cap"
            )

    def sweep(self) -> int:
        now = time.time()
        expired = [t for t, b in self._blobs.items() if b.expires_at < now]
        for t in expired:
            self._blobs.pop(t, None)
        return len(expired)

    async def _cleanup_loop(self, interval_sec: int = 60) -> None:
        while True:
            try:
                await asyncio.sleep(interval_sec)
                n = self.sweep()
                if n:
                    logger.debug(f"BlobStore: swept {n} expired blob(s)")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"BlobStore cleanup loop error: {e}")

    async def start(self, interval_sec: int = 60) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval_sec))

    async def stop(self) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._cleanup_task = None


# --- Module-level singleton ----------------------------------------------
#
# Built lazily so tests can construct a fresh store with custom limits and
# reset the singleton between cases. Production wiring (lifespan in
# ``selene_agent.py``) calls :func:`get_blob_store` then ``await store.start()``.

_blob_store: Optional[BlobStore] = None


def get_blob_store() -> BlobStore:
    global _blob_store
    if _blob_store is None:
        _blob_store = BlobStore(
            ttl_sec=getattr(config, "COMPANION_BLOB_TTL_SEC", 600),
            max_bytes=getattr(config, "COMPANION_BLOB_MAX_BYTES", 10 * 1024 * 1024),
        )
    return _blob_store


def reset_blob_store_for_testing() -> None:
    """Drop the singleton + clear pending uploads — tests use this between cases."""
    global _blob_store
    _blob_store = None
    _pending_uploads.clear()


def _build_image_url(request: Request, token: str) -> str:
    """Build a URL that other in-network services (vision LLM) can fetch.

    Uses the configured Docker-internal base URL so HavenCore services can
    reach the blob without going through the user's reverse proxy. Falls
    back to the request's own base URL when no internal base is configured
    (mostly tests).
    """
    base = getattr(config, "AGENT_INTERNAL_BASE_URL", "") or ""
    base = base.rstrip("/")
    if not base:
        base = str(request.base_url).rstrip("/")
    return f"{base}/api/companion/blob/{token}"


# --- Endpoints -----------------------------------------------------------


@router.post("/companion/upload")
async def upload(
    request: Request,
    tool_call_id: str = Form(...),
    file: UploadFile = File(...),
    device_id: Optional[str] = Form(None),
):
    """Receive a companion-captured image and resolve the waiting tool call.

    Returns 410 when no orchestrator is currently awaiting this
    ``tool_call_id`` — usually means the per-tool timeout already fired or
    the LLM never asked for a photo.
    """
    fut = _pending_uploads.get(tool_call_id)
    if fut is None or fut.done():
        # Clean up any stale done-future entry so /upload retries don't 410
        # forever after a cancelled wait.
        _pending_uploads.pop(tool_call_id, None)
        raise HTTPException(
            status_code=410,
            detail="No tool call is waiting for this upload (timed out or unknown id)",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    max_bytes = getattr(config, "COMPANION_BLOB_MAX_BYTES", 10 * 1024 * 1024)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} > {max_bytes} bytes)",
        )

    mime = file.content_type or "application/octet-stream"
    store = get_blob_store()
    token, expires_at = store.put(
        data=data, mime=mime, tool_call_id=tool_call_id, device_id=device_id,
    )
    image_url = _build_image_url(request, token)
    captured_at = time.time()

    payload = {
        "image_url": image_url,
        "mime": mime,
        "captured_at": captured_at,
        "device_id": device_id,
        "blob_token": token,
        "size_bytes": len(data),
    }

    if not fut.done():
        fut.set_result(payload)

    logger.info(
        f"companion upload accepted: tool_call_id={tool_call_id} bytes={len(data)} "
        f"mime={mime} device_id={device_id} token={token[:8]}…"
    )

    return JSONResponse(
        content={
            "ok": True,
            "image_url": image_url,
            "expires_at": expires_at,
        }
    )


@router.get("/companion/blob/{token}")
async def get_blob(token: str):
    """Stream a previously-uploaded blob back to the caller."""
    store = get_blob_store()
    blob = store.get(token)
    if blob is None:
        raise HTTPException(status_code=404, detail="Blob not found or expired")
    return Response(content=blob.data, media_type=blob.mime)
