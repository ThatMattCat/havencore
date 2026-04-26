"""Burst-capture frames from a Home Assistant camera entity.

Uses HA's `/api/camera_proxy/<entity_id>` REST endpoint, which returns a
single JPEG per call. We hit it N times sequentially with a configurable
sleep between requests — small N (default 6), low interval (500ms), so the
burst takes ~3s. A single aiohttp.ClientSession is reused across the burst
to avoid TCP/TLS setup cost per frame.

Reuses HAOS_URL / HAOS_TOKEN, the same env vars the agent's
mcp_homeassistant_tools and mcp_mqtt_tools already consume. The example
.env value is `https://localhost:8123/api`; we strip a trailing `/api`
the same way HomeAssistantClient does so the URL works unchanged.
"""

import asyncio
import logging
import os

import aiohttp
import cv2
import numpy as np


logger = logging.getLogger("face-recognition.ha_snapshot")


HAOS_URL = os.getenv("HAOS_URL", "")
HAOS_TOKEN = os.getenv("HAOS_TOKEN", "")
HA_REQUEST_TIMEOUT_SEC = float(os.getenv("FACE_REC_HA_TIMEOUT_SEC", "10"))


def _normalize_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base


class HASnapshotError(RuntimeError):
    """Raised when HA capture is unrecoverable (auth, missing entity, network)."""


async def capture_burst(camera: str, n: int, interval_ms: int) -> list[np.ndarray]:
    """Pull N frames from `camera_proxy/<camera>`, decode to BGR.

    Returns the decoded frames in capture order. Frames that fail to decode
    are skipped (logged, not fatal). The first request must succeed —
    otherwise the camera entity is wrong, HA is down, or auth is bad, and
    the rest of the burst will fail the same way; we raise immediately.

    Subsequent transient failures are logged and skipped; if zero frames
    decode successfully we return [] and let the pipeline treat it as a
    no-frames outcome.
    """
    if not HAOS_URL or not HAOS_TOKEN:
        raise HASnapshotError("HAOS_URL / HAOS_TOKEN not configured")

    base = _normalize_base_url(HAOS_URL)
    url = f"{base}/api/camera_proxy/{camera}"
    headers = {"Authorization": f"Bearer {HAOS_TOKEN}"}
    timeout = aiohttp.ClientTimeout(total=HA_REQUEST_TIMEOUT_SEC)

    frames: list[np.ndarray] = []
    interval_sec = max(0.0, interval_ms / 1000.0)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for i in range(n):
            try:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        msg = f"HA camera_proxy returned {resp.status} for {camera}"
                        if i == 0:
                            raise HASnapshotError(msg)
                        logger.warning("%s (frame %d/%d, skipping)", msg, i + 1, n)
                        continue
                    payload = await resp.read()
            except aiohttp.ClientError as e:
                if i == 0:
                    raise HASnapshotError(f"HA camera_proxy network error: {e}") from e
                logger.warning("HA camera_proxy network error on frame %d/%d: %s", i + 1, n, e)
                if i < n - 1:
                    await asyncio.sleep(interval_sec)
                continue

            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Failed to decode JPEG from camera_proxy frame %d/%d", i + 1, n)
            else:
                frames.append(frame)

            if i < n - 1:
                await asyncio.sleep(interval_sec)

    logger.info("Captured %d/%d frames from camera %s", len(frames), n, camera)
    return frames
