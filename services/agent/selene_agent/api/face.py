"""Thin pass-through proxy for the face-recognition service.

Lives in the agent so the SvelteKit dashboard stays single-port (6002) and
nginx config doesn't need to change. All endpoints below mirror the
face-recognition HTTP surface 1:1; only the path prefix changes:

  agent /api/face/*   →   face-recognition /api/*

The proxy reads FACE_REC_API_BASE (same env mcp_face_tools uses) and falls
back to the compose-internal hostname. It deliberately doesn't add auth,
caching, or rewriting — face-recognition isn't exposed off the LAN, and
adding indirection here would just be a place for the two surfaces to
drift.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from selene_agent.utils import logger as custom_logger


logger = custom_logger.get_logger('loki')

router = APIRouter(prefix="/face", tags=["face"])


FACE_REC_API_BASE = os.getenv("FACE_REC_API_BASE", "http://face-recognition:6006").rstrip("/")
JSON_TIMEOUT = aiohttp.ClientTimeout(total=30)
# Confirm re-runs detect+embed on the saved snapshot (~100ms on warm GPU).
# Bumped slightly for cold-cache safety.
CONFIRM_TIMEOUT = aiohttp.ClientTimeout(total=15)
# Image enrollment uploads run insightface on the upload too.
UPLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)
# Streaming: detection snapshots and face_images are small (<2MB) but the
# total covers connect + first byte + drain.
STREAM_TIMEOUT = aiohttp.ClientTimeout(total=30)


async def _forward_json(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: aiohttp.ClientTimeout = JSON_TIMEOUT,
) -> Any:
    """Forward a JSON-in / JSON-out request and bubble up the upstream status.

    Upstream JSON errors (FastAPI validation / our HTTPException detail)
    pass through unchanged so the dashboard sees the real message instead
    of a generic 502.
    """
    url = f"{FACE_REC_API_BASE}{path}"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method, url, json=json_body, params=params,
            ) as resp:
                # 204 has no body; everything else we expect JSON.
                if resp.status == 204:
                    return Response(status_code=204)
                payload = await resp.json(content_type=None)
                if resp.status >= 400:
                    return JSONResponse(status_code=resp.status, content=payload)
                return JSONResponse(status_code=resp.status, content=payload)
    except aiohttp.ClientError as e:
        logger.error("face-rec proxy %s %s failed: %s", method, path, e)
        raise HTTPException(status_code=502, detail=f"face-recognition unreachable: {e}")


async def _stream_bytes(path: str) -> StreamingResponse:
    """Stream an upstream binary response (JPEG) chunk-by-chunk.

    Opens the aiohttp session inside an async generator so the connection
    survives until the StreamingResponse is fully drained. Closing the
    session prematurely would truncate the response.
    """
    url = f"{FACE_REC_API_BASE}{path}"
    session = aiohttp.ClientSession(timeout=STREAM_TIMEOUT)
    try:
        resp = await session.get(url)
    except aiohttp.ClientError as e:
        await session.close()
        logger.error("face-rec stream GET %s failed: %s", path, e)
        raise HTTPException(status_code=502, detail=f"face-recognition unreachable: {e}")

    if resp.status >= 400:
        try:
            body = await resp.text()
        finally:
            resp.release()
            await session.close()
        # Forward the upstream JSON error if the caller can use it; else raw text.
        try:
            import json as _json
            return JSONResponse(status_code=resp.status, content=_json.loads(body))
        except Exception:
            raise HTTPException(status_code=resp.status, detail=body[:500])

    media_type = resp.headers.get("content-type", "application/octet-stream")

    async def _iter():
        try:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                yield chunk
        finally:
            resp.release()
            await session.close()

    return StreamingResponse(_iter(), media_type=media_type)


# ---------- People ----------

@router.get("/people")
async def list_people():
    return await _forward_json("GET", "/api/people")


@router.post("/people")
async def create_person(body: Dict[str, Any]):
    return await _forward_json("POST", "/api/people", json_body=body)


@router.get("/people/{person_id}")
async def get_person(person_id: str):
    return await _forward_json("GET", f"/api/people/{person_id}")


@router.patch("/people/{person_id}")
async def update_person(person_id: str, body: Dict[str, Any]):
    return await _forward_json("PATCH", f"/api/people/{person_id}", json_body=body)


@router.delete("/people/{person_id}")
async def delete_person(person_id: str):
    return await _forward_json("DELETE", f"/api/people/{person_id}")


# ---------- Person images ----------

@router.post("/people/{person_id}/images")
async def enroll_image(person_id: str, request: Request):
    """Forward a multipart upload as multipart.

    aiohttp accepts the raw body + the original Content-Type (which carries
    the boundary), so we re-stream the request body unmodified rather than
    re-parsing + re-encoding the form. Cheaper and avoids a multipart
    round-trip that could lose field ordering.
    """
    content_type = request.headers.get("content-type")
    if not content_type or not content_type.startswith("multipart/"):
        raise HTTPException(
            status_code=400, detail="multipart/form-data required",
        )
    body = await request.body()
    url = f"{FACE_REC_API_BASE}/api/people/{person_id}/images"
    try:
        async with aiohttp.ClientSession(timeout=UPLOAD_TIMEOUT) as session:
            async with session.post(
                url, data=body, headers={"content-type": content_type},
            ) as resp:
                payload = await resp.json(content_type=None)
                return JSONResponse(status_code=resp.status, content=payload)
    except aiohttp.ClientError as e:
        logger.error("face-rec proxy upload failed: %s", e)
        raise HTTPException(status_code=502, detail=f"face-recognition unreachable: {e}")


@router.delete("/people/{person_id}/images/{face_image_id}")
async def delete_face_image(person_id: str, face_image_id: str):
    return await _forward_json(
        "DELETE", f"/api/people/{person_id}/images/{face_image_id}",
    )


@router.post("/people/{person_id}/images/{face_image_id}/set-primary")
async def set_primary(person_id: str, face_image_id: str):
    return await _forward_json(
        "POST", f"/api/people/{person_id}/images/{face_image_id}/set-primary",
    )


@router.post("/people/{person_id}/enroll-from-camera")
async def enroll_from_camera(person_id: str, body: Dict[str, Any]):
    return await _forward_json(
        "POST", f"/api/people/{person_id}/enroll-from-camera",
        json_body=body, timeout=UPLOAD_TIMEOUT,
    )


# ---------- Face images (flat) ----------

@router.get("/face_images/{face_image_id}/bytes")
async def stream_face_image(face_image_id: str):
    return await _stream_bytes(f"/api/face_images/{face_image_id}/bytes")


# ---------- Detections ----------

@router.get("/detections")
async def list_detections(request: Request):
    # Forward query string verbatim — keeps unknowns_only / review_state /
    # since_seconds_ago filters in sync without enumerating them here.
    return await _forward_json(
        "GET", "/api/detections",
        params=dict(request.query_params),
    )


@router.get("/detections/{detection_id}/snapshot")
async def stream_detection_snapshot(detection_id: str):
    return await _stream_bytes(f"/api/detections/{detection_id}/snapshot")


@router.post("/detections/{detection_id}/confirm")
async def confirm_detection(detection_id: str, body: Dict[str, Any]):
    return await _forward_json(
        "POST", f"/api/detections/{detection_id}/confirm",
        json_body=body, timeout=CONFIRM_TIMEOUT,
    )


@router.post("/detections/{detection_id}/reject")
async def reject_detection(detection_id: str):
    return await _forward_json("POST", f"/api/detections/{detection_id}/reject")


@router.post("/detections/bulk-delete")
async def bulk_delete_detections(body: Dict[str, Any]):
    return await _forward_json(
        "POST", "/api/detections/bulk-delete", json_body=body,
    )


# ---------- Admin (rescan / job polling) ----------

# Rescan-unknowns walks every unknown row, re-embeds, and queries Qdrant.
# That's an expensive backfill — kick is fast, the work happens in a
# background asyncio task. Polling reads the in-memory job dict.
RESCAN_KICK_TIMEOUT = aiohttp.ClientTimeout(total=10)


@router.post("/admin/rescan-unknowns")
async def rescan_unknowns():
    return await _forward_json(
        "POST", "/api/admin/rescan-unknowns", timeout=RESCAN_KICK_TIMEOUT,
    )


@router.get("/admin/jobs/{job_id}")
async def get_admin_job(job_id: str):
    return await _forward_json("GET", f"/api/admin/jobs/{job_id}")


# ---------- Cameras (for the review-queue dropdown) ----------

@router.get("/cameras")
async def list_cameras():
    return await _forward_json("GET", "/api/cameras")


# ---------- Health ----------

@router.get("/health")
async def health():
    return await _forward_json("GET", "/health")
