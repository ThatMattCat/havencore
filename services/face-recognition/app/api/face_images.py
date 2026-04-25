"""Flat face_images bytes-streaming endpoint.

Lives in its own module so the URL stays flat (`/api/face_images/{id}/bytes`)
without colliding with the person-scoped routes under `/api/people/...`.
The dashboard and any other consumer references face_image rows by id, so
the URL doesn't need to know which person they belong to.
"""

import logging
import uuid

from fastapi import APIRouter, HTTPException, Path as PathParam
from fastapi.responses import FileResponse

from api.people import _resolve_face_image_path
from db import db


logger = logging.getLogger("face-recognition.api.face_images")

router = APIRouter(prefix="/api/face_images", tags=["face_images"])


@router.get("/{face_image_id}/bytes")
async def stream_face_image_bytes(
    face_image_id: uuid.UUID = PathParam(...),
):
    """Stream a face_image JPEG by id.

    Identity URL — no filesystem path leaks to the client. Tolerates both
    absolute (current enrollment + auto-improvement) and relative paths
    (forward-compatible with the step-8 path normalization).
    """
    img = await db.get_face_image_by_id(face_image_id)
    if img is None:
        raise HTTPException(status_code=404, detail="face image not found")
    abs_path = _resolve_face_image_path(img["path"])
    if not abs_path.exists():
        logger.warning("face_image %s row exists but file missing: %s", face_image_id, abs_path)
        raise HTTPException(status_code=410, detail="face image file missing on disk")
    return FileResponse(abs_path, media_type="image/jpeg")
