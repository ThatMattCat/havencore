"""On-demand face identification.

`POST /api/identify` accepts a single image (multipart `file`), runs detect+
embed, and queries the existing Qdrant gallery for the nearest enrolled
face. Unlike the `/api/trigger` pipeline, this does not store the snapshot,
write a `face_detections` row, or publish MQTT — it is the agent's
companion-camera side-channel asking "who is this?" for one frame.

Returns 200 even when no enrolled face matches (face-not-recognized is a
normal result, not an error). The HTTP error path is reserved for upstream
problems the caller can fix (bad image, embedder not ready).
"""

import io
import logging
import uuid

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import APIRouter, File, HTTPException, UploadFile

import config
from db import db
from embedder import embedder
from face_qdrant import vector_store
from quality import score_face


logger = logging.getLogger("face-recognition.api.identify")

router = APIRouter(prefix="/api", tags=["identify"])


@router.post("/identify")
async def identify(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    img = _decode_with_exif_rotation(raw)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    faces = embedder.detect_and_embed(img)
    if not faces:
        return {"found": False, "face_count": 0}

    # Pick the highest-quality face when multiple are visible. Falls back to
    # detector confidence when the quality score floors at -1.
    best_face = None
    best_quality = -1.0
    for face in faces:
        q = score_face(img, face)
        if q > best_quality:
            best_quality = q
            best_face = face

    hits = vector_store.query(best_face.normed_embedding, limit=1)
    if not hits:
        return {"found": False, "face_count": len(faces)}

    top = hits[0]
    confidence = float(top.score)
    if confidence <= config.MATCH_THRESHOLD:
        return {
            "found": False,
            "face_count": len(faces),
            "confidence": confidence,
        }

    payload = top.payload or {}
    raw_pid = payload.get("person_id")
    person_id: uuid.UUID | None = None
    if raw_pid:
        try:
            person_id = uuid.UUID(raw_pid)
        except ValueError:
            logger.warning(
                "Qdrant point %s has invalid person_id=%r", top.id, raw_pid,
            )

    name: str | None = None
    if person_id is not None:
        name = await db.get_person_name(person_id)

    if name is None:
        # Match fired but the person row is gone. Surface as not-found rather
        # than returning a dangling id.
        logger.warning(
            "Qdrant point %s matched but person %s is missing; "
            "returning not-found",
            top.id, person_id,
        )
        return {
            "found": False,
            "face_count": len(faces),
            "confidence": confidence,
        }

    return {
        "found": True,
        "name": name,
        "person_id": str(person_id),
        "confidence": confidence,
        "face_count": len(faces),
    }


def _decode_with_exif_rotation(raw: bytes) -> "np.ndarray | None":
    """Decode JPEG bytes to BGR honoring EXIF orientation.

    ``cv2.imdecode`` reads pixel data straight off the JFIF segment and
    ignores the EXIF orientation tag, so phone-camera portraits (which
    are stored as landscape bytes plus a "rotate 90" EXIF flag) come out
    sideways. RetinaFace at det_size=1280 misses sideways faces often
    enough to break this endpoint for the most common companion-app
    input. PIL's ``exif_transpose`` applies the orientation in pixel
    space; we then hand the corrected RGB array to OpenCV converted to
    BGR so the rest of the pipeline (embedder, quality scoring) sees the
    same colorspace it always has.
    """
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rotated = ImageOps.exif_transpose(im)
            if rotated.mode != "RGB":
                rotated = rotated.convert("RGB")
            arr_rgb = np.asarray(rotated)
    except (UnidentifiedImageError, OSError) as e:
        logger.warning("PIL decode failed, falling back to cv2.imdecode: %s", e)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
