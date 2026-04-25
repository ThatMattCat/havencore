"""People + enrollment endpoints.

Atomicity note: enrollment writes to disk → Qdrant → Postgres in that
order. A failure between Qdrant upsert and DB insert leaves an orphan
Qdrant point + JPEG (the row never lands). The Step 8 admin
`rebuild-embeddings` endpoint will reconcile drift; for v1 we accept
this and surface the failure as a 500 so the client can retry.
"""

import asyncio
import logging
import uuid
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Path as PathParam, UploadFile

import config
import ha_snapshot
from db import db
from embedder import embedder
from face_qdrant import vector_store
from models import (
    CameraEnrollmentResult,
    EnrollFromCameraRequest,
    EnrollmentResult,
    FaceImageDeleteResult,
    FaceImageOut,
    PersonCreate,
    PersonDeleteResult,
    PersonDetailOut,
    PersonOut,
    PersonUpdate,
)
from quality import score_face


logger = logging.getLogger("face-recognition.api.people")

router = APIRouter(prefix="/api/people", tags=["people"])


ENROLLMENT_DIR = Path(config.SNAPSHOT_DIR) / "enrollments"


def _select_best_face_in_burst(frames: list[np.ndarray]):
    """Single-pass best-quality face across a burst.

    Detect + score every face in every frame, drop anything below
    QUALITY_FLOOR, return the single highest-quality face. Synchronous so
    the caller can offload the whole thing to a thread (insightface inference
    is CPU/CUDA-blocking from asyncio's perspective).

    Mirrors `pipeline._detect_and_score_sync` but for the simpler enrollment
    case — we want one face, not the top-K mean used in detection.
    """
    best_face = None
    best_quality = -1.0
    best_frame_idx = -1
    faces_kept = 0
    for idx, frame in enumerate(frames):
        for face in embedder.detect_and_embed(frame):
            q = score_face(frame, face)
            if q < config.QUALITY_FLOOR:
                continue
            faces_kept += 1
            if q > best_quality:
                best_quality = q
                best_face = face
                best_frame_idx = idx
    return best_face, best_quality, best_frame_idx, faces_kept


async def _persist_enrollment(
    *,
    person_id: uuid.UUID,
    frame: np.ndarray,
    embedding: np.ndarray,
    quality_score: float,
    is_primary: bool,
    source: str,
) -> dict:
    """Shared write path for both upload and camera enrollment.

    file → Qdrant → DB ordering matches the existing enrollment endpoint.
    Returns the inserted face_images row dict + the qdrant_point_id.
    """
    face_image_id = uuid.uuid4()
    qdrant_point_id = uuid.uuid4()

    person_dir = ENROLLMENT_DIR / str(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    img_path = person_dir / f"{face_image_id}.jpg"
    if not cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise HTTPException(status_code=500, detail="failed to write enrollment image to disk")

    try:
        vector_store.upsert_point(
            point_id=str(qdrant_point_id),
            vector=embedding.tolist(),
            payload={
                "person_id": str(person_id),
                "face_image_id": str(face_image_id),
                "source": source,
            },
        )
    except Exception as e:
        img_path.unlink(missing_ok=True)
        logger.exception("Qdrant upsert failed during enrollment")
        raise HTTPException(status_code=500, detail=f"qdrant upsert failed: {e}")

    try:
        row = await db.insert_face_image(
            face_image_id=face_image_id,
            person_id=person_id,
            path=str(img_path),
            qdrant_point_id=qdrant_point_id,
            source=source,
            quality_score=quality_score,
            is_primary=is_primary,
        )
    except Exception as e:
        img_path.unlink(missing_ok=True)
        logger.exception("DB insert failed after Qdrant upsert (orphan point %s)", qdrant_point_id)
        raise HTTPException(status_code=500, detail=f"db insert failed: {e}")

    return {"row": row, "qdrant_point_id": qdrant_point_id}


@router.post("", response_model=PersonOut, status_code=201)
async def create_person(payload: PersonCreate) -> PersonOut:
    try:
        row = await db.create_person(
            name=payload.name,
            access_level=payload.access_level,
            notes=payload.notes,
        )
    except asyncpg.UniqueViolationError:
        raise HTTPException(status_code=409, detail=f"person '{payload.name}' already exists")
    return PersonOut(**row)


@router.get("", response_model=list[PersonOut])
async def list_people() -> list[PersonOut]:
    rows = await db.list_people()
    return [PersonOut(**r) for r in rows]


@router.get("/{person_id}", response_model=PersonDetailOut)
async def get_person(person_id: uuid.UUID = PathParam(...)) -> PersonDetailOut:
    person = await db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")
    images = await db.list_face_images(person_id)
    return PersonDetailOut(
        **person,
        images=[FaceImageOut(**img) for img in images],
    )


@router.patch("/{person_id}", response_model=PersonOut)
async def update_person(
    payload: PersonUpdate,
    person_id: uuid.UUID = PathParam(...),
) -> PersonOut:
    if payload.access_level is None and payload.notes is None:
        raise HTTPException(status_code=400, detail="no fields to update")
    row = await db.update_person(
        person_id=person_id,
        access_level=payload.access_level,
        notes=payload.notes,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="person not found")
    return PersonOut(**row)


@router.post(
    "/{person_id}/images",
    response_model=EnrollmentResult,
    status_code=201,
)
async def enroll_image(
    person_id: uuid.UUID = PathParam(...),
    file: UploadFile = File(...),
    is_primary: bool = Form(False),
    source: str = Form("upload"),
) -> EnrollmentResult:
    person = await db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    faces = embedder.detect_and_embed(img)
    if not faces:
        raise HTTPException(status_code=422, detail="no face detected in image")

    # Single uploaded image: detector confidence is the best signal we have.
    # Multi-frame burst enrollment uses the real quality.score_face() instead.
    faces_sorted = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    best = faces_sorted[0]
    quality_score = float(best.det_score)

    persisted = await _persist_enrollment(
        person_id=person_id,
        frame=img,
        embedding=best.normed_embedding,
        quality_score=quality_score,
        is_primary=is_primary,
        source=source,
    )

    return EnrollmentResult(
        id=persisted["row"]["id"],
        qdrant_point_id=persisted["qdrant_point_id"],
        quality_score=quality_score,
        faces_detected=len(faces),
    )


@router.post(
    "/{person_id}/enroll-from-camera",
    response_model=CameraEnrollmentResult,
    status_code=201,
)
async def enroll_from_camera(
    payload: EnrollFromCameraRequest,
    person_id: uuid.UUID = PathParam(...),
) -> CameraEnrollmentResult:
    """Burst-capture from an HA camera and enroll the best face.

    Reuses the same burst settings as the detection pipeline (FACE_REC_BURST_*).
    Does NOT persist a face_detections row and does NOT publish MQTT — this
    is enrollment, not detection. The agent's `face_enroll_person` tool calls
    this when its `source` arg is `camera:<entity_id>`.
    """
    person = await db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")

    try:
        frames = await ha_snapshot.capture_burst(
            camera=payload.camera,
            n=config.BURST_FRAMES,
            interval_ms=config.BURST_INTERVAL_MS,
        )
    except ha_snapshot.HASnapshotError as e:
        logger.warning("HA capture failed for camera=%s: %s", payload.camera, e)
        raise HTTPException(status_code=502, detail=f"home assistant capture failed: {e}")

    if not frames:
        raise HTTPException(status_code=502, detail="no frames captured from camera")

    best_face, best_quality, best_frame_idx, faces_kept = await asyncio.to_thread(
        _select_best_face_in_burst, frames
    )

    if best_face is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"no face cleared quality floor ({config.QUALITY_FLOOR:.2f}) "
                f"across {len(frames)} frames"
            ),
        )

    persisted = await _persist_enrollment(
        person_id=person_id,
        frame=frames[best_frame_idx],
        embedding=best_face.normed_embedding,
        quality_score=best_quality,
        is_primary=payload.is_primary,
        source="agent_enroll",
    )

    return CameraEnrollmentResult(
        id=persisted["row"]["id"],
        qdrant_point_id=persisted["qdrant_point_id"],
        quality_score=best_quality,
        frames_processed=len(frames),
        faces_kept=faces_kept,
    )


@router.delete("/{person_id}", response_model=PersonDeleteResult)
async def delete_person(
    person_id: uuid.UUID = PathParam(...),
) -> PersonDeleteResult:
    """Delete a person, their enrollment images, and their Qdrant points.

    Cascade order: collect file paths from face_images BEFORE deleting the
    person row (so we still have them after the FK cascade fires), then
    delete the person row, then unlink files, then drop Qdrant points by
    payload filter. Detection rows survive with person_id NULL'd
    (ON DELETE SET NULL on face_detections.person_id).
    """
    images = await db.delete_person(person_id)
    if images is None:
        raise HTTPException(status_code=404, detail="person not found")

    for img in images:
        try:
            Path(img["path"]).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to unlink %s during person delete: %s", img["path"], e)

    try:
        vector_store.delete_by_person(str(person_id))
    except Exception as e:
        # Person + files are gone; orphan points get caught by the step-8
        # rebuild-embeddings reconcile pass. Log loudly so it's noticed.
        logger.exception("Qdrant cascade-delete failed for person %s: %s", person_id, e)

    return PersonDeleteResult(
        id=person_id,
        images_removed=len(images),
        qdrant_points_removed=len(images),
    )


@router.delete(
    "/{person_id}/images/{face_image_id}",
    response_model=FaceImageDeleteResult,
)
async def delete_face_image(
    person_id: uuid.UUID = PathParam(...),
    face_image_id: uuid.UUID = PathParam(...),
) -> FaceImageDeleteResult:
    """Delete one face_image row, its file, and its Qdrant point.

    Refuses to delete a row marked is_primary — caller should set-primary
    on a different image first so the gallery always has a primary.
    """
    img = await db.get_face_image(person_id=person_id, face_image_id=face_image_id)
    if img is None:
        raise HTTPException(status_code=404, detail="face image not found")
    if img["is_primary"]:
        raise HTTPException(
            status_code=409,
            detail="cannot delete primary image; set another image as primary first",
        )

    deleted = await db.delete_face_image(person_id=person_id, face_image_id=face_image_id)
    if deleted is None:
        # Lost a race or concurrent edit flipped it to primary — surface the
        # collision rather than silently 404'ing.
        raise HTTPException(status_code=409, detail="face image could not be deleted")

    abs_path = _resolve_face_image_path(deleted["path"])
    try:
        abs_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Failed to unlink %s during image delete: %s", abs_path, e)

    try:
        vector_store.delete_point(str(deleted["qdrant_point_id"]))
    except Exception as e:
        logger.warning(
            "Failed to delete Qdrant point %s: %s", deleted["qdrant_point_id"], e,
        )

    return FaceImageDeleteResult(id=face_image_id)


@router.post(
    "/{person_id}/images/{face_image_id}/set-primary",
    response_model=FaceImageOut,
)
async def set_primary_face_image(
    person_id: uuid.UUID = PathParam(...),
    face_image_id: uuid.UUID = PathParam(...),
) -> FaceImageOut:
    """Atomically swap the primary marker to `face_image_id`."""
    row = await db.set_primary_face_image(
        person_id=person_id, face_image_id=face_image_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="face image not found")
    return FaceImageOut(**row)


def _resolve_face_image_path(path: str) -> Path:
    """Resolve a face_images.path to an absolute filesystem path.

    Absolute paths (existing enrollment + auto-improvement rows) are returned
    as-is. Relative paths resolve under SNAPSHOT_DIR (forward-compatible with
    the step-8 path normalization that will move enrollment paths to relative).
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(config.SNAPSHOT_DIR) / p
