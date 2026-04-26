"""Detection-side endpoints.

`POST /api/trigger` runs the full pipeline once on demand for a given HA
camera entity. `GET /api/detections` surfaces the persisted history with
filters used by the agent's `face_who_is_at` / `face_recent_visitors` tools
and the SvelteKit `/people/detections` + `/people/unknowns` routes.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Path as PathParam, Query
from fastapi.responses import FileResponse

import config
import pipeline
from api.people import _persist_enrollment, _resolve_face_image_path
from db import db
from embedder import embedder
from ha_snapshot import HASnapshotError
from models import (
    BulkDeleteResult,
    BulkDeleteUnknownsRequest,
    ConfirmDetectionRequest,
    ConfirmDetectionResult,
    DetectionOut,
    PipelineResult,
)
from quality import score_face


logger = logging.getLogger("face-recognition.api.detections")

router = APIRouter(prefix="/api", tags=["detections"])


REVIEW_STATES = ("auto", "confirmed", "rejected", "pending")


@router.post("/trigger", response_model=PipelineResult)
async def trigger(
    camera: str = Query(..., description="HA camera entity_id (e.g. front_door)"),
) -> PipelineResult:
    event_id = uuid.uuid4()
    captured_at = datetime.now(timezone.utc)
    try:
        return await pipeline.process_event(camera, event_id, captured_at)
    except HASnapshotError as e:
        # Auth / wrong entity / network — caller can fix and retry.
        logger.warning("HA capture failed for camera=%s: %s", camera, e)
        raise HTTPException(status_code=502, detail=f"home assistant capture failed: {e}")


@router.get("/detections", response_model=list[DetectionOut])
async def list_detections(
    camera: Optional[str] = Query(None, description="Filter by HA camera entity_id"),
    since_seconds_ago: Optional[int] = Query(
        None, ge=1, description="Only detections within the last N seconds"
    ),
    person_id: Optional[uuid.UUID] = Query(None, description="Filter to one person"),
    review_state: Optional[str] = Query(
        None, description="Filter by review_state: auto|confirmed|rejected|pending"
    ),
    unknowns_only: bool = Query(
        False,
        description="Shorthand: person_id IS NULL AND review_state != 'rejected'",
    ),
    limit: int = Query(20, ge=1, le=200),
) -> list[DetectionOut]:
    if review_state is not None and review_state not in REVIEW_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"review_state must be one of {REVIEW_STATES}",
        )
    rows = await db.list_detections(
        camera=camera,
        since_seconds_ago=since_seconds_ago,
        person_id=person_id,
        limit=limit,
        review_state=review_state,
        unknowns_only=unknowns_only,
    )
    return [DetectionOut(**r) for r in rows]


@router.get("/detections/{detection_id}/snapshot")
async def stream_detection_snapshot(
    detection_id: uuid.UUID = PathParam(...),
):
    """Stream a detection's snapshot JPEG.

    `face_detections.snapshot_path` is stored relative to SNAPSHOT_DIR
    (step 4) but tolerates absolute paths in case future rows are written
    that way. Identity URL — the on-disk path stays internal.
    """
    detection = await db.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")
    abs_path = _resolve_snapshot_path(detection["snapshot_path"])
    if not abs_path.exists():
        logger.warning(
            "detection %s row exists but snapshot missing: %s", detection_id, abs_path,
        )
        raise HTTPException(status_code=410, detail="snapshot file missing on disk")
    return FileResponse(abs_path, media_type="image/jpeg")


@router.post(
    "/detections/{detection_id}/confirm",
    response_model=ConfirmDetectionResult,
)
async def confirm_detection(
    payload: ConfirmDetectionRequest,
    detection_id: uuid.UUID = PathParam(...),
) -> ConfirmDetectionResult:
    """Confirm an unknown detection: re-embed from disk, attach to a person.

    Body must specify exactly one of `person_id` or `name`. With `name`,
    a new person is created (or an existing one with that name is reused).
    The snapshot JPEG is read from disk, run back through detect+embed,
    and the highest-quality face is persisted to that person's gallery
    via `_persist_enrollment` (file → Qdrant → DB). The detection row is
    then updated: person_id set, review_state='confirmed',
    embedding_contributed=true on success.

    No new column is added to face_detections — re-embedding from the saved
    snapshot keeps the schema unchanged at the cost of ~100ms per confirm.
    """
    if (payload.person_id is None) == (payload.name is None):
        raise HTTPException(
            status_code=400,
            detail="exactly one of person_id or name is required",
        )

    detection = await db.get_detection(detection_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")

    # Resolve target person — by id, or get-or-create by name.
    if payload.person_id is not None:
        person = await db.get_person(payload.person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person not found")
        target_id: uuid.UUID = payload.person_id
        target_name: str = person["name"]
    else:
        try:
            row = await db.create_person(
                name=payload.name, access_level="unknown", notes=None,
            )
            target_id = row["id"]
            target_name = row["name"]
        except asyncpg.UniqueViolationError:
            # Match by exact name; the caller asked us to use that label.
            existing = await _find_person_by_name(payload.name)
            if existing is None:
                raise HTTPException(
                    status_code=500,
                    detail="name conflict but person could not be resolved",
                )
            target_id = existing["id"]
            target_name = existing["name"]

    # Re-embed from the saved snapshot. Re-running detect+embed (rather than
    # re-using the pipeline's mean-of-top-K embedding) keeps the schema
    # change-free and tolerates the case where the original event scored a
    # face but didn't store it as a face_image.
    abs_snapshot = _resolve_snapshot_path(detection["snapshot_path"])
    if not abs_snapshot.exists():
        raise HTTPException(status_code=410, detail="snapshot file missing on disk")

    img = cv2.imread(str(abs_snapshot))
    if img is None:
        raise HTTPException(
            status_code=422, detail="snapshot could not be decoded for re-embedding",
        )

    best_face, best_quality, faces_detected = await asyncio.to_thread(
        _best_face_in_frame, img,
    )

    embedding_contributed = False
    if best_face is not None:
        await _persist_enrollment(
            person_id=target_id,
            frame=img,
            embedding=best_face.normed_embedding,
            quality_score=best_quality,
            is_primary=False,
            source="detection_confirmed",
        )
        embedding_contributed = True
    else:
        logger.warning(
            "detection %s: re-embed found no face above QUALITY_FLOOR; "
            "marking confirmed but skipping embedding contribution",
            detection_id,
        )

    updated = await db.confirm_detection(
        detection_id=detection_id,
        person_id=target_id,
        embedding_contributed=embedding_contributed,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="detection not found")

    return ConfirmDetectionResult(
        detection_id=detection_id,
        person_id=target_id,
        person_name=target_name,
        embedding_contributed=embedding_contributed,
        quality_score=best_quality if best_face is not None else None,
        faces_detected=faces_detected,
    )


@router.post("/detections/bulk-delete", response_model=BulkDeleteResult)
async def bulk_delete_unknowns(payload: BulkDeleteUnknownsRequest) -> BulkDeleteResult:
    """Mass-delete unknown detections (file + DB row) by scope.

    Mirrors the retention sweeper's list → unlink → batch-delete pattern
    (`app/retention.py`): file unlinks are best-effort and counted; the DB
    DELETE is the source of truth. Caller surfaces the totals.
    """
    rows = await db.list_unknown_detection_paths(
        only_rejected=(payload.scope == "rejected"),
    )
    if not rows:
        return BulkDeleteResult(rows_deleted=0, files_unlinked=0, scope=payload.scope)

    files_unlinked = 0
    ids: list[uuid.UUID] = []
    for row in rows:
        ids.append(row["id"])
        abs_path = _resolve_snapshot_path(row["snapshot_path"])
        try:
            if abs_path.exists():
                abs_path.unlink()
                files_unlinked += 1
        except Exception as e:
            logger.warning(
                "bulk-delete: failed to unlink %s for detection %s: %s",
                abs_path, row["id"], e,
            )

    rows_deleted = await db.delete_detections_by_ids(ids)
    logger.info(
        "bulk-delete scope=%s: %d rows deleted, %d files unlinked",
        payload.scope, rows_deleted, files_unlinked,
    )
    return BulkDeleteResult(
        rows_deleted=rows_deleted,
        files_unlinked=files_unlinked,
        scope=payload.scope,
    )


@router.post("/detections/{detection_id}/reject", response_model=DetectionOut)
async def reject_detection(
    detection_id: uuid.UUID = PathParam(...),
) -> DetectionOut:
    """Mark an unknown detection rejected so it stops appearing in the queue.

    Doesn't touch person_id or the snapshot file — retention sweeps still
    age it out per RETENTION_UNKNOWN_DAYS.
    """
    row = await db.reject_detection(detection_id)
    if row is None:
        raise HTTPException(status_code=404, detail="detection not found")
    enriched = await db.get_detection(detection_id)
    return DetectionOut(**(enriched or row))


def _best_face_in_frame(img: np.ndarray):
    """Single-frame variant of pipeline._detect_and_score_sync.

    Returns (best_face, best_quality, faces_detected). best_face is None
    when no face cleared QUALITY_FLOOR.
    """
    faces = embedder.detect_and_embed(img)
    best_face = None
    best_quality = -1.0
    for face in faces:
        q = score_face(img, face)
        if q < config.QUALITY_FLOOR:
            continue
        if q > best_quality:
            best_quality = q
            best_face = face
    return best_face, best_quality, len(faces)


def _resolve_snapshot_path(snapshot_path: str) -> Path:
    """Detection snapshots are stored relative to SNAPSHOT_DIR (step 4),
    but tolerate absolute paths to stay forward-compatible.
    """
    p = Path(snapshot_path)
    if p.is_absolute():
        return p
    return Path(config.SNAPSHOT_DIR) / p


async def _find_person_by_name(name: str) -> Optional[dict]:
    """Tiny helper for the confirm-by-name race path. Avoids adding a public
    DB method just for this one fallback.
    """
    assert db.pool is not None
    async with db.pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name FROM people WHERE name = $1", name,
        )
    return dict(row) if row else None
