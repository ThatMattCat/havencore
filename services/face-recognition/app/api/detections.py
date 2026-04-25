"""Detection-side endpoints.

`POST /api/trigger` runs the full pipeline once on demand for a given HA
camera entity. `GET /api/detections` surfaces the persisted history with
filters used by the agent's `face_who_is_at` / `face_recent_visitors` tools
and (later) the SvelteKit `/people/detections` route.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

import pipeline
from db import db
from ha_snapshot import HASnapshotError
from models import DetectionOut, PipelineResult


logger = logging.getLogger("face-recognition.api.detections")

router = APIRouter(prefix="/api", tags=["detections"])


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
    limit: int = Query(20, ge=1, le=200),
) -> list[DetectionOut]:
    rows = await db.list_detections(
        camera=camera,
        since_seconds_ago=since_seconds_ago,
        person_id=person_id,
        limit=limit,
    )
    return [DetectionOut(**r) for r in rows]
