"""Detection-side endpoints.

Step 4 surfaces a single endpoint, `POST /api/trigger`, that runs the full
pipeline once on demand for a given HA camera entity. MQTT-driven triggers
land in step 5; the history/review endpoints land in step 7. Both reuse
`pipeline.process_event`.
"""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

import pipeline
from ha_snapshot import HASnapshotError
from models import PipelineResult


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
