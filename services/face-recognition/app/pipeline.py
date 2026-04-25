"""Per-event detection pipeline.

  burst capture → detect+embed each frame → quality-score each face
  → drop anything below QUALITY_FLOOR → keep top-K by quality across all
  frames → mean of L2-normalized embeddings (re-normalized) → Qdrant knn
  → identified vs unknown decision → persist (snapshot + face_detections row)
  → continuous-improvement (3 gates + FIFO eviction) when identified.

Decoupled from FastAPI and MQTT so the same `process_event` is used by both
the manual /api/trigger endpoint (step 4) and the MQTT bridge (step 5).

v1 limitation (carried over from the design doc): if multiple distinct
people appear in a single event, mean-embedding the top-K faces produces a
nonsense identity. Disambiguating is explicitly v2 — see
"Out of scope for v1" in the plan.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional


# (camera, mode) → coroutine. Pipeline calls this at lifecycle transitions;
# the MQTT bridge supplies one that publishes to haven/face/status. The HTTP
# trigger endpoint passes None (status is only meaningful for async/event-
# driven callers). Never raises into the pipeline — emit failures are logged.
StatusEmitter = Callable[[str, str], Awaitable[None]]

import cv2
import numpy as np

import config
import ha_snapshot
from db import db
from embedder import embedder
from face_qdrant import vector_store
from models import PipelineResult
from quality import score_face


logger = logging.getLogger("face-recognition.pipeline")


SNAPSHOT_DIR = Path(config.SNAPSHOT_DIR)
AUTO_IMPROVE_DIR = SNAPSHOT_DIR / "auto"
TOP_K_FACES = 3


@dataclass
class _Candidate:
    """One scored face surviving the quality floor, with its source frame index."""
    frame_idx: int
    face: object  # insightface Face: .bbox, .kps, .det_score, .normed_embedding
    quality: float


def _select_top_k(candidates: list[_Candidate], k: int) -> list[_Candidate]:
    return sorted(candidates, key=lambda c: c.quality, reverse=True)[:k]


def _mean_normalized_embedding(candidates: list[_Candidate]) -> np.ndarray:
    stacked = np.stack([c.face.normed_embedding for c in candidates], axis=0)
    mean = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-8:
        # Degenerate (vectors canceled) — fall back to the single best face's embedding.
        return candidates[0].face.normed_embedding
    return mean / norm


def _save_snapshot(frame: np.ndarray, captured_at: datetime, event_id: uuid.UUID) -> str:
    """Write the best frame to disk; return path RELATIVE to SNAPSHOT_DIR.

    Date components are taken from the UTC representation of `captured_at`
    so triggers from different timezones (e.g. HA's local TZ) all file into
    the same date dir as same-instant UTC-side detections. The DB row's
    captured_at is unaffected — TIMESTAMPTZ normalizes internally.
    """
    captured_at_utc = (
        captured_at.astimezone(timezone.utc) if captured_at.tzinfo
        else captured_at.replace(tzinfo=timezone.utc)
    )
    rel_dir = Path(captured_at_utc.strftime("%Y/%m/%d"))
    abs_dir = SNAPSHOT_DIR / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    rel_path = rel_dir / f"{event_id}.jpg"
    abs_path = SNAPSHOT_DIR / rel_path
    if not cv2.imwrite(str(abs_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write snapshot to {abs_path}")
    return str(rel_path)


def _save_auto_improvement_image(
    frame: np.ndarray, person_id: uuid.UUID, face_image_id: uuid.UUID
) -> Path:
    """Save the source frame as an auto-improvement face_image. Absolute path.

    Stored as full frame (matches enrollment convention) so the step-8
    re-embedding admin endpoint can re-run detection without depending on
    pre-cropped data.
    """
    person_dir = AUTO_IMPROVE_DIR / str(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    abs_path = person_dir / f"{face_image_id}.jpg"
    if not cv2.imwrite(str(abs_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to write auto-improvement image to {abs_path}")
    return abs_path


async def _maybe_contribute_embedding(
    *,
    person_id: uuid.UUID,
    detection_id: uuid.UUID,
    best_candidate: _Candidate,
    best_frame: np.ndarray,
    best_quality: float,
    confidence: float,
) -> bool:
    """Run the three continuous-improvement gates; return True if a new
    embedding was added for this person.

    File → Qdrant → DB ordering, same as enrollment. FIFO cap eviction
    happens BEFORE the new write so we don't briefly exceed the cap.
    """
    if best_quality < config.IMPROVEMENT_QUALITY_FLOOR:
        return False
    if confidence < config.IMPROVEMENT_THRESHOLD:
        return False

    current = await db.count_face_images_for_person(person_id)
    if current >= config.MAX_EMBEDDINGS_PER_PERSON:
        evicted = await db.evict_oldest_non_primary_face_image(person_id)
        if evicted is None:
            logger.warning(
                "person %s at embedding cap with no non-primary images to evict; "
                "skipping continuous improvement",
                person_id,
            )
            return False
        # Best-effort cleanup; orphans get caught by step-8 reconcile pass.
        try:
            Path(evicted["path"]).unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to unlink evicted file %s: %s", evicted["path"], e)
        try:
            vector_store.delete_point(str(evicted["qdrant_point_id"]))
        except Exception as e:
            logger.warning(
                "Failed to delete evicted Qdrant point %s: %s",
                evicted["qdrant_point_id"], e,
            )

    face_image_id = uuid.uuid4()
    qdrant_point_id = uuid.uuid4()

    abs_path = _save_auto_improvement_image(best_frame, person_id, face_image_id)

    try:
        vector_store.upsert_point(
            point_id=str(qdrant_point_id),
            vector=best_candidate.face.normed_embedding.tolist(),
            payload={
                "person_id": str(person_id),
                "face_image_id": str(face_image_id),
                "source": "detection_auto",
            },
        )
    except Exception as e:
        abs_path.unlink(missing_ok=True)
        logger.exception("Qdrant upsert failed during continuous improvement: %s", e)
        return False

    try:
        await db.insert_face_image_for_detection(
            face_image_id=face_image_id,
            person_id=person_id,
            path=str(abs_path),
            qdrant_point_id=qdrant_point_id,
            source="detection_auto",
            quality_score=best_quality,
            detection_id=detection_id,
        )
    except Exception as e:
        abs_path.unlink(missing_ok=True)
        logger.exception(
            "DB insert failed after Qdrant upsert during continuous improvement "
            "(orphan point %s): %s", qdrant_point_id, e,
        )
        return False

    logger.info(
        "Contributed new embedding for person %s (quality=%.2f, confidence=%.2f)",
        person_id, best_quality, confidence,
    )
    return True


def _detect_and_score_sync(frames: list[np.ndarray]) -> list[_Candidate]:
    """Detect + quality-score every face in every frame. Filters by floor.

    Wrapped so the caller can offload it to a thread (insightface inference
    is blocking C/CUDA work).
    """
    candidates: list[_Candidate] = []
    for idx, frame in enumerate(frames):
        faces = embedder.detect_and_embed(frame)
        for face in faces:
            q = score_face(frame, face)
            if q >= config.QUALITY_FLOOR:
                candidates.append(_Candidate(frame_idx=idx, face=face, quality=q))
    return candidates


async def process_event(
    camera: str,
    event_id: uuid.UUID,
    captured_at: datetime,
    *,
    status_emitter: Optional[StatusEmitter] = None,
) -> PipelineResult:
    """Run one trigger end-to-end. Capture failures raise; expected outcomes
    (no_frames, no_face, unknown, identified) all return a PipelineResult.

    `status_emitter`, if supplied, is called at lifecycle transitions:
    capturing → matching → idle. The terminal `idle` is fired in a
    `try/finally` so it always runs, even if HA capture or downstream IO
    raises. Emit failures never propagate.
    """

    async def _emit(mode: str) -> None:
        if status_emitter is None:
            return
        try:
            await status_emitter(camera, mode)
        except Exception as e:
            logger.warning("status emit (%s) failed: %s", mode, e)

    await _emit("capturing")
    try:
        frames = await ha_snapshot.capture_burst(
            camera=camera,
            n=config.BURST_FRAMES,
            interval_ms=config.BURST_INTERVAL_MS,
        )

        if not frames:
            logger.info("event %s: no frames captured for camera %s", event_id, camera)
            return PipelineResult(
                outcome="no_frames",
                event_id=event_id,
                captured_at=captured_at,
                camera=camera,
                frames_processed=0,
                faces_kept=0,
            )

        await _emit("matching")
        # insightface inference is CPU-blocking from asyncio's perspective;
        # offload so the FastAPI event loop stays responsive.
        candidates = await asyncio.to_thread(_detect_and_score_sync, frames)

        if not candidates:
            logger.info(
                "event %s: %d frames, no face cleared quality floor",
                event_id, len(frames),
            )
            return PipelineResult(
                outcome="no_face",
                event_id=event_id,
                captured_at=captured_at,
                camera=camera,
                frames_processed=len(frames),
                faces_kept=0,
            )

        top = _select_top_k(candidates, TOP_K_FACES)
        best = top[0]
        best_frame = frames[best.frame_idx]
        best_quality = best.quality
        mean_emb = _mean_normalized_embedding(top)

        # Qdrant cosine score == cosine similarity for unit vectors.
        # Plan threshold "cosine distance < MATCH_THRESHOLD" is equivalent at the
        # 0.5 boundary; we treat MATCH_THRESHOLD as a similarity floor to keep
        # the comparison direction obvious.
        hits = vector_store.query(mean_emb, limit=3)
        person_id: Optional[uuid.UUID] = None
        person_name: Optional[str] = None
        confidence: Optional[float] = None

        if hits:
            top_hit = hits[0]
            confidence = float(top_hit.score)
            if confidence > config.MATCH_THRESHOLD:
                payload = top_hit.payload or {}
                raw_pid = payload.get("person_id")
                if raw_pid:
                    try:
                        person_id = uuid.UUID(raw_pid)
                    except ValueError:
                        logger.warning(
                            "Qdrant payload has invalid person_id=%r on point %s",
                            raw_pid, top_hit.id,
                        )
                        person_id = None
                if person_id is not None:
                    person_name = await db.get_person_name(person_id)
                    if person_name is None:
                        # Qdrant has the point but Postgres lost the person —
                        # treat as unknown rather than returning a stale id.
                        logger.warning(
                            "Qdrant point %s references unknown person_id %s; "
                            "treating event as unknown",
                            top_hit.id, person_id,
                        )
                        person_id = None

        snapshot_rel_path = _save_snapshot(best_frame, captured_at, event_id)

        detection = await db.insert_face_detection(
            event_id=event_id,
            camera=camera,
            captured_at=captured_at,
            person_id=person_id,
            confidence=confidence,
            quality_score=best_quality,
            snapshot_path=snapshot_rel_path,
        )
        detection_id: uuid.UUID = detection["id"]

        embedding_contributed = False
        if person_id is not None and confidence is not None:
            embedding_contributed = await _maybe_contribute_embedding(
                person_id=person_id,
                detection_id=detection_id,
                best_candidate=best,
                best_frame=best_frame,
                best_quality=best_quality,
                confidence=confidence,
            )

        outcome = "identified" if person_id is not None else "unknown"
        logger.info(
            "event %s on %s: outcome=%s person=%s conf=%s quality=%.2f frames=%d kept=%d contributed=%s",
            event_id, camera, outcome, person_name,
            f"{confidence:.3f}" if confidence is not None else None,
            best_quality, len(frames), len(top), embedding_contributed,
        )

        return PipelineResult(
            outcome=outcome,
            event_id=event_id,
            captured_at=captured_at,
            camera=camera,
            detection_id=detection_id,
            person_id=person_id,
            person_name=person_name,
            confidence=confidence,
            quality_score=best_quality,
            snapshot_path=snapshot_rel_path,
            frames_processed=len(frames),
            faces_kept=len(top),
            embedding_contributed=embedding_contributed,
        )
    finally:
        await _emit("idle")
