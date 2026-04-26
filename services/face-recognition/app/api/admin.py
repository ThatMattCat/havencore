"""Admin / operator endpoints.

Trigger maintenance jobs out-of-band:
  - retention sweep (commit E) — early eviction, useful right after lowering
    a retention env var
  - rebuild-embeddings (this commit) — full re-embed of every face_images
    row from disk; recovers from drift after a model swap, an interrupted
    enrollment, or any divergence between Postgres and Qdrant
"""

import asyncio
import logging
import time
import uuid as _uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import APIRouter, HTTPException

import config
import pipeline
import retention
from db import db
from embedder import embedder
from face_qdrant import vector_store
from quality import score_face


logger = logging.getLogger("face-recognition.api.admin")

router = APIRouter(prefix="/api/admin", tags=["admin"])


SNAPSHOT_DIR = Path(config.SNAPSHOT_DIR)
# In-memory job registry. One process, one rebuild at a time is the assumed
# usage; the lock enforces it. If a deeper queue ever shows up, swap this
# for a Postgres-backed job table.
_jobs: Dict[str, Dict[str, Any]] = {}
_rebuild_lock = asyncio.Lock()
_rescan_lock = asyncio.Lock()
RESCAN_BATCH = 200


def _resolve(path: str) -> Path:
    """Same resolver as api/people.py — tolerate absolute + relative paths."""
    p = Path(path)
    if p.is_absolute():
        return p
    return SNAPSHOT_DIR / p


@router.post("/retention/sweep")
async def trigger_retention_sweep():
    """Run a single retention sweep right now and return the result."""
    result = await retention.run_once()
    retention.sweeper.last_result = result
    return result


@router.post("/rebuild-embeddings")
async def start_rebuild_embeddings():
    """Kick off a full re-embed of every face_images row.

    Use cases:
      - A model bump changed embedding shape or distribution; the Qdrant
        index needs to match the model now in use.
      - Enrollment failed mid-way and orphan Qdrant points were left.
      - Routine sanity check that on-disk JPEGs still embed the way the
        DB thinks they do.

    Returns immediately with a job_id. Poll GET /api/admin/jobs/{job_id}
    for status. Only one rebuild can run at a time per process; a second
    request while one is in flight returns 409.
    """
    if _rebuild_lock.locked():
        # Surface the existing job so the caller can poll it instead.
        in_flight = next(
            (jid for jid, j in _jobs.items() if j["status"] == "running"), None,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "a rebuild is already running",
                "job_id": in_flight,
            },
        )

    job_id = str(_uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "rebuild_embeddings",
        "status": "running",
        "phase": "pending",
        "started_at": time.time(),
        "finished_at": None,
        "elapsed_ms": 0,
        "totals": {
            "images": 0,
            "re_embedded": 0,
            "missing_files": 0,
            "no_face": 0,
            "embed_failed": 0,
            "orphan_points_removed": 0,
        },
        "errors": [],
    }
    asyncio.create_task(_run_rebuild(job_id))
    return {"job_id": job_id, "status": "running"}


@router.post("/rescan-unknowns")
async def start_rescan_unknowns():
    """Re-match every unknown detection against the current Qdrant index.

    Useful after confirming several unknowns as a known person — the new
    embeddings raise the chance that older near-misses now clear
    MATCH_THRESHOLD. High-quality matches that also clear the live
    pipeline's IMPROVEMENT_QUALITY_FLOOR / IMPROVEMENT_THRESHOLD will
    additionally feed the gallery (FIFO-bounded by MAX_EMBEDDINGS_PER_PERSON).

    Returns immediately with a job_id. Poll GET /api/admin/jobs/{job_id}.
    Single-flight per process; second concurrent request returns 409.
    """
    if _rescan_lock.locked():
        in_flight = next(
            (jid for jid, j in _jobs.items()
             if j["status"] == "running" and j.get("type") == "rescan_unknowns"),
            None,
        )
        raise HTTPException(
            status_code=409,
            detail={
                "error": "a rescan is already running",
                "job_id": in_flight,
            },
        )

    job_id = str(_uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "rescan_unknowns",
        "status": "running",
        "phase": "pending",
        "started_at": time.time(),
        "finished_at": None,
        "elapsed_ms": 0,
        "totals": {
            "examined": 0,
            "matched": 0,
            "no_match": 0,
            "contributed": 0,
            "skipped_missing_snapshot": 0,
            "errors": 0,
        },
        "errors": [],
    }
    asyncio.create_task(_run_rescan(job_id))
    return {"job_id": job_id, "status": "running"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] == "running":
        job["elapsed_ms"] = int((time.time() - job["started_at"]) * 1000)
    return job


@router.get("/jobs")
async def list_jobs(limit: int = 20):
    """Most-recent jobs first; capped to keep the response small."""
    items = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
    return {"jobs": items[:limit]}


async def _run_rebuild(job_id: str) -> None:
    """The actual worker. All errors are captured into the job record so
    the caller can see them via the poll endpoint instead of digging in
    server logs.
    """
    job = _jobs[job_id]
    async with _rebuild_lock:
        try:
            job["phase"] = "loading"
            rows = await db.list_all_face_images()
            job["totals"]["images"] = len(rows)
            db_image_ids = {str(r["id"]) for r in rows}

            job["phase"] = "embedding"
            for row in rows:
                fi_id: _uuid.UUID = row["id"]
                qd_id: _uuid.UUID = row["qdrant_point_id"]
                person_id: _uuid.UUID = row["person_id"]
                rel_or_abs = row["path"]
                source = row["source"]

                abs_path = _resolve(rel_or_abs)
                if not abs_path.exists():
                    job["totals"]["missing_files"] += 1
                    job["errors"].append({
                        "face_image_id": str(fi_id),
                        "reason": "file_missing",
                        "path": str(abs_path),
                    })
                    continue

                try:
                    img = cv2.imread(str(abs_path))
                    if img is None:
                        raise ValueError("cv2.imread returned None")
                    best, q = await asyncio.to_thread(_best_face_in_frame, img)
                except Exception as e:
                    job["totals"]["embed_failed"] += 1
                    job["errors"].append({
                        "face_image_id": str(fi_id),
                        "reason": "decode_or_inference_failed",
                        "detail": str(e)[:200],
                    })
                    continue

                if best is None:
                    job["totals"]["no_face"] += 1
                    job["errors"].append({
                        "face_image_id": str(fi_id),
                        "reason": "no_face_above_quality_floor",
                    })
                    continue

                try:
                    # Same point id → in-place replace (Qdrant upsert semantics).
                    vector_store.upsert_point(
                        point_id=str(qd_id),
                        vector=best.normed_embedding.tolist(),
                        payload={
                            "person_id": str(person_id),
                            "face_image_id": str(fi_id),
                            "source": source,
                        },
                    )
                    await db.update_face_image_quality(fi_id, float(q))
                    job["totals"]["re_embedded"] += 1
                except Exception as e:
                    job["totals"]["embed_failed"] += 1
                    job["errors"].append({
                        "face_image_id": str(fi_id),
                        "reason": "qdrant_or_db_write_failed",
                        "detail": str(e)[:200],
                    })

            job["phase"] = "cleaning_orphans"
            try:
                pairs = await asyncio.to_thread(vector_store.scroll_all_payload)
                orphan_ids: List[str] = []
                for point_id, payload in pairs:
                    fi = payload.get("face_image_id")
                    if fi and fi not in db_image_ids:
                        orphan_ids.append(point_id)
                if orphan_ids:
                    removed = await asyncio.to_thread(
                        vector_store.delete_points, orphan_ids,
                    )
                    job["totals"]["orphan_points_removed"] = int(removed)
            except Exception as e:
                job["errors"].append({
                    "face_image_id": None,
                    "reason": "orphan_scan_failed",
                    "detail": str(e)[:200],
                })

            job["status"] = "done"
            job["phase"] = "done"
        except Exception as e:
            logger.exception("Rebuild job %s crashed: %s", job_id, e)
            job["status"] = "error"
            job["phase"] = "error"
            job["errors"].append({
                "face_image_id": None,
                "reason": "fatal",
                "detail": str(e)[:500],
            })
        finally:
            job["finished_at"] = time.time()
            job["elapsed_ms"] = int((job["finished_at"] - job["started_at"]) * 1000)


async def _run_rescan(job_id: str) -> None:
    """Walk all unknown detections, re-embed from disk, query Qdrant, and
    confirm matches above MATCH_THRESHOLD. High-quality matches also
    contribute to the gallery via the live pipeline's improvement gates.

    Pagination strategy: we re-query `unknowns_only=True` each iteration.
    Matched rows drop out of that filter (review_state flips to 'confirmed'),
    so the queue naturally drains. If a full batch of `RESCAN_BATCH` rows
    yields zero matches we've made one full pass with nothing more to do,
    so we exit.
    """
    job = _jobs[job_id]
    async with _rescan_lock:
        try:
            job["phase"] = "scanning"
            while True:
                rows = await db.list_detections(
                    camera=None,
                    since_seconds_ago=None,
                    person_id=None,
                    limit=RESCAN_BATCH,
                    review_state=None,
                    unknowns_only=True,
                )
                if not rows:
                    break

                progress = False
                for row in rows:
                    job["totals"]["examined"] += 1
                    detection_id = row["id"]
                    abs_path = _resolve(row["snapshot_path"])
                    if not abs_path.exists():
                        job["totals"]["skipped_missing_snapshot"] += 1
                        continue

                    try:
                        img = await asyncio.to_thread(cv2.imread, str(abs_path))
                        if img is None:
                            raise ValueError("cv2.imread returned None")
                        best, q = await asyncio.to_thread(_best_face_in_frame, img)
                    except Exception as e:
                        job["totals"]["errors"] += 1
                        job["errors"].append({
                            "detection_id": str(detection_id),
                            "reason": "decode_or_inference_failed",
                            "detail": str(e)[:200],
                        })
                        continue

                    if best is None:
                        job["totals"]["no_match"] += 1
                        continue

                    try:
                        hits = await asyncio.to_thread(
                            vector_store.query, best.normed_embedding, 3,
                        )
                    except Exception as e:
                        job["totals"]["errors"] += 1
                        job["errors"].append({
                            "detection_id": str(detection_id),
                            "reason": "qdrant_query_failed",
                            "detail": str(e)[:200],
                        })
                        continue

                    if not hits or hits[0].score <= config.MATCH_THRESHOLD:
                        job["totals"]["no_match"] += 1
                        continue

                    pid_raw = (hits[0].payload or {}).get("person_id")
                    try:
                        pid = _uuid.UUID(pid_raw)
                    except (TypeError, ValueError):
                        job["totals"]["errors"] += 1
                        job["errors"].append({
                            "detection_id": str(detection_id),
                            "reason": "invalid_person_id_in_payload",
                            "detail": str(pid_raw)[:80],
                        })
                        continue

                    person = await db.get_person(pid)
                    if person is None:
                        # Stale Qdrant point pointing at a deleted person.
                        job["totals"]["no_match"] += 1
                        continue

                    confidence = float(hits[0].score)
                    await db.confirm_detection(
                        detection_id=detection_id,
                        person_id=pid,
                        embedding_contributed=False,
                    )
                    job["totals"]["matched"] += 1
                    progress = True

                    try:
                        contributed = await pipeline.contribute_embedding_for_detection(
                            person_id=pid,
                            detection_id=detection_id,
                            face=best,
                            frame=img,
                            quality=float(q),
                            confidence=confidence,
                        )
                    except Exception as e:
                        logger.exception(
                            "rescan: contribute_embedding raised for detection %s",
                            detection_id,
                        )
                        contributed = False
                        job["totals"]["errors"] += 1
                        job["errors"].append({
                            "detection_id": str(detection_id),
                            "reason": "contribute_failed",
                            "detail": str(e)[:200],
                        })

                    if contributed:
                        job["totals"]["contributed"] += 1
                        # Flip the row's flag so the UI shows the contribution.
                        await db.confirm_detection(
                            detection_id=detection_id,
                            person_id=pid,
                            embedding_contributed=True,
                        )

                if not progress:
                    # Full batch examined, nothing matched — done.
                    break

            job["status"] = "done"
            job["phase"] = "done"
        except Exception as e:
            logger.exception("Rescan job %s crashed: %s", job_id, e)
            job["status"] = "error"
            job["phase"] = "error"
            job["errors"].append({
                "detection_id": None,
                "reason": "fatal",
                "detail": str(e)[:500],
            })
        finally:
            job["finished_at"] = time.time()
            job["elapsed_ms"] = int((job["finished_at"] - job["started_at"]) * 1000)


def _best_face_in_frame(img):
    """Single-frame detect+score (mirrors the helper in api/detections.py).

    Returns (best_face, quality) or (None, -1.0) if nothing clears the
    QUALITY_FLOOR. Synchronous so the caller can offload it via asyncio.
    """
    faces = embedder.detect_and_embed(img)
    best = None
    best_q = -1.0
    for f in faces:
        q = score_face(img, f)
        if q < config.QUALITY_FLOOR:
            continue
        if q > best_q:
            best_q = q
            best = f
    return best, best_q
