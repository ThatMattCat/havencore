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
