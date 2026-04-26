"""Periodic retention sweeper for detection snapshots.

Runs as a single asyncio task started by the FastAPI lifespan. Each sweep:
  1. Asks the DB for detections older than the per-bucket retention window
     (unknowns vs identified — typically kept different lengths).
  2. Unlinks each snapshot file (relative-under-SNAPSHOT_DIR, with a
     defensive absolute-path branch for any pre-step-8 rows still around).
  3. Deletes the rows in a single batch so the UI's review queue and
     detection timeline don't surface "snapshot missing" rows.

Auto-improvement face_images are intentionally NOT touched here — those
are part of the gallery and only get evicted by the FIFO cap in the
pipeline. Enrollment images are similarly persistent.

Idempotent: a sweep with nothing to do is a single SQL fetch.
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from db import db


logger = logging.getLogger("face-recognition.retention")


SNAPSHOT_DIR = Path(config.SNAPSHOT_DIR)
# Cap per-sweep work so a long backlog doesn't monopolize the event loop.
# Subsequent sweeps continue draining on the configured cadence.
SWEEP_BATCH_LIMIT = 1000


def _resolve(path: str) -> Path:
    """Match the resolver in api/people.py — handle both relative + absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return SNAPSHOT_DIR / p


async def run_once() -> dict:
    """Execute a single sweep. Safe to call from outside the periodic loop."""
    rows = await db.list_aged_detections(
        unknown_max_age_days=config.RETENTION_UNKNOWN_DAYS,
        known_max_age_days=config.RETENTION_KNOWN_DAYS,
        limit=SWEEP_BATCH_LIMIT,
    )
    if not rows:
        return {
            "swept_at": datetime.now(timezone.utc).isoformat(),
            "rows_examined": 0,
            "files_unlinked": 0,
            "rows_deleted": 0,
        }

    files_unlinked = 0
    for row in rows:
        try:
            path = _resolve(row["snapshot_path"])
            if path.exists():
                path.unlink()
                files_unlinked += 1
        except Exception as e:
            logger.warning(
                "Failed to unlink snapshot for detection %s (%s): %s",
                row["id"], row["snapshot_path"], e,
            )

    deleted = await db.delete_detections_by_ids([r["id"] for r in rows])
    logger.info(
        "Retention sweep: %d rows examined, %d files unlinked, %d rows deleted",
        len(rows), files_unlinked, deleted,
    )
    return {
        "swept_at": datetime.now(timezone.utc).isoformat(),
        "rows_examined": len(rows),
        "files_unlinked": files_unlinked,
        "rows_deleted": deleted,
    }


class RetentionSweeper:
    """Owns the periodic asyncio task. One instance per service process."""

    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self.last_result: Optional[dict] = None
        self._running = False

    async def start(self) -> None:
        if config.RETENTION_SWEEP_ON_STARTUP:
            try:
                self.last_result = await run_once()
            except Exception as e:
                logger.exception("Startup retention sweep failed: %s", e)

        interval_min = config.RETENTION_SWEEP_INTERVAL_MIN
        if interval_min <= 0:
            logger.warning(
                "FACE_REC_RETENTION_SWEEP_INTERVAL_MIN=0 — periodic retention disabled",
            )
            return

        self._running = True
        self._task = asyncio.create_task(self._loop(interval_min * 60))
        logger.info("Retention sweeper started (interval=%dm)", interval_min)

    async def _loop(self, interval_sec: int) -> None:
        try:
            while self._running:
                await asyncio.sleep(interval_sec)
                if not self._running:
                    break
                try:
                    self.last_result = await run_once()
                except Exception as e:
                    logger.exception("Retention sweep failed: %s", e)
        except asyncio.CancelledError:
            # Normal shutdown path — don't log as an error.
            pass

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Retention sweeper stopped")

    def info(self) -> dict:
        return {
            "enabled": config.RETENTION_SWEEP_INTERVAL_MIN > 0,
            "interval_min": config.RETENTION_SWEEP_INTERVAL_MIN,
            "unknown_days": config.RETENTION_UNKNOWN_DAYS,
            "known_days": config.RETENTION_KNOWN_DAYS,
            "last_sweep": self.last_result,
        }


sweeper = RetentionSweeper()
