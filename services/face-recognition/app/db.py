"""Postgres access for the face-recognition service.

Owns an asyncpg pool and runs idempotent migrations on startup so existing
HavenCore deployments pick up the face tables without a DB rebuild. Fresh
deployments get the same schema via services/postgres/init.sql; both paths
must stay in sync.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import asyncpg


logger = logging.getLogger("face-recognition.db")


# gen_random_uuid() lives in pgcrypto. init.sql enables it too; we re-issue
# here so databases that predate the face tables get the extension.
MIGRATION_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS people (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  access_level TEXT NOT NULL DEFAULT 'unknown',
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id UUID NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  qdrant_point_id UUID NOT NULL,
  is_primary BOOLEAN DEFAULT false,
  source TEXT NOT NULL,
  quality_score REAL,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_face_images_person ON face_images(person_id);

CREATE TABLE IF NOT EXISTS face_detections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id UUID NOT NULL,
  camera TEXT NOT NULL,
  captured_at TIMESTAMPTZ DEFAULT now(),
  person_id UUID REFERENCES people(id) ON DELETE SET NULL,
  confidence REAL,
  quality_score REAL,
  snapshot_path TEXT NOT NULL,
  review_state TEXT NOT NULL DEFAULT 'auto',
  embedding_contributed BOOLEAN DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_face_detections_captured ON face_detections(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_face_detections_unknown
  ON face_detections(review_state) WHERE person_id IS NULL;
"""


class Database:
    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None
        self._config = {
            "host": os.getenv("POSTGRES_HOST", "postgres"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "havencore"),
            "user": os.getenv("POSTGRES_USER", "havencore"),
            "password": os.getenv("POSTGRES_PASSWORD", "havencore_password"),
        }

    async def initialize(self, max_retries: int = 10, retry_delay: int = 5) -> None:
        for attempt in range(1, max_retries + 1):
            try:
                self.pool = await asyncpg.create_pool(
                    **self._config,
                    min_size=1,
                    max_size=5,
                    command_timeout=30,
                )
                logger.info("asyncpg pool initialized (host=%s db=%s)",
                            self._config["host"], self._config["database"])
                return
            except Exception as e:
                logger.warning("Postgres connect attempt %d/%d failed: %s",
                               attempt, max_retries, e)
                if attempt == max_retries:
                    raise
                await asyncio.sleep(retry_delay)

    async def migrate(self) -> None:
        assert self.pool is not None, "call initialize() before migrate()"
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(MIGRATION_SQL)
        logger.info("Face-recognition migrations applied")

    async def health(self) -> bool:
        if self.pool is None:
            return False
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning("Postgres health check failed: %s", e)
            return False

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            logger.info("asyncpg pool closed")

    # --- people CRUD ---------------------------------------------------

    async def create_person(
        self, name: str, access_level: str, notes: Optional[str]
    ) -> dict[str, Any]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO people (name, access_level, notes)
                VALUES ($1, $2, $3)
                RETURNING id, name, access_level, notes, created_at, updated_at
                """,
                name, access_level, notes,
            )
        return dict(row) | {"image_count": 0}

    async def list_people(self) -> list[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT p.id, p.name, p.access_level, p.notes,
                       p.created_at, p.updated_at,
                       COUNT(fi.id) AS image_count
                FROM people p
                LEFT JOIN face_images fi ON fi.person_id = p.id
                GROUP BY p.id
                ORDER BY p.created_at DESC
                """
            )
        return [dict(r) for r in rows]

    async def get_person(self, person_id: UUID) -> Optional[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT p.id, p.name, p.access_level, p.notes,
                       p.created_at, p.updated_at,
                       COUNT(fi.id) AS image_count
                FROM people p
                LEFT JOIN face_images fi ON fi.person_id = p.id
                WHERE p.id = $1
                GROUP BY p.id
                """,
                person_id,
            )
        return dict(row) if row else None

    async def list_face_images(self, person_id: UUID) -> list[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, path, is_primary, source, quality_score, created_at
                FROM face_images
                WHERE person_id = $1
                ORDER BY is_primary DESC, created_at DESC
                """,
                person_id,
            )
        return [dict(r) for r in rows]

    async def insert_face_image(
        self,
        face_image_id: UUID,
        person_id: UUID,
        path: str,
        qdrant_point_id: UUID,
        source: str,
        quality_score: Optional[float],
        is_primary: bool,
    ) -> dict[str, Any]:
        """Insert one face_images row, optionally swapping primary in the same tx."""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                if is_primary:
                    await conn.execute(
                        "UPDATE face_images SET is_primary = false WHERE person_id = $1",
                        person_id,
                    )
                row = await conn.fetchrow(
                    """
                    INSERT INTO face_images
                        (id, person_id, path, qdrant_point_id, source,
                         quality_score, is_primary)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id, path, is_primary, source, quality_score, created_at
                    """,
                    face_image_id, person_id, path, qdrant_point_id, source,
                    quality_score, is_primary,
                )
        return dict(row)

    async def get_person_name(self, person_id: UUID) -> Optional[str]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT name FROM people WHERE id = $1", person_id
            )

    async def delete_person(self, person_id: UUID) -> Optional[list[dict[str, Any]]]:
        """Delete a person; return their face_images rows for cleanup.

        Returns the (path, qdrant_point_id) of every face_image so the caller
        can unlink files and remove Qdrant points. Returns None if the person
        doesn't exist. The face_images rows themselves are removed by the FK
        cascade. Detection rows keep their snapshot but lose the foreign key
        (ON DELETE SET NULL on face_detections.person_id).
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                images = await conn.fetch(
                    "SELECT path, qdrant_point_id FROM face_images WHERE person_id = $1",
                    person_id,
                )
                deleted = await conn.fetchval(
                    "DELETE FROM people WHERE id = $1 RETURNING id", person_id,
                )
        if deleted is None:
            return None
        return [dict(r) for r in images]

    async def get_face_image(
        self, person_id: UUID, face_image_id: UUID
    ) -> Optional[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, person_id, path, qdrant_point_id, is_primary,
                       source, quality_score, created_at
                FROM face_images
                WHERE id = $1 AND person_id = $2
                """,
                face_image_id, person_id,
            )
        return dict(row) if row else None

    async def get_face_image_by_id(
        self, face_image_id: UUID
    ) -> Optional[dict[str, Any]]:
        """Same as get_face_image but doesn't require knowing the person_id.

        Used by the bytes-streaming endpoint, which gets only an image id.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, person_id, path, qdrant_point_id, is_primary,
                       source, quality_score, created_at
                FROM face_images
                WHERE id = $1
                """,
                face_image_id,
            )
        return dict(row) if row else None

    async def delete_face_image(
        self, person_id: UUID, face_image_id: UUID
    ) -> Optional[dict[str, Any]]:
        """Delete one face_image row; return path + qdrant_point_id for cleanup.

        Returns None if the row doesn't belong to `person_id` (or doesn't
        exist). Refuses to delete a row marked is_primary so the gallery
        always has a primary as long as it has any images — the caller
        should set-primary on another image first.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                DELETE FROM face_images
                WHERE id = $1 AND person_id = $2 AND is_primary = false
                RETURNING id, path, qdrant_point_id
                """,
                face_image_id, person_id,
            )
        return dict(row) if row else None

    async def set_primary_face_image(
        self, person_id: UUID, face_image_id: UUID
    ) -> Optional[dict[str, Any]]:
        """Atomically swap the primary marker to `face_image_id`.

        Returns the new primary's row, or None if the image doesn't exist
        for this person.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                exists = await conn.fetchval(
                    "SELECT 1 FROM face_images WHERE id = $1 AND person_id = $2",
                    face_image_id, person_id,
                )
                if not exists:
                    return None
                await conn.execute(
                    "UPDATE face_images SET is_primary = false WHERE person_id = $1",
                    person_id,
                )
                row = await conn.fetchrow(
                    """
                    UPDATE face_images SET is_primary = true
                    WHERE id = $1 AND person_id = $2
                    RETURNING id, path, is_primary, source, quality_score, created_at
                    """,
                    face_image_id, person_id,
                )
        return dict(row) if row else None

    async def update_person(
        self,
        person_id: UUID,
        access_level: Optional[str],
        notes: Optional[str],
    ) -> Optional[dict[str, Any]]:
        """Partial update. Returns None if person doesn't exist.

        COALESCE preserves existing column values for omitted fields. Caller
        should prevalidate `access_level` against ACCESS_LEVEL_PATTERN.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE people
                SET access_level = COALESCE($2, access_level),
                    notes = COALESCE($3, notes),
                    updated_at = now()
                WHERE id = $1
                RETURNING id, name, access_level, notes, created_at, updated_at
                """,
                person_id, access_level, notes,
            )
        if row is None:
            return None
        async with self.pool.acquire() as conn:
            count = int(await conn.fetchval(
                "SELECT COUNT(*) FROM face_images WHERE person_id = $1", person_id,
            ))
        return dict(row) | {"image_count": count}

    # --- detections ----------------------------------------------------

    async def insert_face_detection(
        self,
        event_id: UUID,
        camera: str,
        captured_at: datetime,
        person_id: Optional[UUID],
        confidence: Optional[float],
        quality_score: float,
        snapshot_path: str,
    ) -> dict[str, Any]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO face_detections
                    (event_id, camera, captured_at, person_id, confidence,
                     quality_score, snapshot_path)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id, event_id, camera, captured_at, person_id,
                          confidence, quality_score, snapshot_path,
                          review_state, embedding_contributed
                """,
                event_id, camera, captured_at, person_id, confidence,
                quality_score, snapshot_path,
            )
        return dict(row)

    async def list_detections(
        self,
        camera: Optional[str],
        since_seconds_ago: Optional[int],
        person_id: Optional[UUID],
        limit: int,
        review_state: Optional[str] = None,
        unknowns_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Filtered detection history, newest first.

        `since_seconds_ago` is converted to an absolute timestamp here rather
        than passed through SQL so the query plan stays parameterized cleanly.
        LEFT JOIN people surfaces person_name (NULL for unknown rows).
        `unknowns_only=True` is shorthand for "person_id IS NULL AND
        review_state != 'rejected'" — the queue the /people/unknowns UI
        watches.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.id, d.event_id, d.camera, d.captured_at,
                       d.person_id, p.name AS person_name,
                       d.confidence, d.quality_score, d.snapshot_path,
                       d.review_state, d.embedding_contributed
                FROM face_detections d
                LEFT JOIN people p ON d.person_id = p.id
                WHERE ($1::text IS NULL OR d.camera = $1)
                  AND ($2::int IS NULL OR d.captured_at >= now() - make_interval(secs => $2))
                  AND ($3::uuid IS NULL OR d.person_id = $3)
                  AND ($4::text IS NULL OR d.review_state = $4)
                  AND (NOT $5::bool OR (d.person_id IS NULL AND d.review_state != 'rejected'))
                ORDER BY d.captured_at DESC
                LIMIT $6
                """,
                camera, since_seconds_ago, person_id, review_state, unknowns_only, limit,
            )
        return [dict(r) for r in rows]

    async def get_detection(self, detection_id: UUID) -> Optional[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT d.id, d.event_id, d.camera, d.captured_at,
                       d.person_id, p.name AS person_name,
                       d.confidence, d.quality_score, d.snapshot_path,
                       d.review_state, d.embedding_contributed
                FROM face_detections d
                LEFT JOIN people p ON d.person_id = p.id
                WHERE d.id = $1
                """,
                detection_id,
            )
        return dict(row) if row else None

    async def confirm_detection(
        self,
        detection_id: UUID,
        person_id: UUID,
        embedding_contributed: bool,
    ) -> Optional[dict[str, Any]]:
        """Mark a detection confirmed and attach a person.

        Used by /api/detections/{id}/confirm after the snapshot has been
        re-embedded and (optionally) added to the person's gallery via
        _persist_enrollment.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE face_detections
                SET person_id = $2,
                    review_state = 'confirmed',
                    embedding_contributed = embedding_contributed OR $3
                WHERE id = $1
                RETURNING id, event_id, camera, captured_at, person_id,
                          confidence, quality_score, snapshot_path,
                          review_state, embedding_contributed
                """,
                detection_id, person_id, embedding_contributed,
            )
        return dict(row) if row else None

    async def reject_detection(
        self, detection_id: UUID
    ) -> Optional[dict[str, Any]]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE face_detections
                SET review_state = 'rejected'
                WHERE id = $1
                RETURNING id, event_id, camera, captured_at, person_id,
                          confidence, quality_score, snapshot_path,
                          review_state, embedding_contributed
                """,
                detection_id,
            )
        return dict(row) if row else None

    # --- continuous improvement ---------------------------------------

    async def count_face_images_for_person(self, person_id: UUID) -> int:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return int(
                await conn.fetchval(
                    "SELECT COUNT(*) FROM face_images WHERE person_id = $1",
                    person_id,
                )
            )

    async def evict_oldest_non_primary_face_image(
        self, person_id: UUID
    ) -> Optional[dict[str, Any]]:
        """Delete the oldest non-primary face_images row for `person_id`.

        Returns the deleted row's `path` and `qdrant_point_id` so the caller
        can clean up the file and Qdrant point. Returns None when every
        face_image for this person is marked primary (caller should skip
        improvement in that case).
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                DELETE FROM face_images
                WHERE id = (
                    SELECT id FROM face_images
                    WHERE person_id = $1 AND is_primary = false
                    ORDER BY created_at ASC
                    LIMIT 1
                )
                RETURNING id, path, qdrant_point_id
                """,
                person_id,
            )
        return dict(row) if row else None

    async def insert_face_image_for_detection(
        self,
        face_image_id: UUID,
        person_id: UUID,
        path: str,
        qdrant_point_id: UUID,
        source: str,
        quality_score: float,
        detection_id: UUID,
    ) -> dict[str, Any]:
        """Insert auto-improvement face_images row + flag the detection.

        Both writes happen in one transaction so a detection is never
        marked `embedding_contributed=true` without a corresponding
        face_images row landing.
        """
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO face_images
                        (id, person_id, path, qdrant_point_id, source,
                         quality_score, is_primary)
                    VALUES ($1, $2, $3, $4, $5, $6, false)
                    RETURNING id, path, is_primary, source, quality_score, created_at
                    """,
                    face_image_id, person_id, path, qdrant_point_id,
                    source, quality_score,
                )
                await conn.execute(
                    "UPDATE face_detections SET embedding_contributed = true WHERE id = $1",
                    detection_id,
                )
        return dict(row)


db = Database()
