"""Postgres access for the face-recognition service.

Owns an asyncpg pool and runs idempotent migrations on startup so existing
HavenCore deployments pick up the face tables without a DB rebuild. Fresh
deployments get the same schema via services/postgres/init.sql; both paths
must stay in sync.
"""

import asyncio
import logging
import os
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


db = Database()
