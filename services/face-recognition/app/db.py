"""Postgres access for the face-recognition service.

Owns an asyncpg pool and runs idempotent migrations on startup so existing
HavenCore deployments pick up the face tables without a DB rebuild. Fresh
deployments get the same schema via services/postgres/init.sql; both paths
must stay in sync.
"""

import asyncio
import logging
import os
from typing import Optional

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


db = Database()
