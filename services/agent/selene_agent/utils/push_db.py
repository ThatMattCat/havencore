"""Companion-app push device registry.

Stores UnifiedPush endpoints (one per device) so the autonomy engine can
fan-out notifications via the user's distributor of choice (typically
ntfy). Reuses the asyncpg pool owned by conversation_db.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger('loki')

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS push_devices (
    device_id     UUID PRIMARY KEY,
    device_label  TEXT NOT NULL,
    endpoint      TEXT NOT NULL,
    platform      TEXT NOT NULL DEFAULT 'android',
    registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_push_devices_endpoint ON push_devices (endpoint);
"""


class PushDB:
    async def ensure_schema(self) -> None:
        pool = conversation_db.pool
        if not pool:
            logger.warning("push_db.ensure_schema: pool not initialized, skipping")
            return
        async with pool.acquire() as conn:
            await conn.execute(ENSURE_TABLE_SQL)

    async def upsert_device(
        self,
        device_id: str,
        device_label: str,
        endpoint: str,
        platform: str = "android",
    ) -> None:
        pool = conversation_db.pool
        if not pool:
            logger.warning("push_db.upsert_device: pool not initialized")
            return
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO push_devices (device_id, device_label, endpoint, platform)
                VALUES ($1::uuid, $2, $3, $4)
                ON CONFLICT (device_id) DO UPDATE SET
                    device_label = EXCLUDED.device_label,
                    endpoint     = EXCLUDED.endpoint,
                    platform     = EXCLUDED.platform,
                    last_seen_at = now()
                """,
                device_id,
                device_label,
                endpoint,
                platform,
            )

    async def delete_device(self, device_id: str) -> bool:
        pool = conversation_db.pool
        if not pool:
            return False
        async with pool.acquire() as conn:
            status = await conn.execute(
                "DELETE FROM push_devices WHERE device_id = $1::uuid",
                device_id,
            )
        # asyncpg returns 'DELETE <n>' as the status string
        try:
            return int(status.split()[-1]) > 0
        except (ValueError, IndexError):
            return False

    async def list_devices(self) -> List[Dict[str, Any]]:
        pool = conversation_db.pool
        if not pool:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT device_id, device_label, endpoint, platform,
                       registered_at, last_seen_at
                FROM push_devices
                ORDER BY registered_at ASC
                """
            )
        return [
            {
                "device_id": str(r["device_id"]),
                "device_label": r["device_label"],
                "endpoint": r["endpoint"],
                "platform": r["platform"],
                "registered_at": _iso(r["registered_at"]),
                "last_seen_at": _iso(r["last_seen_at"]),
            }
            for r in rows
        ]


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


push_db = PushDB()
