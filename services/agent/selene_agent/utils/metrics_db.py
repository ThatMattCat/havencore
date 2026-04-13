"""Per-turn agent metrics persistence.

Reuses the asyncpg pool owned by conversation_db (same Postgres container).
"""
import json
from typing import Any, Dict, List, Optional

from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger('loki')

ENSURE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS turn_metrics (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    llm_ms INTEGER NOT NULL,
    tool_ms_total INTEGER NOT NULL,
    total_ms INTEGER NOT NULL,
    iterations INTEGER NOT NULL,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_created_at ON turn_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_session ON turn_metrics (session_id);
"""


class MetricsDB:
    async def ensure_schema(self) -> None:
        pool = conversation_db.pool
        if not pool:
            logger.warning("metrics_db.ensure_schema: pool not initialized, skipping")
            return
        async with pool.acquire() as conn:
            await conn.execute(ENSURE_TABLE_SQL)

    async def record_turn(self, session_id: Optional[str], payload: Dict[str, Any]) -> None:
        pool = conversation_db.pool
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO turn_metrics
                        (session_id, llm_ms, tool_ms_total, total_ms, iterations, tool_calls)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    session_id,
                    int(payload.get("llm_ms", 0)),
                    int(payload.get("tool_ms_total", 0)),
                    int(payload.get("total_ms", 0)),
                    int(payload.get("iterations", 0)),
                    json.dumps(payload.get("tool_calls", [])),
                )
        except Exception as e:
            logger.error(f"Failed to record turn metric: {e}")

    async def fetch_recent_turns(self, limit: int = 50) -> List[Dict[str, Any]]:
        pool = conversation_db.pool
        if not pool:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, created_at, llm_ms, tool_ms_total,
                       total_ms, iterations, tool_calls
                FROM turn_metrics
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "created_at": r["created_at"].isoformat(),
                "llm_ms": r["llm_ms"],
                "tool_ms_total": r["tool_ms_total"],
                "total_ms": r["total_ms"],
                "iterations": r["iterations"],
                "tool_calls": json.loads(r["tool_calls"]) if isinstance(r["tool_calls"], str) else r["tool_calls"],
            }
            for r in rows
        ]

    async def summary(self, days: int = 7) -> Dict[str, Any]:
        pool = conversation_db.pool
        if not pool:
            return {}
        async with pool.acquire() as conn:
            agg = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)::int AS turns,
                    COALESCE(AVG(llm_ms), 0)::int AS avg_llm_ms,
                    COALESCE(AVG(total_ms), 0)::int AS avg_total_ms,
                    COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms), 0)::int AS p95_total_ms
                FROM turn_metrics
                WHERE created_at >= NOW() - ($1 || ' days')::interval
                """,
                str(days),
            )
            turns_today = await conn.fetchval(
                """
                SELECT COUNT(*)::int FROM turn_metrics
                WHERE created_at::date = CURRENT_DATE
                """
            )
            per_day_rows = await conn.fetch(
                """
                SELECT created_at::date AS day, COUNT(*)::int AS turns
                FROM turn_metrics
                WHERE created_at >= NOW() - ($1 || ' days')::interval
                GROUP BY day
                ORDER BY day ASC
                """,
                str(days * 2),
            )
        return {
            "turns": agg["turns"] if agg else 0,
            "avg_llm_ms": agg["avg_llm_ms"] if agg else 0,
            "avg_total_ms": agg["avg_total_ms"] if agg else 0,
            "p95_total_ms": agg["p95_total_ms"] if agg else 0,
            "turns_today": turns_today or 0,
            "per_day": [
                {"day": r["day"].isoformat(), "turns": r["turns"]}
                for r in per_day_rows
            ],
        }

    async def top_tools(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        pool = conversation_db.pool
        if not pool:
            return []
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    tc->>'name' AS name,
                    COUNT(*)::int AS count,
                    COALESCE(AVG((tc->>'ms')::int), 0)::int AS avg_ms
                FROM turn_metrics,
                     LATERAL jsonb_array_elements(tool_calls) AS tc
                WHERE created_at >= NOW() - ($1 || ' days')::interval
                GROUP BY name
                ORDER BY count DESC
                LIMIT $2
                """,
                str(days),
                limit,
            )
        return [
            {"name": r["name"], "count": r["count"], "avg_ms": r["avg_ms"]}
            for r in rows
        ]


metrics_db = MetricsDB()
