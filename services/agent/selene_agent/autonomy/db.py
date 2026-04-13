"""Autonomy persistence: ``agenda_items`` and ``autonomy_runs``.

Reuses the asyncpg pool owned by ``conversation_db`` (same Postgres container
as the rest of the agent), matching the pattern in ``metrics_db``.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from selene_agent.autonomy import schedule
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger('loki')


ENSURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agenda_items (
    id UUID PRIMARY KEY,
    kind TEXT NOT NULL,
    schedule_cron TEXT,
    next_fire_at TIMESTAMP WITH TIME ZONE,
    last_fired_at TIMESTAMP WITH TIME ZONE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    autonomy_level TEXT NOT NULL DEFAULT 'notify',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_by TEXT NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agenda_items_next_fire_at ON agenda_items (next_fire_at);
CREATE INDEX IF NOT EXISTS idx_agenda_items_kind ON agenda_items (kind);

CREATE TABLE IF NOT EXISTS autonomy_runs (
    id UUID PRIMARY KEY,
    agenda_item_id UUID REFERENCES agenda_items(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL,
    summary TEXT,
    severity TEXT,
    signature_hash TEXT,
    notified_via TEXT,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_autonomy_runs_triggered_at ON autonomy_runs (triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_autonomy_runs_signature ON autonomy_runs (signature_hash);
CREATE INDEX IF NOT EXISTS idx_autonomy_runs_kind ON autonomy_runs (kind);
"""


# --- agenda_items --------------------------------------------------------

async def ensure_schema() -> None:
    pool = conversation_db.pool
    if not pool:
        logger.warning("autonomy.db.ensure_schema: pool not initialized, skipping")
        return
    async with pool.acquire() as conn:
        await conn.execute(ENSURE_SCHEMA_SQL)


async def ensure_default_agenda() -> None:
    """Create (or update) the two system-owned default agenda rows.

    Env-configured cron expressions are authoritative; we update the row's
    ``schedule_cron`` each startup so operators can tune via .env without
    touching the DB. We leave ``enabled`` alone so a manual pause survives
    restart.
    """
    pool = conversation_db.pool
    if not pool:
        return
    now = datetime.now(timezone.utc)

    defaults = [
        {
            "kind": "briefing",
            "cron": config.AUTONOMY_BRIEFING_CRON,
            "autonomy_level": "notify",
            "cfg": {
                "email_to": config.AUTONOMY_BRIEFING_EMAIL_TO,
                "camera_entities": config.AUTONOMY_BRIEFING_CAMERA_ENTITIES,
                "window_hours": 10,
            },
        },
        {
            "kind": "anomaly_sweep",
            "cron": config.AUTONOMY_ANOMALY_CRON,
            "autonomy_level": "notify",
            "cfg": {
                "ha_notify_target": config.AUTONOMY_HA_NOTIFY_TARGET,
                "watch_domains": config.AUTONOMY_ANOMALY_WATCH_DOMAINS,
                "cooldown_min": config.AUTONOMY_ANOMALY_COOLDOWN_MIN,
            },
        },
        {
            "kind": "memory_review",
            "cron": config.AUTONOMY_MEMORY_REVIEW_CRON,
            "autonomy_level": "observe",
            "cfg": {
                "max_scan": config.AUTONOMY_MEMORY_MAX_SCAN,
                "llm_call_cap": config.AUTONOMY_MEMORY_LLM_CALL_CAP,
            },
        },
    ]

    async with pool.acquire() as conn:
        for d in defaults:
            try:
                next_fire = schedule.next_fire_at(d["cron"], after=now)
            except Exception as e:
                logger.error(f"Invalid cron for default {d['kind']}: {d['cron']!r}: {e}")
                continue

            row = await conn.fetchrow(
                """
                SELECT id, schedule_cron FROM agenda_items
                WHERE kind = $1 AND created_by = 'system'
                LIMIT 1
                """,
                d["kind"],
            )
            if row is None:
                new_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO agenda_items
                        (id, kind, schedule_cron, next_fire_at, config,
                         autonomy_level, enabled, created_by)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, TRUE, 'system')
                    """,
                    new_id,
                    d["kind"],
                    d["cron"],
                    next_fire,
                    json.dumps(d["cfg"]),
                    d["autonomy_level"],
                )
                logger.info(f"Seeded default agenda item: {d['kind']} (cron={d['cron']})")
            else:
                # Update config + cron from env; preserve enabled flag.
                await conn.execute(
                    """
                    UPDATE agenda_items
                       SET schedule_cron = $2,
                           config = $3::jsonb,
                           autonomy_level = $4,
                           next_fire_at = CASE
                               WHEN schedule_cron IS DISTINCT FROM $2 OR next_fire_at IS NULL
                                   THEN $5
                               ELSE next_fire_at
                           END
                     WHERE id = $1
                    """,
                    row["id"],
                    d["cron"],
                    json.dumps(d["cfg"]),
                    d["autonomy_level"],
                    next_fire,
                )


def _row_to_agenda(row) -> Dict[str, Any]:
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "schedule_cron": row["schedule_cron"],
        "next_fire_at": row["next_fire_at"],
        "last_fired_at": row["last_fired_at"],
        "config": json.loads(row["config"]) if isinstance(row["config"], str) else (row["config"] or {}),
        "autonomy_level": row["autonomy_level"],
        "enabled": row["enabled"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
    }


async def list_due_items(now_utc: datetime) -> List[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, kind, schedule_cron, next_fire_at, last_fired_at,
                   config, autonomy_level, enabled, created_by, created_at
            FROM agenda_items
            WHERE enabled = TRUE
              AND next_fire_at IS NOT NULL
              AND next_fire_at <= $1
            ORDER BY next_fire_at ASC
            """,
            now_utc,
        )
    return [_row_to_agenda(r) for r in rows]


async def list_all_items() -> List[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, kind, schedule_cron, next_fire_at, last_fired_at,
                   config, autonomy_level, enabled, created_by, created_at
            FROM agenda_items
            ORDER BY next_fire_at ASC NULLS LAST
            """
        )
    return [_row_to_agenda(r) for r in rows]


async def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, kind, schedule_cron, next_fire_at, last_fired_at,
                   config, autonomy_level, enabled, created_by, created_at
            FROM agenda_items WHERE id = $1
            """,
            uuid.UUID(item_id),
        )
    return _row_to_agenda(row) if row else None


async def advance_item(item_id: str, fired_at: datetime, next_at: datetime) -> None:
    pool = conversation_db.pool
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE agenda_items
               SET last_fired_at = $2,
                   next_fire_at = $3
             WHERE id = $1
            """,
            uuid.UUID(item_id),
            fired_at,
            next_at,
        )


# --- autonomy_runs -------------------------------------------------------

async def insert_run(run: Dict[str, Any]) -> str:
    """Insert a run row and return the new ``id``.

    Expected keys: ``agenda_item_id``, ``kind``, ``status``; optional:
    ``summary``, ``severity``, ``signature_hash``, ``notified_via``,
    ``messages``, ``metrics``, ``error``, ``triggered_at``, ``completed_at``.
    """
    pool = conversation_db.pool
    if not pool:
        return ""
    run_id = run.get("id") or str(uuid.uuid4())
    triggered_at = run.get("triggered_at") or datetime.now(timezone.utc)
    completed_at = run.get("completed_at")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO autonomy_runs
                (id, agenda_item_id, kind, triggered_at, completed_at, status,
                 summary, severity, signature_hash, notified_via, messages,
                 metrics, error)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb,
                    $12::jsonb, $13)
            """,
            uuid.UUID(run_id),
            uuid.UUID(run["agenda_item_id"]) if run.get("agenda_item_id") else None,
            run["kind"],
            triggered_at,
            completed_at,
            run["status"],
            run.get("summary"),
            run.get("severity"),
            run.get("signature_hash"),
            run.get("notified_via"),
            json.dumps(run.get("messages", [])),
            json.dumps(run.get("metrics", {})),
            run.get("error"),
        )
    return run_id


async def count_runs_since(since: datetime) -> int:
    pool = conversation_db.pool
    if not pool:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM autonomy_runs
            WHERE triggered_at >= $1
              AND status IN ('ok', 'error')
            """,
            since,
        )
    return int(val or 0)


async def last_run_for_signature(signature_hash: str, since: datetime) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, triggered_at, severity, notified_via, status
            FROM autonomy_runs
            WHERE signature_hash = $1
              AND triggered_at >= $2
              AND status = 'ok'
            ORDER BY triggered_at DESC
            LIMIT 1
            """,
            signature_hash,
            since,
        )
    if not row:
        return None
    return {
        "id": str(row["id"]),
        "triggered_at": row["triggered_at"],
        "severity": row["severity"],
        "notified_via": row["notified_via"],
        "status": row["status"],
    }


def _row_to_run(row, include_messages: bool = False) -> Dict[str, Any]:
    out = {
        "id": str(row["id"]),
        "agenda_item_id": str(row["agenda_item_id"]) if row["agenda_item_id"] else None,
        "kind": row["kind"],
        "triggered_at": row["triggered_at"].isoformat() if row["triggered_at"] else None,
        "completed_at": row["completed_at"].isoformat() if row["completed_at"] else None,
        "status": row["status"],
        "summary": row["summary"],
        "severity": row["severity"],
        "signature_hash": row["signature_hash"],
        "notified_via": row["notified_via"],
        "error": row["error"],
        "metrics": json.loads(row["metrics"]) if isinstance(row["metrics"], str) else (row["metrics"] or {}),
    }
    if include_messages:
        msgs = row["messages"]
        out["messages"] = json.loads(msgs) if isinstance(msgs, str) else (msgs or [])
    return out


async def list_runs(limit: int = 50, include_messages: bool = False) -> List[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, agenda_item_id, kind, triggered_at, completed_at,
                   status, summary, severity, signature_hash, notified_via,
                   messages, metrics, error
            FROM autonomy_runs
            ORDER BY triggered_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [_row_to_run(r, include_messages=include_messages) for r in rows]
