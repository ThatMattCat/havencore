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

-- v3 columns (idempotent; safe to re-run).
ALTER TABLE agenda_items ADD COLUMN IF NOT EXISTS trigger_spec JSONB;
ALTER TABLE agenda_items ADD COLUMN IF NOT EXISTS name TEXT;
ALTER TABLE agenda_items ALTER COLUMN schedule_cron DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conname = 'agenda_items_trigger_or_cron'
    ) THEN
        ALTER TABLE agenda_items
          ADD CONSTRAINT agenda_items_trigger_or_cron
          CHECK (schedule_cron IS NOT NULL OR trigger_spec IS NOT NULL);
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_agenda_items_trigger_source
    ON agenda_items ((trigger_spec->>'source'))
    WHERE trigger_spec IS NOT NULL;

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

-- v3 columns.
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS scheduled_for TIMESTAMP WITH TIME ZONE;
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS trigger_source TEXT;
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS trigger_event JSONB;

-- v4 columns (confirmation flow + action audit trail).
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS confirmation_token TEXT;
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS confirmation_prompt_id TEXT;
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS confirmation_response TEXT;
ALTER TABLE autonomy_runs ADD COLUMN IF NOT EXISTS action_audit JSONB;

CREATE INDEX IF NOT EXISTS idx_autonomy_runs_awaiting
    ON autonomy_runs (triggered_at)
    WHERE status = 'awaiting_confirmation';

CREATE INDEX IF NOT EXISTS idx_autonomy_runs_scheduled
    ON autonomy_runs (scheduled_for)
    WHERE status = 'scheduled';
CREATE INDEX IF NOT EXISTS idx_autonomy_runs_trigger_source
    ON autonomy_runs (trigger_source) WHERE trigger_source IS NOT NULL;

-- LISTEN/NOTIFY on insert so the WS live-feed can stream runs.
CREATE OR REPLACE FUNCTION notify_autonomy_run() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('autonomy_runs_ch', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS autonomy_runs_notify ON autonomy_runs;
CREATE TRIGGER autonomy_runs_notify
    AFTER INSERT ON autonomy_runs
    FOR EACH ROW EXECUTE FUNCTION notify_autonomy_run();

-- camera_zones: maps a raw camera entity_id to a generic zone slug
-- (front_door, backyard, driveway, etc.). The autonomy sensor-event
-- normalizer reads this so the LLM reasons about zones, not camera names.
CREATE TABLE IF NOT EXISTS camera_zones (
    camera_entity TEXT PRIMARY KEY,
    zone          TEXT NOT NULL,
    zone_label    TEXT,
    notes         TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_camera_zones_zone ON camera_zones (zone);

CREATE OR REPLACE FUNCTION notify_camera_zones() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('camera_zones_ch', COALESCE(NEW.camera_entity, OLD.camera_entity));
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS camera_zones_notify ON camera_zones;
CREATE TRIGGER camera_zones_notify
    AFTER INSERT OR UPDATE OR DELETE ON camera_zones
    FOR EACH ROW EXECUTE FUNCTION notify_camera_zones();
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
                "notify_to": config.AUTONOMY_BRIEFING_NOTIFY_TO,
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


AGENDA_COLUMNS = (
    "id, kind, name, schedule_cron, trigger_spec, next_fire_at, last_fired_at, "
    "config, autonomy_level, enabled, created_by, created_at"
)


def _maybe_json(val):
    if val is None:
        return None
    return json.loads(val) if isinstance(val, str) else val


def _row_to_agenda(row) -> Dict[str, Any]:
    cfg = _maybe_json(row["config"]) or {}
    trigger = _maybe_json(row["trigger_spec"])
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "name": row["name"],
        "schedule_cron": row["schedule_cron"],
        "trigger_spec": trigger,
        "next_fire_at": row["next_fire_at"],
        "last_fired_at": row["last_fired_at"],
        "config": cfg,
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
            f"""
            SELECT {AGENDA_COLUMNS}
            FROM agenda_items
            WHERE enabled = TRUE
              AND schedule_cron IS NOT NULL
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
            f"""
            SELECT {AGENDA_COLUMNS}
            FROM agenda_items
            ORDER BY next_fire_at ASC NULLS LAST, created_at ASC
            """
        )
    return [_row_to_agenda(r) for r in rows]


async def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f"""
            SELECT {AGENDA_COLUMNS}
            FROM agenda_items WHERE id = $1
            """,
            uuid.UUID(item_id),
        )
    return _row_to_agenda(row) if row else None


async def list_webhook_items(name: str) -> List[Dict[str, Any]]:
    """Enabled items whose trigger_spec targets the webhook ``name``."""
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {AGENDA_COLUMNS}
            FROM agenda_items
            WHERE enabled = TRUE
              AND trigger_spec->>'source' = 'webhook'
              AND trigger_spec->'match'->>'name' = $1
            """,
            name,
        )
    return [_row_to_agenda(r) for r in rows]


async def list_mqtt_items() -> List[Dict[str, Any]]:
    """Enabled items whose trigger_spec source is MQTT."""
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT {AGENDA_COLUMNS}
            FROM agenda_items
            WHERE enabled = TRUE
              AND trigger_spec->>'source' = 'mqtt'
            """
        )
    return [_row_to_agenda(r) for r in rows]


async def create_item(
    *,
    kind: str,
    name: Optional[str] = None,
    schedule_cron: Optional[str] = None,
    trigger_spec: Optional[Dict[str, Any]] = None,
    next_fire_at: Optional[datetime] = None,
    cfg: Optional[Dict[str, Any]] = None,
    autonomy_level: str = "notify",
    enabled: bool = True,
    created_by: str = "user",
) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    if not schedule_cron and not trigger_spec:
        raise ValueError("agenda item requires schedule_cron or trigger_spec")
    new_id = str(uuid.uuid4())
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agenda_items
                (id, kind, name, schedule_cron, trigger_spec, next_fire_at,
                 config, autonomy_level, enabled, created_by)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9, $10)
            """,
            uuid.UUID(new_id),
            kind,
            name,
            schedule_cron,
            json.dumps(trigger_spec) if trigger_spec is not None else None,
            next_fire_at,
            json.dumps(cfg or {}),
            autonomy_level,
            enabled,
            created_by,
        )
    return await get_item(new_id)


async def update_item(item_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Apply a sparse update to an agenda item. Returns the updated row.

    Accepted keys: name, schedule_cron, trigger_spec, config, autonomy_level,
    enabled, next_fire_at.
    """
    pool = conversation_db.pool
    if not pool:
        return None
    allowed = {
        "name", "schedule_cron", "trigger_spec", "config",
        "autonomy_level", "enabled", "next_fire_at",
    }
    sets: List[str] = []
    args: List[Any] = []
    for key, val in patch.items():
        if key not in allowed:
            continue
        args.append(
            json.dumps(val) if key in ("trigger_spec", "config") and val is not None else val
        )
        cast = "::jsonb" if key in ("trigger_spec", "config") else ""
        sets.append(f"{key} = ${len(args)}{cast}")
    if not sets:
        return await get_item(item_id)
    args.append(uuid.UUID(item_id))
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE agenda_items SET {', '.join(sets)} WHERE id = ${len(args)}",
            *args,
        )
    return await get_item(item_id)


async def delete_item(item_id: str) -> bool:
    """Delete an agenda item plus any still-scheduled deferred runs for it."""
    pool = conversation_db.pool
    if not pool:
        return False
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM autonomy_runs
                WHERE agenda_item_id = $1 AND status = 'scheduled'
                """,
                uuid.UUID(item_id),
            )
            result = await conn.execute(
                "DELETE FROM agenda_items WHERE id = $1",
                uuid.UUID(item_id),
            )
    return result.endswith(" 1")


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
    ``messages``, ``metrics``, ``error``, ``triggered_at``, ``completed_at``,
    ``scheduled_for``, ``trigger_source``, ``trigger_event``.
    """
    pool = conversation_db.pool
    if not pool:
        return ""
    run_id = run.get("id") or str(uuid.uuid4())
    triggered_at = run.get("triggered_at") or datetime.now(timezone.utc)
    completed_at = run.get("completed_at")
    trigger_event = run.get("trigger_event")
    action_audit = run.get("action_audit")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO autonomy_runs
                (id, agenda_item_id, kind, triggered_at, completed_at, status,
                 summary, severity, signature_hash, notified_via, messages,
                 metrics, error, scheduled_for, trigger_source, trigger_event,
                 confirmation_token, confirmation_prompt_id, confirmation_response,
                 action_audit)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb,
                    $12::jsonb, $13, $14, $15, $16::jsonb,
                    $17, $18, $19, $20::jsonb)
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
            run.get("scheduled_for"),
            run.get("trigger_source"),
            json.dumps(trigger_event) if trigger_event is not None else None,
            run.get("confirmation_token"),
            run.get("confirmation_prompt_id"),
            run.get("confirmation_response"),
            json.dumps(action_audit) if action_audit is not None else None,
        )
    return run_id


async def list_scheduled_runs_due(now_utc: datetime) -> List[Dict[str, Any]]:
    """Deferred runs whose ``scheduled_for`` has arrived, still ``scheduled``."""
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, agenda_item_id, kind, triggered_at, status,
                   scheduled_for, trigger_source, trigger_event
            FROM autonomy_runs
            WHERE status = 'scheduled'
              AND scheduled_for IS NOT NULL
              AND scheduled_for <= $1
            ORDER BY scheduled_for ASC
            """,
            now_utc,
        )
    out = []
    for r in rows:
        out.append({
            "id": str(r["id"]),
            "agenda_item_id": str(r["agenda_item_id"]) if r["agenda_item_id"] else None,
            "kind": r["kind"],
            "triggered_at": r["triggered_at"],
            "status": r["status"],
            "scheduled_for": r["scheduled_for"],
            "trigger_source": r["trigger_source"],
            "trigger_event": _maybe_json(r["trigger_event"]),
        })
    return out


async def claim_scheduled_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Atomically claim a deferred run for dispatch.

    Deletes the ``scheduled`` placeholder row and returns its
    ``agenda_item_id`` / ``trigger_source`` / ``trigger_event`` so the caller
    can re-dispatch without racing another tick. Returns ``None`` if the row
    was already claimed (lost the race) or never existed.
    """
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            DELETE FROM autonomy_runs
             WHERE id = $1 AND status = 'scheduled'
         RETURNING agenda_item_id, trigger_source, trigger_event
            """,
            uuid.UUID(run_id),
        )
    if row is None:
        return None
    return {
        "agenda_item_id": str(row["agenda_item_id"]) if row["agenda_item_id"] else None,
        "trigger_source": row["trigger_source"],
        "trigger_event": _maybe_json(row["trigger_event"]),
    }


async def finalize_run(run_id: str, patch: Dict[str, Any]) -> None:
    """Update a claimed run row with final status/summary/etc."""
    pool = conversation_db.pool
    if not pool:
        return
    allowed = {
        "status", "summary", "severity", "signature_hash", "notified_via",
        "messages", "metrics", "error", "completed_at",
        "confirmation_token", "confirmation_prompt_id", "confirmation_response",
        "action_audit",
    }
    sets: List[str] = []
    args: List[Any] = []
    for key, val in patch.items():
        if key not in allowed:
            continue
        if key in ("messages", "metrics"):
            args.append(json.dumps(val if val is not None else ([] if key == "messages" else {})))
            sets.append(f"{key} = ${len(args)}::jsonb")
        elif key == "action_audit":
            args.append(json.dumps(val) if val is not None else None)
            sets.append(f"{key} = ${len(args)}::jsonb")
        else:
            args.append(val)
            sets.append(f"{key} = ${len(args)}")
    if not sets:
        return
    args.append(uuid.UUID(run_id))
    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE autonomy_runs SET {', '.join(sets)} WHERE id = ${len(args)}",
            *args,
        )


async def count_events_last_hour(item_id: str) -> int:
    pool = conversation_db.pool
    if not pool:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM autonomy_runs
            WHERE agenda_item_id = $1
              AND trigger_source IN ('webhook', 'mqtt')
              AND triggered_at >= NOW() - INTERVAL '1 hour'
            """,
            uuid.UUID(item_id),
        )
    return int(val or 0)


async def count_deferred_runs() -> int:
    pool = conversation_db.pool
    if not pool:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT COUNT(*)::int FROM autonomy_runs WHERE status = 'scheduled'"
        )
    return int(val or 0)


async def get_run(run_id: str, *, include_messages: bool = True) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, agenda_item_id, kind, triggered_at, completed_at,
                   status, summary, severity, signature_hash, notified_via,
                   messages, metrics, error, scheduled_for, trigger_source,
                   trigger_event, confirmation_token, confirmation_prompt_id,
                   confirmation_response, action_audit
            FROM autonomy_runs WHERE id = $1
            """,
            uuid.UUID(run_id),
        )
    return _row_to_run(row, include_messages=include_messages) if row else None


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
    keys = row.keys() if hasattr(row, "keys") else []
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
        "metrics": _maybe_json(row["metrics"]) or {},
    }
    if "scheduled_for" in keys:
        sf = row["scheduled_for"]
        out["scheduled_for"] = sf.isoformat() if sf else None
    if "trigger_source" in keys:
        out["trigger_source"] = row["trigger_source"]
    if "trigger_event" in keys:
        out["trigger_event"] = _maybe_json(row["trigger_event"])
    if "confirmation_token" in keys:
        out["confirmation_token"] = row["confirmation_token"]
    if "confirmation_prompt_id" in keys:
        out["confirmation_prompt_id"] = row["confirmation_prompt_id"]
    if "confirmation_response" in keys:
        out["confirmation_response"] = row["confirmation_response"]
    if "action_audit" in keys:
        out["action_audit"] = _maybe_json(row["action_audit"])
    if include_messages:
        out["messages"] = _maybe_json(row["messages"]) or []
    return out


async def list_runs(
    limit: int = 50,
    include_messages: bool = False,
    *,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    trigger_source: Optional[str] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return []
    clauses: List[str] = []
    args: List[Any] = []
    if kind:
        args.append(kind)
        clauses.append(f"kind = ${len(args)}")
    if status:
        args.append(status)
        clauses.append(f"status = ${len(args)}")
    if trigger_source:
        args.append(trigger_source)
        clauses.append(f"trigger_source = ${len(args)}")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    lim_placeholder = f"${len(args)}"
    args.append(offset)
    off_placeholder = f"${len(args)}"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, agenda_item_id, kind, triggered_at, completed_at,
                   status, summary, severity, signature_hash, notified_via,
                   messages, metrics, error, scheduled_for, trigger_source,
                   trigger_event, confirmation_token, confirmation_prompt_id,
                   confirmation_response, action_audit
            FROM autonomy_runs
            {where}
            ORDER BY triggered_at DESC
            LIMIT {lim_placeholder} OFFSET {off_placeholder}
            """,
            *args,
        )
    return [_row_to_run(r, include_messages=include_messages) for r in rows]


async def list_awaiting_confirmation_runs() -> List[Dict[str, Any]]:
    """All runs currently in ``awaiting_confirmation`` status, newest first."""
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, agenda_item_id, kind, triggered_at, completed_at,
                   status, summary, severity, signature_hash, notified_via,
                   messages, metrics, error, scheduled_for, trigger_source,
                   trigger_event, confirmation_token, confirmation_prompt_id,
                   confirmation_response, action_audit
            FROM autonomy_runs
            WHERE status = 'awaiting_confirmation'
            ORDER BY triggered_at DESC
            """
        )
    return [_row_to_run(r, include_messages=False) for r in rows]


async def claim_confirmation_timeout(
    run_id: str, timeout_at: datetime
) -> bool:
    """Atomically transition a run from ``awaiting_confirmation`` to
    ``confirmation_timeout``. Returns True on success, False if already moved.
    """
    pool = conversation_db.pool
    if not pool:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE autonomy_runs
               SET status = 'confirmation_timeout',
                   confirmation_response = 'timeout',
                   completed_at = $2
             WHERE id = $1 AND status = 'awaiting_confirmation'
            """,
            uuid.UUID(run_id),
            timeout_at,
        )
    return result.endswith(" 1")


async def list_expired_confirmations(now_utc: datetime) -> List[Dict[str, Any]]:
    """Awaiting-confirmation runs whose deadline (triggered_at + timeout) has
    passed. The per-item timeout is read from the item's config; we return the
    rows with their agenda item config so the engine can evaluate each.
    """
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT r.id AS run_id, r.triggered_at,
                   COALESCE((ai.config->>'confirmation_timeout_sec')::int, 300) AS tout
            FROM autonomy_runs r
            LEFT JOIN agenda_items ai ON ai.id = r.agenda_item_id
            WHERE r.status = 'awaiting_confirmation'
            """
        )
    out: List[Dict[str, Any]] = []
    for r in rows:
        triggered = r["triggered_at"]
        if triggered is None:
            continue
        if triggered + timedelta(seconds=int(r["tout"])) <= now_utc:
            out.append({"run_id": str(r["run_id"])})
    return out


async def record_action_audit(run_id: str, audit: List[Dict[str, Any]]) -> None:
    """Overwrite the ``action_audit`` column with the provided list."""
    await finalize_run(run_id, {"action_audit": audit})


async def count_confirmation_timeouts_since(since: datetime) -> int:
    pool = conversation_db.pool
    if not pool:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            """
            SELECT COUNT(*)::int FROM autonomy_runs
            WHERE status = 'confirmation_timeout'
              AND triggered_at >= $1
            """,
            since,
        )
    return int(val or 0)


async def count_awaiting_confirmation() -> int:
    pool = conversation_db.pool
    if not pool:
        return 0
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT COUNT(*)::int FROM autonomy_runs WHERE status = 'awaiting_confirmation'"
        )
    return int(val or 0)


# --- camera_zones --------------------------------------------------------

async def list_camera_zones() -> List[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT camera_entity, zone, zone_label, notes, updated_at "
            "FROM camera_zones ORDER BY zone, camera_entity"
        )
    return [
        {
            "camera_entity": r["camera_entity"],
            "zone": r["zone"],
            "zone_label": r["zone_label"],
            "notes": r["notes"],
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        }
        for r in rows
    ]


async def get_camera_zone(camera_entity: str) -> Optional[Dict[str, Any]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT camera_entity, zone, zone_label, notes "
            "FROM camera_zones WHERE camera_entity = $1",
            camera_entity,
        )
    if not row:
        return None
    return {
        "camera_entity": row["camera_entity"],
        "zone": row["zone"],
        "zone_label": row["zone_label"],
        "notes": row["notes"],
    }


async def upsert_camera_zone(
    camera_entity: str,
    *,
    zone: str,
    zone_label: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    pool = conversation_db.pool
    if not pool:
        raise RuntimeError("db not initialized")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO camera_zones (camera_entity, zone, zone_label, notes, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (camera_entity) DO UPDATE SET
                zone = EXCLUDED.zone,
                zone_label = EXCLUDED.zone_label,
                notes = EXCLUDED.notes,
                updated_at = NOW()
            """,
            camera_entity,
            zone,
            zone_label,
            notes,
        )
    return {"camera_entity": camera_entity, "zone": zone, "zone_label": zone_label, "notes": notes}


async def delete_camera_zone(camera_entity: str) -> bool:
    pool = conversation_db.pool
    if not pool:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM camera_zones WHERE camera_entity = $1", camera_entity,
        )
    return result.endswith(" 1")


async def count_runs_by_trigger_source_last(hours: int = 24) -> Dict[str, int]:
    pool = conversation_db.pool
    if not pool:
        return {}
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT COALESCE(trigger_source, 'cron') AS src, COUNT(*)::int AS n
            FROM autonomy_runs
            WHERE triggered_at >= NOW() - make_interval(hours => $1)
            GROUP BY COALESCE(trigger_source, 'cron')
            """,
            hours,
        )
    return {r["src"]: int(r["n"]) for r in rows}
