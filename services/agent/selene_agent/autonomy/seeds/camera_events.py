"""Default agenda items that wire camera/sensor events into watch_llm.

These items subscribe to the ``haven/<domain>/<kind>`` MQTT topic schema
emitted by face-recognition (and, in the future, vehicle/motion/doorbell
sources). They are idempotent — re-running ``ensure_seeds()`` only inserts
rows that don't already exist with the matching ``created_by`` marker. We
deliberately do NOT update existing seeded rows on subsequent boots: once
the operator has tuned an item via the dashboard, our defaults shouldn't
clobber that.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger("loki")


_CREATED_BY = "system_camera"


_FACE_TRIGGER_PROMPT_NOTE = (
    "Camera events. Treat known residents at expected zones as nominal. "
    "Treat unknown subjects, low-quality matches, or unusual times as "
    "potentially worth notifying. Use 'speaker' only when residents are "
    "plausibly home; 'signal' for time-sensitive when away; 'silent' when "
    "nothing actionable."
)


SEEDS = [
    {
        "name": "face_identified_triage",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "enabled": True,
        # Narrow topic — never match haven/face/status (capturing/matching/idle
        # lifecycle pings) which the LLM has nothing useful to triage.
        "trigger_spec": {
            "source": "mqtt",
            "match": {"topic": "haven/face/identified"},
        },
        "config": {
            "subject": _FACE_TRIGGER_PROMPT_NOTE,
            "gather": {
                "memories_k": 3,
                "presence": True,
                "recent_visitors_hours": 6,
            },
            "notify": {"channel": "signal"},
            "severity_floor": "low",
            "cooldown_min": 30,
            "attach_snapshot": True,
            "event_rate_limit": "20/min",
        },
    },
    {
        "name": "face_unknown_triage",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "enabled": True,
        "trigger_spec": {
            "source": "mqtt",
            "match": {"topic": "haven/face/unknown"},
        },
        "config": {
            "subject": _FACE_TRIGGER_PROMPT_NOTE,
            "gather": {
                "memories_k": 3,
                "presence": True,
                "recent_visitors_hours": 6,
            },
            "notify": {"channel": "signal"},
            "severity_floor": "low",
            "cooldown_min": 15,
            "attach_snapshot": True,
            "event_rate_limit": "20/min",
        },
    },
    {
        # "Person sensor tripped, but no face was identifiable" — backyard
        # cat-walking, hooded delivery driver, etc. The LLM uses presence,
        # zone, and time to judge; later, a vision AI gather step can
        # examine the snapshot for additional context (clothing, objects).
        "name": "face_no_face_triage",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        "enabled": True,
        "trigger_spec": {
            "source": "mqtt",
            "match": {"topic": "haven/face/no_face"},
        },
        "config": {
            "subject": (
                "Person sensor triggered with no identifiable face. Treat as "
                "nominal when a resident is home and the zone matches their "
                "typical activity (e.g. backyard during cat walks). Treat as "
                "potentially worth notifying when residents are away or the "
                "zone is sensitive."
            ),
            "gather": {
                "memories_k": 3,
                "presence": True,
                "recent_visitors_hours": 6,
            },
            "notify": {"channel": "signal"},
            # No-face events are inherently noisy — wildlife, shadows. Set a
            # higher floor so only escalated severity actually pages.
            "severity_floor": "med",
            "cooldown_min": 45,
            "attach_snapshot": True,
            "event_rate_limit": "10/min",
        },
    },
    {
        "name": "vehicle_event_triage",
        "kind": "watch_llm",
        "autonomy_level": "notify",
        # Off by default — only useful once an LPR / vehicle source publishes
        # on haven/vehicles/+. Operator flips this on when ready.
        "enabled": False,
        "trigger_spec": {
            "source": "mqtt",
            "match": {"topic": "haven/vehicles/+"},
        },
        "config": {
            "subject": (
                "Vehicle events. Generalize across deployments — reason about "
                "zone, plate (when present), and time, not specific cameras."
            ),
            "gather": {
                "memories_k": 3,
                "presence": True,
            },
            "notify": {"channel": "signal"},
            "severity_floor": "low",
            "cooldown_min": 15,
            "attach_snapshot": True,
            "event_rate_limit": "10/min",
        },
    },
]


# Seed names superseded by a subsequent revision. We delete them once at
# boot so the new seeds below can take over without an orphaned row matching
# the wider haven/face/+ pattern (which used to spam watch_llm on status
# pings). User-edited rows are protected by the created_by filter.
_RETIRED_SEED_NAMES = ["face_event_triage"]


async def ensure_seeds() -> None:
    pool = conversation_db.pool
    if not pool:
        logger.warning("[seeds.camera_events] db pool not initialized; skipping")
        return
    async with pool.acquire() as conn:
        for retired in _RETIRED_SEED_NAMES:
            result = await conn.execute(
                "DELETE FROM agenda_items "
                "WHERE name = $1 AND created_by = $2",
                retired, _CREATED_BY,
            )
            if result.endswith(" 1"):
                logger.info(
                    f"[seeds.camera_events] removed retired seed {retired}"
                )
        for seed in SEEDS:
            existing = await conn.fetchval(
                "SELECT 1 FROM agenda_items "
                "WHERE name = $1 AND created_by = $2 LIMIT 1",
                seed["name"], _CREATED_BY,
            )
            if existing:
                continue
            await _insert_seed(conn, seed)
            logger.info(f"[seeds.camera_events] inserted {seed['name']}")


async def _insert_seed(conn, seed: Dict[str, Any]) -> None:
    new_id = str(uuid.uuid4())
    await conn.execute(
        """
        INSERT INTO agenda_items
            (id, kind, name, schedule_cron, trigger_spec, next_fire_at,
             config, autonomy_level, enabled, created_by)
        VALUES ($1, $2, $3, NULL, $4::jsonb, NULL, $5::jsonb, $6, $7, $8)
        """,
        uuid.UUID(new_id),
        seed["kind"],
        seed["name"],
        json.dumps(seed["trigger_spec"]),
        json.dumps(seed["config"]),
        seed["autonomy_level"],
        seed["enabled"],
        _CREATED_BY,
    )
