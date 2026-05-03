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


_NO_FACE_SCENE_PROMPT = (
    "Pay particular attention to whether this looks like a resident, pet, "
    "delivery driver, or wildlife — those are the common no-face triggers we "
    "want to disambiguate. Note clothing, posture, animals, vehicles, packages, "
    "and anything unusual for a residential property. 2-3 sentences. No "
    "speculation about intent."
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
                "scene_description": True,
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
                "scene_description": True,
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
        # zone, time, and the vision AI scene description (resident vs. pet
        # vs. delivery driver vs. wildlife) to judge.
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
                "scene_description": True,
                "scene_description_prompt": _NO_FACE_SCENE_PROMPT,
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
        await _migrate_scene_description(conn)


async def _migrate_scene_description(conn) -> None:
    """One-time patch: add ``scene_description`` (and an optional
    ``scene_description_prompt`` override) to the gather block of any existing
    system-seeded face_*_triage rows that don't already have the key.

    Mirrors the retired-seed pattern — operator-edited rows are protected by
    the ``created_by`` filter, and rows where the operator has already
    explicitly set or unset ``scene_description`` via the dashboard are
    protected by the missing-key check.
    """
    for seed in SEEDS:
        gather = (seed.get("config") or {}).get("gather") or {}
        if "scene_description" not in gather:
            continue
        row = await conn.fetchrow(
            "SELECT id, config FROM agenda_items "
            "WHERE name = $1 AND created_by = $2 LIMIT 1",
            seed["name"], _CREATED_BY,
        )
        if not row:
            continue
        existing_cfg = row["config"]
        if isinstance(existing_cfg, str):
            existing_cfg = json.loads(existing_cfg)
        existing_gather = (existing_cfg or {}).get("gather") or {}
        if "scene_description" in existing_gather:
            continue
        new_gather = dict(existing_gather)
        new_gather["scene_description"] = gather["scene_description"]
        if "scene_description_prompt" in gather:
            new_gather["scene_description_prompt"] = gather["scene_description_prompt"]
        new_cfg = dict(existing_cfg or {})
        new_cfg["gather"] = new_gather
        await conn.execute(
            "UPDATE agenda_items SET config = $1::jsonb WHERE id = $2",
            json.dumps(new_cfg),
            row["id"],
        )
        logger.info(
            f"[seeds.camera_events] patched {seed['name']} with scene_description"
        )


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
