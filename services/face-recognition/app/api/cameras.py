"""Camera discovery against Home Assistant.

Queries HA's ``/api/states`` and surfaces every ``binary_sensor.*_person``
entity with the corresponding ``camera.<base>_fluent`` it would trigger.
``camera_exists`` flags any naming-convention drift so a future camera that
breaks the convention can be spotted before the MQTT bridge silently fails
on it.

Reuses HAOS_URL / HAOS_TOKEN — same env vars the agent's
``mcp_homeassistant_tools`` and our ``ha_snapshot`` consume — and the same
trailing-``/api`` strip rule so the example .env value works unchanged.
"""

import logging
import os

import aiohttp
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


logger = logging.getLogger("face-recognition.api.cameras")

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


HAOS_URL = os.getenv("HAOS_URL", "")
HAOS_TOKEN = os.getenv("HAOS_TOKEN", "")
HA_REQUEST_TIMEOUT_SEC = float(os.getenv("FACE_REC_HA_TIMEOUT_SEC", "10"))

PERSON_SENSOR_PREFIX = "binary_sensor."
PERSON_SENSOR_SUFFIX = "_person"
CAMERA_PREFIX = "camera."
CAMERA_SUFFIX = "_fluent"


class CameraDiscovery(BaseModel):
    sensor_entity: str
    camera_entity: str
    camera_exists: bool
    current_state: str


def _normalize_base_url(raw: str) -> str:
    base = raw.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    return base


def _derive_camera_entity(sensor_entity: str) -> str:
    base = sensor_entity[len(PERSON_SENSOR_PREFIX):-len(PERSON_SENSOR_SUFFIX)]
    return f"{CAMERA_PREFIX}{base}{CAMERA_SUFFIX}"


@router.get("", response_model=list[CameraDiscovery])
async def discover_cameras() -> list[CameraDiscovery]:
    if not HAOS_URL or not HAOS_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="HAOS_URL / HAOS_TOKEN not configured",
        )

    base = _normalize_base_url(HAOS_URL)
    headers = {"Authorization": f"Bearer {HAOS_TOKEN}"}
    timeout = aiohttp.ClientTimeout(total=HA_REQUEST_TIMEOUT_SEC)

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(f"{base}/api/states") as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=502,
                        detail=f"home assistant returned {resp.status}",
                    )
                states = await resp.json()
    except aiohttp.ClientError as e:
        logger.warning("HA /api/states fetch failed: %s", e)
        raise HTTPException(status_code=502, detail=f"home assistant network error: {e}")

    camera_ids = {
        s.get("entity_id") for s in states
        if isinstance(s.get("entity_id"), str)
        and s["entity_id"].startswith(CAMERA_PREFIX)
    }

    discoveries: list[CameraDiscovery] = []
    for s in states:
        entity_id = s.get("entity_id")
        if not isinstance(entity_id, str):
            continue
        if not (
            entity_id.startswith(PERSON_SENSOR_PREFIX)
            and entity_id.endswith(PERSON_SENSOR_SUFFIX)
        ):
            continue
        camera_entity = _derive_camera_entity(entity_id)
        discoveries.append(
            CameraDiscovery(
                sensor_entity=entity_id,
                camera_entity=camera_entity,
                camera_exists=camera_entity in camera_ids,
                current_state=str(s.get("state", "unknown")),
            )
        )

    discoveries.sort(key=lambda d: d.sensor_entity)
    return discoveries
