"""Camera ↔ zone mapping API.

Cameras come from face-recognition's ``/api/cameras`` discovery (which itself
walks HA's camera entities); zones come from the autonomy ``camera_zones``
table. We left-join the two so the dashboard can render "every camera you
could face-detect with, and the zone slug it currently maps to".

Why this lives here instead of under ``/api/face/``: zone mapping is an
autonomy concern, not a face-rec concern — the same mapping will apply once
vehicle/motion/doorbell event sources start publishing on
``haven/<domain>/<kind>``. Keeping it under ``/api/cameras`` reflects that.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger("loki")


router = APIRouter(prefix="/cameras", tags=["cameras"])


FACE_REC_API_BASE = os.getenv(
    "FACE_REC_API_BASE", "http://face-recognition:6006"
).rstrip("/")
_FACE_TIMEOUT = aiohttp.ClientTimeout(total=10)


class CameraRow(BaseModel):
    camera_entity: str
    sensor_entity: Optional[str] = None
    camera_exists: bool = True
    current_state: Optional[str] = None
    zone: Optional[str] = None
    zone_label: Optional[str] = None
    notes: Optional[str] = None
    updated_at: Optional[str] = None


class ZoneAssignment(BaseModel):
    zone: str = Field(min_length=1, max_length=64)
    zone_label: Optional[str] = Field(default=None, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=500)


async def _list_face_cameras() -> List[Dict[str, Any]]:
    """Pull the discovered cameras from face-rec. Returns [] if unreachable so
    the page can still render zones-only mode if the user wants to manage
    them ahead of face-rec coming online.
    """
    url = f"{FACE_REC_API_BASE}/api/cameras"
    try:
        async with aiohttp.ClientSession(timeout=_FACE_TIMEOUT) as session:
            async with session.get(url) as resp:
                if resp.status >= 400:
                    logger.warning(
                        f"[cameras] face-rec /api/cameras returned {resp.status}"
                    )
                    return []
                data = await resp.json(content_type=None)
                if isinstance(data, list):
                    return data
                return []
    except aiohttp.ClientError as e:
        logger.warning(f"[cameras] face-rec unreachable: {e}")
        return []


@router.get("")
async def list_cameras() -> Dict[str, Any]:
    """Return discovered cameras left-joined with their zone assignment.

    The response also includes orphan ``camera_zones`` rows whose
    ``camera_entity`` is no longer reported by face-rec — useful when a
    camera is offline / removed but the user still wants to see (and
    optionally clean up) the assignment.
    """
    face_rows = await _list_face_cameras()
    zone_rows = await autonomy_db.list_camera_zones()
    zones_by_entity = {z["camera_entity"]: z for z in zone_rows}

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for fr in face_rows:
        cam = fr.get("camera_entity") or fr.get("entity_id") or ""
        if not cam:
            continue
        seen.add(cam)
        z = zones_by_entity.get(cam) or {}
        out.append({
            "camera_entity": cam,
            "sensor_entity": fr.get("sensor_entity"),
            "camera_exists": fr.get("camera_exists", True),
            "current_state": fr.get("current_state"),
            "zone": z.get("zone"),
            "zone_label": z.get("zone_label"),
            "notes": z.get("notes"),
            "updated_at": z.get("updated_at"),
        })
    # Orphan zones: assignments that no face-rec camera matches anymore.
    for cam, z in zones_by_entity.items():
        if cam in seen:
            continue
        out.append({
            "camera_entity": cam,
            "sensor_entity": None,
            "camera_exists": False,
            "current_state": None,
            "zone": z.get("zone"),
            "zone_label": z.get("zone_label"),
            "notes": z.get("notes"),
            "updated_at": z.get("updated_at"),
        })

    distinct_zones = sorted({z["zone"] for z in zone_rows if z.get("zone")})
    return {"cameras": out, "zones": distinct_zones}


@router.put("/{entity:path}/zone")
async def set_camera_zone(entity: str, body: ZoneAssignment, request: Request):
    if not entity:
        raise HTTPException(status_code=400, detail="camera_entity required")
    row = await autonomy_db.upsert_camera_zone(
        entity,
        zone=body.zone.strip(),
        zone_label=(body.zone_label or "").strip() or None,
        notes=(body.notes or "").strip() or None,
    )
    return row


@router.delete("/{entity:path}/zone")
async def clear_camera_zone(entity: str):
    if not entity:
        raise HTTPException(status_code=400, detail="camera_entity required")
    deleted = await autonomy_db.delete_camera_zone(entity)
    return {"camera_entity": entity, "deleted": deleted}
