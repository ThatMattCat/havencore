"""Companion-app push device registration.

Endpoints:
  POST   /api/push/register             upsert a UnifiedPush endpoint
  DELETE /api/push/register/{device_id} remove a registered endpoint
  GET    /api/push/register             list all registered devices

LAN-only, no auth gate (consistent with the rest of /api/*).
"""
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from selene_agent.utils import logger as custom_logger
from selene_agent.utils.push_db import push_db

logger = custom_logger.get_logger('loki')
router = APIRouter()


class PushRegisterReq(BaseModel):
    device_id: str
    device_label: str = Field(min_length=1, max_length=120)
    endpoint: str
    platform: str = "android"


@router.post("/push/register")
async def register(body: PushRegisterReq):
    try:
        uuid.UUID(body.device_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="device_id must be a UUID")
    parsed = urlparse(body.endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="endpoint must be http(s)://...")
    await push_db.upsert_device(
        body.device_id, body.device_label, body.endpoint, body.platform
    )
    logger.info(
        f"[push] registered device={body.device_id} label={body.device_label!r} "
        f"endpoint={body.endpoint}"
    )
    return {"ok": True}


@router.delete("/push/register/{device_id}")
async def deregister(device_id: str):
    try:
        uuid.UUID(device_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="device_id must be a UUID")
    deleted = await push_db.delete_device(device_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="device not registered")
    logger.info(f"[push] deregistered device={device_id}")
    return {"ok": True}


@router.get("/push/register")
async def list_registered():
    devices = await push_db.list_devices()
    return {"devices": devices}
