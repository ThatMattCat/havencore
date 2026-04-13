"""Autonomy engine REST API."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from selene_agent.autonomy import db as autonomy_db

router = APIRouter()


def _engine(req: Request):
    engine = getattr(req.app.state, "autonomy_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="autonomy engine not initialized")
    return engine


@router.get("/autonomy/status")
async def status(req: Request):
    return await _engine(req).status()


@router.post("/autonomy/pause")
async def pause(req: Request):
    _engine(req).pause()
    return {"paused": True}


@router.post("/autonomy/resume")
async def resume(req: Request):
    _engine(req).resume()
    return {"paused": False}


@router.get("/autonomy/items")
async def items():
    rows = await autonomy_db.list_all_items()
    # Normalize datetimes for JSON.
    out = []
    for r in rows:
        out.append({
            **r,
            "next_fire_at": r["next_fire_at"].isoformat() if r.get("next_fire_at") else None,
            "last_fired_at": r["last_fired_at"].isoformat() if r.get("last_fired_at") else None,
            "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
        })
    return {"items": out}


@router.get("/autonomy/runs")
async def runs(limit: int = 50, include_messages: int = 0):
    data = await autonomy_db.list_runs(limit=limit, include_messages=bool(include_messages))
    return {"runs": data, "limit": limit}


@router.post("/autonomy/trigger/{agenda_item_id}")
async def trigger(agenda_item_id: str, req: Request):
    result = await _engine(req).trigger(agenda_item_id)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="agenda item not found")
    return result
