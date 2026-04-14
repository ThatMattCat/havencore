"""Autonomy engine REST + WS API."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from selene_agent.autonomy import db as autonomy_db
from selene_agent.autonomy import schedule as autonomy_schedule
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger('loki')

router = APIRouter()
ws_router = APIRouter()


# --- helpers --------------------------------------------------------------

def _engine(req: Request):
    engine = getattr(req.app.state, "autonomy_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="autonomy engine not initialized")
    return engine


def _serialize_item(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **r,
        "next_fire_at": r["next_fire_at"].isoformat() if r.get("next_fire_at") else None,
        "last_fired_at": r["last_fired_at"].isoformat() if r.get("last_fired_at") else None,
        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
    }


# --- schemas --------------------------------------------------------------

class AgendaCreate(BaseModel):
    kind: str
    name: Optional[str] = None
    schedule_cron: Optional[str] = None
    trigger_spec: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    autonomy_level: str = "notify"
    enabled: bool = True


class AgendaPatch(BaseModel):
    name: Optional[str] = None
    schedule_cron: Optional[str] = None
    trigger_spec: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    autonomy_level: Optional[str] = None
    enabled: Optional[bool] = None


# --- existing endpoints ---------------------------------------------------

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
    return {"items": [_serialize_item(r) for r in rows]}


@router.get("/autonomy/runs")
async def runs(
    limit: int = 50,
    include_messages: int = 0,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    trigger_source: Optional[str] = None,
    offset: int = 0,
):
    data = await autonomy_db.list_runs(
        limit=max(1, min(500, limit)),
        include_messages=bool(include_messages),
        kind=kind,
        status=status,
        trigger_source=trigger_source,
        offset=max(0, offset),
    )
    return {"runs": data, "limit": limit, "offset": offset}


@router.post("/autonomy/trigger/{agenda_item_id}")
async def trigger(
    agenda_item_id: str,
    req: Request,
    bypass_quiet: bool = Query(False),
):
    result = await _engine(req).trigger(agenda_item_id, bypass_quiet=bypass_quiet)
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="agenda item not found")
    return result


# --- v3 CRUD --------------------------------------------------------------

def _validate_agenda(body: Dict[str, Any]) -> None:
    kind = body.get("kind")
    if kind not in ("briefing", "anomaly_sweep", "memory_review", "reminder", "watch", "routine"):
        raise HTTPException(status_code=400, detail=f"unknown kind: {kind}")
    cron = body.get("schedule_cron")
    trigger_spec = body.get("trigger_spec")
    if not cron and not trigger_spec:
        raise HTTPException(status_code=400, detail="item requires schedule_cron or trigger_spec")
    if cron:
        try:
            autonomy_schedule.next_fire_at(cron, after=datetime.now(timezone.utc))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid cron: {e}")
    if trigger_spec:
        src = (trigger_spec or {}).get("source")
        if src not in ("mqtt", "webhook"):
            raise HTTPException(status_code=400, detail="trigger_spec.source must be 'mqtt' or 'webhook'")
        match = (trigger_spec or {}).get("match") or {}
        if src == "mqtt" and not match.get("topic"):
            raise HTTPException(status_code=400, detail="mqtt trigger_spec requires match.topic")
        if src == "webhook" and not match.get("name"):
            raise HTTPException(status_code=400, detail="webhook trigger_spec requires match.name")


@router.post("/autonomy/items")
async def create_item(body: AgendaCreate, req: Request):
    body_dict = body.model_dump()
    _validate_agenda(body_dict)
    next_fire = None
    if body.schedule_cron:
        next_fire = autonomy_schedule.next_fire_at(
            body.schedule_cron, after=datetime.now(timezone.utc)
        )
    created = await autonomy_db.create_item(
        kind=body.kind,
        name=body.name,
        schedule_cron=body.schedule_cron,
        trigger_spec=body.trigger_spec,
        next_fire_at=next_fire,
        cfg=body.config,
        autonomy_level=body.autonomy_level,
        enabled=body.enabled,
        created_by="user",
    )
    if created is None:
        raise HTTPException(status_code=503, detail="database not ready")
    _engine(req).notify_agenda_changed()
    return {"item": _serialize_item(created)}


@router.patch("/autonomy/items/{item_id}")
async def patch_item(item_id: str, body: AgendaPatch, req: Request):
    patch = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None or k == "enabled"}
    if not patch:
        current = await autonomy_db.get_item(item_id)
        if current is None:
            raise HTTPException(status_code=404, detail="not found")
        return {"item": _serialize_item(current)}

    # Re-validate if shape-relevant fields changed.
    current = await autonomy_db.get_item(item_id)
    if current is None:
        raise HTTPException(status_code=404, detail="not found")
    effective = {**current, **patch}
    if "schedule_cron" in patch or "trigger_spec" in patch:
        _validate_agenda({
            "kind": effective.get("kind"),
            "schedule_cron": effective.get("schedule_cron"),
            "trigger_spec": effective.get("trigger_spec"),
        })
        if patch.get("schedule_cron"):
            patch["next_fire_at"] = autonomy_schedule.next_fire_at(
                patch["schedule_cron"], after=datetime.now(timezone.utc)
            )
    updated = await autonomy_db.update_item(item_id, patch)
    if updated is None:
        raise HTTPException(status_code=404, detail="not found")
    _engine(req).notify_agenda_changed()
    return {"item": _serialize_item(updated)}


@router.delete("/autonomy/items/{item_id}")
async def delete_item(item_id: str, req: Request):
    ok = await autonomy_db.delete_item(item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    _engine(req).notify_agenda_changed()
    return {"deleted": True}


# --- v3 webhook intake ----------------------------------------------------

@router.post("/autonomy/webhook/{name}")
async def webhook(name: str, req: Request):
    """Reactive webhook intake — matches body against any enabled items that
    target ``name``. ``name`` is the shared secret (home-lab trust model).
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    items = await autonomy_db.list_webhook_items(name)
    fired: List[Dict[str, Any]] = []
    engine = _engine(req)
    event = {"source": "webhook", "name": name, "payload": body if isinstance(body, dict) else {"value": body}}
    for item in items:
        result = await engine.trigger_event(item["id"], source="webhook", event=event)
        fired.append({"item_id": item["id"], "status": result.get("status")})
    return {"matched": len(items), "fired": fired}


# --- v3 reactive sources health ------------------------------------------

@router.get("/autonomy/events/summary")
async def events_summary(req: Request):
    engine = _engine(req)
    counts = await autonomy_db.count_runs_by_trigger_source_last(24)
    listener = getattr(engine, "_mqtt_listener", None)
    return {
        "runs_last_24h_by_source": counts,
        "mqtt": {
            "connected": bool(listener and listener.is_connected()),
            "subscribed_topics": listener.subscribed_topics() if listener else [],
        },
        "webhook": {
            "items": [
                {"id": i["id"], "name": (i.get("trigger_spec") or {}).get("match", {}).get("name")}
                for i in await autonomy_db.list_all_items()
                if (i.get("trigger_spec") or {}).get("source") == "webhook"
            ],
        },
        "deferred_runs_pending": await autonomy_db.count_deferred_runs(),
    }


# --- v3 live WS feed ------------------------------------------------------

@ws_router.websocket("/autonomy/runs")
async def ws_runs(ws: WebSocket):
    """Stream every newly-inserted autonomy run over WebSocket.

    Backed by a Postgres LISTEN on channel ``autonomy_runs_ch`` (a trigger on
    ``autonomy_runs`` emits the new row's id on each INSERT).
    """
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def _notify(conn, pid, channel, payload):
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass

    pool = conversation_db.pool
    if pool is None:
        await ws.send_text(json.dumps({"type": "error", "error": "db not ready"}))
        await ws.close()
        return

    listener_conn = await pool.acquire()
    try:
        await listener_conn.add_listener("autonomy_runs_ch", _notify)
        # Prime with the most recent 25 runs so fresh clients have context.
        recent = await autonomy_db.list_runs(limit=25, include_messages=False)
        for r in reversed(recent):
            await ws.send_text(json.dumps({"type": "run", "run": r}))

        async def _pump_from_ws():
            # Drain client messages (ignored) so disconnects are noticed.
            while True:
                try:
                    await ws.receive_text()
                except WebSocketDisconnect:
                    raise
                except Exception:
                    raise

        pump_task = asyncio.create_task(_pump_from_ws())
        try:
            while True:
                get_task = asyncio.create_task(queue.get())
                done, pending = await asyncio.wait(
                    {get_task, pump_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if pump_task in done:
                    get_task.cancel()
                    break
                run_id = get_task.result()
                run = await autonomy_db.get_run(run_id, include_messages=False)
                if run is not None:
                    try:
                        await ws.send_text(json.dumps({"type": "run", "run": run}))
                    except Exception:
                        break
        finally:
            pump_task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[autonomy WS] error: {e}")
    finally:
        try:
            await listener_conn.remove_listener("autonomy_runs_ch", _notify)
        except Exception:
            pass
        await pool.release(listener_conn)
        try:
            await ws.close()
        except Exception:
            pass
