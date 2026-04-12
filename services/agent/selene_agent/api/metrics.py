"""Agent metrics API — per-turn timings and aggregates."""

from fastapi import APIRouter

from selene_agent.utils.metrics_db import metrics_db

router = APIRouter()


@router.get("/metrics/turns")
async def recent_turns(limit: int = 50):
    turns = await metrics_db.fetch_recent_turns(limit=limit)
    return {"turns": turns, "limit": limit}


@router.get("/metrics/summary")
async def summary(days: int = 7):
    data = await metrics_db.summary(days=days)
    return data


@router.get("/metrics/top-tools")
async def top_tools(days: int = 7, limit: int = 10):
    tools = await metrics_db.top_tools(days=days, limit=limit)
    return {"tools": tools, "days": days}
