"""Agent-level settings endpoints (operational phase, etc.)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from selene_agent.utils import agent_state
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter(tags=["agent"])


class PhasePayload(BaseModel):
    phase: str


@router.get("/agent/phase")
async def get_phase():
    phase = await agent_state.get_agent_phase()
    row = await agent_state.get_state("agent_phase")
    since: Optional[str] = None
    if row:
        _, updated_at = row
        since = updated_at.isoformat() if updated_at else None
    return {"phase": phase, "since": since}


@router.post("/agent/phase")
async def set_phase(body: PhasePayload):
    if body.phase not in agent_state.VALID_PHASES:
        raise HTTPException(
            400,
            f"invalid phase: {body.phase!r} (expected {list(agent_state.VALID_PHASES)})",
        )
    updated_at = await agent_state.set_agent_phase(body.phase)

    # Nudge active sessions to rebuild their system prompt on the next turn.
    try:
        from selene_agent.selene_agent import app
        from selene_agent.utils.session_pool import SessionOrchestratorPool
        pool: Optional[SessionOrchestratorPool] = getattr(app.state, "session_pool", None)
        if pool is not None:
            await pool.rebuild_system_prompts()
    except Exception as e:
        logger.warning(f"failed to nudge session pool after phase change: {e}")

    return {"phase": body.phase, "since": updated_at.isoformat()}
