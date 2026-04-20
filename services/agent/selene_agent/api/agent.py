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


# ---------- LLM provider ----------

class LLMProviderPayload(BaseModel):
    provider: str


@router.get("/system/llm-provider")
async def get_llm_provider():
    """Return the current agent-LLM provider, its model, and the full set of
    valid provider names for UI dropdowns."""
    name = await agent_state.get_llm_provider_name()
    row = await agent_state.get_state("llm_provider")
    since: Optional[str] = None
    if row:
        _, updated_at = row
        since = updated_at.isoformat() if updated_at else None

    model: Optional[str] = None
    try:
        from selene_agent.selene_agent import app
        provider = getattr(app.state, "provider", None)
        if provider is not None:
            model = getattr(provider, "model", None)
    except Exception as e:
        logger.debug(f"get_llm_provider: app.state lookup failed ({e})")

    return {
        "provider": name,
        "model": model,
        "valid": list(agent_state.VALID_LLM_PROVIDERS),
        "since": since,
    }


@router.post("/system/llm-provider")
async def set_llm_provider(body: LLMProviderPayload):
    """Persist the selected provider and hot-swap ``app.state.provider``.

    Existing sessions pick up the new provider on their next turn via the
    provider-getter closure — no session rebuild needed.
    """
    if body.provider not in agent_state.VALID_LLM_PROVIDERS:
        raise HTTPException(
            400,
            f"invalid provider: {body.provider!r} "
            f"(expected {list(agent_state.VALID_LLM_PROVIDERS)})",
        )

    updated_at = await agent_state.set_llm_provider_name(body.provider)

    model: Optional[str] = None
    try:
        from selene_agent.selene_agent import app
        from selene_agent.providers import build_provider
        vllm_model = getattr(app.state, "model_name", None) or "gpt-3.5-turbo"
        provider = build_provider(body.provider, vllm_model=vllm_model)
        app.state.provider = provider
        model = getattr(provider, "model", None)
        logger.info(f"LLM provider swapped to {provider.name} ({model})")
    except Exception as e:
        logger.error(f"failed to swap provider to {body.provider!r}: {e}")
        raise HTTPException(500, f"failed to swap provider: {e}")

    return {
        "provider": body.provider,
        "model": model,
        "since": updated_at.isoformat(),
    }
