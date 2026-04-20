"""Small key/value state table for agent-level settings (operational phase, etc.).

Reuses the asyncpg pool owned by ``conversation_db``. Values are strings; the
caller is responsible for any parsing. Designed for settings that need to
survive restarts but aren't worth their own table.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger
from selene_agent.utils.conversation_db import conversation_db

logger = custom_logger.get_logger('loki')


ENSURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


VALID_PHASES = ("learning", "operating")
VALID_LLM_PROVIDERS = ("vllm", "anthropic", "openai")


async def ensure_schema() -> None:
    pool = conversation_db.pool
    if not pool:
        logger.warning("agent_state.ensure_schema: pool not initialized, skipping")
        return
    async with pool.acquire() as conn:
        await conn.execute(ENSURE_SCHEMA_SQL)


async def get_state(key: str) -> Optional[Tuple[str, datetime]]:
    pool = conversation_db.pool
    if not pool:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value, updated_at FROM agent_state WHERE key = $1",
            key,
        )
    if not row:
        return None
    return row["value"], row["updated_at"]


async def set_state(key: str, value: str) -> datetime:
    pool = conversation_db.pool
    if not pool:
        raise RuntimeError("conversation_db pool not initialized")
    now = datetime.now(timezone.utc)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_state (key, value, updated_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """,
            key, value, now,
        )
    return now


# ---------- Phase helpers ----------

_AGENT_PHASE_KEY = "agent_phase"


async def get_agent_phase() -> str:
    """Return the current operational phase. Falls back to
    ``config.AGENT_PHASE_DEFAULT`` when the row is absent or the DB pool is
    unavailable (pre-startup, migrations, tests)."""
    default = getattr(config, "AGENT_PHASE_DEFAULT", "learning")
    try:
        row = await get_state(_AGENT_PHASE_KEY)
    except Exception as e:
        logger.debug(f"get_agent_phase: DB read failed ({e}); using default")
        return default
    if not row:
        return default
    value, _ = row
    if value not in VALID_PHASES:
        logger.warning(f"get_agent_phase: invalid stored value {value!r}; using default")
        return default
    return value


async def set_agent_phase(phase: str) -> datetime:
    if phase not in VALID_PHASES:
        raise ValueError(f"invalid phase: {phase!r} (expected one of {VALID_PHASES})")
    return await set_state(_AGENT_PHASE_KEY, phase)


# ---------- LLM provider helpers ----------

_LLM_PROVIDER_KEY = "llm_provider"


async def get_llm_provider_name() -> str:
    """Return the current agent-LLM provider name. Falls back to
    ``config.LLM_PROVIDER_DEFAULT`` when the row is absent or the DB pool is
    unavailable (pre-startup, migrations, tests)."""
    default = getattr(config, "LLM_PROVIDER_DEFAULT", "vllm")
    try:
        row = await get_state(_LLM_PROVIDER_KEY)
    except Exception as e:
        logger.debug(f"get_llm_provider_name: DB read failed ({e}); using default")
        return default
    if not row:
        return default
    value, _ = row
    if value not in VALID_LLM_PROVIDERS:
        logger.warning(
            f"get_llm_provider_name: invalid stored value {value!r}; using default"
        )
        return default
    return value


async def set_llm_provider_name(provider: str) -> datetime:
    if provider not in VALID_LLM_PROVIDERS:
        raise ValueError(
            f"invalid provider: {provider!r} (expected one of {VALID_LLM_PROVIDERS})"
        )
    return await set_state(_LLM_PROVIDER_KEY, provider)
