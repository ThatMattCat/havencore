"""Cron scheduling helpers for the autonomy engine.

All timestamps are stored in UTC. Cron strings are interpreted in
``config.CURRENT_TIMEZONE`` (matching the convention used by the user-facing
orchestrator) before being converted to UTC for storage.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from croniter import croniter

from selene_agent.utils import config


def _local_tz() -> ZoneInfo:
    return ZoneInfo(config.CURRENT_TIMEZONE or "UTC")


def next_fire_at(cron_expr: str, *, after: Optional[datetime] = None) -> datetime:
    """Return the next firing time for ``cron_expr`` as a UTC-aware datetime.

    ``after`` may be naive or tz-aware; if naive, it is treated as UTC.
    """
    if after is None:
        after = datetime.now(timezone.utc)
    elif after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)

    local_after = after.astimezone(_local_tz())
    itr = croniter(cron_expr, local_after)
    local_next = itr.get_next(datetime)
    if local_next.tzinfo is None:
        local_next = local_next.replace(tzinfo=_local_tz())
    return local_next.astimezone(timezone.utc)


def validate_cron(cron_expr: str) -> bool:
    """Cheap syntactic validation."""
    try:
        croniter(cron_expr)
        return True
    except Exception:
        return False
