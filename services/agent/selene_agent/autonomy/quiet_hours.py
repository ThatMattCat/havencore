"""Quiet-hours evaluator.

A tz-aware window specified by ``{start: "HH:MM", end: "HH:MM", policy}``
against ``config.CURRENT_TIMEZONE``. Supports cross-midnight windows
(e.g. 22:00 → 07:00). On malformed input, reports "not quiet" so a bad
config fails open rather than silencing the engine.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from selene_agent.utils import config


def _zone() -> Optional[Any]:
    tz_name = (config.CURRENT_TIMEZONE or "").strip() or "UTC"
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return None


def _parse_hhmm(s: str) -> Optional[time]:
    try:
        hh, mm = s.split(":", 1)
        return time(int(hh), int(mm))
    except Exception:
        return None


def _normalize(spec: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(spec, dict):
        return None
    start = _parse_hhmm(str(spec.get("start") or ""))
    end = _parse_hhmm(str(spec.get("end") or ""))
    if start is None or end is None:
        return None
    policy = spec.get("policy") or "defer"
    if policy not in ("defer", "drop"):
        policy = "defer"
    return {"start": start, "end": end, "policy": policy}


def _is_in_window(local_now: datetime, start: time, end: time) -> bool:
    t = local_now.time()
    if start == end:
        return False
    if start < end:
        return start <= t < end
    # Cross-midnight: quiet when t >= start OR t < end.
    return t >= start or t < end


def policy(spec: Any) -> str:
    """Extract the configured policy, defaulting to ``defer``."""
    n = _normalize(spec)
    return n["policy"] if n else "defer"


def is_quiet(now: datetime, spec: Any) -> bool:
    n = _normalize(spec)
    if n is None:
        return False
    tz = _zone()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(tz) if tz else now
    return _is_in_window(local_now, n["start"], n["end"])


def next_end_at(now: datetime, spec: Any) -> Optional[datetime]:
    """Return the next time the quiet window ends, in UTC.

    Returns None if the spec is malformed.
    """
    n = _normalize(spec)
    if n is None:
        return None
    tz = _zone()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(tz) if tz else now
    end = n["end"]
    candidate = local_now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate = candidate + timedelta(days=1)
    return candidate.astimezone(timezone.utc)
