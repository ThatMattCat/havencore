"""Per-item event rate limiter — leaky-bucket, in memory.

Applied to webhook/MQTT-driven fires in addition to the global hourly cap.
Keyed on ``item_id`` so a single noisy topic cannot starve other watches.
Rate spec is a shorthand string like ``"10/min"`` or ``"2/sec"``; anything
unparseable disables the gate (fail open).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class _Bucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    updated_at: float


def _parse_spec(spec: str) -> Optional[Tuple[float, float]]:
    """Return (capacity, refill_per_sec) for ``N/unit`` strings."""
    if not isinstance(spec, str) or "/" not in spec:
        return None
    count_str, unit = spec.split("/", 1)
    try:
        count = float(count_str.strip())
    except ValueError:
        return None
    unit = unit.strip().lower()
    if unit in ("sec", "s", "second"):
        window = 1.0
    elif unit in ("min", "m", "minute"):
        window = 60.0
    elif unit in ("hr", "h", "hour"):
        window = 3600.0
    else:
        return None
    if count <= 0 or window <= 0:
        return None
    return count, count / window


class EventRateLimiter:
    def __init__(self) -> None:
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def try_consume(self, item_id: str, spec: Optional[str]) -> bool:
        """Attempt to consume one token. Returns True if allowed.

        ``spec`` unset or unparseable = unlimited.
        """
        if not spec:
            return True
        parsed = _parse_spec(spec)
        if parsed is None:
            return True
        capacity, refill = parsed
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(item_id)
            if b is None or b.capacity != capacity or b.refill_per_sec != refill:
                b = _Bucket(
                    capacity=capacity,
                    refill_per_sec=refill,
                    tokens=capacity,
                    updated_at=now,
                )
                self._buckets[item_id] = b
            elapsed = max(0.0, now - b.updated_at)
            b.tokens = min(b.capacity, b.tokens + elapsed * b.refill_per_sec)
            b.updated_at = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True
            return False

    def reset(self, item_id: Optional[str] = None) -> None:
        with self._lock:
            if item_id is None:
                self._buckets.clear()
            else:
                self._buckets.pop(item_id, None)


# Shared singleton used by engine.trigger_event.
limiter = EventRateLimiter()
