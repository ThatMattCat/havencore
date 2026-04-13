"""Pure functions for memory importance dynamics.

No Qdrant, no I/O. Unit-testable in isolation.
"""
from __future__ import annotations

import math
from datetime import datetime


def compute_importance_effective(
    *,
    base_importance: float,
    created_at: datetime,
    access_count: int,
    now: datetime,
    half_life_days: float,
    access_coef: float,
) -> float:
    """Effective importance = base * exp(-age / half_life) + coef * log(1 + accesses).

    Clamped to [0, 10]. Half-life controls decay; access_coef controls boost
    per order-of-magnitude of retrievals.
    """
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    decay = math.exp(-age_days / half_life_days) if half_life_days > 0 else 1.0
    boost = access_coef * math.log(1 + max(0, access_count))
    value = base_importance * decay + boost
    return max(0.0, min(10.0, value))
