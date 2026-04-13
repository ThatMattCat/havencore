"""Tests for the pure-function decay/boost math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from selene_agent.autonomy import memory_math


def test_importance_effective_fresh_entry_returns_near_base():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now  # zero age
    got = memory_math.compute_importance_effective(
        base_importance=3,
        created_at=created,
        access_count=0,
        now=now,
        half_life_days=60,
        access_coef=0.5,
    )
    # decay=1, boost=0 -> 3.0
    assert got == pytest.approx(3.0, abs=0.01)


def test_importance_effective_halves_after_half_life():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now - timedelta(days=60)
    got = memory_math.compute_importance_effective(
        base_importance=4,
        created_at=created,
        access_count=0,
        now=now,
        half_life_days=60,
        access_coef=0.5,
    )
    # 4 * e^(-60/60) ~= 4 * 0.3679 ~= 1.47
    assert got == pytest.approx(4 * 2.718281828 ** -1, abs=0.01)


def test_importance_effective_access_boost_is_logarithmic():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now
    base = memory_math.compute_importance_effective(
        base_importance=2, created_at=created, access_count=0,
        now=now, half_life_days=60, access_coef=0.5,
    )
    more = memory_math.compute_importance_effective(
        base_importance=2, created_at=created, access_count=10,
        now=now, half_life_days=60, access_coef=0.5,
    )
    # log(1+10) * 0.5 ~= 1.2
    import math
    assert (more - base) == pytest.approx(0.5 * math.log(1 + 10), abs=0.01)


def test_importance_effective_clamps_to_zero_ten():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now - timedelta(days=3650)  # very old
    got = memory_math.compute_importance_effective(
        base_importance=1, created_at=created, access_count=0,
        now=now, half_life_days=60, access_coef=0.5,
    )
    assert got >= 0.0
    got_hi = memory_math.compute_importance_effective(
        base_importance=5, created_at=now, access_count=10**8,
        now=now, half_life_days=60, access_coef=0.5,
    )
    assert got_hi <= 10.0
