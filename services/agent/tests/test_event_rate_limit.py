"""Tests for the per-item event rate limiter."""
from __future__ import annotations

import time

from selene_agent.autonomy.event_rate_limit import EventRateLimiter


def test_no_spec_allows_all():
    lim = EventRateLimiter()
    for _ in range(50):
        assert lim.try_consume("item-a", None) is True


def test_malformed_spec_allows_all():
    lim = EventRateLimiter()
    assert lim.try_consume("item-a", "garbage") is True
    assert lim.try_consume("item-a", "10/eons") is True
    assert lim.try_consume("item-a", "-1/sec") is True


def test_burst_capacity_enforced_per_minute():
    lim = EventRateLimiter()
    spec = "3/min"
    # First 3 consume; 4th blocks.
    assert lim.try_consume("item-a", spec) is True
    assert lim.try_consume("item-a", spec) is True
    assert lim.try_consume("item-a", spec) is True
    assert lim.try_consume("item-a", spec) is False


def test_separate_items_have_independent_buckets():
    lim = EventRateLimiter()
    spec = "1/sec"
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("b", spec) is True
    assert lim.try_consume("a", spec) is False
    assert lim.try_consume("b", spec) is False


def test_sec_shorthand_refills():
    lim = EventRateLimiter()
    spec = "2/sec"
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is False
    # Refills at 2 tokens/sec → after ~0.6s, one token available.
    time.sleep(0.6)
    assert lim.try_consume("a", spec) is True


def test_hour_shorthand_parses():
    lim = EventRateLimiter()
    # 3/hr = 3 burst, ~0.00083 tok/s refill — just confirm parse + first 3 pass.
    spec = "3/hr"
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is False


def test_reset_clears_bucket():
    lim = EventRateLimiter()
    spec = "1/min"
    assert lim.try_consume("a", spec) is True
    assert lim.try_consume("a", spec) is False
    lim.reset("a")
    assert lim.try_consume("a", spec) is True


def test_spec_change_reinits_bucket():
    lim = EventRateLimiter()
    assert lim.try_consume("a", "1/min") is True
    assert lim.try_consume("a", "1/min") is False
    # Switching to a wider spec resets the bucket for this item.
    assert lim.try_consume("a", "5/min") is True
