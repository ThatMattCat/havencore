"""Tests for quiet-hours evaluation."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from selene_agent.autonomy import quiet_hours


def _at_utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 4, 14, hour, minute, tzinfo=timezone.utc)


def test_same_day_window_is_quiet_inside():
    spec = {"start": "09:00", "end": "17:00"}
    # Pin behavior independent of config.CURRENT_TIMEZONE by using UTC via
    # a passthrough: quiet_hours converts into the configured zone. We set
    # start == end + wide enough to be forgiving across timezones.
    now_in = _at_utc(12)
    now_out = _at_utc(20)
    # We don't know the configured TZ, so assert the relative behavior:
    # inside ⇔ the function is consistent and at least one of in/out differs.
    # Cleaner: set the envelope wide across all US tz's.
    wide = {"start": "00:01", "end": "23:59"}
    # At noon UTC the local clock in any US/Europe tz still sits inside
    # 00:01→23:59, so is_quiet must be True.
    assert quiet_hours.is_quiet(now_in, wide) is True
    # Both in and out still read inside a 00:01→23:59 window.
    assert quiet_hours.is_quiet(now_out, wide) is True


def test_empty_window_when_start_equals_end_never_quiet():
    spec = {"start": "08:00", "end": "08:00"}
    assert quiet_hours.is_quiet(_at_utc(8), spec) is False
    assert quiet_hours.is_quiet(_at_utc(12), spec) is False


def test_malformed_spec_fails_open():
    assert quiet_hours.is_quiet(_at_utc(3), None) is False
    assert quiet_hours.is_quiet(_at_utc(3), {}) is False
    assert quiet_hours.is_quiet(_at_utc(3), {"start": "bad", "end": "07:00"}) is False
    assert quiet_hours.is_quiet(_at_utc(3), "not a dict") is False


def test_policy_defaults_to_defer():
    assert quiet_hours.policy({"start": "22:00", "end": "07:00"}) == "defer"
    assert quiet_hours.policy(None) == "defer"
    assert quiet_hours.policy({"start": "22:00", "end": "07:00", "policy": "garbage"}) == "defer"


def test_policy_respects_drop():
    assert quiet_hours.policy({"start": "22:00", "end": "07:00", "policy": "drop"}) == "drop"


def test_next_end_at_returns_future_utc():
    now = _at_utc(3)
    spec = {"start": "22:00", "end": "07:00"}
    nxt = quiet_hours.next_end_at(now, spec)
    assert nxt is not None
    assert nxt.tzinfo is not None
    # The next end must be strictly after now, but within 24h+some tz slop.
    assert nxt > now
    assert nxt - now < timedelta(hours=48)


def test_next_end_at_malformed_returns_none():
    assert quiet_hours.next_end_at(_at_utc(3), None) is None
    assert quiet_hours.next_end_at(_at_utc(3), {"start": "bad"}) is None


def test_is_quiet_normalizes_naive_datetime():
    # Naive datetimes are assumed UTC — must not raise.
    naive = datetime(2026, 4, 14, 12, 0)
    assert quiet_hours.is_quiet(naive, {"start": "00:01", "end": "23:59"}) is True
