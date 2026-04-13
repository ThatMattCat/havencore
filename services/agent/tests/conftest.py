"""Shared pytest fixtures for the agent test suite."""
from __future__ import annotations

import pytest


@pytest.fixture
def frozen_now():
    """A fixed UTC datetime for deterministic age math in tests."""
    from datetime import datetime, timezone
    return datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
