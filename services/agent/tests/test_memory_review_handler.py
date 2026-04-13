"""Tests for the memory_review handler pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _stub_point(pid, text, importance, tier="L2", created="2026-04-13T00:00:00+00:00",
                access_count=0, source_ids=None, importance_effective=None):
    p = MagicMock()
    p.id = pid
    p.payload = {
        "text": text,
        "timestamp": created,
        "importance": importance,
        "importance_effective": importance_effective
            if importance_effective is not None else importance,
        "access_count": access_count,
        "tier": tier,
        "source_ids": source_ids or [],
        "tags": [],
    }
    p.vector = [0.0] * 8
    return p


@pytest.fixture
def qdrant_stub():
    c = MagicMock()
    c.scroll = MagicMock()
    c.set_payload = MagicMock()
    c.upsert = MagicMock()
    c.delete = MagicMock()
    return c


@pytest.mark.asyncio
async def test_scan_and_decay_updates_importance_effective(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # Two L2 points: one fresh, one 60d old, same base importance.
    pts = [
        _stub_point("fresh", "x", importance=4,
                    created="2026-04-13T00:00:00+00:00"),
        _stub_point("old", "x", importance=4,
                    created="2026-02-12T00:00:00+00:00"),
    ]
    qdrant_stub.scroll.return_value = (pts, None)

    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l2_scanned": 0, "importance_adjusted": 0}
    await memory_review._scan_and_decay(qdrant_stub, stats)
    assert stats["l2_scanned"] == 2
    assert stats["importance_adjusted"] == 2
    # set_payload was called with per-id updates; collect values.
    calls = qdrant_stub.set_payload.call_args_list
    got = {}
    for call in calls:
        payload = call.kwargs["payload"]
        for pid in call.kwargs["points"]:
            got[pid] = payload["importance_effective"]
    assert got["fresh"] == pytest.approx(4.0, abs=0.01)
    # 60 days -> half-life decay ~= 4 * 1/e ~= 1.47
    assert got["old"] == pytest.approx(4 * 2.718281828 ** -1, abs=0.05)
