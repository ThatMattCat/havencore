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


@pytest.mark.asyncio
async def test_cluster_step_creates_l3_with_source_ids(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review
    from selene_agent.autonomy import memory_clustering

    # 6 L2 points, all recent. Pretend HDBSCAN clusters them into label 0.
    pts = [_stub_point(f"e{i}", "text", 3) for i in range(6)]
    for p in pts:
        p.vector = [float(i) for i in range(8)]
    qdrant_stub.scroll.return_value = (pts, None)

    monkeypatch.setattr(memory_clustering, "cluster_vectors", lambda v, **k: [0] * len(v))
    async def _summarize(**kw):
        return {"summary": "unified topic", "tags": ["t1", "t2"], "rationale": "because"}
    monkeypatch.setattr(memory_clustering, "summarize_cluster", _summarize)

    # Embedding service stub.
    monkeypatch.setattr(memory_review, "_embed", lambda text: [0.0] * 1024)

    stats = {"l3_created": 0, "clusters_found": 0, "noise_points": 0, "llm_calls": 0}
    since = datetime(2026, 4, 1, tzinfo=timezone.utc)
    await memory_review._cluster_to_l3(
        qdrant_stub, stats,
        since=since, llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l3_created"] == 1
    assert stats["clusters_found"] == 1
    # upsert was called once with a PointStruct whose payload has tier=L3 + source_ids.
    upsert_call = qdrant_stub.upsert.call_args
    point = upsert_call.kwargs["points"][0]
    assert point.payload["tier"] == "L3"
    assert set(point.payload["source_ids"]) == {f"e{i}" for i in range(6)}


@pytest.mark.asyncio
async def test_cluster_step_skips_when_too_few_new_points(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    qdrant_stub.scroll.return_value = ([_stub_point("a", "x", 3)], None)
    stats = {"l3_created": 0, "clusters_found": 0, "noise_points": 0, "llm_calls": 0}
    await memory_review._cluster_to_l3(
        qdrant_stub, stats,
        since=datetime(2026, 4, 1, tzinfo=timezone.utc),
        llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l3_created"] == 0
    qdrant_stub.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_propose_l4_flags_eligible_l3(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # L3 candidate: old enough, important enough, accessed enough.
    old_enough = "2026-03-15T00:00:00+00:00"
    candidate = _stub_point(
        "l3a", "core preference", importance=5,
        tier="L3", created=old_enough, access_count=5,
        importance_effective=5.0,
    )
    qdrant_stub.scroll.return_value = ([candidate], None)
    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l4_proposed": 0, "llm_calls": 0}
    await memory_review._propose_l4(
        qdrant_stub, stats,
        llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l4_proposed"] == 1
    call = qdrant_stub.set_payload.call_args
    payload = call.kwargs["payload"]
    assert payload["pending_l4_approval"] is True
    assert payload["proposed_at"] is not None


@pytest.mark.asyncio
async def test_prune_respects_source_protection(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # L3 references "protected". Both "protected" and "unprotected" are
    # stale+low-importance L2, but only "unprotected" should be deleted.
    l3 = _stub_point("l3", "x", 3, tier="L3", source_ids=["protected"])
    l2_protected = _stub_point(
        "protected", "p", importance=1, tier="L2",
        created="2025-09-01T00:00:00+00:00",
        importance_effective=0.1,
    )
    l2_free = _stub_point(
        "unprotected", "u", importance=1, tier="L2",
        created="2025-09-01T00:00:00+00:00",
        importance_effective=0.1,
    )

    def _scroll_side_effect(**kw):
        flt = kw.get("scroll_filter")
        # Detect tier being filtered by stringifying the filter.
        s = str(flt)
        if "'L3'" in s:
            return ([l3], None)
        return ([l2_protected, l2_free], None)

    qdrant_stub.scroll.side_effect = _scroll_side_effect
    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l2_pruned": 0}
    await memory_review._prune_l2(qdrant_stub, stats)
    assert stats["l2_pruned"] == 1
    delete_call = qdrant_stub.delete.call_args
    ids = delete_call.kwargs["points_selector"].points
    assert ids == ["unprotected"]
