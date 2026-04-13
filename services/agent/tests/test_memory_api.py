"""Tests for /api/memory/* routes — L4 CRUD and proposal queue."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from selene_agent.api.memory import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_list_l4_returns_active_entries(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        p = MagicMock()
        p.id, p.payload = "x", {
            "text": "core", "importance": 5, "importance_effective": 5.0,
            "tier": "L4", "timestamp": "2026-04-01T00:00:00+00:00",
            "tags": [], "pending_l4_approval": False,
        }
        c.scroll.return_value = ([p], None)
        qc.return_value = c
        r = client.get("/api/memory/l4")
    assert r.status_code == 200
    body = r.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["id"] == "x"


def test_approve_promotes_to_l4_and_invalidates_cache(client):
    from selene_agent.utils import l4_context
    with patch("selene_agent.api.memory._qdrant_client") as qc, \
         patch.object(l4_context, "invalidate_cache") as inv:
        c = MagicMock()
        qc.return_value = c
        r = client.post(f"/api/memory/l4/proposals/{uuid.uuid4()}/approve")
    assert r.status_code == 200
    c.set_payload.assert_called_once()
    payload = c.set_payload.call_args.kwargs["payload"]
    assert payload["tier"] == "L4"
    assert payload["pending_l4_approval"] is False
    inv.assert_called_once()


def test_reject_clears_flag_without_promoting(client):
    from selene_agent.utils import l4_context
    with patch("selene_agent.api.memory._qdrant_client") as qc, \
         patch.object(l4_context, "invalidate_cache") as inv:
        c = MagicMock()
        qc.return_value = c
        r = client.post(f"/api/memory/l4/proposals/{uuid.uuid4()}/reject")
    assert r.status_code == 200
    payload = c.set_payload.call_args.kwargs["payload"]
    assert "tier" not in payload  # stays L3
    assert payload["pending_l4_approval"] is False
    inv.assert_called_once()


def test_list_l3_paginates(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        pts = []
        for i in range(3):
            p = MagicMock()
            p.id = f"l{i}"
            p.payload = {
                "text": f"t{i}", "importance": 3, "importance_effective": 3.0,
                "tier": "L3", "timestamp": "2026-04-01T00:00:00+00:00",
                "tags": [], "source_ids": [f"s{i}a", f"s{i}b"],
            }
            pts.append(p)
        c.scroll.return_value = (pts, None)
        qc.return_value = c
        r = client.get("/api/memory/l3?limit=2&offset=1")
    assert r.status_code == 200
    assert len(r.json()["entries"]) == 2


def test_l3_sources_returns_source_l2_entries(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        l3 = MagicMock()
        l3.id, l3.payload = "l3x", {"source_ids": ["a", "b"], "tier": "L3"}
        c.retrieve.side_effect = [[l3], [MagicMock(id="a", payload={"text": "A", "tier": "L2"}),
                                           MagicMock(id="b", payload={"text": "B", "tier": "L2"})]]
        qc.return_value = c
        r = client.get("/api/memory/l3/l3x/sources")
    assert r.status_code == 200
    texts = [s["text"] for s in r.json()["sources"]]
    assert set(texts) == {"A", "B"}


def test_delete_l3_removes_consolidated_entry(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        qc.return_value = c
        r = client.delete("/api/memory/l3/abc")
    assert r.status_code == 200
    c.delete.assert_called_once()


def test_stats_returns_tier_counts(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        c.count.side_effect = [MagicMock(count=n) for n in (100, 10, 2, 3)]
        qc.return_value = c
        r = client.get("/api/memory/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["l2_count"] == 100
    assert body["l3_count"] == 10
    assert body["l4_count"] == 2
    assert body["pending_proposals"] == 3
