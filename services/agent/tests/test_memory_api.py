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
