"""Tests for GET /api/conversations/{session_id} with optional ?id= flush filter."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def app(monkeypatch):
    from selene_agent.api import conversations as conv_api

    stored = [
        {
            "id": 101,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello from flush-1"},
            ],
            "created_at": "2026-04-18T10:00:00+00:00",
            "metadata": {"device_name": "Matts Laptop", "message_count": 2},
        },
        {
            "id": 102,
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello from flush-2"},
            ],
            "created_at": "2026-04-18T11:00:00+00:00",
            "metadata": {"device_name": "Matts Desktop", "message_count": 2},
        },
    ]

    async def fake_get(session_id, limit=100, flush_id=None):
        if session_id != "sid-abc":
            return []
        if flush_id is None:
            return sorted(stored, key=lambda r: r["created_at"], reverse=True)
        return [r for r in stored if r["id"] == flush_id]

    monkeypatch.setattr(conv_api.conversation_db, "get_conversation_history", fake_get)

    app = FastAPI()
    app.include_router(conv_api.router, prefix="/api")
    return app


def test_get_conversation_without_id_returns_all_flushes(app):
    client = TestClient(app)
    r = client.get("/api/conversations/sid-abc")
    assert r.status_code == 200
    body = r.json()["conversation"]
    assert [row["id"] for row in body] == [102, 101]  # newest first


def test_get_conversation_with_id_returns_single_flush(app):
    client = TestClient(app)
    r = client.get("/api/conversations/sid-abc?id=101")
    assert r.status_code == 200
    body = r.json()["conversation"]
    assert len(body) == 1
    assert body[0]["id"] == 101
    assert body[0]["metadata"]["device_name"] == "Matts Laptop"


def test_get_conversation_with_mismatched_id_404s(app):
    client = TestClient(app)
    r = client.get("/api/conversations/sid-abc?id=9999")
    assert r.status_code == 404


def test_get_conversation_unknown_session_404s(app):
    client = TestClient(app)
    r = client.get("/api/conversations/does-not-exist")
    assert r.status_code == 404
