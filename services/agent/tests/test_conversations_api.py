"""Tests for /api/conversations/{session_id} — GET (with optional ?id=) and DELETE."""
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
            "session_id": "sid-abc",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello from flush-1"},
            ],
            "created_at": "2026-04-18T10:00:00+00:00",
            "metadata": {"device_name": "Matts Laptop", "message_count": 2},
        },
        {
            "id": 102,
            "session_id": "sid-abc",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello from flush-2"},
            ],
            "created_at": "2026-04-18T11:00:00+00:00",
            "metadata": {"device_name": "Matts Desktop", "message_count": 2},
        },
    ]

    async def fake_get(session_id, limit=100, flush_id=None):
        rows = [r for r in stored if r["session_id"] == session_id]
        if not rows:
            return []
        if flush_id is None:
            return sorted(rows, key=lambda r: r["created_at"], reverse=True)
        return [r for r in rows if r["id"] == flush_id]

    async def fake_delete(session_id, flush_id):
        before = len(stored)
        stored[:] = [
            r for r in stored
            if not (r["session_id"] == session_id and r["id"] == flush_id)
        ]
        return before - len(stored)

    monkeypatch.setattr(conv_api.conversation_db, "get_conversation_history", fake_get)
    monkeypatch.setattr(conv_api.conversation_db, "delete_conversation_history", fake_delete)

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


def test_delete_conversation_removes_one_flush(app):
    client = TestClient(app)
    r = client.delete("/api/conversations/sid-abc?id=101")
    assert r.status_code == 200
    assert r.json() == {"deleted": 1}

    # The remaining flush is still fetchable; the deleted one 404s.
    r = client.get("/api/conversations/sid-abc")
    assert r.status_code == 200
    assert [row["id"] for row in r.json()["conversation"]] == [102]

    r = client.get("/api/conversations/sid-abc?id=101")
    assert r.status_code == 404


def test_delete_conversation_requires_id(app):
    client = TestClient(app)
    r = client.delete("/api/conversations/sid-abc")
    # Missing required ?id= → FastAPI returns 422 (validation error).
    assert r.status_code == 422


def test_delete_conversation_mismatched_id_404s(app):
    client = TestClient(app)
    r = client.delete("/api/conversations/sid-abc?id=9999")
    assert r.status_code == 404


def test_delete_conversation_unknown_session_404s(app):
    client = TestClient(app)
    r = client.delete("/api/conversations/does-not-exist?id=101")
    # The (session_id, id) pair doesn't match any row, so deleted=0 → 404.
    assert r.status_code == 404
