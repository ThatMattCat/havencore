"""Tests for X-Idle-Timeout header (REST) and WS idle_timeout field."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.orchestrator import AgentEvent, AgentOrchestrator, EventType
from selene_agent.utils import config


class _FakePool:
    def __init__(self):
        self.orch = AgentOrchestrator(
            client=MagicMock(),
            mcp_manager=MagicMock(),
            model_name="test-model",
            tools=[],
            session_id="pinned-sid",
        )
        self.orch.messages = [{"role": "system", "content": "sys"}]

        async def fake_run(user_message):
            yield AgentEvent(type=EventType.DONE, data={"content": "ok"})

        self.orch.run = fake_run  # type: ignore[assignment]

    async def get_or_create(self, sid):
        return self.orch

    def lock_for(self, sid):
        import asyncio
        return asyncio.Lock()


@pytest.fixture
def app_and_pool(monkeypatch):
    from selene_agent.api.chat import router, ws_router

    async def noop(*a, **kw):
        return None

    monkeypatch.setattr(
        "selene_agent.api.chat.metrics_db.record_turn", noop
    )

    app = FastAPI()
    pool = _FakePool()
    app.state.session_pool = pool
    app.include_router(router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    return app, pool


def test_rest_x_idle_timeout_header_applied(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Idle-Timeout": "60"},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.idle_timeout_override == 60


def test_rest_bad_x_idle_timeout_ignored(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Idle-Timeout": "abc"},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.idle_timeout_override is None


def test_rest_clamps_x_idle_timeout(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Idle-Timeout": "99999"},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.idle_timeout_override == config.CONVERSATION_TIMEOUT_MAX


def test_rest_missing_header_leaves_override_none(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert pool.orch.idle_timeout_override is None


def test_ws_first_frame_idle_timeout_applied(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "session", "session_id": "pinned-sid", "idle_timeout": 45})
        # Server announces the session before any subsequent activity.
        announcement = ws.receive_json()
        assert announcement["type"] == "session"
        assert pool.orch.idle_timeout_override == 45


def test_ws_mid_stream_idle_timeout_update(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "session", "session_id": "pinned-sid", "idle_timeout": 30})
        ws.receive_json()  # session announcement
        assert pool.orch.idle_timeout_override == 30

        # Mid-stream update: second session frame bumps the override.
        ws.send_json({"type": "session", "idle_timeout": 120})
        assert pool.orch.idle_timeout_override == 120


def test_ws_bad_idle_timeout_in_frame_ignored(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "session", "session_id": "pinned-sid", "idle_timeout": "nope"})
        ws.receive_json()
        assert pool.orch.idle_timeout_override is None
