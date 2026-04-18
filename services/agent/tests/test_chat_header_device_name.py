"""Tests for X-Device-Name header (REST) and WS device_name field."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.orchestrator import AgentEvent, AgentOrchestrator, EventType


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

    monkeypatch.setattr("selene_agent.api.chat.metrics_db.record_turn", noop)

    app = FastAPI()
    pool = _FakePool()
    app.state.session_pool = pool
    app.include_router(router, prefix="/api")
    app.include_router(ws_router, prefix="/ws")
    return app, pool


def test_rest_x_device_name_header_applied(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Device-Name": "Kitchen Speaker"},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.device_name == "Kitchen Speaker"


def test_rest_missing_header_leaves_device_name_none(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert pool.orch.device_name is None


def test_rest_empty_header_preserves_existing_name(app_and_pool):
    app, pool = app_and_pool
    pool.orch.device_name = "Office"
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Device-Name": ""},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.device_name == "Office"


def test_rest_rename_via_header(app_and_pool):
    app, pool = app_and_pool
    pool.orch.device_name = "kitchen"
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Device-Name": "Kitchen Speaker"},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.device_name == "Kitchen Speaker"


def test_rest_oversized_header_truncated(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    r = client.post(
        "/api/chat",
        headers={"X-Device-Name": "x" * 200},
        json={"message": "hi"},
    )
    assert r.status_code == 200
    assert pool.orch.device_name == "x" * 64


def test_ws_first_frame_device_name_applied(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "type": "session",
            "session_id": "pinned-sid",
            "device_name": "Office",
        })
        announcement = ws.receive_json()
        assert announcement["type"] == "session"
        assert pool.orch.device_name == "Office"


def test_ws_first_frame_without_device_name_leaves_none(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"type": "session", "session_id": "pinned-sid"})
        ws.receive_json()
        assert pool.orch.device_name is None


def test_ws_mid_stream_device_name_update(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "type": "session",
            "session_id": "pinned-sid",
            "device_name": "WebTab",
        })
        ws.receive_json()
        assert pool.orch.device_name == "WebTab"

        ws.send_json({"type": "session", "device_name": "WebTab2"})
        assert pool.orch.device_name == "WebTab2"


def test_ws_mid_stream_omitted_field_preserves_name(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "type": "session",
            "session_id": "pinned-sid",
            "device_name": "Stable",
        })
        ws.receive_json()

        # A mid-stream session frame without device_name should not clobber.
        ws.send_json({"type": "session", "idle_timeout": 60})
        assert pool.orch.device_name == "Stable"


def test_ws_non_string_device_name_in_frame_ignored(app_and_pool):
    app, pool = app_and_pool
    client = TestClient(app)

    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({
            "type": "session",
            "session_id": "pinned-sid",
            "device_name": 123,
        })
        ws.receive_json()
        assert pool.orch.device_name is None
