"""The act-confirmation token must never reach a read-only client surface.

``confirmation_token`` is a bearer secret: whoever holds it can approve a
parked ``act`` run, which fires real Home Assistant actuators. It used to be
serialized by ``autonomy_db._row_to_run`` unconditionally, so
``GET /api/autonomy/runs`` and both send sites of the ``/ws/autonomy/runs``
feed published it verbatim — while ``/runs/awaiting`` and ``/runs/{id}``
popped it back off with comments insisting it had to stay hidden.

These tests drive the *real* db serializer (only the asyncpg pool is faked),
so they fail the moment the token is re-added to any of those payloads.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.api import autonomy as autonomy_api
from selene_agent.utils.conversation_db import conversation_db

SECRET = "s3cret-confirmation-token-value"


# --- fake asyncpg pool ----------------------------------------------------

def _run_row(status: str = "awaiting_confirmation", kind: str = "act") -> dict:
    """An ``autonomy_runs`` row as asyncpg would hand it back.

    A plain dict satisfies everything ``_row_to_run`` asks of a Record
    (``.keys()`` + ``__getitem__``). The token is present here on purpose: the
    point of the test is that the serializer drops it, not that some SELECT
    happens to omit the column.
    """
    now = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "id": uuid.uuid4(),
        "agenda_item_id": uuid.uuid4(),
        "kind": kind,
        "triggered_at": now,
        "completed_at": None,
        "status": status,
        "summary": "act: 1 pending",
        "severity": None,
        "signature_hash": "sig",
        "notified_via": "signal",
        "messages": [],
        "metrics": {},
        "error": None,
        "scheduled_for": None,
        "trigger_source": "cron",
        "trigger_event": None,
        "confirmation_token": SECRET,
        "confirmation_prompt_id": None,
        "confirmation_response": None,
        "action_audit": [{"tool": "ha_control_light", "outcome": "pending"}],
    }


class _FakeAcquire:
    """asyncpg's PoolAcquireContext is both awaitable and an async CM; the
    autonomy code uses it both ways (``async with`` in db.py, bare ``await``
    in the WS handler)."""

    def __init__(self, conn):
        self._conn = conn

    def __await__(self):
        async def _get():
            return self._conn
        return _get().__await__()

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)

    async def release(self, conn):
        return None


class _FakeConn:
    def __init__(self, rows, notify_with=None):
        self.rows = rows
        # When set, adding the LISTEN callback immediately pushes this run id
        # onto the handler's queue — that is the "live update" path.
        self.notify_with = notify_with
        self.removed = False

    async def fetch(self, query, *args):
        return list(self.rows)

    async def fetchrow(self, query, *args):
        want = str(args[0]) if args else None
        for row in self.rows:
            if str(row["id"]) == want:
                return row
        return None

    async def add_listener(self, channel, callback):
        if self.notify_with is not None:
            callback(None, 0, channel, self.notify_with)

    async def remove_listener(self, channel, callback):
        self.removed = True


@pytest.fixture
def rest_client(monkeypatch):
    row = _run_row()
    monkeypatch.setattr(conversation_db, "pool", _FakePool(_FakeConn([row])))
    app = FastAPI()
    app.include_router(autonomy_api.router, prefix="/api")
    return TestClient(app), row


# --- REST ------------------------------------------------------------------

def test_runs_listing_does_not_leak_confirmation_token(rest_client):
    client, row = rest_client
    resp = client.get("/api/autonomy/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["runs"]) == 1
    # The run really is the parked act run whose token we planted.
    assert body["runs"][0]["status"] == "awaiting_confirmation"
    assert "confirmation_token" not in body["runs"][0]
    # Belt and braces: the secret must not appear anywhere in the payload.
    assert SECRET not in resp.text


def test_awaiting_listing_does_not_leak_confirmation_token(rest_client):
    client, _ = rest_client
    resp = client.get("/api/autonomy/runs/awaiting")
    assert resp.status_code == 200
    assert "confirmation_token" not in resp.json()["runs"][0]
    assert SECRET not in resp.text


def test_single_run_does_not_leak_confirmation_token(rest_client):
    client, row = rest_client
    resp = client.get(f"/api/autonomy/runs/{row['id']}")
    assert resp.status_code == 200
    assert "confirmation_token" not in resp.json()["run"]
    assert SECRET not in resp.text


# --- WebSocket feed --------------------------------------------------------

def test_ws_feed_does_not_leak_token_on_primed_or_live_rows(monkeypatch):
    """Both send sites of /ws/autonomy/runs: the primed backlog and the
    LISTEN-driven live update."""
    primed = _run_row()
    live = _run_row()
    conn = _FakeConn([primed, live], notify_with=str(live["id"]))
    monkeypatch.setattr(conversation_db, "pool", _FakePool(conn))

    app = FastAPI()
    app.include_router(autonomy_api.ws_router, prefix="/ws")
    client = TestClient(app)

    with client.websocket_connect("/ws/autonomy/runs") as ws:
        # Priming replays list_runs (both rows), then the live update lands.
        seen = [ws.receive_text() for _ in range(3)]

    assert len(seen) == 3
    live_frame = json.loads(seen[-1])
    assert live_frame["type"] == "run"
    assert live_frame["run"]["id"] == str(live["id"])
    for raw in seen:
        frame = json.loads(raw)
        assert frame["type"] == "run"
        assert "confirmation_token" not in frame["run"]
        assert SECRET not in raw


# --- serializer contract ---------------------------------------------------

def test_row_to_run_hides_token_by_default_and_opts_in_explicitly():
    from selene_agent.autonomy.db import _row_to_run

    row = _run_row()
    assert "confirmation_token" not in _row_to_run(row)
    # The confirm path — the only caller that validates the token — can ask.
    assert _row_to_run(row, include_token=True)["confirmation_token"] == SECRET
