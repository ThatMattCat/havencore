"""Tests for MetricsDB.record_turn / fetch_recent_turns device_name plumbing."""
from __future__ import annotations

import datetime as _dt
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from selene_agent.utils.metrics_db import ENSURE_TABLE_SQL, MetricsDB


class _FakeConn:
    def __init__(self):
        self.execute = AsyncMock()
        self.fetch = AsyncMock(return_value=[])


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        @asynccontextmanager
        async def _ctx():
            yield self._conn
        return _ctx()


@pytest.fixture
def fake_conn(monkeypatch):
    conn = _FakeConn()
    pool = _FakePool(conn)
    monkeypatch.setattr(
        "selene_agent.utils.metrics_db.conversation_db.pool",
        pool,
    )
    return conn


def test_ensure_table_sql_includes_device_name_alter():
    """The ALTER must be present so live deployments gain the column at startup."""
    assert "ADD COLUMN IF NOT EXISTS device_name TEXT" in ENSURE_TABLE_SQL


async def test_record_turn_writes_device_name(fake_conn):
    db = MetricsDB()
    payload = {
        "llm_ms": 100,
        "tool_ms_total": 50,
        "total_ms": 200,
        "iterations": 1,
        "tool_calls": [],
    }
    await db.record_turn("sess-1", payload, device_name="Kitchen Speaker")

    fake_conn.execute.assert_awaited_once()
    args = fake_conn.execute.call_args.args
    # Positional args after the SQL: session_id, llm_ms, tool_ms_total,
    # total_ms, iterations, tool_calls(json), device_name.
    assert args[1] == "sess-1"
    assert args[-1] == "Kitchen Speaker"


async def test_record_turn_default_device_name_is_null(fake_conn):
    db = MetricsDB()
    payload = {
        "llm_ms": 1,
        "tool_ms_total": 0,
        "total_ms": 1,
        "iterations": 1,
        "tool_calls": [],
    }
    await db.record_turn("sess-2", payload)

    fake_conn.execute.assert_awaited_once()
    args = fake_conn.execute.call_args.args
    assert args[-1] is None


async def test_fetch_recent_turns_includes_device_name(fake_conn):
    fake_conn.fetch.return_value = [
        {
            "id": 1,
            "session_id": "sess-1",
            "created_at": _dt.datetime(2026, 4, 18, 9, 14, 0),
            "llm_ms": 100,
            "tool_ms_total": 50,
            "total_ms": 200,
            "iterations": 1,
            "tool_calls": [],
            "device_name": "Kitchen Speaker",
        },
        {
            "id": 2,
            "session_id": "sess-2",
            "created_at": _dt.datetime(2026, 4, 18, 9, 15, 0),
            "llm_ms": 100,
            "tool_ms_total": 0,
            "total_ms": 100,
            "iterations": 1,
            "tool_calls": [],
            "device_name": None,
        },
    ]
    db = MetricsDB()
    rows = await db.fetch_recent_turns(limit=2)

    assert len(rows) == 2
    assert rows[0]["device_name"] == "Kitchen Speaker"
    assert rows[1]["device_name"] is None
