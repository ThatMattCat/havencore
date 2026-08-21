"""Quiet-hours deferral coalescing (#44).

Exercises the real SQL against the live Postgres, since coalescing is a single
atomic UPDATE — an in-memory fake would pass either way.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils.conversation_db import conversation_db


@pytest.fixture
async def pool(monkeypatch):
    """A pool bound to *this* test's event loop.

    pytest-asyncio hands each test a fresh loop, so a shared asyncpg pool from
    an earlier test is already closed. Build one per test and point the module
    the code under test reads (``conversation_db.pool``) at it.
    """
    import asyncpg

    try:
        p = await asyncpg.create_pool(**conversation_db.db_config, min_size=1, max_size=4)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"postgres unavailable: {e}")
    monkeypatch.setattr(conversation_db, "pool", p)
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
async def item_id(pool):
    """A throwaway agenda item; runs FK to it.

    Inserted **disabled** and never given a next_fire_at so the live engine
    cannot pick it up mid-test, and dropped again on teardown.
    """
    iid = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO agenda_items (id, kind, schedule_cron, config, autonomy_level, enabled)
               VALUES ($1, 'act', '0 8 * * *', '{}'::jsonb, 'act', false)""",
            iid,
        )
    yield str(iid)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM autonomy_runs WHERE agenda_item_id = $1", iid)
        await conn.execute("DELETE FROM agenda_items WHERE id = $1", iid)


async def _mk_run(pool, item_id, status, **extra):
    run_id = await autonomy_db.insert_run(
        {"agenda_item_id": item_id, "kind": "act", "status": status, **extra}
    )
    return run_id


async def test_repeated_deferrals_coalesce_into_one_row(pool, item_id):
    """A long quiet window must leave exactly one pending placeholder."""
    quiet_end = datetime.now(timezone.utc) + timedelta(hours=8)

    assert await autonomy_db.coalesce_deferred_run(item_id) is False  # nothing yet
    await _mk_run(pool, item_id, "scheduled", scheduled_for=quiet_end,
                  summary="deferred by quiet hours")

    for _ in range(35):
        assert await autonomy_db.coalesce_deferred_run(item_id) is True

    async with pool.acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM autonomy_runs WHERE agenda_item_id=$1 AND status='scheduled'",
            uuid.UUID(item_id),
        )
        coalesced = await conn.fetchval(
            """SELECT (metrics->>'coalesced')::int FROM autonomy_runs
               WHERE agenda_item_id=$1 AND status='scheduled'""",
            uuid.UUID(item_id),
        )
    assert n == 1, f"expected 1 pending deferral, got {n}"
    assert coalesced == 36, f"suppressed count not recorded: {coalesced}"


async def test_coalesce_ignores_non_scheduled_runs(pool, item_id):
    """Completed history must not absorb a new deferral."""
    await _mk_run(pool, item_id, "ok", completed_at=datetime.now(timezone.utc))
    assert await autonomy_db.coalesce_deferred_run(item_id) is False


async def test_coalesce_is_per_item(pool, item_id):
    """One item's pending deferral must not swallow another item's."""
    other = uuid.uuid4()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO agenda_items (id, kind, schedule_cron, config, autonomy_level, enabled)
               VALUES ($1, 'routine', '0 8 * * *', '{}'::jsonb, 'notify', false)""",
            other,
        )
    try:
        await _mk_run(pool, item_id, "scheduled",
                      scheduled_for=datetime.now(timezone.utc) + timedelta(hours=1))
        assert await autonomy_db.coalesce_deferred_run(str(other)) is False
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM autonomy_runs WHERE agenda_item_id = $1", other)
            await conn.execute("DELETE FROM agenda_items WHERE id = $1", other)
