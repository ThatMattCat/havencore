"""Atomic confirm claim for approved act runs (#43).

Exercises the real SQL against the live Postgres, since the bug is precisely
about atomicity — an in-memory fake would pass either way.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

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


async def test_only_one_concurrent_confirm_wins(pool, item_id):
    """Two simultaneous approvals: exactly one may execute the actions."""
    run_id = await _mk_run(pool, item_id, "awaiting_confirmation")

    results = await asyncio.gather(
        autonomy_db.claim_confirmation(run_id),
        autonomy_db.claim_confirmation(run_id),
    )
    assert sorted(results) == [False, True], f"both confirms claimed the run: {results}"

    row = await autonomy_db.get_run(run_id, include_messages=False)
    assert row["status"] == "confirming"


async def test_confirm_cannot_claim_after_timeout_sweep(pool, item_id):
    """An approve arriving after the sweep gave up must not execute."""
    run_id = await _mk_run(pool, item_id, "awaiting_confirmation")

    assert await autonomy_db.claim_confirmation_timeout(run_id, datetime.now(timezone.utc))
    assert await autonomy_db.claim_confirmation(run_id) is False

    row = await autonomy_db.get_run(run_id, include_messages=False)
    assert row["status"] == "confirmation_timeout"


async def test_timeout_sweep_cannot_steal_a_claimed_confirm(pool, item_id):
    """And the reverse: once confirming, the sweep must not reclaim it."""
    run_id = await _mk_run(pool, item_id, "awaiting_confirmation")

    assert await autonomy_db.claim_confirmation(run_id) is True
    assert await autonomy_db.claim_confirmation_timeout(run_id, datetime.now(timezone.utc)) is False


async def test_claim_rejects_non_awaiting_states(pool, item_id):
    for status in ("ok", "error", "confirmation_denied", "scheduled"):
        run_id = await _mk_run(pool, item_id, status)
        assert await autonomy_db.claim_confirmation(run_id) is False, status
