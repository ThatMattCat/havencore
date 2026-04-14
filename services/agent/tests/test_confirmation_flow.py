"""Tests for the engine's confirmation flow — resume_confirmed_run."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_engine():
    from selene_agent.autonomy.engine import AutonomyEngine

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})
    engine = AutonomyEngine(
        client=MagicMock(), mcp_manager=mcp, model_name="m", base_tools=[]
    )
    return engine, mcp


@pytest.mark.asyncio
async def test_resume_rejects_unknown_run(monkeypatch):
    engine, _ = _make_engine()
    from selene_agent.autonomy import db as autonomy_db
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=None))
    result = await engine.resume_confirmed_run("nope", approved=True, token="t")
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_resume_rejects_non_awaiting_state(monkeypatch):
    engine, _ = _make_engine()
    from selene_agent.autonomy import db as autonomy_db
    monkeypatch.setattr(
        autonomy_db, "get_run",
        AsyncMock(return_value={"id": "r1", "status": "ok", "action_audit": []}),
    )
    result = await engine.resume_confirmed_run("r1", approved=True, token="t")
    assert result["status"] == "invalid_state"
    assert result["current"] == "ok"


@pytest.mark.asyncio
async def test_resume_denied_finalizes_and_advances(monkeypatch):
    engine, _mcp = _make_engine()
    from selene_agent.autonomy import db as autonomy_db

    run_row = {
        "id": "r2",
        "status": "awaiting_confirmation",
        "agenda_item_id": "i2",
        "confirmation_token": "tok",
        "action_audit": [{"tool": "x", "args": {}, "outcome": "pending"}],
    }
    item_row = {
        "id": "i2", "kind": "act",
        "config": {"action_allow_list": ["x"]},
        "schedule_cron": None,
    }
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=run_row))
    monkeypatch.setattr(autonomy_db, "get_item", AsyncMock(return_value=item_row))
    fin = AsyncMock()
    monkeypatch.setattr(autonomy_db, "finalize_run", fin)
    monkeypatch.setattr(engine, "_advance", AsyncMock())

    result = await engine.resume_confirmed_run("r2", approved=False, token="tok")
    assert result["status"] == "confirmation_denied"
    args, _ = fin.await_args
    assert args[0] == "r2"
    assert args[1]["status"] == "confirmation_denied"
    assert args[1]["confirmation_response"] == "denied"


@pytest.mark.asyncio
async def test_resume_invalid_token_rejected(monkeypatch):
    engine, _ = _make_engine()
    from selene_agent.autonomy import db as autonomy_db

    run_row = {
        "id": "r3",
        "status": "awaiting_confirmation",
        "agenda_item_id": "i3",
        "confirmation_token": "correct",
        "action_audit": [],
    }
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=run_row))
    fin = AsyncMock()
    monkeypatch.setattr(autonomy_db, "finalize_run", fin)
    result = await engine.resume_confirmed_run("r3", approved=True, token="wrong")
    assert result["status"] == "invalid_token"
    fin.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_approved_executes_and_finalizes(monkeypatch):
    engine, mcp = _make_engine()
    from selene_agent.autonomy import db as autonomy_db

    run_row = {
        "id": "r4",
        "status": "awaiting_confirmation",
        "agenda_item_id": "i4",
        "confirmation_token": "tok",
        "action_audit": [
            {"tool": "ha_control_light", "args": {"state": "on"}, "outcome": "pending"}
        ],
    }
    item_row = {
        "id": "i4", "kind": "act",
        "config": {"action_allow_list": ["ha_control_light"]},
        "schedule_cron": None,
    }
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=run_row))
    monkeypatch.setattr(autonomy_db, "get_item", AsyncMock(return_value=item_row))
    fin = AsyncMock()
    monkeypatch.setattr(autonomy_db, "finalize_run", fin)
    monkeypatch.setattr(engine, "_advance", AsyncMock())

    result = await engine.resume_confirmed_run("r4", approved=True, token="tok")
    assert result["status"] == "ok"
    mcp.execute_tool.assert_awaited_once_with(
        "ha_control_light", {"state": "on"}
    )
    args, _ = fin.await_args
    assert args[1]["confirmation_response"] == "approved"
    assert args[1]["status"] == "ok"


@pytest.mark.asyncio
async def test_timeout_sweep_claims_expired(monkeypatch):
    engine, _ = _make_engine()
    from selene_agent.autonomy import db as autonomy_db

    monkeypatch.setattr(
        autonomy_db,
        "list_expired_confirmations",
        AsyncMock(return_value=[{"run_id": "x"}, {"run_id": "y"}]),
    )
    claim = AsyncMock(return_value=True)
    monkeypatch.setattr(autonomy_db, "claim_confirmation_timeout", claim)
    await engine._sweep_confirmation_timeouts(datetime.now(timezone.utc))
    assert claim.await_count == 2
