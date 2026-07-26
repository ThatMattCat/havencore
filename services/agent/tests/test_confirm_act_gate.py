"""The act tier's kill switch is re-checked at confirm time (issue #84).

``handlers/act.handle`` refuses to *plan* when ``AUTONOMY_ACT_ENABLED`` is off,
but a plan parks in ``awaiting_confirmation`` and the approve deep-link stays
live afterwards. The flag is read from the environment at import, so turning it
off means recreating the agent container — which the parked row survives,
because it lives in Postgres. A paused engine is worse still: ``_tick`` returns
before ``_sweep_confirmation_timeouts``, so the parked run never even times out.

Either way, an approval arriving after the operator shut the tier down used to
fire every actuator in the plan. Now it is refused and the run is finalized.

Only the DB layer is faked here; the engine, the gate and the HTTP endpoint are
the real ones.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.api import autonomy as autonomy_api
from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils import config

TOKEN = "the-real-confirmation-token"
RUN_ID = "b0b1c2d3-0000-4000-8000-000000000084"


def _run_row():
    return {
        "id": RUN_ID,
        "status": "awaiting_confirmation",
        "agenda_item_id": "i1",
        "confirmation_token": TOKEN,
        "action_audit": [
            {"tool": "ha_control_light", "args": {"state": "on"}, "outcome": "pending"},
            {"tool": "ha_nope", "args": {}, "outcome": "skipped_not_allowed"},
        ],
    }


def _item_row():
    return {
        "id": "i1",
        "kind": "act",
        "config": {"action_allow_list": ["ha_control_light"]},
        "schedule_cron": None,
    }


@pytest.fixture
def gate(monkeypatch):
    """Real engine + real endpoint with only ``autonomy_db`` faked.

    Yields a handle with ``post()`` (token-authorized confirm), the engine, the
    mocked MCP manager, the recorded ``finalize_run`` patches, and an ordering
    log so a test can pin the gate relative to the atomic claim.
    """
    from selene_agent.autonomy.engine import AutonomyEngine

    # Pin both halves of the gate rather than inheriting the container's .env,
    # so these tests assert the code and not the deployment's flags.
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "AUTONOMY_ENABLED", True, raising=False)

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})
    engine = AutonomyEngine(
        client=MagicMock(), mcp_manager=mcp, model_name="m", base_tools=[]
    )
    monkeypatch.setattr(engine, "_advance", AsyncMock())

    order: list[str] = []
    patches: list[dict] = []

    async def _claim(run_id):
        order.append("claim")
        return True

    async def _finalize(run_id, patch):
        order.append("finalize")
        patches.append(patch)

    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=_run_row()))
    monkeypatch.setattr(autonomy_db, "get_item", AsyncMock(return_value=_item_row()))
    monkeypatch.setattr(autonomy_db, "claim_confirmation", _claim)
    monkeypatch.setattr(autonomy_db, "finalize_run", _finalize)

    app = FastAPI()
    app.state.autonomy_engine = engine
    app.include_router(autonomy_api.router, prefix="/api")
    client = TestClient(app)

    def post(approved=True):
        return client.post(
            f"/api/autonomy/runs/{RUN_ID}/confirm",
            json={"approved": approved, "token": TOKEN},
        )

    return SimpleNamespace(
        post=post, engine=engine, mcp=mcp, patches=patches, order=order
    )


# --- the kill switch ------------------------------------------------------

def test_disabled_tier_does_not_execute_an_approved_plan(gate, monkeypatch):
    """The #84 hole: approve a run parked before the tier was switched off."""
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    gate.post(approved=True)
    gate.mcp.execute_tool.assert_not_awaited()


def test_disabled_tier_finalizes_the_run(gate, monkeypatch):
    """Blocked is a terminal outcome, not a run left parked for the reaper."""
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    gate.post(approved=True)

    assert len(gate.patches) == 1
    patch = gate.patches[0]
    assert patch["status"] == "error"
    assert patch["confirmation_response"] == "blocked"
    assert patch["completed_at"] is not None
    assert "AUTONOMY_ACT_ENABLED" in patch["error"]
    # The plan-time gate reports the same error string, so a disabled tier
    # reads identically whichever phase it trips in.
    assert patch["error"] == "AUTONOMY_ACT_ENABLED is false"
    # ...and it is emphatically not still awaiting confirmation.
    assert patch["status"] != "awaiting_confirmation"


def test_disabled_tier_marks_the_pending_audit_steps_skipped(gate, monkeypatch):
    """The audit trail has to show the steps did not run."""
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    gate.post(approved=True)

    audit = gate.patches[0]["action_audit"]
    assert [a["outcome"] for a in audit] == ["skipped_denied", "skipped_not_allowed"]
    assert not any(a["outcome"] == "executed" for a in audit)


def test_gate_runs_after_the_atomic_claim(gate, monkeypatch):
    """The claim (9a815ef) actuates nothing, so we take it before deciding.

    Owning the row is what makes the finalize exclusive — gating ahead of the
    CAS would mean writing a row the timeout sweep could still grab.
    """
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    gate.post(approved=True)

    assert gate.order == ["claim", "finalize"]
    assert gate.patches[0]["confirmation_response"] == "blocked"


# --- the pause state ------------------------------------------------------

def test_paused_engine_does_not_execute_an_approved_plan(gate):
    gate.engine.pause()
    gate.post(approved=True)
    gate.mcp.execute_tool.assert_not_awaited()
    assert gate.patches[0]["error"] == "autonomy engine paused"
    assert gate.patches[0]["status"] == "error"


def test_autonomy_kill_switch_env_counts_as_paused(gate, monkeypatch):
    """``is_paused()`` is ``self.paused or not AUTONOMY_ENABLED`` — both halves."""
    monkeypatch.setattr(config, "AUTONOMY_ENABLED", False, raising=False)
    gate.post(approved=True)
    gate.mcp.execute_tool.assert_not_awaited()
    assert gate.patches[0]["error"] == "autonomy engine paused"


@pytest.mark.asyncio
async def test_engine_returns_act_disabled_status(gate, monkeypatch):
    """Engine-level contract, independent of the HTTP layer."""
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    result = await gate.engine.resume_confirmed_run(
        RUN_ID, approved=True, token=TOKEN
    )
    assert result["status"] == "act_disabled"
    assert result["reason"] == "AUTONOMY_ACT_ENABLED is false"
    gate.mcp.execute_tool.assert_not_awaited()


# --- the HTTP contract ----------------------------------------------------

def test_endpoint_returns_409_when_the_tier_is_disabled(gate, monkeypatch):
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    resp = gate.post(approved=True)
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "AUTONOMY_ACT_ENABLED" in detail
    assert "finalized" in detail


def test_endpoint_returns_409_when_paused(gate):
    gate.engine.pause()
    resp = gate.post(approved=True)
    assert resp.status_code == 409
    assert "paused" in resp.json()["detail"]


# --- regression guards ----------------------------------------------------
#
# These two pass against the pre-fix code as well — by construction, since they
# assert the *unchanged* behaviour the gate must not break. They are here to
# catch a gate that is too eager, not to demonstrate the bug.

def test_enabled_tier_still_executes_normally(gate):
    """REGRESSION GUARD (passes pre-fix): the happy path is untouched."""
    resp = gate.post(approved=True)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    gate.mcp.execute_tool.assert_awaited_once_with("ha_control_light", {"state": "on"})


def test_deny_still_works_when_the_tier_is_disabled(gate, monkeypatch):
    """REGRESSION GUARD (passes pre-fix): denial is always safe, so it stays
    ahead of the gate — an operator must be able to clear a parked run even
    with the tier switched off."""
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", False, raising=False)
    resp = gate.post(approved=False)
    assert resp.status_code == 200
    assert resp.json()["status"] == "confirmation_denied"
    assert gate.patches[0]["status"] == "confirmation_denied"
    gate.mcp.execute_tool.assert_not_awaited()
