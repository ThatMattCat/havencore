"""Authorization for POST /api/autonomy/runs/{id}/confirm.

Approving a parked ``act`` run fires real actuators, so it needs a proof. There
are exactly two:

* the confirmation token from the notification deep-link (companion app /
  Signal), which works from anywhere, and
* a browser ``Origin`` that is allowlisted or equal to the request's own
  ``Host`` (same-host auto-allow, for TLS/hostname fronts) — the dashboard's
  Approve button, which has no token to send because nothing publishes it any
  more.

The absent-Origin rule here is the **inverse** of ``WebSocketOriginGuard``'s: a
missing Origin there means "non-browser client, allow"; here it means "not a
browser, so prove it with the token". Without that inversion a bare
``curl -X POST .../confirm -d '{"approved":true}'`` approves anything whose
run_id the caller has seen — and run_ids are broadcast on the WS feed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from selene_agent.api import autonomy as autonomy_api
from selene_agent.autonomy import db as autonomy_db
from selene_agent.utils import config

ALLOWED_ORIGIN = "http://localhost:6002"
EVIL_ORIGIN = "http://evil.example"
TOKEN = "the-real-confirmation-token"

RUN_ID = "b0b1c2d3-0000-4000-8000-000000000001"


def _run_row():
    return {
        "id": RUN_ID,
        "status": "awaiting_confirmation",
        "agenda_item_id": "i1",
        "confirmation_token": TOKEN,
        "action_audit": [
            {"tool": "ha_control_light", "args": {"state": "on"}, "outcome": "pending"}
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
def confirm(monkeypatch):
    """A client for the real endpoint + real engine, with only the DB faked.

    Returns ``(post, mcp)`` where ``post(body, origin=...)`` issues the confirm
    request and ``mcp.execute_tool`` records whether actuators actually fired.
    """
    from selene_agent.autonomy.engine import AutonomyEngine

    # Pin the allowlist so the test doesn't depend on the container's
    # HOST_IP_ADDRESS / AGENT_CORS_ORIGINS. Same for the act tier's kill
    # switch, which the confirm path re-checks before executing (#84) — these
    # tests are about authorization, not about whether the tier is on.
    monkeypatch.setattr(config, "AGENT_CORS_ORIGINS", ALLOWED_ORIGIN, raising=False)
    monkeypatch.setattr(config, "AUTONOMY_ACT_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "AUTONOMY_ENABLED", True, raising=False)

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})
    engine = AutonomyEngine(
        client=MagicMock(), mcp_manager=mcp, model_name="m", base_tools=[]
    )
    monkeypatch.setattr(engine, "_advance", AsyncMock())

    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=_run_row()))
    monkeypatch.setattr(autonomy_db, "get_item", AsyncMock(return_value=_item_row()))
    monkeypatch.setattr(
        autonomy_db, "claim_confirmation", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(autonomy_db, "finalize_run", AsyncMock())

    app = FastAPI()
    app.state.autonomy_engine = engine
    app.include_router(autonomy_api.router, prefix="/api")
    client = TestClient(app)

    def post(body, origin=None):
        headers = {} if origin is None else {"Origin": origin}
        return client.post(
            f"/api/autonomy/runs/{RUN_ID}/confirm", json=body, headers=headers
        )

    return post, mcp


# --- the inverted absent-Origin rule --------------------------------------

def test_tokenless_confirm_without_origin_is_rejected(confirm):
    """The #77 hole: `curl -X POST .../confirm -d '{"approved":true}'`."""
    post, mcp = confirm
    resp = post({"approved": True})
    assert resp.status_code == 403
    assert "token" in resp.json()["detail"].lower()
    mcp.execute_tool.assert_not_awaited()


def test_tokenless_confirm_from_allowlisted_origin_is_accepted(confirm):
    """The dashboard's Approve button — same-origin POST, no token."""
    post, mcp = confirm
    resp = post({"approved": True}, origin=ALLOWED_ORIGIN)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mcp.execute_tool.assert_awaited_once_with("ha_control_light", {"state": "on"})


def test_tokenless_deny_from_allowlisted_origin_is_accepted(confirm):
    """Deny is an authorized decision too — the dashboard must keep working."""
    post, mcp = confirm
    resp = post({"approved": False}, origin=ALLOWED_ORIGIN)
    assert resp.status_code == 200
    assert resp.json()["status"] == "confirmation_denied"
    mcp.execute_tool.assert_not_awaited()


def test_tokenless_confirm_from_foreign_origin_is_rejected(confirm):
    post, mcp = confirm
    resp = post({"approved": True}, origin=EVIL_ORIGIN)
    assert resp.status_code == 403
    mcp.execute_tool.assert_not_awaited()


def test_tokenless_confirm_from_same_host_origin_is_accepted(confirm):
    """A TLS/hostname front: Origin equals the request's own Host (TestClient
    sends `Host: testserver`) but is NOT in the pinned allowlist — the Approve
    button must keep working with zero AGENT_CORS_ORIGINS config."""
    post, mcp = confirm
    resp = post({"approved": True}, origin="http://testserver")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mcp.execute_tool.assert_awaited_once()


def test_explicit_null_token_is_treated_as_no_token(confirm):
    """The dashboard sends `token: null` when there's no deep-link param."""
    post, mcp = confirm
    assert post({"approved": True, "token": None}).status_code == 403
    mcp.execute_tool.assert_not_awaited()


# --- the token stays sufficient, whatever the origin ----------------------

def test_correct_token_without_origin_is_accepted(confirm):
    """The companion / Signal deep-link path: no browser, real token."""
    post, mcp = confirm
    resp = post({"approved": True, "token": TOKEN})
    assert resp.status_code == 200
    mcp.execute_tool.assert_awaited_once()


def test_correct_token_from_foreign_origin_is_accepted(confirm):
    post, mcp = confirm
    resp = post({"approved": True, "token": TOKEN}, origin=EVIL_ORIGIN)
    assert resp.status_code == 200
    mcp.execute_tool.assert_awaited_once()


# --- a wrong token is a red flag, not a fallback --------------------------

def test_wrong_token_is_rejected_even_from_allowlisted_origin(confirm):
    post, mcp = confirm
    resp = post({"approved": True, "token": "guessed"}, origin=ALLOWED_ORIGIN)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "invalid token"
    mcp.execute_tool.assert_not_awaited()


def test_empty_token_does_not_match_an_unset_stored_token(confirm, monkeypatch):
    """Degenerate compare_digest('', '') must not authorize anything."""
    row = _run_row()
    row["confirmation_token"] = None
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=row))
    post, mcp = confirm
    assert post({"approved": True, "token": ""}, origin=EVIL_ORIGIN).status_code == 403
    mcp.execute_tool.assert_not_awaited()


# --- engine default -------------------------------------------------------

@pytest.mark.asyncio
async def test_engine_requires_a_token_by_default(monkeypatch):
    """Non-HTTP callers get no origin exemption: allow_tokenless defaults off."""
    from selene_agent.autonomy.engine import AutonomyEngine

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})
    engine = AutonomyEngine(
        client=MagicMock(), mcp_manager=mcp, model_name="m", base_tools=[]
    )
    monkeypatch.setattr(autonomy_db, "get_run", AsyncMock(return_value=_run_row()))
    claim = AsyncMock(return_value=True)
    monkeypatch.setattr(autonomy_db, "claim_confirmation", claim)

    result = await engine.resume_confirmed_run(RUN_ID, approved=True)
    assert result["status"] == "token_required"
    claim.assert_not_awaited()
    mcp.execute_tool.assert_not_awaited()
