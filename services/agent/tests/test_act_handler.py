"""Tests for the act handler — two-phase plan validation + execute."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _plan_result(content: str):
    return MagicMock(content=content, messages=[], metrics={}, error=None, status="ok")


@pytest.mark.asyncio
async def test_empty_prompt_returns_error():
    from selene_agent.autonomy.handlers import act

    item = {"id": "a0", "kind": "act", "config": {"action_allow_list": ["x"]}}
    result = await act.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "prompt" in result["error"]


@pytest.mark.asyncio
async def test_empty_allow_list_returns_error():
    from selene_agent.autonomy.handlers import act

    item = {"id": "a1", "kind": "act", "config": {"prompt": "do things"}}
    result = await act.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "allow_list" in result["error"]


@pytest.mark.asyncio
async def test_disabled_tier_returns_error(monkeypatch):
    from selene_agent.autonomy.handlers import act
    from selene_agent.utils import config as cfg

    monkeypatch.setattr(cfg, "AUTONOMY_ACT_ENABLED", False, raising=False)
    item = {
        "id": "a2",
        "kind": "act",
        "config": {"prompt": "x", "action_allow_list": ["ha_control_light"]},
    }
    result = await act.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "disabled" in result["summary"]


@pytest.mark.asyncio
async def test_require_confirmation_parks_run(monkeypatch):
    from selene_agent.autonomy.handlers import act
    from selene_agent.utils import config as cfg

    monkeypatch.setattr(cfg, "AUTONOMY_ACT_ENABLED", True, raising=False)

    plan_json = (
        '{"steps": [{"tool": "ha_control_light", '
        '"args": {"entity_id": "light.lamp", "state": "on"}, '
        '"rationale": "goal"}, '
        '{"tool": "evil_tool", "args": {}, "rationale": "blocked"}], '
        '"reasoning": "ok"}'
    )
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_plan_result(plan_json))
    monkeypatch.setattr(act, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "a3",
        "kind": "act",
        "autonomy_level": "act",
        "config": {
            "prompt": "turn on the lamp",
            "action_allow_list": ["ha_control_light"],
            "require_confirmation": True,
        },
    }
    result = await act.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "awaiting_confirmation"
    assert result["confirmation_token"]
    audit = result["action_audit"]
    assert len(audit) == 2
    pending = [a for a in audit if a["outcome"] == "pending"]
    blocked = [a for a in audit if a["outcome"] == "skipped_not_allowed"]
    assert len(pending) == 1 and pending[0]["tool"] == "ha_control_light"
    assert len(blocked) == 1 and blocked[0]["tool"] == "evil_tool"


@pytest.mark.asyncio
async def test_inline_execute_when_confirmation_disabled(monkeypatch):
    from selene_agent.autonomy.handlers import act
    from selene_agent.utils import config as cfg

    monkeypatch.setattr(cfg, "AUTONOMY_ACT_ENABLED", True, raising=False)

    plan_json = (
        '{"steps": [{"tool": "ha_control_light", "args": {"state": "on"}, '
        '"rationale": "goal"}], "reasoning": "ok"}'
    )
    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_plan_result(plan_json))
    monkeypatch.setattr(act, "AutonomousTurn", lambda **kw: fake_turn)

    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})

    item = {
        "id": "a4",
        "kind": "act",
        "autonomy_level": "act",
        "config": {
            "prompt": "turn on",
            "action_allow_list": ["ha_control_light"],
            "require_confirmation": False,
        },
    }
    result = await act.handle(
        item, client=None, mcp_manager=mcp, model_name="m", base_tools=[]
    )
    assert result["status"] == "ok"
    assert result["action_audit"][0]["outcome"] == "executed"
    mcp.execute_tool.assert_awaited_once_with(
        "ha_control_light", {"state": "on"}
    )


@pytest.mark.asyncio
async def test_execute_audit_marks_errors_but_continues():
    from selene_agent.autonomy.handlers import act

    audit = [
        {"tool": "t1", "args": {}, "rationale": "", "outcome": "pending"},
        {"tool": "t2", "args": {}, "rationale": "", "outcome": "pending"},
        {"tool": "t3", "args": {}, "rationale": "", "outcome": "skipped_not_allowed"},
    ]
    mcp = MagicMock()
    call_count = {"n": 0}

    async def exec_tool(name, args):
        call_count["n"] += 1
        if name == "t1":
            raise RuntimeError("boom")
        return {"ok": True}

    mcp.execute_tool = AsyncMock(side_effect=exec_tool)
    result = await act.execute_audit(audit, mcp, strict=False)
    assert result[0]["outcome"] == "error"
    assert "boom" in result[0]["error"]
    assert result[1]["outcome"] == "executed"
    # Skipped untouched.
    assert result[2]["outcome"] == "skipped_not_allowed"
    assert call_count["n"] == 2  # t3 was never called


@pytest.mark.asyncio
async def test_execute_approved_runs_pending_steps():
    from selene_agent.autonomy.handlers import act

    run_row = {
        "action_audit": [
            {"tool": "ha_control_light", "args": {"state": "off"}, "outcome": "pending"},
        ]
    }
    item = {"id": "a5", "config": {"action_allow_list": ["ha_control_light"]}}
    mcp = MagicMock()
    mcp.execute_tool = AsyncMock(return_value={"success": True})
    result = await act.execute_approved(run_row, item, mcp)
    assert result["status"] == "ok"
    assert result["action_audit"][0]["outcome"] == "executed"


@pytest.mark.asyncio
async def test_invalid_plan_json_returns_error(monkeypatch):
    from selene_agent.autonomy.handlers import act
    from selene_agent.utils import config as cfg

    monkeypatch.setattr(cfg, "AUTONOMY_ACT_ENABLED", True, raising=False)

    fake_turn = MagicMock()
    fake_turn.run = AsyncMock(return_value=_plan_result("not json"))
    monkeypatch.setattr(act, "AutonomousTurn", lambda **kw: fake_turn)

    item = {
        "id": "a6",
        "kind": "act",
        "autonomy_level": "act",
        "config": {
            "prompt": "do it",
            "action_allow_list": ["ha_control_light"],
        },
    }
    result = await act.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert "invalid plan" in result["summary"]
