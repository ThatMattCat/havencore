"""PATCH /autonomy/items must honor explicit nulls for the trigger fields.

Switching an item's trigger type in the dashboard has to *clear* the trigger it
switched away from. Previously both the form and the endpoint dropped None, so
an item that moved cron -> mqtt kept its stale schedule_cron/next_fire_at and
fired on both paths.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from selene_agent.api import autonomy as autonomy_api


class _Req:
    """Stand-in for the FastAPI Request; _engine() is monkeypatched, so the
    endpoint never actually reaches into it."""


@pytest.fixture
def captured(monkeypatch):
    """Capture the patch dict that reaches the DB layer."""
    seen = {}
    current = {
        "id": "11111111-1111-1111-1111-111111111111",
        "kind": "watch",
        "schedule_cron": "0 * * * *",
        "trigger_spec": None,
        "next_fire_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "config": {},
        "autonomy_level": "notify",
        "enabled": True,
        "name": "door watch",
    }

    async def _get_item(item_id):
        return dict(current)

    async def _update_item(item_id, patch):
        seen.update(patch)
        return {**current, **patch}

    monkeypatch.setattr(autonomy_api.autonomy_db, "get_item", _get_item)
    monkeypatch.setattr(autonomy_api.autonomy_db, "update_item", _update_item)
    monkeypatch.setattr(autonomy_api, "_engine", lambda req: type("E", (), {"notify_agenda_changed": lambda self: None})())
    monkeypatch.setattr(autonomy_api, "_serialize_item", lambda row: row)
    return seen


async def test_explicit_null_clears_cron_and_next_fire_at(captured):
    """cron -> mqtt: the old schedule and its next_fire_at must both go."""
    body = autonomy_api.AgendaPatch(
        schedule_cron=None,
        trigger_spec={"source": "mqtt", "match": {"topic": "home/door/+"}},
    )
    await autonomy_api.patch_item("11111111-1111-1111-1111-111111111111", body, _Req())

    assert "schedule_cron" in captured, "explicit null was dropped from the patch"
    assert captured["schedule_cron"] is None
    assert captured["next_fire_at"] is None, "stale next_fire_at keeps the cron firing"
    assert captured["trigger_spec"]["source"] == "mqtt"


async def test_explicit_null_clears_trigger_spec(captured):
    """mqtt -> cron: the old trigger_spec must be cleared."""
    body = autonomy_api.AgendaPatch(schedule_cron="0 8 * * *", trigger_spec=None)
    await autonomy_api.patch_item("11111111-1111-1111-1111-111111111111", body, _Req())

    assert "trigger_spec" in captured
    assert captured["trigger_spec"] is None
    assert captured["schedule_cron"] == "0 8 * * *"
    assert captured["next_fire_at"] is not None


async def test_unset_fields_are_still_untouched(captured):
    """A patch that only renames must not clear either trigger."""
    body = autonomy_api.AgendaPatch(name="renamed")
    await autonomy_api.patch_item("11111111-1111-1111-1111-111111111111", body, _Req())

    assert captured == {"name": "renamed"}


async def test_clearing_both_triggers_is_rejected(captured):
    """An item with neither a schedule nor a trigger can never fire."""
    from fastapi import HTTPException

    body = autonomy_api.AgendaPatch(schedule_cron=None, trigger_spec=None)
    with pytest.raises(HTTPException) as exc:
        await autonomy_api.patch_item("11111111-1111-1111-1111-111111111111", body, _Req())
    assert exc.value.status_code == 400
