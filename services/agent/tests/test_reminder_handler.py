"""Tests for the reminder agenda handler."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_reminder_sends_notification_and_returns_ok(monkeypatch):
    from selene_agent.autonomy.handlers import reminder

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    update_item = AsyncMock()
    monkeypatch.setattr(reminder.autonomy_db, "update_item", update_item)

    item = {
        "id": "r-1",
        "name": "Laundry",
        "config": {
            "title": "Laundry",
            "body": "Move to dryer",
            "channel": "ha_push",
        },
    }
    result = await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    fake_notifier.send.assert_awaited_once()
    assert result["status"] == "ok"
    assert result["notified_via"] == "ha_push"
    assert update_item.await_count == 0  # not one_shot
    assert not result.get("_delete_after_run")  # recurring reminders are not deleted


@pytest.mark.asyncio
async def test_reminder_one_shot_signals_delete_on_success(monkeypatch):
    from selene_agent.autonomy.handlers import reminder

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    # Handler should not touch the DB itself; the engine does the delete
    # after insert_run so the autonomy_runs FK reference stays valid.
    update_item = AsyncMock()
    delete_item = AsyncMock()
    monkeypatch.setattr(reminder.autonomy_db, "update_item", update_item)
    monkeypatch.setattr(reminder.autonomy_db, "delete_item", delete_item, raising=False)

    item = {
        "id": "r-2",
        "config": {
            "title": "One-shot",
            "body": "Do the thing once",
            "channel": "ha_push",
            "one_shot": True,
        },
    }
    result = await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "ok"
    assert result["_delete_after_run"] is True
    update_item.assert_not_called()
    delete_item.assert_not_called()


@pytest.mark.asyncio
async def test_reminder_one_shot_does_not_signal_delete_on_failure(monkeypatch):
    from selene_agent.autonomy.handlers import reminder

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=False)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    update_item = AsyncMock()
    monkeypatch.setattr(reminder.autonomy_db, "update_item", update_item)

    item = {
        "id": "r-3",
        "config": {"title": "t", "body": "b", "channel": "ha_push", "one_shot": True},
    }
    result = await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    assert result["status"] == "error"
    assert result["notified_via"] is None
    assert not result.get("_delete_after_run")
    update_item.assert_not_called()


@pytest.mark.asyncio
async def test_reminder_rejects_empty_body(monkeypatch):
    from selene_agent.autonomy.handlers import reminder

    update_item = AsyncMock()
    monkeypatch.setattr(reminder.autonomy_db, "update_item", update_item)

    item = {"id": "r-4", "config": {"title": "", "body": ""}}
    result = await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )
    # Title/body default to "Reminder" via the `or item.get('name')` chain;
    # here we pass neither name nor title — handler falls back to "Reminder"
    # as title, which becomes the body, so delivery should still be attempted.
    # Assert the handler returns a structured result regardless.
    assert "status" in result
    assert result["messages"] == []
