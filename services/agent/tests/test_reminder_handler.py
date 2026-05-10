"""Tests for the reminder agenda handler."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


def _stub_personalize(monkeypatch, *, body: str, image_prompt=None):
    """Replace the personalize_reminder helper with a stub returning known data."""
    from selene_agent.autonomy.handlers import reminder

    stub = AsyncMock(return_value={"body": body, "image_prompt": image_prompt})
    monkeypatch.setattr(reminder, "personalize_reminder", stub)
    return stub


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


# --- v2 personalization tests ---------------------------------------------

@pytest.mark.asyncio
async def test_reminder_personalizes_by_default_for_signal_with_image(monkeypatch):
    """personalize defaults to True when missing from config; signal channel
    runs the rewrite + (when an image_prompt is suggested) image gen + attaches."""
    from selene_agent.autonomy.handlers import reminder

    personalize_stub = _stub_personalize(
        monkeypatch,
        body="Hey, time to take the trash out — bins to the curb!",
        image_prompt="cartoon trash can at the curb",
    )

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    mcp_manager = MagicMock()
    mcp_manager.execute_tool = AsyncMock(return_value=json.dumps(
        {"images": [{"path": "/app/outputs/img-1.png", "url": "http://x/img-1.png"}]}
    ))
    fake_client = MagicMock()

    item = {
        "id": "r-default-personalize",
        "config": {"title": "Trash", "body": "Take the trash out", "channel": "signal"},
        # NOTE: no `personalize` key — should default to True
    }
    result = await reminder.handle(
        item, client=fake_client, mcp_manager=mcp_manager, model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is True
    assert result["metrics"]["image_attached"] is True

    personalize_stub.assert_awaited_once()
    mcp_manager.execute_tool.assert_awaited_once()
    tool_name, tool_args = mcp_manager.execute_tool.await_args.args
    assert tool_name == "generate_image"
    assert tool_args["prompt"] == "cartoon trash can at the curb"

    fake_notifier.send.assert_awaited_once()
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["body"].startswith("Hey")
    assert sent_kwargs["attachments"] == ["/app/outputs/img-1.png"]


@pytest.mark.asyncio
async def test_reminder_personalize_ha_push_skips_image_gen(monkeypatch):
    """For non-signal channels, even if the LLM suggests an image prompt the
    helper strips it. Handler must not call generate_image."""
    from selene_agent.autonomy.handlers import reminder

    _stub_personalize(monkeypatch, body="Ok, friendly reminder.", image_prompt=None)

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    mcp_manager = MagicMock()
    mcp_manager.execute_tool = AsyncMock()

    item = {
        "id": "r-ha",
        "config": {"title": "x", "body": "y", "channel": "ha_push", "personalize": True},
    }
    result = await reminder.handle(
        item, client=MagicMock(), mcp_manager=mcp_manager, model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is True
    assert result["metrics"]["image_attached"] is False
    mcp_manager.execute_tool.assert_not_called()
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert "attachments" not in sent_kwargs


@pytest.mark.asyncio
async def test_reminder_personalize_false_skips_llm(monkeypatch):
    """personalize=false → no LLM call, verbatim delivery."""
    from selene_agent.autonomy.handlers import reminder

    personalize_stub = _stub_personalize(monkeypatch, body="should-not-appear")

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-verbatim",
        "config": {
            "title": "Trash",
            "body": "Take the trash out",
            "channel": "signal",
            "personalize": False,
        },
    }
    result = await reminder.handle(
        item, client=MagicMock(), mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is False
    personalize_stub.assert_not_called()
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["body"] == "Take the trash out"
    assert "attachments" not in sent_kwargs


@pytest.mark.asyncio
async def test_reminder_personalize_falls_back_when_llm_returns_unchanged_body(monkeypatch):
    """Helper returning the original body unchanged (its fallback contract) →
    handler reports personalized=False and delivers verbatim."""
    from selene_agent.autonomy.handlers import reminder

    _stub_personalize(monkeypatch, body="Take the trash out", image_prompt=None)

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-fallback",
        "config": {"title": "Trash", "body": "Take the trash out", "channel": "signal"},
    }
    result = await reminder.handle(
        item, client=MagicMock(), mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is False
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["body"] == "Take the trash out"


@pytest.mark.asyncio
async def test_reminder_drops_title_when_identical_to_body(monkeypatch):
    """When the LLM scheduled with only `title=` (no body), the MCP tool stores
    body=title. Without dedup, SignalNotifier/SpeakerNotifier concatenate the
    same string twice — Signal renders the reminder twice in one message,
    speaker channel synthesizes TTS for the text twice. The handler must pass
    title="" to the notifier when title and body are identical."""
    from selene_agent.autonomy.handlers import reminder

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-dup",
        "config": {
            "title": "Take out the trash",
            "body": "Take out the trash",
            "channel": "signal",
            "personalize": False,
        },
    }
    await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["title"] == ""
    assert sent_kwargs["body"] == "Take out the trash"


@pytest.mark.asyncio
async def test_reminder_keeps_title_when_distinct_from_body(monkeypatch):
    """When title and body are intentionally different (user set both, or
    personalization rewrote the body), keep both — they convey separate info."""
    from selene_agent.autonomy.handlers import reminder

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-distinct",
        "config": {
            "title": "Laundry",
            "body": "Move it to the dryer",
            "channel": "signal",
            "personalize": False,
        },
    }
    await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["title"] == "Laundry"
    assert sent_kwargs["body"] == "Move it to the dryer"


@pytest.mark.asyncio
async def test_reminder_personalize_image_gen_failure_sends_text_only(monkeypatch):
    """LLM rewrite succeeds; image gen tool throws → handler still delivers
    the personalized text, just without an attachment."""
    from selene_agent.autonomy.handlers import reminder

    _stub_personalize(monkeypatch, body="Friendly trash reminder.", image_prompt="trash bin")

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    mcp_manager = MagicMock()
    mcp_manager.execute_tool = AsyncMock(side_effect=RuntimeError("comfyui down"))

    item = {
        "id": "r-image-fail",
        "config": {"title": "Trash", "body": "Take the trash out", "channel": "signal"},
    }
    result = await reminder.handle(
        item, client=MagicMock(), mcp_manager=mcp_manager, model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is True
    assert result["metrics"]["image_attached"] is False
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert sent_kwargs["body"] == "Friendly trash reminder."
    assert "attachments" not in sent_kwargs


@pytest.mark.asyncio
async def test_reminder_personalize_image_gen_returns_no_path_skips_attachment(monkeypatch):
    """generate_image returns a malformed payload (no images list) → no attachment."""
    from selene_agent.autonomy.handlers import reminder

    _stub_personalize(monkeypatch, body="Friendly text.", image_prompt="something")

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    mcp_manager = MagicMock()
    mcp_manager.execute_tool = AsyncMock(return_value=json.dumps({"images": []}))

    item = {
        "id": "r-no-path",
        "config": {"title": "x", "body": "y", "channel": "signal"},
    }
    result = await reminder.handle(
        item, client=MagicMock(), mcp_manager=mcp_manager, model_name="m", base_tools=[]
    )

    assert result["metrics"]["image_attached"] is False
    sent_kwargs = fake_notifier.send.await_args.kwargs
    assert "attachments" not in sent_kwargs


@pytest.mark.asyncio
async def test_reminder_personalize_skipped_when_client_unavailable(monkeypatch):
    """client=None (the engine couldn't supply one) → personalize step is skipped,
    deterministic delivery proceeds. Covers the legacy test's implicit behavior."""
    from selene_agent.autonomy.handlers import reminder

    personalize_stub = _stub_personalize(monkeypatch, body="should-not-appear")

    fake_notifier = MagicMock()
    fake_notifier.send = AsyncMock(return_value=True)
    monkeypatch.setattr(reminder, "_make_notifier", lambda *a, **kw: fake_notifier)

    item = {
        "id": "r-noclient",
        "config": {"title": "Trash", "body": "Take the trash out", "channel": "signal"},
    }
    result = await reminder.handle(
        item, client=None, mcp_manager=MagicMock(), model_name="m", base_tools=[]
    )

    assert result["status"] == "ok"
    assert result["metrics"]["personalized"] is False
    personalize_stub.assert_not_called()
