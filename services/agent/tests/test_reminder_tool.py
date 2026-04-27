"""Tests for the reminder MCP tool (selene_agent.modules.mcp_reminder_tools)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# --- _resolve_when ---------------------------------------------------------

def test_resolve_when_in_seconds_produces_one_shot_cron():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    now = datetime(2026, 4, 27, 14, 30, 0, tzinfo=timezone.utc)
    cron, one_shot = _resolve_when({"in_seconds": 90}, now_utc=now)

    assert one_shot is True
    # Pure 5-field cron: minute hour day month dow
    parts = cron.split()
    assert len(parts) == 5
    assert parts[4] == "*"
    # Day-of-week wildcard means it won't repeat across weeks; year is implicit
    # so the engine handler must use one_shot=True to disable after firing.


def test_resolve_when_at_iso_produces_one_shot_cron():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    now = datetime(2026, 4, 27, 14, 30, 0, tzinfo=timezone.utc)
    future = (now + timedelta(hours=2)).isoformat()
    cron, one_shot = _resolve_when({"at": future}, now_utc=now)

    assert one_shot is True
    assert len(cron.split()) == 5


def test_resolve_when_cron_passthrough_recurring():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    cron, one_shot = _resolve_when({"cron": "0 18 * * 0"})

    assert one_shot is False
    assert cron == "0 18 * * 0"


def test_resolve_when_requires_exactly_one_field():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    with pytest.raises(ValueError, match="exactly one"):
        _resolve_when({})

    with pytest.raises(ValueError, match="exactly one"):
        _resolve_when({"in_seconds": 60, "cron": "* * * * *"})


def test_resolve_when_rejects_past_at():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    now = datetime(2026, 4, 27, 14, 30, 0, tzinfo=timezone.utc)
    past = (now - timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="future"):
        _resolve_when({"at": past}, now_utc=now)


def test_resolve_when_rejects_zero_or_negative_seconds():
    from selene_agent.modules.mcp_reminder_tools.mcp_server import _resolve_when

    with pytest.raises(ValueError, match="> 0"):
        _resolve_when({"in_seconds": 0})
    with pytest.raises(ValueError, match="> 0"):
        _resolve_when({"in_seconds": -10})


# --- schedule_reminder integration with mocked HTTP ------------------------

class _FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._body = json.dumps(body)

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in capturing the last call."""

    def __init__(self, response: _FakeResponse):
        self.response = response
        self.last_call: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        self.last_call = {"method": "POST", "url": url, "json": json}
        return self.response

    def get(self, url, timeout=None):
        self.last_call = {"method": "GET", "url": url}
        return self.response

    def delete(self, url, timeout=None):
        self.last_call = {"method": "DELETE", "url": url}
        return self.response


@pytest.mark.asyncio
async def test_schedule_reminder_defaults_channel_to_signal():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(200, {"item": {"id": "abc-123", "next_fire_at": "2026-04-27T15:30:00+00:00"}}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.schedule_reminder({
            "title": "Take the trash out",
            "in_seconds": 3600,
        })

    assert result["status"] == "ok"
    assert result["id"] == "abc-123"
    assert result["channel"] == "signal"
    assert result["one_shot"] is True
    posted = fake.last_call["json"]
    assert posted["kind"] == "reminder"
    assert posted["config"]["channel"] == "signal"
    assert posted["config"]["title"] == "Take the trash out"
    assert posted["config"]["body"] == "Take the trash out"  # body falls back to title
    assert posted["config"]["one_shot"] is True


@pytest.mark.asyncio
async def test_schedule_reminder_personalize_defaults_to_true():
    """Caller omits personalize → cfg payload sent to /api/autonomy/items has personalize=True."""
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(200, {"item": {"id": "p-1", "next_fire_at": None}}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.schedule_reminder({
            "title": "Take the trash out",
            "in_seconds": 3600,
        })

    assert result["status"] == "ok"
    assert result["personalize"] is True
    assert fake.last_call["json"]["config"]["personalize"] is True


@pytest.mark.asyncio
async def test_schedule_reminder_personalize_false_propagates():
    """Explicit personalize=false → cfg payload preserves it."""
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(200, {"item": {"id": "p-2", "next_fire_at": None}}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.schedule_reminder({
            "title": "Verbatim",
            "in_seconds": 3600,
            "personalize": False,
        })

    assert result["personalize"] is False
    assert fake.last_call["json"]["config"]["personalize"] is False


@pytest.mark.asyncio
async def test_list_reminders_surfaces_personalize_default_true_for_legacy_items():
    """Existing rows without `personalize` in config should be reported as personalize=True
    so the LLM's mental model matches the handler's default."""
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    body = {"items": [
        # Legacy row: no personalize key
        {"id": "1", "kind": "reminder", "enabled": True, "schedule_cron": "0 9 * * *",
         "next_fire_at": None, "config": {"title": "old", "channel": "signal"}},
        # New row: explicit personalize=false
        {"id": "2", "kind": "reminder", "enabled": True, "schedule_cron": "0 0 * * *",
         "next_fire_at": None,
         "config": {"title": "verbatim", "channel": "signal", "personalize": False}},
    ]}
    fake = _FakeSession(_FakeResponse(200, body))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.list_reminders({})

    by_id = {r["id"]: r for r in result["reminders"]}
    assert by_id["1"]["personalize"] is True
    assert by_id["2"]["personalize"] is False


@pytest.mark.asyncio
async def test_schedule_reminder_recurring_keeps_one_shot_false():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(200, {"item": {"id": "rec-1", "next_fire_at": None}}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.schedule_reminder({
            "title": "Sunday trash night",
            "cron": "0 18 * * 0",
            "channel": "ha_push",
        })

    assert result["status"] == "ok"
    assert result["one_shot"] is False
    assert result["cron"] == "0 18 * * 0"
    assert fake.last_call["json"]["config"]["channel"] == "ha_push"
    assert fake.last_call["json"]["config"]["one_shot"] is False


@pytest.mark.asyncio
async def test_schedule_reminder_rejects_unknown_channel():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    server = mcp_server.ReminderToolsServer()
    result = await server.schedule_reminder({
        "title": "x",
        "in_seconds": 60,
        "channel": "carrier_pigeon",
    })
    assert result["status"] == "error"
    assert "channel" in result["error"]


@pytest.mark.asyncio
async def test_schedule_reminder_propagates_api_error():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(400, {"detail": "invalid cron: bad"}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.schedule_reminder({
            "title": "x",
            "cron": "not a cron",
        })
    assert result["status"] == "error"
    assert "400" in result["error"]


# --- list_reminders + cancel_reminder --------------------------------------

@pytest.mark.asyncio
async def test_list_reminders_filters_to_kind_reminder_and_enabled():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    body = {"items": [
        {"id": "1", "kind": "reminder", "enabled": True, "schedule_cron": "0 9 * * *",
         "next_fire_at": "2026-04-28T09:00:00+00:00",
         "config": {"title": "Morning meds", "channel": "ha_push", "one_shot": False}},
        {"id": "2", "kind": "reminder", "enabled": False, "schedule_cron": "0 0 * * *",
         "next_fire_at": None,
         "config": {"title": "Old", "channel": "signal", "one_shot": True}},
        {"id": "3", "kind": "briefing", "enabled": True, "schedule_cron": "0 7 * * *",
         "next_fire_at": "2026-04-28T07:00:00+00:00", "config": {}},
    ]}
    fake = _FakeSession(_FakeResponse(200, body))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.list_reminders({})

    assert result["status"] == "ok"
    assert result["count"] == 1
    assert result["reminders"][0]["id"] == "1"
    assert result["reminders"][0]["title"] == "Morning meds"


@pytest.mark.asyncio
async def test_list_reminders_include_disabled_returns_all_reminder_kind():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    body = {"items": [
        {"id": "1", "kind": "reminder", "enabled": True, "schedule_cron": "0 9 * * *",
         "next_fire_at": None, "config": {"title": "a", "channel": "signal"}},
        {"id": "2", "kind": "reminder", "enabled": False, "schedule_cron": "0 0 * * *",
         "next_fire_at": None, "config": {"title": "b", "channel": "signal"}},
        {"id": "3", "kind": "briefing", "enabled": True, "schedule_cron": "0 7 * * *",
         "next_fire_at": None, "config": {}},
    ]}
    fake = _FakeSession(_FakeResponse(200, body))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.list_reminders({"include_disabled": True})

    assert result["count"] == 2
    assert {r["id"] for r in result["reminders"]} == {"1", "2"}


@pytest.mark.asyncio
async def test_cancel_reminder_returns_404_as_error():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(404, {"detail": "not found"}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.cancel_reminder({"id": "missing"})

    assert result["status"] == "error"
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_cancel_reminder_success():
    from selene_agent.modules.mcp_reminder_tools import mcp_server

    fake = _FakeSession(_FakeResponse(200, {"deleted": True}))
    server = mcp_server.ReminderToolsServer()

    with patch.object(mcp_server.aiohttp, "ClientSession", return_value=fake):
        result = await server.cancel_reminder({"id": "abc-123"})

    assert result["status"] == "ok"
    assert result["deleted"] is True
    assert fake.last_call["method"] == "DELETE"
    assert fake.last_call["url"].endswith("/api/autonomy/items/abc-123")
