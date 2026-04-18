"""Tests for _apply_device_name helper and the orchestrator.device_name field."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from selene_agent.api.chat import DEVICE_NAME_MAX_LEN, _apply_device_name
from selene_agent.orchestrator import AgentOrchestrator


def _make_orch(device_name=None):
    orch = AgentOrchestrator(
        client=MagicMock(),
        mcp_manager=MagicMock(),
        model_name="test-model",
        tools=[],
        session_id="sess-under-test",
    )
    if device_name is not None:
        orch.device_name = device_name
    return orch


def test_device_name_field_defaults_none():
    orch = _make_orch()
    assert orch.device_name is None


def test_apply_device_name_plain_ascii():
    orch = _make_orch()
    _apply_device_name(orch, "Kitchen Speaker")
    assert orch.device_name == "Kitchen Speaker"


def test_apply_device_name_trims_whitespace():
    orch = _make_orch()
    _apply_device_name(orch, "  Office  ")
    assert orch.device_name == "Office"


def test_apply_device_name_none_is_noop_preserves_existing():
    orch = _make_orch(device_name="Living Room")
    _apply_device_name(orch, None)
    assert orch.device_name == "Living Room"


def test_apply_device_name_empty_is_noop_preserves_existing():
    orch = _make_orch(device_name="Living Room")
    _apply_device_name(orch, "")
    assert orch.device_name == "Living Room"


def test_apply_device_name_whitespace_only_is_noop_preserves_existing():
    orch = _make_orch(device_name="Living Room")
    _apply_device_name(orch, "   \t  ")
    assert orch.device_name == "Living Room"


def test_apply_device_name_non_string_ignored_preserves_existing():
    orch = _make_orch(device_name="Living Room")
    for bad in (123, 12.5, [], {}, ["Office"]):
        _apply_device_name(orch, bad)
    assert orch.device_name == "Living Room"


def test_apply_device_name_strips_control_chars():
    orch = _make_orch()
    _apply_device_name(orch, "Kit\x00chen\x07Speaker\x7f")
    assert orch.device_name == "KitchenSpeaker"


def test_apply_device_name_only_control_chars_is_noop():
    orch = _make_orch(device_name="prior")
    _apply_device_name(orch, "\x00\x01\x02")
    assert orch.device_name == "prior"


def test_apply_device_name_at_cap_preserved_exactly():
    orch = _make_orch()
    name = "x" * DEVICE_NAME_MAX_LEN
    _apply_device_name(orch, name)
    assert orch.device_name == name
    assert len(orch.device_name) == DEVICE_NAME_MAX_LEN


def test_apply_device_name_over_cap_truncated():
    orch = _make_orch()
    _apply_device_name(orch, "x" * (DEVICE_NAME_MAX_LEN + 25))
    assert orch.device_name == "x" * DEVICE_NAME_MAX_LEN


def test_apply_device_name_unicode_and_emoji_preserved():
    orch = _make_orch()
    _apply_device_name(orch, "Küche 🔊")
    assert orch.device_name == "Küche 🔊"


def test_apply_device_name_rename_last_write_wins():
    orch = _make_orch(device_name="kitchen")
    _apply_device_name(orch, "Kitchen Speaker")
    assert orch.device_name == "Kitchen Speaker"
    _apply_device_name(orch, "Kitchen Hub")
    assert orch.device_name == "Kitchen Hub"
