"""Tests for the v3 tool_gating.narrow() helper and override filtering."""
from __future__ import annotations

import pytest

from selene_agent.autonomy import tool_gating


def test_narrow_returns_subset_of_tier():
    result = tool_gating.narrow("notify", ["search_memories", "ha_send_notification"])
    assert result == {"search_memories", "ha_send_notification"}


def test_narrow_empty_override_returns_empty_set():
    assert tool_gating.narrow("notify", []) == set()


def test_narrow_raises_on_names_outside_tier():
    # create_memory is in V1_DENY and never in any tier.
    with pytest.raises(ValueError) as ei:
        tool_gating.narrow("notify", ["search_memories", "create_memory"])
    assert "create_memory" in str(ei.value)


def test_narrow_rejects_notify_tool_under_observe_tier():
    with pytest.raises(ValueError):
        tool_gating.narrow("observe", ["ha_send_notification"])


def test_filter_tools_honors_override():
    tools = [
        {"type": "function", "function": {"name": "search_memories"}},
        {"type": "function", "function": {"name": "ha_send_notification"}},
        {"type": "function", "function": {"name": "brave_search"}},
    ]
    out = tool_gating.filter_tools(tools, "notify", override=["search_memories"])
    names = [(t.get("function") or {}).get("name") for t in out]
    assert names == ["search_memories"]


def test_filter_tools_drops_denied_even_if_in_override():
    # narrow() rejects denied tools first; we verify via the wrapper.
    tools = [{"type": "function", "function": {"name": "create_memory"}}]
    with pytest.raises(ValueError):
        tool_gating.filter_tools(tools, "notify", override=["create_memory"])


def test_filter_tools_no_override_returns_full_tier():
    tools = [
        {"type": "function", "function": {"name": "search_memories"}},
        {"type": "function", "function": {"name": "ha_send_notification"}},
        {"type": "function", "function": {"name": "create_memory"}},
    ]
    out = tool_gating.filter_tools(tools, "notify")
    names = {(t.get("function") or {}).get("name") for t in out}
    assert "search_memories" in names
    assert "ha_send_notification" in names
    assert "create_memory" not in names  # in V1_DENY
