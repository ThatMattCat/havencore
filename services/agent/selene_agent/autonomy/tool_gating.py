"""Autonomy tool-gating allow-lists.

Filtered at ``AutonomousTurn`` construction, not delegated to the LLM.
Maintain the per-tier sets here so they are easy to audit.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

# Read-only / safe-to-read HA tools + memory + general-knowledge tools.
OBSERVE_TOOLS: Set[str] = {
    # Home Assistant — read
    "ha_get_domain_entity_states",
    "ha_get_entities_in_area",
    "ha_get_entity_history",
    "ha_get_domain_services",
    "ha_get_presence",
    "ha_get_calendar_events",
    "ha_list_areas",
    "ha_evaluate_template",
    # Memory — read
    "search_memories",
    # General knowledge / reference
    "brave_search",
    "wolfram_alpha",
    "get_weather_forecast",
    "search_wikipedia",
    "query_multimodal_api",
    "fetch",
    # MQTT — camera snapshot capture (triggers HA script, waits on MQTT topic)
    "get_camera_snapshots",
}

# Notifier tools are only added at the ``notify`` tier.
NOTIFY_ONLY_TOOLS: Set[str] = {
    "send_signal_message",
    "ha_send_notification",
}

# Tools that NEVER run under any autonomy tier in v1 (explicit denial for audit).
V1_DENY: Set[str] = {
    "ha_control_light",
    "ha_control_switch",
    "ha_control_media_player",
    "ha_control_climate",
    "ha_execute_service",
    "ha_activate_scene",
    "ha_trigger_script",
    "ha_trigger_automation",
    "ha_toggle_automation",
    "create_memory",
    "delete_memory",
    "play_media",
    "pause_media",
}


def tier_allow_list(tier: str) -> Set[str]:
    """Return the set of tool names allowed for a given autonomy tier."""
    if tier == "observe":
        return set(OBSERVE_TOOLS)
    if tier == "notify":
        return OBSERVE_TOOLS | NOTIFY_ONLY_TOOLS
    # Unknown tiers get nothing.
    return set()


def filter_tools(
    tools: List[Dict[str, Any]],
    tier: str,
    override: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Filter an OpenAI-format tools list by the given tier's allow-list.

    Items explicitly in ``V1_DENY`` are always dropped regardless of tier —
    this is a defense-in-depth check on top of the allow-list.

    When ``override`` is supplied, the allow-list narrows to
    ``tier_allow_list(tier) ∩ override``. Overrides can only *restrict* the
    tier; any name outside the tier raises ``ValueError`` (fail loudly —
    the v3 routine handler relies on this to reject misconfigured items).
    """
    if override is None:
        allowed = tier_allow_list(tier)
    else:
        allowed = narrow(tier, override)
    out: List[Dict[str, Any]] = []
    for tool in tools:
        name = (tool.get("function") or {}).get("name") or tool.get("name")
        if not name:
            continue
        if name in V1_DENY:
            continue
        if name in allowed:
            out.append(tool)
    return out


def narrow(tier: str, override_names: Iterable[str]) -> Set[str]:
    """Return the intersection of ``tier_allow_list(tier)`` with ``override_names``.

    Raises ``ValueError`` if any requested name is outside the tier allow-list
    so that misconfigured routines fail at construction rather than silently
    running with a narrower toolset than the operator intended.
    """
    allowed = tier_allow_list(tier)
    requested = {n for n in override_names if n}
    extra = requested - allowed
    if extra:
        raise ValueError(
            f"tools_override contains names outside tier '{tier}': "
            + ", ".join(sorted(extra))
        )
    return requested
