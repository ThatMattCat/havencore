"""Autonomy tool-gating allow-lists.

Filtered at ``AutonomousTurn`` construction, not delegated to the LLM.
Maintain the per-tier sets here so they are easy to audit.

Tiers:
- ``observe`` — read-only HA + memory + general knowledge (safe for any LLM turn).
- ``notify`` — observe + notifier tools (Signal, HA push). Handlers usually
  invoke notifiers directly rather than through the LLM, but keeping the
  allow-list set here means the surface is auditable in one place.
- ``speak`` — same tool surface as ``notify`` (speaker delivery happens via
  the SpeakerNotifier, not an LLM tool). Listed as its own tier so the engine
  can distinguish it for metrics / UI / quiet-hours policy.
- ``act`` (v4) — observe + notify + a bounded set of HA actuator tools, and
  only under a per-item ``action_allow_list`` (enforced via ``narrow()``).
  An ``act`` item must always supply an override — running with the full
  surface is an anti-pattern.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

# Read-only / safe-to-read HA tools + memory + general-knowledge tools.
OBSERVE_TOOLS: Set[str] = {
    # Home Assistant — read
    "ha_list_entities",
    "ha_get_entity_history",
    "ha_list_services",
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

# v4 ``act`` tier — actuator tools. The per-item ``action_allow_list`` always
# narrows this set; the handler validates non-empty overrides at item
# validation time so operators cannot accidentally grant the full surface.
# Explicitly excluded (out of v4 scope): ha_trigger_automation, ha_toggle_automation,
# create_memory, delete_memory.
ACT_TOOLS: Set[str] = {
    "ha_control_light",
    "ha_control_switch",
    "ha_control_climate",
    "ha_control_media_player",
    "ha_activate_scene",
    "ha_trigger_script",
    "ha_execute_service",
    "mass_play_media",
    "mass_playback_control",
}

# Tools that never reach the LLM under any autonomy tier (defense-in-depth
# block on top of the allow-list). v4 move: ACT_TOOLS are removed from this
# global deny set — they're gated per-tier below. Everything still-denied
# has no autonomy use case in v4.
V1_DENY: Set[str] = {
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
    if tier in ("notify", "speak"):
        # ``speak`` is a delivery-channel distinction — the tool surface
        # matches ``notify`` (SpeakerNotifier invokes MA outside the LLM turn).
        return OBSERVE_TOOLS | NOTIFY_ONLY_TOOLS
    if tier == "act":
        return OBSERVE_TOOLS | NOTIFY_ONLY_TOOLS | ACT_TOOLS
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
