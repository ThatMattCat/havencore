"""Deterministic event matcher for v3 reactive agenda items.

The matcher is intentionally small: strict source equality, single-level
``+`` wildcard for MQTT topics, and an AND subset-filter on payload keys.
Handlers may evaluate richer conditions *after* dispatch; this layer only
answers "is this event in scope for this item?"
"""
from __future__ import annotations

from typing import Any, Dict


def _topic_matches(pattern: str, topic: str) -> bool:
    """MQTT topic match with single-level ``+`` wildcard only.

    ``+`` matches exactly one level; ``#`` is intentionally not supported
    (would invite wide fan-out on a single misconfigured rule).
    """
    p_parts = pattern.split("/")
    t_parts = topic.split("/")
    if len(p_parts) != len(t_parts):
        return False
    for p, t in zip(p_parts, t_parts):
        if p == "+":
            continue
        if p != t:
            return False
    return True


def _payload_subset(expected: Any, incoming: Any) -> bool:
    """Every key in ``expected`` must be equal (recursively) in ``incoming``.

    Non-dict ``expected`` is compared by strict equality. Non-dict
    ``incoming`` under a dict expectation is treated as no-match.
    """
    if isinstance(expected, dict):
        if not isinstance(incoming, dict):
            return False
        for key, val in expected.items():
            if key not in incoming:
                return False
            if not _payload_subset(val, incoming[key]):
                return False
        return True
    return expected == incoming


def match(trigger_spec: Dict[str, Any], event: Dict[str, Any]) -> bool:
    """Return True iff ``event`` satisfies ``trigger_spec``.

    ``trigger_spec`` shape::

        {"source": "mqtt"|"webhook", "match": {...}}

    Event shape::

        mqtt:    {"topic": str, "payload": Any}
        webhook: {"name": str,  "payload": dict}
    """
    if not isinstance(trigger_spec, dict) or not isinstance(event, dict):
        return False
    source = trigger_spec.get("source")
    if source != event.get("source") and source not in (event.get("source"), None):
        # Allow callers to omit ``source`` on the event when they already
        # know which source they're dispatching from (webhook/mqtt endpoints
        # stamp their own source internally before calling this helper).
        pass
    spec_match = trigger_spec.get("match") or {}
    if source == "mqtt":
        topic_pattern = spec_match.get("topic")
        if not topic_pattern:
            return False
        if not _topic_matches(topic_pattern, event.get("topic", "")):
            return False
        payload_filter = spec_match.get("payload")
        if payload_filter is None:
            return True
        return _payload_subset(payload_filter, event.get("payload"))
    if source == "webhook":
        name = spec_match.get("name")
        if not name or name != event.get("name"):
            return False
        payload_filter = spec_match.get("payload")
        if payload_filter is None:
            return True
        return _payload_subset(payload_filter, event.get("payload"))
    return False
