"""Tests for the v3 reactive trigger matcher."""
from __future__ import annotations

from selene_agent.autonomy import trigger_match


# --- MQTT topic matching -------------------------------------------------

def test_mqtt_exact_topic_matches():
    spec = {"source": "mqtt", "match": {"topic": "home/door/front/state"}}
    event = {"source": "mqtt", "topic": "home/door/front/state", "payload": {}}
    assert trigger_match.match(spec, event) is True


def test_mqtt_plus_wildcard_matches_single_level():
    spec = {"source": "mqtt", "match": {"topic": "home/door/+/state"}}
    event = {"source": "mqtt", "topic": "home/door/back/state", "payload": {}}
    assert trigger_match.match(spec, event) is True


def test_mqtt_plus_wildcard_rejects_level_count_mismatch():
    spec = {"source": "mqtt", "match": {"topic": "home/+/state"}}
    event = {"source": "mqtt", "topic": "home/door/front/state", "payload": {}}
    assert trigger_match.match(spec, event) is False


def test_mqtt_hash_is_not_supported():
    spec = {"source": "mqtt", "match": {"topic": "home/#"}}
    event = {"source": "mqtt", "topic": "home/door/front/state", "payload": {}}
    # '#' is a literal, so it won't match a real multi-level topic.
    assert trigger_match.match(spec, event) is False


def test_mqtt_topic_mismatch_returns_false():
    spec = {"source": "mqtt", "match": {"topic": "home/door/front/state"}}
    event = {"source": "mqtt", "topic": "home/door/back/state", "payload": {}}
    assert trigger_match.match(spec, event) is False


# --- Payload subset matching --------------------------------------------

def test_mqtt_payload_subset_matches_when_keys_equal():
    spec = {
        "source": "mqtt",
        "match": {"topic": "home/door/front/state", "payload": {"state": "open"}},
    }
    event = {
        "source": "mqtt",
        "topic": "home/door/front/state",
        "payload": {"state": "open", "battery": 90},
    }
    assert trigger_match.match(spec, event) is True


def test_mqtt_payload_missing_key_rejects():
    spec = {
        "source": "mqtt",
        "match": {"topic": "home/+/state", "payload": {"state": "open"}},
    }
    event = {"source": "mqtt", "topic": "home/door/state", "payload": {"battery": 90}}
    assert trigger_match.match(spec, event) is False


def test_mqtt_payload_value_mismatch_rejects():
    spec = {
        "source": "mqtt",
        "match": {"topic": "home/+/state", "payload": {"state": "open"}},
    }
    event = {"source": "mqtt", "topic": "home/door/state", "payload": {"state": "closed"}}
    assert trigger_match.match(spec, event) is False


def test_mqtt_payload_recursive_subset_matches():
    spec = {
        "source": "mqtt",
        "match": {"topic": "a/b", "payload": {"outer": {"inner": 1}}},
    }
    event = {
        "source": "mqtt",
        "topic": "a/b",
        "payload": {"outer": {"inner": 1, "extra": 2}, "other": 3},
    }
    assert trigger_match.match(spec, event) is True


def test_mqtt_empty_payload_filter_always_matches():
    spec = {"source": "mqtt", "match": {"topic": "a/b"}}
    event = {"source": "mqtt", "topic": "a/b", "payload": {"anything": True}}
    assert trigger_match.match(spec, event) is True


# --- Webhook matching ---------------------------------------------------

def test_webhook_name_match():
    spec = {"source": "webhook", "match": {"name": "front-door"}}
    event = {"source": "webhook", "name": "front-door", "payload": {}}
    assert trigger_match.match(spec, event) is True


def test_webhook_name_mismatch_rejects():
    spec = {"source": "webhook", "match": {"name": "front-door"}}
    event = {"source": "webhook", "name": "back-door", "payload": {}}
    assert trigger_match.match(spec, event) is False


def test_webhook_payload_filter_applied():
    spec = {
        "source": "webhook",
        "match": {"name": "door", "payload": {"action": "open"}},
    }
    ok = {"source": "webhook", "name": "door", "payload": {"action": "open"}}
    no = {"source": "webhook", "name": "door", "payload": {"action": "close"}}
    assert trigger_match.match(spec, ok) is True
    assert trigger_match.match(spec, no) is False


# --- Edge cases ---------------------------------------------------------

def test_non_dict_inputs_return_false():
    assert trigger_match.match(None, {}) is False
    assert trigger_match.match({}, None) is False
    assert trigger_match.match("nope", {}) is False


def test_unknown_source_returns_false():
    spec = {"source": "email", "match": {}}
    event = {"source": "email"}
    assert trigger_match.match(spec, event) is False
