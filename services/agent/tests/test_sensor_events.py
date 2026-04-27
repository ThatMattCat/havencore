"""Tests for the autonomy sensor-event normalizer."""
from __future__ import annotations

import pytest

from selene_agent.autonomy import sensor_events


# --- Topic parsing -------------------------------------------------------

def test_parse_topic_face_unknown():
    assert sensor_events.parse_topic("haven/face/unknown") == ("face", "unknown")


def test_parse_topic_face_identified():
    assert sensor_events.parse_topic("haven/face/identified") == ("face", "identified")


def test_parse_topic_vehicle_kind():
    assert sensor_events.parse_topic("haven/vehicle/identified") == ("vehicle", "identified")


def test_parse_topic_rejects_trigger_subtopics():
    # haven/face/trigger/<camera> is an *input* to face-rec, not a normalized
    # output event — must not be picked up by the normalizer.
    assert sensor_events.parse_topic("haven/face/trigger/camera.x") is None


def test_parse_topic_rejects_non_haven_prefix():
    assert sensor_events.parse_topic("home/door/state") is None


def test_parse_topic_rejects_unknown_domain():
    assert sensor_events.parse_topic("haven/foobar/something") is None


# --- Face normalizer -----------------------------------------------------

@pytest.mark.asyncio
async def test_normalize_face_identified_full_payload():
    sensor_events.get_zone_cache().set_for_test({
        "camera.front_duo_3_fluent": ("front_door", "Front Door"),
    })
    payload = {
        "event_id": "evt-1",
        "camera": "camera.front_duo_3_fluent",
        "person_id": "p-1",
        "person_name": "Matt",
        "confidence": 0.91,
        "quality_score": 0.78,
        "snapshot_path": "2026/04/27/abc.jpg",
        "captured_at": "2026-04-27T18:00:00+00:00",
        "detection_id": "det-1",
    }
    out = await sensor_events.normalize("haven/face/identified", payload)
    assert out is not None
    assert out["source"] == "mqtt"
    assert out["topic"] == "haven/face/identified"
    se = out["sensor_event"]
    assert se["domain"] == "face"
    assert se["kind"] == "identified"
    assert se["zone"] == "front_door"
    assert se["zone_label"] == "Front Door"
    assert se["camera_entity"] == "camera.front_duo_3_fluent"
    assert se["subject"]["identity"] == "Matt"
    assert se["subject"]["confidence"] == pytest.approx(0.91)
    assert se["subject"]["quality"] == pytest.approx(0.78)
    assert se["snapshot_url"] is not None
    assert "/api/face/detections/det-1/snapshot" in se["snapshot_url"]
    assert se["raw"] == payload


@pytest.mark.asyncio
async def test_normalize_face_unknown_no_subject_identity():
    sensor_events.get_zone_cache().set_for_test({
        "camera.front_duo_3_fluent": ("front_door", None),
    })
    payload = {
        "event_id": "evt-2",
        "camera": "camera.front_duo_3_fluent",
        "snapshot_path": "2026/04/27/x.jpg",
        "quality_score": 0.6,
        "captured_at": "2026-04-27T18:01:00+00:00",
    }
    out = await sensor_events.normalize("haven/face/unknown", payload)
    se = out["sensor_event"]
    assert se["kind"] == "unknown"
    assert se["zone"] == "front_door"
    assert se["subject"]["type"] == "person"
    assert se["subject"]["identity"] is None
    assert se["subject"]["quality"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_normalize_face_unknown_camera_no_zone_mapped():
    sensor_events.get_zone_cache().set_for_test({})  # empty
    payload = {
        "event_id": "evt-3",
        "camera": "camera.unmapped",
        "snapshot_path": "x.jpg",
        "captured_at": "2026-04-27T18:02:00+00:00",
    }
    out = await sensor_events.normalize("haven/face/unknown", payload)
    se = out["sensor_event"]
    assert se["zone"] is None
    assert se["camera_entity"] == "camera.unmapped"


@pytest.mark.asyncio
async def test_normalize_face_no_face_kind():
    """haven/face/no_face — person sensor tripped, no face cleared the
    quality floor. Subject must flag no_face=True so the LLM knows the
    quality data is absent on purpose, not missing."""
    sensor_events.get_zone_cache().set_for_test({
        "camera.backyard_left_cam_fluent": ("backyard", "Backyard"),
    })
    payload = {
        "event_id": "evt-nf",
        "detection_id": "det-nf",
        "camera": "camera.backyard_left_cam_fluent",
        "snapshot_path": "2026/04/27/x.jpg",
        "frames_processed": 6,
        "captured_at": "2026-04-27T18:05:00+00:00",
    }
    out = await sensor_events.normalize("haven/face/no_face", payload)
    se = out["sensor_event"]
    assert se["kind"] == "no_face"
    assert se["zone"] == "backyard"
    assert se["subject"]["type"] == "unknown"
    assert se["subject"]["identity"] is None
    assert se["subject"]["no_face"] is True
    assert "/api/face/detections/det-nf/snapshot" in se["snapshot_url"]


@pytest.mark.asyncio
async def test_normalize_face_no_detection_id_means_no_snapshot_url():
    sensor_events.get_zone_cache().set_for_test({})
    payload = {
        "event_id": "evt-4",
        "camera": "camera.x",
        "captured_at": "2026-04-27T18:03:00+00:00",
    }
    out = await sensor_events.normalize("haven/face/unknown", payload)
    assert out["sensor_event"]["snapshot_url"] is None


# --- Topic schema fallthrough --------------------------------------------

@pytest.mark.asyncio
async def test_normalize_returns_none_for_non_haven_topic():
    out = await sensor_events.normalize("home/door/state", {"state": "open"})
    assert out is None


@pytest.mark.asyncio
async def test_normalize_passthrough_for_undefined_domain_kind():
    """Vehicle/motion/doorbell don't have a dedicated normalizer yet — they
    should still produce a SensorEvent rather than dropping the message."""
    sensor_events.get_zone_cache().set_for_test({
        "camera.driveway": ("driveway", "Driveway"),
    })
    payload = {
        "event_id": "v-1",
        "camera": "camera.driveway",
        "plate": "ABC123",
    }
    out = await sensor_events.normalize("haven/vehicle/unknown", payload)
    assert out is not None
    se = out["sensor_event"]
    assert se["domain"] == "vehicle"
    assert se["kind"] == "unknown"
    assert se["zone"] == "driveway"
    assert se["raw"]["plate"] == "ABC123"


@pytest.mark.asyncio
async def test_normalize_handles_non_dict_payload():
    """Some publishers may emit a JSON string or scalar; the normalizer must
    not crash and should wrap the value into raw.value."""
    sensor_events.get_zone_cache().set_for_test({})
    out = await sensor_events.normalize("haven/motion/detected", "ping")
    assert out is not None
    assert out["sensor_event"]["raw"] == {"value": "ping"}


# --- Zone cache fuzzy match ----------------------------------------------

@pytest.mark.asyncio
async def test_zone_lookup_falls_back_to_camera_prefix():
    """Some publishers may send a bare slug ('front_door') instead of the
    full HA entity ('camera.front_door'); the cache should still match."""
    sensor_events.get_zone_cache().set_for_test({
        "camera.front_door": ("front_door", "Front Door"),
    })
    zone, label = await sensor_events.get_zone_cache().lookup("front_door")
    assert zone == "front_door"
    assert label == "Front Door"
