"""FOV-aware re-embed paths (issue #55).

The live pipeline tiles `panoramic_dual_lens` frames before InsightFace
because a single full-frame pass downscales a 7680x2160 Reolink panorama
~6x at DET_SIZE, shrinking the faces the tiler exists for below the
detector's floor. Every *re*-embed path used to skip that decision and run
plain single-pass detection on the same full frame:

  * POST /api/detections/{id}/confirm      (api/detections.py)
  * the rescan-unknowns admin job          (api/admin.py)
  * POST /api/people/{id}/enroll-from-camera (api/people.py)

All three seeded cameras in `db.CAMERA_SEED_SQL` are panoramic, so this was
the deployment's only realistic path: confirms landed with
`embedding_contributed=false`, and when a second, closer person shared the
panorama the full-frame pass could enroll *their* embedding under the
confirmed name.

Same house style as `test_tiling.py` / `test_person_delete_cleanup.py`:
pure-function, no GPU / Postgres / Qdrant. `embedder.detect_and_embed` is
faked per test so we control exactly which faces "appear" at which frame
width, and the async endpoints are driven with `asyncio.run`.
"""

import asyncio
import time
import uuid
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import config
import pipeline
from api import admin, detections, people
from models import ConfirmDetectionRequest, EnrollFromCameraRequest


PANO_CAMERA = "camera.front_duo_3_clear"

# A 4000-wide frame splits at half (2000) with 5% overlap → both tiles are
# 2200 wide, the right one starting at x=1800. Mirrors test_tiling.py.
FULL_W = 4000
TILE_W = 2200
RIGHT_OFFSET = 1800
FRAME_H = 200


# ---- fakes ---------------------------------------------------------------


def _make_face(bbox, det_score=0.9, embedding=None):
    """Mimic the insightface Face surface the re-embed paths touch."""
    if embedding is None:
        embedding = [1.0, 0.0, 0.0, 0.0]
    return SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float32),
        kps=None,
        det_score=float(det_score),
        normed_embedding=np.array(embedding, dtype=np.float32),
    )


def _make_detector(returns_by_width):
    """Fake `embedder.detect_and_embed` keyed by the width of the array it is
    handed, so a test can say "nothing is visible in the full 4000px frame,
    but there is a face in the 2200px tile" — precisely the condition tiling
    was added for.

    `returns_by_width` maps width → list of face-lists, consumed in call
    order for that width. Unlisted widths (and exhausted scripts) yield no
    faces. `.calls` records the width of every call in order.
    """
    calls: list[int] = []
    seen: dict[int, int] = {}

    def fake(frame):
        w = int(frame.shape[1])
        calls.append(w)
        idx = seen.get(w, 0)
        seen[w] = idx + 1
        script = returns_by_width.get(w, [])
        return list(script[idx]) if idx < len(script) else []

    fake.calls = calls
    return fake


def _install(monkeypatch, detector, *, fov_type="panoramic_dual_lens"):
    """Wire a fake detector + flat quality score + camera row into pipeline."""
    monkeypatch.setattr(pipeline.embedder, "detect_and_embed", detector)
    # Well above QUALITY_FLOOR (0.40) — these tests are about *finding* the
    # face, not about the quality policy.
    monkeypatch.setattr(pipeline, "score_face", lambda frame, face: 0.8)

    async def _get_camera(entity_id):
        if fov_type is None:
            return None
        return {"entity_id": entity_id, "fov_type": fov_type}

    monkeypatch.setattr(pipeline.db, "get_camera", _get_camera)


def _panorama(tmp_path, rel="2026/01/02/snap.jpg"):
    """Write a real full-panorama JPEG so cv2.imread returns a 4000px frame."""
    abs_path = tmp_path / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((FRAME_H, FULL_W, 3), dtype=np.uint8)
    assert cv2.imwrite(str(abs_path), img)
    return rel


# ---- confirm -------------------------------------------------------------


def _confirm_fixture(monkeypatch, tmp_path, rel):
    """Stub every DB/Qdrant touchpoint of the confirm endpoint and return the
    list that captures `_persist_enrollment` kwargs."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path))
    person_id = uuid.uuid4()
    detection_id = uuid.uuid4()

    async def _get_detection(_id):
        return {"id": detection_id, "snapshot_path": rel, "camera": PANO_CAMERA}

    async def _get_person(_id):
        return {"id": person_id, "name": "Ada"}

    async def _confirm(**kwargs):
        return {"id": detection_id}

    monkeypatch.setattr(detections.db, "get_detection", _get_detection)
    monkeypatch.setattr(detections.db, "get_person", _get_person)
    monkeypatch.setattr(detections.db, "confirm_detection", _confirm)

    enrolled: list[dict] = []

    async def _persist(**kwargs):
        enrolled.append(kwargs)
        return {"row": {"id": uuid.uuid4()}, "qdrant_point_id": uuid.uuid4()}

    monkeypatch.setattr(detections, "_persist_enrollment", _persist)
    return person_id, detection_id, enrolled


def test_confirm_tiles_panoramic_snapshot(monkeypatch, tmp_path):
    """Core regression: the face is invisible in the full frame and only
    detectable inside a tile. Pre-fix the confirm marked the row confirmed
    with embedding_contributed=false and the operator's label never reached
    the gallery."""
    rel = _panorama(tmp_path)
    detector = _make_detector({TILE_W: [[_make_face([100, 20, 200, 120])], []]})
    _install(monkeypatch, detector)
    person_id, detection_id, enrolled = _confirm_fixture(monkeypatch, tmp_path, rel)

    result = asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    # Detection ran per tile, never once on the full 4000px frame.
    assert detector.calls == [TILE_W, TILE_W]
    assert result.embedding_contributed is True
    assert result.quality_score == pytest.approx(0.8)
    assert len(enrolled) == 1


def test_confirm_panoramic_bboxes_are_full_frame_coords(monkeypatch, tmp_path):
    """A face found in the RIGHT tile must come back in full-frame space.
    Tile-local coordinates would put the face 1800px off, which is the
    "enrolls the wrong region/person" failure mode."""
    rel = _panorama(tmp_path)
    subject = _make_face([100, 20, 300, 120])  # tile-local coords
    detector = _make_detector({TILE_W: [[], [subject]]})  # left empty, right hit
    _install(monkeypatch, detector)
    person_id, detection_id, enrolled = _confirm_fixture(monkeypatch, tmp_path, rel)

    result = asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    assert result.embedding_contributed is True
    assert len(enrolled) == 1
    # `_translate_face` rewrites the Face in place: the tile-local
    # [100, 20, 300, 120] must read [1900, 20, 2100, 120] in panorama space.
    assert subject.bbox.tolist() == [
        100.0 + RIGHT_OFFSET, 20.0, 300.0 + RIGHT_OFFSET, 120.0,
    ]


def test_confirm_panoramic_ignores_the_full_frame_decoy(monkeypatch, tmp_path):
    """Two people in the panorama: a closer decoy the full-frame pass can
    resolve, and the actual subject only visible in a tile. Pre-fix the
    single full-frame pass enrolled the DECOY's embedding under the
    operator's confirmed name, poisoning matching."""
    rel = _panorama(tmp_path)
    decoy = _make_face([10, 10, 60, 60], det_score=0.99, embedding=[0.0, 1.0, 0.0, 0.0])
    subject = _make_face(
        [100, 20, 300, 120], det_score=0.7, embedding=[1.0, 0.0, 0.0, 0.0]
    )
    detector = _make_detector({FULL_W: [[decoy]], TILE_W: [[], [subject]]})
    _install(monkeypatch, detector)
    person_id, detection_id, enrolled = _confirm_fixture(monkeypatch, tmp_path, rel)

    asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    assert len(enrolled) == 1
    assert enrolled[0]["embedding"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # And the subject's bbox is expressed in panorama coords, not tile-local.
    assert subject.bbox.tolist() == [
        100.0 + RIGHT_OFFSET, 20.0, 300.0 + RIGHT_OFFSET, 120.0,
    ]


def test_confirm_standard_camera_stays_single_pass(monkeypatch, tmp_path):
    """Guard-rail: a non-panoramic camera must NOT be tiled — one call, on
    the whole frame."""
    rel = _panorama(tmp_path)
    detector = _make_detector({FULL_W: [[_make_face([10, 10, 60, 60])]]})
    _install(monkeypatch, detector, fov_type="standard")
    person_id, detection_id, enrolled = _confirm_fixture(monkeypatch, tmp_path, rel)

    result = asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    assert detector.calls == [FULL_W]
    assert result.embedding_contributed is True
    assert len(enrolled) == 1


def test_confirm_survives_missing_camera_row(monkeypatch, tmp_path):
    """A detection whose camera row was deleted (or was never registered)
    falls back to 'standard' and still completes — the confirm endpoint must
    not 500 because of a camera lookup."""
    rel = _panorama(tmp_path)
    detector = _make_detector({FULL_W: [[_make_face([10, 10, 60, 60])]]})
    _install(monkeypatch, detector, fov_type=None)  # get_camera returns None
    person_id, detection_id, _ = _confirm_fixture(monkeypatch, tmp_path, rel)

    result = asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    assert detector.calls == [FULL_W]
    assert result.embedding_contributed is True


def test_confirm_survives_camera_lookup_failure(monkeypatch, tmp_path):
    """A DB hiccup during the FOV lookup degrades to 'standard' instead of
    propagating out of the endpoint."""
    rel = _panorama(tmp_path)
    detector = _make_detector({FULL_W: [[_make_face([10, 10, 60, 60])]]})
    _install(monkeypatch, detector)

    async def _boom(entity_id):
        raise RuntimeError("pool is closed")

    monkeypatch.setattr(pipeline.db, "get_camera", _boom)
    person_id, detection_id, _ = _confirm_fixture(monkeypatch, tmp_path, rel)

    result = asyncio.run(
        detections.confirm_detection(
            ConfirmDetectionRequest(person_id=person_id), detection_id
        )
    )

    assert detector.calls == [FULL_W]
    assert result.embedding_contributed is True


# ---- resolve_fov_type ----------------------------------------------------


def test_resolve_fov_type_defaults_to_standard_for_null_camera(monkeypatch):
    assert asyncio.run(pipeline.resolve_fov_type(None)) == "standard"


def test_resolve_fov_type_reads_the_camera_row(monkeypatch):
    async def _get_camera(entity_id):
        return {"entity_id": entity_id, "fov_type": "panoramic_dual_lens"}

    monkeypatch.setattr(pipeline.db, "get_camera", _get_camera)
    assert (
        asyncio.run(pipeline.resolve_fov_type(PANO_CAMERA)) == "panoramic_dual_lens"
    )


# ---- rescan-unknowns -----------------------------------------------------


def _rescan_job(job_id):
    return {
        "job_id": job_id,
        "type": "rescan_unknowns",
        "status": "running",
        "phase": "pending",
        "started_at": time.time(),
        "finished_at": None,
        "elapsed_ms": 0,
        "totals": {
            "examined": 0,
            "matched": 0,
            "no_match": 0,
            "contributed": 0,
            "skipped_missing_snapshot": 0,
            "errors": 0,
        },
        "errors": [],
    }


def test_rescan_tiles_panoramic_snapshots(monkeypatch, tmp_path):
    """The rescan job's query embedding must come from a tiled detection.
    Pre-fix it re-detected single-pass on the full panorama, found nothing,
    and counted every unknown as no_match — making the job near-useless for
    the only camera type this deployment has."""
    rel = _panorama(tmp_path, rel="2026/01/03/unknown.jpg")
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(admin, "SNAPSHOT_DIR", tmp_path)

    subject = _make_face([100, 20, 300, 120], embedding=[1.0, 0.0, 0.0, 0.0])
    detector = _make_detector({TILE_W: [[subject], []]})
    _install(monkeypatch, detector)

    detection_id = uuid.uuid4()
    person_id = uuid.uuid4()
    batches = [[{"id": detection_id, "snapshot_path": rel, "camera": PANO_CAMERA}], []]

    async def _list_detections(**kwargs):
        return batches.pop(0) if batches else []

    async def _get_person(pid):
        return {"id": pid, "name": "Ada"}

    confirmed: list[dict] = []

    async def _confirm(**kwargs):
        confirmed.append(kwargs)
        return {"id": detection_id}

    async def _contribute(**kwargs):
        return True

    monkeypatch.setattr(admin.db, "list_detections", _list_detections)
    monkeypatch.setattr(admin.db, "get_person", _get_person)
    monkeypatch.setattr(admin.db, "confirm_detection", _confirm)
    monkeypatch.setattr(
        admin.pipeline, "contribute_embedding_for_detection", _contribute
    )
    monkeypatch.setattr(
        admin.vector_store,
        "query",
        lambda vec, k: [
            SimpleNamespace(score=0.95, payload={"person_id": str(person_id)})
        ],
    )

    job_id = "test-rescan"
    monkeypatch.setitem(admin._jobs, job_id, _rescan_job(job_id))
    asyncio.run(admin._run_rescan(job_id))

    job = admin._jobs[job_id]
    assert detector.calls == [TILE_W, TILE_W]
    assert job["totals"]["matched"] == 1
    assert job["totals"]["no_match"] == 0
    assert job["totals"]["contributed"] == 1
    assert confirmed and confirmed[0]["person_id"] == person_id


# ---- enroll-from-camera --------------------------------------------------


def test_enroll_from_camera_tiles_panoramic_burst(monkeypatch, tmp_path):
    """The burst path takes a list of frames rather than one saved frame, but
    needs the same tiling. Pre-fix this raised 422 'no face cleared quality
    floor' for every panoramic camera."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path))
    subject = _make_face([100, 20, 300, 120], embedding=[1.0, 0.0, 0.0, 0.0])
    # Two frames × two tiles = four calls; only the second frame's left tile
    # actually resolves the face.
    detector = _make_detector({TILE_W: [[], [], [subject], []]})
    _install(monkeypatch, detector)

    person_id = uuid.uuid4()
    frames = [np.zeros((FRAME_H, FULL_W, 3), dtype=np.uint8) for _ in range(2)]

    async def _get_person(_id):
        return {"id": person_id, "name": "Ada"}

    async def _capture_burst(**kwargs):
        return frames

    enrolled: list[dict] = []

    async def _persist(**kwargs):
        enrolled.append(kwargs)
        return {"row": {"id": uuid.uuid4()}, "qdrant_point_id": uuid.uuid4()}

    monkeypatch.setattr(people.db, "get_person", _get_person)
    monkeypatch.setattr(people.ha_snapshot, "capture_burst", _capture_burst)
    monkeypatch.setattr(people, "_persist_enrollment", _persist)

    result = asyncio.run(
        people.enroll_from_camera(
            EnrollFromCameraRequest(camera=PANO_CAMERA), person_id
        )
    )

    assert detector.calls == [TILE_W, TILE_W, TILE_W, TILE_W]
    assert result.faces_kept == 1
    assert result.frames_processed == 2
    assert result.quality_score == pytest.approx(0.8)
    # Enrolled from frame index 1 — the frame the face was actually found in.
    assert len(enrolled) == 1
    assert enrolled[0]["frame"] is frames[1]


def test_enroll_from_camera_standard_stays_single_pass(monkeypatch, tmp_path):
    """Guard-rail: a standard camera's burst is detected frame-by-frame, one
    call per frame, no tiling."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path))
    detector = _make_detector({FULL_W: [[], [_make_face([10, 10, 60, 60])]]})
    _install(monkeypatch, detector, fov_type="standard")

    person_id = uuid.uuid4()
    frames = [np.zeros((FRAME_H, FULL_W, 3), dtype=np.uint8) for _ in range(2)]

    async def _get_person(_id):
        return {"id": person_id, "name": "Ada"}

    async def _capture_burst(**kwargs):
        return frames

    async def _persist(**kwargs):
        return {"row": {"id": uuid.uuid4()}, "qdrant_point_id": uuid.uuid4()}

    monkeypatch.setattr(people.db, "get_person", _get_person)
    monkeypatch.setattr(people.ha_snapshot, "capture_burst", _capture_burst)
    monkeypatch.setattr(people, "_persist_enrollment", _persist)

    result = asyncio.run(
        people.enroll_from_camera(
            EnrollFromCameraRequest(camera="camera.doorbell"), person_id
        )
    )

    assert detector.calls == [FULL_W, FULL_W]
    assert result.faces_kept == 1
