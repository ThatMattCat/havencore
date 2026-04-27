"""Tests for the panoramic tiling + IoU dedup helpers in pipeline.py.

Pure-function tests — no GPU, no Postgres, no MQTT. The single InsightFace
dependency (`embedder.detect_and_embed`) gets monkey-patched per test so we
control exactly which faces "appear" in each tile and assert on translation
+ dedup behavior.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import pipeline


# ---- _iou ----------------------------------------------------------------


def test_iou_disjoint_bboxes_is_zero():
    a = (0, 0, 100, 100)
    b = (200, 200, 300, 300)
    assert pipeline._iou(a, b) == 0.0


def test_iou_identical_bboxes_is_one():
    a = (10, 10, 50, 50)
    b = (10, 10, 50, 50)
    assert pipeline._iou(a, b) == pytest.approx(1.0)


def test_iou_half_overlap_horizontal():
    # Both 100x100, B shifted +50 on x. Intersection 50x100=5000,
    # union = 10000+10000-5000 = 15000 → 1/3.
    a = (0, 0, 100, 100)
    b = (50, 0, 150, 100)
    assert pipeline._iou(a, b) == pytest.approx(1 / 3)


def test_iou_touching_edges_no_area():
    # Sharing an edge is zero area intersection.
    a = (0, 0, 100, 100)
    b = (100, 0, 200, 100)
    assert pipeline._iou(a, b) == 0.0


def test_iou_accepts_numpy_arrays():
    a = np.array([0.0, 0.0, 100.0, 100.0])
    b = np.array([50.0, 50.0, 150.0, 150.0])
    # 50x50 intersection = 2500, union = 10000+10000-2500 = 17500 → 1/7.
    assert pipeline._iou(a, b) == pytest.approx(2500 / 17500)


# ---- _translate_face -----------------------------------------------------


def _make_face(bbox, kps=None, det_score=0.9):
    """Mimic the interface insightface.Face exposes (bbox, kps, det_score)."""
    return SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float32),
        kps=None if kps is None else np.array(kps, dtype=np.float32),
        det_score=float(det_score),
    )


def test_translate_face_shifts_bbox_x_only():
    face = _make_face([10, 20, 60, 80])
    pipeline._translate_face(face, offset_x=100)
    assert face.bbox.tolist() == [110.0, 20.0, 160.0, 80.0]


def test_translate_face_shifts_kps_x_only():
    face = _make_face(
        [10, 20, 60, 80],
        kps=[[20, 30], [40, 30], [30, 50], [25, 65], [35, 65]],
    )
    pipeline._translate_face(face, offset_x=100)
    expected_kps = [[120, 30], [140, 30], [130, 50], [125, 65], [135, 65]]
    assert face.kps.tolist() == expected_kps


def test_translate_face_zero_offset_is_noop():
    face = _make_face([10, 20, 60, 80], kps=[[20, 30]] * 5)
    bbox_before = face.bbox.copy()
    kps_before = face.kps.copy()
    pipeline._translate_face(face, offset_x=0)
    assert (face.bbox == bbox_before).all()
    assert (face.kps == kps_before).all()


def test_translate_face_handles_missing_kps():
    # Should not crash when face.kps is None.
    face = _make_face([10, 20, 60, 80], kps=None)
    pipeline._translate_face(face, offset_x=50)
    assert face.bbox.tolist() == [60.0, 20.0, 110.0, 80.0]


# ---- _detect_panoramic ---------------------------------------------------


def test_detect_panoramic_falls_back_for_narrow_frames(monkeypatch):
    """Frames narrower than 1024 are processed in one pass — tiling them
    would lose more to the per-tile resize than it gains."""
    calls = []

    def fake_detect(tile):
        calls.append(tile.shape)
        return [_make_face([10, 10, 50, 50])]

    monkeypatch.setattr(pipeline.embedder, "detect_and_embed", fake_detect)
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    faces = pipeline._detect_panoramic(frame)

    assert len(calls) == 1
    assert calls[0] == (480, 800, 3)
    assert len(faces) == 1


def _scripted_detect(*per_call_returns):
    """Build a fake embedder.detect_and_embed that returns a different face
    list per call, keyed by call order. _detect_panoramic always invokes
    the left tile first, then the right, so per_call_returns[0] is left and
    per_call_returns[1] is right."""
    state = {"i": 0, "shapes": []}

    def fake(tile):
        state["shapes"].append(tile.shape)
        idx = state["i"]
        state["i"] += 1
        return per_call_returns[idx]

    return fake, state


def test_detect_panoramic_runs_per_tile_and_translates_right_bboxes(monkeypatch):
    """A 4000-wide frame splits at half (2000) with 5% overlap on each side
    → left tile [0:2200], right tile [1800:4000] (both width 2200). Faces
    from the right tile must come back with bboxes shifted +1800 along x."""
    fake, state = _scripted_detect(
        [_make_face([100, 50, 200, 150], det_score=0.95)],   # left
        [_make_face([100, 200, 300, 400], det_score=0.85)],  # right
    )
    monkeypatch.setattr(pipeline.embedder, "detect_and_embed", fake)
    frame = np.zeros((1080, 4000, 3), dtype=np.uint8)
    faces = pipeline._detect_panoramic(frame)

    # Both tiles ran, both 2200 wide.
    assert [s[1] for s in state["shapes"]] == [2200, 2200]
    # Two faces, both retained (no overlap).
    assert len(faces) == 2
    bboxes = sorted(f.bbox.tolist() for f in faces)
    # Left face untranslated; right face shifted by 1800 (=half - overlap).
    assert bboxes == [[100.0, 50.0, 200.0, 150.0], [1900.0, 200.0, 2100.0, 400.0]]


def test_detect_panoramic_dedupes_overlap_zone(monkeypatch):
    """A face landing in the seam shows up in both tiles. The dedup keeps
    the higher det_score and drops the duplicate."""
    # Same face: left-tile-local [2050,100,2150,200] (also global since left
    # tile starts at x=0); right-tile-local [250,100,350,200] → global
    # [2050,100,2150,200] after +1800 translation.
    fake, _ = _scripted_detect(
        [_make_face([2050, 100, 2150, 200], det_score=0.92)],  # left
        [_make_face([250, 100, 350, 200], det_score=0.80)],    # right
    )
    monkeypatch.setattr(pipeline.embedder, "detect_and_embed", fake)
    frame = np.zeros((1080, 4000, 3), dtype=np.uint8)
    faces = pipeline._detect_panoramic(frame)

    assert len(faces) == 1
    assert faces[0].det_score == pytest.approx(0.92)
    assert faces[0].bbox.tolist() == [2050.0, 100.0, 2150.0, 200.0]


def test_detect_panoramic_keeps_distinct_faces_in_overlap_zone(monkeypatch):
    """Two genuinely different faces, both happening to land inside the seam
    overlap, must both survive — IoU between them is below threshold."""
    fake, _ = _scripted_detect(
        [_make_face([2050, 100, 2150, 200], det_score=0.90)],  # left
        # Different face in the right tile, no spatial overlap with the left.
        [_make_face([100, 500, 200, 600], det_score=0.85)],    # right
    )
    monkeypatch.setattr(pipeline.embedder, "detect_and_embed", fake)
    frame = np.zeros((1080, 4000, 3), dtype=np.uint8)
    faces = pipeline._detect_panoramic(frame)

    assert len(faces) == 2
