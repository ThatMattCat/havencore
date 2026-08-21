"""Tests for on-disk face-image cleanup when a person (or an embedding) is removed.

Pure-function tests in the same spirit as `test_tiling.py` — no GPU, no
Postgres, no Qdrant, no live service. `db.delete_person` /
`db.evict_oldest_non_primary_face_image` are faked, `vector_store` calls are
no-op'd, and SNAPSHOT_DIR is redirected at `tmp_path`, so the only thing
under test is the path resolution + unlink accounting.

Regression cover for issue #40: face_images.path rows are stored RELATIVE to
SNAPSHOT_DIR, so unlinking the raw value resolves against the container CWD
(/app) and silently removes nothing.
"""

import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import config
import pipeline
from api import people


# ---- helpers -------------------------------------------------------------


def _use_snapshot_dir(monkeypatch, tmp_path: Path) -> None:
    """Point every SNAPSHOT_DIR reader at tmp_path."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(pipeline, "SNAPSHOT_DIR", tmp_path)


def _write_face_image(tmp_path: Path, rel: str) -> Path:
    """Create a fake enrollment JPEG at `rel` under the snapshot dir."""
    abs_path = tmp_path / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"\xff\xd8\xff\xd9")
    return abs_path


def _fake_delete_person(monkeypatch, rows: list[dict]):
    async def _delete_person(person_id):
        return rows

    monkeypatch.setattr(people.db, "delete_person", _delete_person)
    monkeypatch.setattr(
        people.vector_store, "delete_by_person", lambda person_id: None
    )


def _row(path: str) -> dict:
    return {"path": path, "qdrant_point_id": uuid.uuid4()}


# ---- delete_person: relative paths ---------------------------------------


def test_delete_person_unlinks_relative_paths_under_snapshot_dir(monkeypatch, tmp_path):
    """The stored path is relative to SNAPSHOT_DIR, not to the process CWD."""
    _use_snapshot_dir(monkeypatch, tmp_path)
    person_id = uuid.uuid4()

    rel_enroll = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    rel_auto = f"auto/{person_id}/{uuid.uuid4()}.jpg"
    f_enroll = _write_face_image(tmp_path, rel_enroll)
    f_auto = _write_face_image(tmp_path, rel_auto)

    _fake_delete_person(monkeypatch, [_row(rel_enroll), _row(rel_auto)])

    result = asyncio.run(people.delete_person(person_id=person_id))

    assert not f_enroll.exists(), "enrollment JPEG survived the person delete"
    assert not f_auto.exists(), "auto-improvement JPEG survived the person delete"
    assert result.images_removed == 2


def test_delete_person_images_removed_counts_only_real_unlinks(monkeypatch, tmp_path):
    """A row whose file is already gone must not inflate images_removed."""
    _use_snapshot_dir(monkeypatch, tmp_path)
    person_id = uuid.uuid4()

    rel_present = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    rel_missing = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    f_present = _write_face_image(tmp_path, rel_present)
    # rel_missing is deliberately never written to disk.

    _fake_delete_person(monkeypatch, [_row(rel_present), _row(rel_missing)])

    result = asyncio.run(people.delete_person(person_id=person_id))

    assert not f_present.exists()
    assert result.images_removed == 1, "already-missing row counted as removed"


def test_delete_person_resolves_legacy_absolute_rows(monkeypatch, tmp_path):
    """Pre-normalization rows stored an absolute path; both forms must work.

    `_resolve_face_image_path` returns absolute inputs as-is, so a legacy row
    and a normalized relative row have to be cleaned up in the same pass.
    """
    _use_snapshot_dir(monkeypatch, tmp_path)
    person_id = uuid.uuid4()

    rel_new = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    rel_legacy = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    f_new = _write_face_image(tmp_path, rel_new)
    f_legacy = _write_face_image(tmp_path, rel_legacy)

    _fake_delete_person(
        monkeypatch, [_row(rel_new), _row(str(f_legacy.resolve()))]
    )

    result = asyncio.run(people.delete_person(person_id=person_id))

    assert not f_new.exists(), "relative row survived the person delete"
    assert not f_legacy.exists(), "legacy absolute row survived the person delete"
    assert result.images_removed == 2


def test_delete_person_logs_and_skips_undeletable_path(monkeypatch, tmp_path, caplog):
    """A path that resolves but can't be unlinked is logged, not counted, not fatal."""
    _use_snapshot_dir(monkeypatch, tmp_path)
    person_id = uuid.uuid4()

    # A directory where a JPEG was expected: unlink() raises instead of
    # quietly no-op'ing, which is exactly the case that must not be counted.
    rel_dir = f"enrollments/{person_id}/{uuid.uuid4()}.jpg"
    (tmp_path / rel_dir).mkdir(parents=True)

    _fake_delete_person(monkeypatch, [_row(rel_dir)])

    with caplog.at_level("WARNING", logger="face-recognition.api.people"):
        result = asyncio.run(people.delete_person(person_id=person_id))

    assert result.images_removed == 0, "failed unlink counted as removed"
    assert (tmp_path / rel_dir).exists()
    assert any("Failed to unlink" in r.message for r in caplog.records), (
        "unlink failure was swallowed without a log line"
    )


# ---- pipeline FIFO eviction ----------------------------------------------


def test_evicted_auto_improvement_file_is_unlinked(monkeypatch, tmp_path):
    """FIFO eviction unlinks the evicted row's file under SNAPSHOT_DIR."""
    _use_snapshot_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "IMPROVEMENT_QUALITY_FLOOR", 0.0)
    monkeypatch.setattr(config, "IMPROVEMENT_THRESHOLD", 0.0)
    monkeypatch.setattr(config, "MAX_EMBEDDINGS_PER_PERSON", 1)

    person_id = uuid.uuid4()
    rel_evicted = f"auto/{person_id}/{uuid.uuid4()}.jpg"
    f_evicted = _write_face_image(tmp_path, rel_evicted)

    async def _count(pid):
        return 5

    async def _evict(pid):
        return {
            "id": uuid.uuid4(),
            "path": rel_evicted,
            "qdrant_point_id": uuid.uuid4(),
        }

    monkeypatch.setattr(pipeline.db, "count_face_images_for_person", _count)
    monkeypatch.setattr(pipeline.db, "evict_oldest_non_primary_face_image", _evict)
    monkeypatch.setattr(pipeline.vector_store, "delete_point", lambda pid: None)
    # Stop right after eviction: the new image never needs to hit disk.
    monkeypatch.setattr(
        pipeline, "_save_auto_improvement_image",
        lambda frame, person_id, face_image_id: Path("auto/new.jpg"),
    )

    def _boom(**kwargs):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(pipeline.vector_store, "upsert_point", _boom)

    candidate = SimpleNamespace(
        face=SimpleNamespace(normed_embedding=np.zeros(512, dtype=np.float32)),
    )

    contributed = asyncio.run(
        pipeline._maybe_contribute_embedding(
            person_id=person_id,
            detection_id=uuid.uuid4(),
            best_candidate=candidate,
            best_frame=np.zeros((8, 8, 3), dtype=np.uint8),
            best_quality=0.9,
            confidence=0.9,
        )
    )

    assert contributed is False
    assert not f_evicted.exists(), "evicted auto-improvement JPEG survived eviction"
