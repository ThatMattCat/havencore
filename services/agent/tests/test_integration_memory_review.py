"""End-to-end test: consolidation run against live Qdrant.

Seeds ~20 synthetic L2 entries across two themes, triggers memory_review
via the engine's manual path, and asserts L3 entries appear carrying the
source texts (new absorb-on-consolidate behavior).

Cleanup runs unconditionally via a pytest fixture teardown, and uses the
`source: "test_integration"` payload tag so leftover rows from interrupted
runs are also swept up.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest


TEST_SOURCE_TAG = "test_integration"


def _qdrant_config():
    return {
        "host": os.getenv("QDRANT_HOST", "qdrant"),
        "port": int(os.getenv("QDRANT_PORT", "6333")),
        "coll": os.getenv("QDRANT_COLLECTION", "user_data"),
        "embed_url": os.getenv("EMBEDDINGS_URL", "http://embeddings:3000"),
    }


def _purge_test_rows(client, coll):
    """Delete every point with payload.source == TEST_SOURCE_TAG."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, PointIdsList,
    )
    flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=TEST_SOURCE_TAG))])
    offset = None
    to_delete: list[str] = []
    while True:
        pts, offset = client.scroll(
            collection_name=coll, scroll_filter=flt,
            limit=512, with_payload=False, offset=offset,
        )
        to_delete.extend(str(p.id) for p in pts)
        if offset is None:
            break
    if to_delete:
        client.delete(collection_name=coll, points_selector=PointIdsList(points=to_delete))
    return len(to_delete)


@pytest.fixture
def qdrant_test_env():
    """Provide qdrant client + config, and sweep test rows on teardown."""
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    cfg = _qdrant_config()
    client = QdrantClient(host=cfg["host"], port=cfg["port"])

    # Pre-sweep in case a prior run left stragglers.
    _purge_test_rows(client, cfg["coll"])

    try:
        yield client, cfg
    finally:
        _purge_test_rows(client, cfg["coll"])


@pytest.mark.asyncio
async def test_end_to_end_consolidation(qdrant_test_env):
    import requests
    from qdrant_client.models import (
        PointStruct, Filter, FieldCondition, MatchValue,
    )

    client, cfg = qdrant_test_env
    coll = cfg["coll"]
    embed_url = cfg["embed_url"]

    # Seed 20 L2 entries across 2 planted themes.
    theme_a = [
        "Matt drinks oat milk in coffee",
        "He dislikes dairy milk in hot drinks",
        "Oat milk again in morning coffee",
        "Oat milk at the cafe today",
        "Standard order: oat latte",
        "Oat milk preference confirmed",
        "Morning coffee with oat milk",
        "Prefers non-dairy milk",
        "Another oat milk coffee",
        "Oat milk remains the default",
    ]
    theme_b = [
        "Matt walks the dog at 7am",
        "Morning dog walk in the park",
        "Dog walk routine 7am",
        "Scheduled dog walk after breakfast",
        "Dog walk timing is consistent",
        "7am dog walk again",
        "Morning park walk with the dog",
        "Dog walk before work",
        "Early dog walk at 7",
        "Morning dog walk continues",
    ]

    # Use an ISO timestamp newer than the last memory_review run so step 3
    # (which filters by ``timestamp >= last_fired_at``) picks these up.
    seed_ts = datetime.now(timezone.utc).isoformat()
    ids: list[str] = []
    for text in theme_a + theme_b:
        r = requests.post(f"{embed_url}/embed", json={"inputs": text})
        r.raise_for_status()
        vec = r.json()[0]
        pid = str(uuid.uuid4())
        ids.append(pid)
        client.upsert(collection_name=coll, points=[PointStruct(
            id=pid, vector=vec, payload={
                "text": text, "tier": "L2", "importance": 3,
                "importance_effective": 3.0,
                "timestamp": seed_ts,
                "tags": [], "source_ids": [], "access_count": 0,
                "last_accessed_at": None, "pending_l4_approval": False,
                "proposed_at": None, "proposal_rationale": None,
                "source": TEST_SOURCE_TAG,
            },
        )])

    # Trigger the run via the live agent's HTTP endpoint. The agent process
    # owns the engine + DB pool (both init in FastAPI lifespan); importing
    # ``app`` from a pytest subprocess gives us an un-started copy with no
    # pool, so we go through the wire instead.
    agent_base = os.getenv("AGENT_BASE_URL", "http://localhost:6002")
    trigger = requests.post(f"{agent_base}/api/memory/runs/trigger", timeout=180)
    trigger.raise_for_status()
    result = trigger.json().get("result", {})
    assert result.get("status") == "ok", f"trigger result: {trigger.json()}"

    # Assert at least one L3 was created that references our seeded L2 ids.
    # New behavior: source L2s are absorbed (deleted) after L3 creation, with
    # text preserved in `source_texts`. We match on source_ids OR source_texts.
    seeded_set = set(ids)
    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    pts, _ = client.scroll(collection_name=coll, scroll_filter=flt,
                           limit=200, with_payload=True)

    def _overlaps(payload: dict) -> bool:
        src_ids = set(str(i) for i in (payload.get("source_ids") or []))
        if src_ids & seeded_set:
            return True
        src_texts = payload.get("source_texts") or []
        for entry in src_texts:
            if isinstance(entry, dict) and str(entry.get("id", "")) in seeded_set:
                return True
        return False

    matching = [p for p in pts if _overlaps(p.payload or {})]
    assert matching, "expected at least one L3 referencing seeded L2 set"

    # After absorption, the seeded L2s should be gone.
    remaining = client.retrieve(
        collection_name=coll, ids=ids, with_payload=False,
    )
    assert not remaining, (
        f"expected seeded L2 rows to be absorbed/deleted, but {len(remaining)} remain"
    )

    # Tag test-generated L3s so the fixture teardown sweeps them too.
    from qdrant_client.models import PointIdsList
    test_l3_ids = [str(p.id) for p in matching]
    client.set_payload(
        collection_name=coll,
        payload={"source": TEST_SOURCE_TAG},
        points=test_l3_ids,
    )
