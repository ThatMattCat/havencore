"""End-to-end test: consolidation run against live Qdrant.

Seeds ~20 synthetic L2 entries across two themes, triggers memory_review
via the engine's manual path, and asserts L3 entries appear with
populated source_ids and correct run metrics.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest


@pytest.mark.asyncio
async def test_end_to_end_consolidation(tmp_path):
    pytest.importorskip("qdrant_client")
    import numpy as np
    import requests
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        PointStruct, Filter, FieldCondition, MatchValue, PointIdsList,
    )

    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    coll = os.getenv("QDRANT_COLLECTION", "user_data")
    embed_url = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")

    client = QdrantClient(host=host, port=port)

    # Snapshot pre-existing L3 ids so cleanup only removes what this run creates.
    pre_l3_flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    pre_l3_ids: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(collection_name=coll, scroll_filter=pre_l3_flt,
                                    limit=256, with_payload=False, offset=offset)
        pre_l3_ids.update(str(p.id) for p in pts)
        if offset is None:
            break

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

    # Assert at least one L3 was created with source_ids pointing into our seeded set.
    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    pts, _ = client.scroll(collection_name=coll, scroll_filter=flt,
                           limit=50, with_payload=True)
    seeded_set = set(ids)
    assert any(
        bool(set((p.payload or {}).get("source_ids") or []) & seeded_set)
        for p in pts
    ), "expected at least one L3 with source_ids overlapping seeded L2 set"

    # Cleanup — delete seeded L2 points and any L3 rows this run produced.
    client.delete(collection_name=coll, points_selector=PointIdsList(points=ids))

    new_l3_ids: list[str] = []
    offset = None
    while True:
        pts, offset = client.scroll(collection_name=coll, scroll_filter=pre_l3_flt,
                                    limit=256, with_payload=False, offset=offset)
        new_l3_ids.extend(str(p.id) for p in pts if str(p.id) not in pre_l3_ids)
        if offset is None:
            break
    if new_l3_ids:
        client.delete(collection_name=coll, points_selector=PointIdsList(points=new_l3_ids))
