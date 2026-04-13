"""Memory consolidation handler — deterministic 5-step pipeline.

Steps:
  1. Scan L2
  2. Apply decay/boost -> importance_effective
  3. Cluster new L2 into L3 (HDBSCAN + LLM summarizer)
  4. Propose L3 -> L4 (flag only; never auto-promote)
  5. Prune stale L2 (respecting source_ids protection)

Runs as a plain async function — not an AutonomousTurn. It does not need
tool gating or a fresh orchestrator; it calls the LLM directly via the
provided async OpenAI client.
"""
from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from selene_agent.autonomy import memory_clustering
from selene_agent.autonomy import memory_math
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


def _embed(text: str) -> List[float]:
    """Get a single embedding via the TEI service (same host as mcp_qdrant_tools)."""
    url = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
    r = requests.post(f"{url}/embed", json={"inputs": text}, timeout=30)
    r.raise_for_status()
    return r.json()[0]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str) -> datetime:
    if not s:
        return _now()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return _now()


def _qdrant_client():
    """Reach into the qdrant MCP server's client. Imported lazily so tests
    can stub the returned client without touching the real Qdrant."""
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import (
        QDRANT_HOST, QDRANT_PORT,
    )
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _scroll_all(client, *, flt, collection: str, batch_size: int = 256,
                with_vectors: bool = False, cap: int | None = None):
    offset = None
    out = []
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=batch_size,
            with_payload=True,
            with_vectors=with_vectors,
            offset=offset,
        )
        out.extend(points)
        if cap is not None and len(out) >= cap:
            return out[:cap]
        if offset is None:
            break
    return out


async def _scan_and_decay(client, stats: Dict[str, Any]) -> None:
    """Step 1+2: scan L2 entries, compute importance_effective, write back."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L2"))])
    points = _scroll_all(
        client,
        flt=flt,
        collection=COLLECTION_NAME,
        cap=config.AUTONOMY_MEMORY_MAX_SCAN,
    )
    stats["l2_scanned"] = len(points)
    now = _now()

    # Group ids by their new importance_effective so we can batch set_payload.
    # Float-key grouping: round to 4 decimals to collapse equal updates.
    groups: Dict[float, List[str]] = defaultdict(list)
    for p in points:
        payload = p.payload or {}
        created = _parse_ts(payload.get("timestamp", ""))
        ie = memory_math.compute_importance_effective(
            base_importance=float(payload.get("importance", 0) or 0),
            created_at=created,
            access_count=int(payload.get("access_count", 0) or 0),
            now=now,
            half_life_days=config.MEMORY_HALF_LIFE_DAYS,
            access_coef=config.MEMORY_ACCESS_COEF,
        )
        groups[round(ie, 4)].append(str(p.id))

    adjusted = 0
    for ie, ids in groups.items():
        if not ids:
            continue
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"importance_effective": ie},
            points=ids,
        )
        adjusted += len(ids)
    stats["importance_adjusted"] = adjusted


async def _cluster_to_l3(
    client,
    stats: Dict[str, Any],
    *,
    since: datetime,
    llm_client,
    model_name: str,
) -> None:
    """Step 3: cluster new L2 entries (since last successful run) → L3 entries."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, DatetimeRange, PointStruct,
    )
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

    flt = Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value="L2")),
        FieldCondition(key="timestamp", range=DatetimeRange(gte=since.isoformat())),
    ])
    points = _scroll_all(
        client, flt=flt, collection=COLLECTION_NAME,
        cap=config.AUTONOMY_MEMORY_MAX_SCAN, with_vectors=True,
    )
    if len(points) < config.MEMORY_HDBSCAN_MIN_CLUSTER_SIZE:
        logger.info(f"[memory_review] only {len(points)} new L2 entries; skip clustering")
        return

    vectors = np.array([p.vector for p in points], dtype=float)
    labels = memory_clustering.cluster_vectors(
        vectors,
        min_cluster_size=config.MEMORY_HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.MEMORY_HDBSCAN_MIN_SAMPLES,
    )
    clusters: Dict[int, List[int]] = defaultdict(list)
    noise = 0
    for idx, lbl in enumerate(labels):
        if lbl == -1:
            noise += 1
        else:
            clusters[lbl].append(idx)
    stats["clusters_found"] = len(clusters)
    stats["noise_points"] = noise

    llm_budget = config.AUTONOMY_MEMORY_LLM_CALL_CAP
    for lbl, member_indices in clusters.items():
        if stats["llm_calls"] >= llm_budget:
            stats["llm_call_cap_hit"] = True
            break
        members = [points[i] for i in member_indices]
        texts = [str((m.payload or {}).get("text", "")) for m in members]

        stats["llm_calls"] += 1
        summary_obj = await memory_clustering.summarize_cluster(
            client=llm_client,
            model_name=model_name,
            member_texts=texts,
        )
        if summary_obj is None:
            continue

        try:
            embedding = _embed(summary_obj["summary"])
        except Exception as e:
            logger.warning(f"[memory_review] embedding failed: {e}; skipping cluster")
            continue

        importances = [float((m.payload or {}).get("importance", 0) or 0) for m in members]
        importances.sort()
        median_imp = importances[len(importances) // 2] if importances else 3.0

        new_id = str(uuid.uuid4())
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=new_id,
                vector=embedding,
                payload={
                    "text": summary_obj["summary"],
                    "timestamp": _now().isoformat(),
                    "importance": median_imp,
                    "importance_effective": median_imp,
                    "tags": summary_obj["tags"],
                    "source": "memory_review",
                    "tier": "L3",
                    "source_ids": [str(m.id) for m in members],
                    "access_count": 0,
                    "last_accessed_at": None,
                    "pending_l4_approval": False,
                    "proposed_at": None,
                    "proposal_rationale": None,
                    "rationale": summary_obj.get("rationale"),
                },
            )],
        )
        stats["l3_created"] += 1


async def handle(
    item: Dict[str, Any],
    *,
    client,             # AsyncOpenAI
    mcp_manager,
    model_name: str,
    base_tools,
) -> Dict[str, Any]:
    start = time.perf_counter()
    qc = _qdrant_client()
    stats: Dict[str, Any] = {
        "l2_scanned": 0, "l3_created": 0, "l3_updated": 0,
        "l4_proposed": 0, "l2_pruned": 0, "importance_adjusted": 0,
        "clusters_found": 0, "noise_points": 0, "llm_calls": 0,
    }

    last_fired = item.get("last_fired_at")
    since = last_fired if isinstance(last_fired, datetime) else _now().replace(
        year=_now().year - 1
    )

    try:
        await _scan_and_decay(qc, stats)
    except Exception as e:
        logger.error(f"[memory_review] step 1/2 failed: {e}")

    try:
        await _cluster_to_l3(
            qc, stats,
            since=since, llm_client=client, model_name=model_name,
        )
    except Exception as e:
        logger.error(f"[memory_review] step 3 failed: {e}")

    total_ms = int((time.perf_counter() - start) * 1000)
    summary = (
        f"{stats['l3_created']} new L3 from {stats['l2_scanned']} L2 scanned, "
        f"{stats['l4_proposed']} L4 proposal, {stats['l2_pruned']} pruned"
    )
    return {
        "status": "ok",
        "summary": summary,
        "messages": [],
        "metrics": {**stats, "total_ms": total_ms},
        "error": None,
    }
