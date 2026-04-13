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

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

from selene_agent.autonomy import memory_math
from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


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


async def handle(
    item: Dict[str, Any],
    *,
    client,             # AsyncOpenAI (unused in steps 1+2, used later)
    mcp_manager,        # unused — handler talks to Qdrant directly
    model_name: str,
    base_tools,
) -> Dict[str, Any]:
    """Top-level entry. Subsequent tasks will flesh out steps 3-5."""
    start = time.perf_counter()
    qc = _qdrant_client()
    stats: Dict[str, Any] = {
        "l2_scanned": 0,
        "l3_created": 0,
        "l3_updated": 0,
        "l4_proposed": 0,
        "l2_pruned": 0,
        "importance_adjusted": 0,
        "clusters_found": 0,
        "noise_points": 0,
        "llm_calls": 0,
    }

    try:
        await _scan_and_decay(qc, stats)
    except Exception as e:
        logger.error(f"[memory_review] step 1/2 failed: {e}")
        return {
            "status": "error",
            "summary": "memory_review: scan/decay failed",
            "messages": [],
            "metrics": {**stats, "total_ms": int((time.perf_counter() - start) * 1000)},
            "error": f"{type(e).__name__}: {e}",
        }

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
