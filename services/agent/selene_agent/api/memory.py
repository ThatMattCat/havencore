"""REST surface backing the /memory dashboard page.

Endpoints use the Qdrant Python client directly; there is no per-request
DB connection pool. Operations are all same-origin and reuse the agent's
existing no-auth pattern (matches /api/autonomy/*).
"""
from __future__ import annotations

import os
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from selene_agent.utils import config
from selene_agent.utils import l4_context
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

router = APIRouter(tags=["memory"])


def _qdrant_client():
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import (
        QDRANT_HOST, QDRANT_PORT,
    )
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _collection() -> str:
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import (
        COLLECTION_NAME,
    )
    return COLLECTION_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed(text: str) -> List[float]:
    url = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
    r = requests.post(f"{url}/embed", json={"inputs": text}, timeout=30)
    r.raise_for_status()
    return r.json()[0]


def _scored_out(p: Any, score: float) -> Dict[str, Any]:
    out = _point_out(p)
    out["score"] = float(score)
    return out


def _point_out(p: Any) -> Dict[str, Any]:
    pl = p.payload or {}
    return {
        "id": str(p.id),
        "text": pl.get("text", ""),
        "importance": pl.get("importance", 0),
        "importance_effective": pl.get("importance_effective", pl.get("importance", 0)),
        "tier": pl.get("tier", "L2"),
        "tags": pl.get("tags", []),
        "timestamp": pl.get("timestamp", ""),
        "source_ids": pl.get("source_ids", []),
        "access_count": pl.get("access_count", 0),
        "last_accessed_at": pl.get("last_accessed_at"),
        "pending_l4_approval": pl.get("pending_l4_approval", False),
        "proposed_at": pl.get("proposed_at"),
        "proposal_rationale": pl.get("proposal_rationale"),
    }


# ---------- L4 CRUD ----------

class L4Create(BaseModel):
    text: str
    importance: int = 5
    tags: List[str] = []


class L4Update(BaseModel):
    text: Optional[str] = None
    importance: Optional[int] = None
    tags: Optional[List[str]] = None


@router.get("/memory/l4")
def list_l4():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = _qdrant_client()
    flt = Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value="L4")),
        FieldCondition(key="pending_l4_approval", match=MatchValue(value=False)),
    ])
    offset = None
    out: List[Any] = []
    while True:
        pts, offset = c.scroll(
            collection_name=_collection(), scroll_filter=flt,
            limit=256, with_payload=True, with_vectors=False, offset=offset,
        )
        out.extend(pts)
        if offset is None:
            break
    return {"entries": [_point_out(p) for p in out]}


@router.post("/memory/l4")
def create_l4(body: L4Create):
    from qdrant_client.models import PointStruct
    c = _qdrant_client()
    new_id = str(_uuid.uuid4())
    vec = _embed(body.text)
    c.upsert(
        collection_name=_collection(),
        points=[PointStruct(id=new_id, vector=vec, payload={
            "text": body.text,
            "timestamp": _now_iso(),
            "importance": body.importance,
            "importance_effective": float(body.importance),
            "tags": body.tags,
            "source": "user_direct",
            "tier": "L4",
            "source_ids": [],
            "access_count": 0,
            "last_accessed_at": None,
            "pending_l4_approval": False,
            "proposed_at": None,
            "proposal_rationale": None,
        })],
    )
    l4_context.invalidate_cache()
    return {"id": new_id}


@router.patch("/memory/l4/{entry_id}")
def update_l4(entry_id: str, body: L4Update):
    c = _qdrant_client()
    payload: Dict[str, Any] = {}
    if body.text is not None:
        payload["text"] = body.text
    if body.importance is not None:
        payload["importance"] = body.importance
        payload["importance_effective"] = float(body.importance)
    if body.tags is not None:
        payload["tags"] = body.tags
    if not payload:
        raise HTTPException(400, "no fields to update")
    c.set_payload(collection_name=_collection(), payload=payload, points=[entry_id])
    l4_context.invalidate_cache()
    return {"id": entry_id, "updated": list(payload.keys())}


@router.delete("/memory/l4/{entry_id}")
def delete_l4(entry_id: str):
    c = _qdrant_client()
    # "Remove from L4" == demote to L3 (do not delete the underlying memory).
    c.set_payload(
        collection_name=_collection(),
        payload={"tier": "L3"},
        points=[entry_id],
    )
    l4_context.invalidate_cache()
    return {"id": entry_id, "demoted_to": "L3"}


# ---------- Proposals ----------

@router.get("/memory/l4/proposals")
def list_proposals():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = _qdrant_client()
    flt = Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value="L3")),
        FieldCondition(key="pending_l4_approval", match=MatchValue(value=True)),
    ])
    offset = None
    out = []
    while True:
        pts, offset = c.scroll(
            collection_name=_collection(), scroll_filter=flt,
            limit=256, with_payload=True, with_vectors=False, offset=offset,
        )
        out.extend(pts)
        if offset is None:
            break
    return {"proposals": [_point_out(p) for p in out]}


@router.post("/memory/l4/proposals/{entry_id}/approve")
def approve_proposal(entry_id: str):
    c = _qdrant_client()
    c.set_payload(
        collection_name=_collection(),
        payload={"tier": "L4", "pending_l4_approval": False},
        points=[entry_id],
    )
    l4_context.invalidate_cache()
    return {"id": entry_id, "promoted_to": "L4"}


@router.post("/memory/l4/proposals/{entry_id}/reject")
def reject_proposal(entry_id: str):
    c = _qdrant_client()
    c.set_payload(
        collection_name=_collection(),
        payload={"pending_l4_approval": False},
        points=[entry_id],
    )
    l4_context.invalidate_cache()
    return {"id": entry_id, "stays_tier": "L3"}


# ---------- L2 browse ----------

@router.get("/memory/l2")
def list_l2(limit: int = 50, offset: int = 0):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = _qdrant_client()
    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L2"))])
    gathered: List[Any] = []
    next_offset = None
    while len(gathered) < offset + limit:
        pts, next_offset = c.scroll(
            collection_name=_collection(), scroll_filter=flt,
            limit=min(256, offset + limit - len(gathered)),
            with_payload=True, with_vectors=False, offset=next_offset,
        )
        gathered.extend(pts)
        if next_offset is None:
            break
    page = gathered[offset: offset + limit]
    return {"entries": [_point_out(p) for p in page], "has_more": next_offset is not None}


@router.delete("/memory/l2/{entry_id}")
def delete_l2(entry_id: str):
    from qdrant_client.models import PointIdsList
    c = _qdrant_client()
    c.delete(collection_name=_collection(),
             points_selector=PointIdsList(points=[entry_id]))
    return {"id": entry_id, "deleted": True}


# ---------- L3 browse ----------

@router.get("/memory/l3")
def list_l3(limit: int = 50, offset: int = 0):
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = _qdrant_client()
    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    # Scroll-based paging: gather `offset+limit`, slice.
    gathered: List[Any] = []
    next_offset = None
    while len(gathered) < offset + limit:
        pts, next_offset = c.scroll(
            collection_name=_collection(), scroll_filter=flt,
            limit=min(256, offset + limit - len(gathered)),
            with_payload=True, with_vectors=False, offset=next_offset,
        )
        gathered.extend(pts)
        if next_offset is None:
            break
    page = gathered[offset: offset + limit]
    return {"entries": [_point_out(p) for p in page], "has_more": next_offset is not None}


@router.get("/memory/l3/{entry_id}/sources")
def l3_sources(entry_id: str):
    c = _qdrant_client()
    l3s = c.retrieve(collection_name=_collection(), ids=[entry_id], with_payload=True)
    if not l3s:
        raise HTTPException(404, "L3 entry not found")
    payload = l3s[0].payload or {}

    # Prefer archived source texts (post-absorption). Fall back to the
    # legacy retrieve-by-id path for L3s created before absorption landed.
    source_texts = payload.get("source_texts") or []
    if source_texts:
        sources = []
        for entry in source_texts:
            if not isinstance(entry, dict):
                continue
            sources.append({
                "id": str(entry.get("id", "")),
                "text": entry.get("text", ""),
                "importance": entry.get("importance", 0),
                "importance_effective": entry.get("importance", 0),
                "tier": "L2",
                "tags": [],
                "timestamp": entry.get("timestamp", ""),
                "source_ids": [],
                "access_count": 0,
                "last_accessed_at": None,
                "pending_l4_approval": False,
                "proposed_at": None,
                "proposal_rationale": None,
                "absorbed": True,
            })
        return {"sources": sources}

    src_ids = payload.get("source_ids") or []
    if not src_ids:
        return {"sources": []}
    sources = c.retrieve(collection_name=_collection(), ids=list(src_ids),
                         with_payload=True)
    return {"sources": [_point_out(s) for s in sources]}


@router.delete("/memory/l3/{entry_id}")
def delete_l3(entry_id: str):
    from qdrant_client.models import PointIdsList
    c = _qdrant_client()
    c.delete(collection_name=_collection(),
             points_selector=PointIdsList(points=[entry_id]))
    return {"id": entry_id, "deleted": True}


# ---------- Semantic search ----------

class SearchRequest(BaseModel):
    q: str
    tiers: List[str] = ["L2", "L3", "L4"]
    limit: int = 25


@router.post("/memory/search")
def search(body: SearchRequest):
    from qdrant_client.models import Filter, FieldCondition, MatchAny

    if not body.q.strip():
        raise HTTPException(400, "query is empty")

    valid_tiers = [t for t in body.tiers if t in ("L2", "L3", "L4")]
    if not valid_tiers:
        raise HTTPException(400, "no valid tiers selected")

    limit = max(1, min(body.limit, 100))

    try:
        vec = _embed(body.q)
    except Exception as e:
        raise HTTPException(502, f"embedding service error: {e}")

    flt = Filter(must=[FieldCondition(key="tier", match=MatchAny(any=valid_tiers))])

    c = _qdrant_client()
    results = c.query_points(
        collection_name=_collection(),
        query=vec,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    ).points

    return {
        "query": body.q,
        "tiers": valid_tiers,
        "results": [_scored_out(r, r.score) for r in results],
    }


# ---------- Runs + stats ----------

@router.get("/memory/runs")
async def list_runs(limit: int = 20):
    from selene_agent.autonomy import db as autonomy_db
    rows = await autonomy_db.list_runs(limit=limit, include_messages=False)
    return {"runs": [r for r in rows if r["kind"] == "memory_review"]}


@router.post("/memory/runs/trigger")
async def trigger_run():
    from selene_agent.autonomy import db as autonomy_db
    # Find the system-owned memory_review agenda item.
    items = await autonomy_db.list_all_items()
    target = next((i for i in items if i["kind"] == "memory_review"), None)
    if target is None:
        raise HTTPException(404, "memory_review agenda item not found")
    # Delegate to the engine via the app state.
    from selene_agent.selene_agent import app
    engine = getattr(app.state, "autonomy_engine", None)
    if engine is None:
        raise HTTPException(503, "autonomy engine not available")
    result = await engine.trigger(target["id"])
    return {"agenda_item_id": target["id"], "result": result}


# ---------- Admin purge (hygiene) ----------

class PurgeRequest(BaseModel):
    tier: Optional[str] = None          # "L2" | "L3" | "L4" | "all"
    source: Optional[str] = None        # payload.source match (e.g. "test")
    ids: Optional[List[str]] = None     # explicit id list
    pending_l4_approval: Optional[bool] = None


@router.post("/memory/admin/purge")
def admin_purge(body: PurgeRequest):
    """Bulk-delete memories by tier + source tag, or by explicit ids.

    Must specify either `ids`, or at least one of `source`/`pending_l4_approval`
    to avoid wiping real data by accident. `tier="all"` is allowed when paired
    with a `source` filter.
    """
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, PointIdsList,
    )

    # Guardrail: must have either explicit ids, or a narrowing filter beyond tier.
    if not body.ids and not body.source and body.pending_l4_approval is None:
        raise HTTPException(
            400,
            "refuse to purge without `ids`, `source`, or `pending_l4_approval` filter",
        )

    c = _qdrant_client()

    if body.ids:
        c.delete(
            collection_name=_collection(),
            points_selector=PointIdsList(points=body.ids),
        )
        l4_context.invalidate_cache()
        return {"deleted_ids": body.ids, "count": len(body.ids)}

    must: List[Any] = []
    if body.tier and body.tier != "all":
        if body.tier not in ("L2", "L3", "L4"):
            raise HTTPException(400, f"invalid tier: {body.tier}")
        must.append(FieldCondition(key="tier", match=MatchValue(value=body.tier)))
    if body.source:
        must.append(FieldCondition(key="source", match=MatchValue(value=body.source)))
    if body.pending_l4_approval is not None:
        must.append(FieldCondition(
            key="pending_l4_approval",
            match=MatchValue(value=body.pending_l4_approval),
        ))

    flt = Filter(must=must)
    # Collect matching ids first so we can return a count.
    offset = None
    to_delete: List[str] = []
    while True:
        pts, offset = c.scroll(
            collection_name=_collection(), scroll_filter=flt,
            limit=512, with_payload=False, with_vectors=False, offset=offset,
        )
        to_delete.extend(str(p.id) for p in pts)
        if offset is None:
            break

    if to_delete:
        c.delete(
            collection_name=_collection(),
            points_selector=PointIdsList(points=to_delete),
        )
        l4_context.invalidate_cache()

    return {"deleted_ids": to_delete, "count": len(to_delete)}


@router.get("/memory/stats")
async def stats():
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    c = _qdrant_client()

    def _count(flt):
        return c.count(collection_name=_collection(), count_filter=flt, exact=True).count

    l2 = _count(Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L2"))]))
    l3 = _count(Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))]))
    l4 = _count(Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value="L4")),
        FieldCondition(key="pending_l4_approval", match=MatchValue(value=False)),
    ]))
    pending = _count(Filter(must=[
        FieldCondition(key="tier", match=MatchValue(value="L3")),
        FieldCondition(key="pending_l4_approval", match=MatchValue(value=True)),
    ]))

    # Approximate token count: ~4 chars per token applied to the rendered block.
    block = await l4_context.build_l4_block()
    l4_est_tokens = max(0, len(block) // 4)

    return {
        "l2_count": l2,
        "l3_count": l3,
        "l4_count": l4,
        "pending_proposals": pending,
        "l4_est_tokens": l4_est_tokens,
    }
