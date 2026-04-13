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
