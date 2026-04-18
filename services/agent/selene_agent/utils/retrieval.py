"""Per-turn semantic retrieval: embed the user message, pull top-K L2/L3
memories, render an ephemeral ``<retrieved_memories>`` block.

L4 is already injected into the system prompt at session init (see
``l4_context.build_l4_block``), so this function excludes L4 to avoid
duplication.

Used by ``AgentOrchestrator.run`` to inject relevant context per turn
without requiring the LLM to call ``search_memories`` itself.
"""
from __future__ import annotations

import os
from typing import List, Optional

import requests

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')


def _qdrant_client():
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import (
        QDRANT_HOST, QDRANT_PORT,
    )
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _embed(text: str, timeout: int = 10) -> List[float]:
    url = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
    r = requests.post(f"{url}/embed", json={"inputs": text}, timeout=timeout)
    r.raise_for_status()
    return r.json()[0]


def get_retrieval_k(phase: str) -> int:
    """Top-K depends on operational phase — learning leans harder on memory."""
    if phase == "learning":
        return int(getattr(config, "MEMORY_RETRIEVAL_TOPK_LEARNING", 5))
    return int(getattr(config, "MEMORY_RETRIEVAL_TOPK_OPERATING", 3))


async def build_retrieval_block(
    user_message: str,
    phase: str = "operating",
) -> Optional[str]:
    """Return a ``<retrieved_memories>`` block for the given user message, or
    ``None`` if retrieval is disabled, empty, or all below the min-score floor.

    Synchronous Qdrant + embeddings calls inside an async function — same
    pattern used by ``l4_context._render`` and ``memory_review``. Fire-and-
    forget; failures are logged and return None so the turn continues.
    """
    if not getattr(config, "MEMORY_RETRIEVAL_ENABLED", True):
        return None

    query = (user_message or "").strip()
    if not query:
        return None

    k = get_retrieval_k(phase)
    if k <= 0:
        return None

    min_score = float(getattr(config, "MEMORY_RETRIEVAL_MIN_SCORE", 0.3))

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

        vec = _embed(query)
        flt = Filter(must=[
            FieldCondition(key="tier", match=MatchAny(any=["L2", "L3"])),
        ])
        client = _qdrant_client()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            query_filter=flt,
            limit=k,
            with_payload=True,
        ).points
    except Exception as e:
        logger.warning(f"retrieval build failed (continuing without): {e}")
        return None

    kept = [r for r in results if float(r.score) >= min_score]
    if not kept:
        return None

    lines: List[str] = ["<retrieved_memories>"]
    for r in kept:
        payload = r.payload or {}
        tier = payload.get("tier", "L2")
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"- [{tier}] {text}")
    lines.append("</retrieved_memories>")
    return "\n".join(lines)
