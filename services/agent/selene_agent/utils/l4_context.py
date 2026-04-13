"""L4 context block builder with in-memory cache.

The cache is invalidated whenever an L4 entry is created, edited, or removed
via the dashboard. See api/memory.py — every mutating endpoint calls
``invalidate_cache`` after a successful Qdrant write.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from selene_agent.utils import config
from selene_agent.utils import logger as custom_logger

logger = custom_logger.get_logger('loki')

_cache_value: Optional[str] = None
_cache_lock = asyncio.Lock()


def invalidate_cache() -> None:
    """Clear the memoized block. Called after any L4 mutation."""
    global _cache_value
    _cache_value = None


def _qdrant_client():
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import (
        QDRANT_HOST, QDRANT_PORT,
    )
    from qdrant_client import QdrantClient
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


async def build_l4_block() -> str:
    """Return the rendered L4 context block (empty string when no entries)."""
    global _cache_value
    if _cache_value is not None:
        return _cache_value
    async with _cache_lock:
        if _cache_value is not None:
            return _cache_value
        try:
            block = await _render()
        except Exception as e:
            logger.warning(f"build_l4_block failed: {e}; returning empty")
            block = ""
        _cache_value = block
        return block


async def _render() -> str:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

    client = _qdrant_client()
    flt = Filter(
        must=[
            FieldCondition(key="tier", match=MatchValue(value="L4")),
            FieldCondition(key="pending_l4_approval", match=MatchValue(value=False)),
        ]
    )
    offset = None
    entries = []
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        entries.extend(points)
        if offset is None:
            break

    if not entries:
        return ""

    def _sort_key(p):
        pl = p.payload or {}
        return (
            -float(pl.get("importance_effective", pl.get("importance", 0)) or 0),
            -_age_seconds(pl.get("timestamp", "")),
        )

    entries.sort(key=_sort_key)
    entries = entries[: max(1, int(config.MEMORY_L4_MAX_ENTRIES or 20))]

    lines: List[str] = ["<persistent_memories>"]
    for p in entries:
        text = str((p.payload or {}).get("text", "")).strip()
        if not text:
            continue
        lines.append(f"- {text}")
    lines.append("</persistent_memories>")
    return "\n".join(lines)


def _age_seconds(ts_iso: str) -> float:
    from datetime import datetime, timezone
    if not ts_iso:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    return (datetime.now(timezone.utc) - dt).total_seconds()
