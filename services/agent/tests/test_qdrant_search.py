"""Tests for v2 changes to qdrant_mcp_server."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "localhost")
    monkeypatch.setenv("QDRANT_PORT", "6333")
    with patch("selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server.QdrantClient") as qc, \
         patch("selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server.requests") as req:
        qc.return_value.get_collection.return_value = True
        req.post.return_value.json.return_value = [[0.0] * 1024]
        req.post.return_value.raise_for_status = MagicMock()
        from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import QdrantMCPServer
        s = QdrantMCPServer()
        s.client = MagicMock()
        s.client.upsert = MagicMock()
        yield s


@pytest.mark.asyncio
async def test_payload_has_v2_fields(server):
    await server._create_memory({"text": "foo", "importance": 3})
    args = server.client.upsert.call_args
    point = args.kwargs["points"][0]
    payload = point.payload
    assert payload["tier"] == "L2"
    assert payload["source_ids"] == []
    assert payload["access_count"] == 0
    assert payload["last_accessed_at"] is None
    assert payload["importance_effective"] == 3
    assert payload["pending_l4_approval"] is False
    assert payload["proposed_at"] is None
    assert payload["proposal_rationale"] is None


def _mk_point(pid, tier, score, text="x"):
    p = MagicMock()
    p.id = pid
    p.score = score
    p.payload = {
        "text": text,
        "timestamp": "2026-04-13T00:00:00+00:00",
        "importance": 3,
        "tags": [],
        "tier": tier,
        "source_ids": [],
    }
    return p


@pytest.mark.asyncio
async def test_search_applies_l3_boost(server, monkeypatch):
    from selene_agent.utils import config
    monkeypatch.setattr(config, "MEMORY_L3_RANK_BOOST", 1.5)
    server.client.query_points = MagicMock()
    server.client.query_points.return_value.points = [
        _mk_point("l2a", "L2", 0.80),
        _mk_point("l3a", "L3", 0.60),
    ]
    out = await server._search_memories({"query": "q", "limit": 5})
    ids = [m["id"] for m in out["results"]]
    # 0.60 * 1.5 = 0.90 > 0.80 -> L3 ranks above L2.
    assert ids[0] == "l3a"
    assert ids[1] == "l2a"


@pytest.mark.asyncio
async def test_search_excludes_l4(server):
    server.client.query_points = MagicMock()
    server.client.query_points.return_value.points = []
    await server._search_memories({"query": "q", "limit": 5})
    filt = server.client.query_points.call_args.kwargs["query_filter"]
    # Walk the filter for a must_not tier='L4'.
    found = False
    if filt and filt.must_not:
        for cond in filt.must_not:
            if getattr(cond, "key", None) == "tier":
                found = True
    assert found, "expected must_not filter on tier='L4'"


@pytest.mark.asyncio
async def test_search_fires_access_update(server):
    server.client.query_points = MagicMock()
    server.client.query_points.return_value.points = [
        _mk_point("a", "L2", 0.9), _mk_point("b", "L2", 0.5),
    ]
    server.client.set_payload = MagicMock()
    await server._search_memories({"query": "q", "limit": 5})
    # Background task scheduled — let it run.
    import asyncio as aio
    await aio.sleep(0)
    await aio.sleep(0)
    assert server.client.set_payload.called
    call = server.client.set_payload.call_args
    assert call.kwargs["collection_name"] == server.collection_name
    payload = call.kwargs["payload"]
    assert "last_accessed_at" in payload
    # Increment is handled via a per-id update path; confirm ids are targeted.
    points = call.kwargs.get("points") or []
    assert set(points) == {"a", "b"}
