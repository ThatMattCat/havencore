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
