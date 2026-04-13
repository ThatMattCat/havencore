"""Tests for L4 block builder and cache."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _point(pid, text, importance, timestamp="2026-04-01T00:00:00+00:00"):
    p = MagicMock()
    p.id = pid
    p.payload = {
        "text": text, "importance": importance,
        "importance_effective": importance, "timestamp": timestamp,
        "tier": "L4",
    }
    return p


@pytest.mark.asyncio
async def test_build_l4_block_empty_returns_empty_string():
    from selene_agent.utils import l4_context

    with patch.object(l4_context, "_qdrant_client") as qc:
        client = MagicMock()
        client.scroll.return_value = ([], None)
        qc.return_value = client
        l4_context.invalidate_cache()
        out = await l4_context.build_l4_block()
        assert out == ""


@pytest.mark.asyncio
async def test_build_l4_block_renders_entries_with_header():
    from selene_agent.utils import l4_context

    with patch.object(l4_context, "_qdrant_client") as qc:
        client = MagicMock()
        client.scroll.return_value = (
            [_point("a", "Household: Matt lives alone.", 5),
             _point("b", "Voice: no emojis in responses.", 4)],
            None,
        )
        qc.return_value = client
        l4_context.invalidate_cache()
        out = await l4_context.build_l4_block()
        assert "<persistent_memories>" in out
        assert "</persistent_memories>" in out
        assert "Matt lives alone" in out
        assert "no emojis" in out
        # Importance desc ordering: "Matt" (imp=5) before "no emojis" (imp=4).
        assert out.index("Matt lives alone") < out.index("no emojis")


@pytest.mark.asyncio
async def test_build_l4_block_caches_until_invalidated():
    from selene_agent.utils import l4_context

    with patch.object(l4_context, "_qdrant_client") as qc:
        client = MagicMock()
        client.scroll.return_value = ([_point("a", "x", 5)], None)
        qc.return_value = client
        l4_context.invalidate_cache()
        out1 = await l4_context.build_l4_block()
        out2 = await l4_context.build_l4_block()
        assert out1 == out2
        # Scroll called only once across the two build calls.
        assert client.scroll.call_count == 1
        l4_context.invalidate_cache()
        await l4_context.build_l4_block()
        assert client.scroll.call_count == 2


@pytest.mark.asyncio
async def test_build_l4_block_respects_max_entries(monkeypatch):
    from selene_agent.utils import config
    from selene_agent.utils import l4_context

    monkeypatch.setattr(config, "MEMORY_L4_MAX_ENTRIES", 2)
    with patch.object(l4_context, "_qdrant_client") as qc:
        client = MagicMock()
        client.scroll.return_value = (
            [_point(f"p{i}", f"Entry {i}", 5 - i) for i in range(5)],
            None,
        )
        qc.return_value = client
        l4_context.invalidate_cache()
        out = await l4_context.build_l4_block()
        # Should include top two by importance.
        assert "Entry 0" in out and "Entry 1" in out
        assert "Entry 2" not in out
