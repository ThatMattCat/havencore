# AutonomyEngine v2 — Memory Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn v1's L2-tier groundwork into a working tiered-memory system with nightly consolidation, user-gated L4 promotion, L4 injection into every prompt, and a SvelteKit `/memory` dashboard for inspection and control.

**Architecture:** New `memory_review` agenda kind runs a deterministic 5-step pipeline (scan L2 → decay/boost → HDBSCAN cluster + LLM summarize → propose L3→L4 flag → prune stale L2 with source protection). `search_memories` gets tier-weighted ranking, L4 exclusion, and async access tracking. `build_l4_block()` prepends an approved set of L4 entries to both user-facing and autonomous system prompts, cached with mutation-based invalidation. New REST surface `/api/memory/*` and SvelteKit `/memory` page provide the approval queue and inspection.

**Tech Stack:** Python 3, FastAPI, asyncpg (via existing `conversation_db` pool), Qdrant Python client, OpenAI async SDK (against vLLM), `hdbscan` for clustering, `numpy` for vector math, SvelteKit + Tailwind for the dashboard, pytest for tests.

**Spec:** `docs/superpowers/specs/2026-04-13-autonomy-v2-memory-design.md`

---

## File Structure

### Created
- `services/agent/selene_agent/autonomy/handlers/memory_review.py` — pipeline entry point + step orchestration
- `services/agent/selene_agent/autonomy/memory_math.py` — pure functions (decay, boost, importance_effective)
- `services/agent/selene_agent/autonomy/memory_clustering.py` — HDBSCAN wrapper + LLM cluster summarizer
- `services/agent/selene_agent/utils/l4_context.py` — L4 block builder + in-memory cache
- `services/agent/selene_agent/api/memory.py` — REST router backing the dashboard
- `services/agent/frontend/src/routes/memory/+page.svelte` — dashboard page
- `services/agent/tests/__init__.py`
- `services/agent/tests/conftest.py`
- `services/agent/tests/test_memory_math.py`
- `services/agent/tests/test_memory_clustering.py`
- `services/agent/tests/test_memory_review_handler.py`
- `services/agent/tests/test_l4_context.py`
- `services/agent/tests/test_qdrant_search.py`
- `services/agent/tests/test_memory_api.py`
- `docs/services/agent/autonomy/memory/README.md`

### Modified
- `services/agent/selene_agent/autonomy/engine.py` — dispatch table gains `memory_review`
- `services/agent/selene_agent/autonomy/db.py` — default agenda seed adds `memory_review`
- `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py` — new payload fields, payload indexes, tier-weighted ranking, L4 exclusion, async access tracking
- `services/agent/selene_agent/orchestrator.py` — prepend L4 block to user system prompt
- `services/agent/selene_agent/autonomy/turn.py` — prepend L4 block to autonomous system prompt
- `services/agent/selene_agent/selene_agent.py` — mount `memory` router
- `services/agent/selene_agent/utils/config.py` — new env vars
- `shared/configs/shared_config.py` — passthrough of new env vars
- `services/agent/pyproject.toml` — add `hdbscan`, `numpy`, `pytest`, `pytest-asyncio`
- `.env.example` — document new vars

---

## Task 1: Scaffold test infrastructure

**Files:**
- Create: `services/agent/tests/__init__.py`
- Create: `services/agent/tests/conftest.py`
- Create: `services/agent/tests/test_sanity.py`
- Modify: `services/agent/pyproject.toml`

- [ ] **Step 1: Add test deps to `pyproject.toml`**

Open `services/agent/pyproject.toml`, locate the `[project]` dependencies section, and add a `[project.optional-dependencies]` block (or extend the existing one). If a `dev` extras group already exists, append to it; otherwise add:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty `tests/__init__.py`**

```bash
touch services/agent/tests/__init__.py
```

- [ ] **Step 3: Create `tests/conftest.py` with basic fixtures**

```python
"""Shared pytest fixtures for the agent test suite."""
from __future__ import annotations

import pytest


@pytest.fixture
def frozen_now():
    """A fixed UTC datetime for deterministic age math in tests."""
    from datetime import datetime, timezone
    return datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc)
```

- [ ] **Step 4: Create `tests/test_sanity.py` — one passing assertion**

```python
def test_sanity():
    assert 1 + 1 == 2
```

- [ ] **Step 5: Install dev deps in the running agent container and run tests**

```bash
docker compose exec -T agent pip install -e '.[dev]'
docker compose exec -T agent pytest -v
```

Expected: `tests/test_sanity.py::test_sanity PASSED`.

- [ ] **Step 6: Commit**

```bash
git add services/agent/pyproject.toml services/agent/tests/
git commit -m "chore(agent): scaffold pytest test suite for v2 memory work"
```

---

## Task 2: Add all v2 configuration knobs

**Files:**
- Modify: `services/agent/selene_agent/utils/config.py`
- Modify: `shared/configs/shared_config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add env-var passthrough in `shared/configs/shared_config.py`**

Locate the section where existing `AUTONOMY_*` vars are defined (search for `AUTONOMY_ENABLED`). Add the new group below the existing autonomy block:

```python
# --- v2 memory consolidation ---
AUTONOMY_MEMORY_REVIEW_CRON = os.getenv("AUTONOMY_MEMORY_REVIEW_CRON", "0 3 * * *")
AUTONOMY_MEMORY_MAX_SCAN = int(os.getenv("AUTONOMY_MEMORY_MAX_SCAN", "5000"))
AUTONOMY_MEMORY_LLM_CALL_CAP = int(os.getenv("AUTONOMY_MEMORY_LLM_CALL_CAP", "20"))

MEMORY_HALF_LIFE_DAYS = float(os.getenv("MEMORY_HALF_LIFE_DAYS", "60"))
MEMORY_ACCESS_COEF = float(os.getenv("MEMORY_ACCESS_COEF", "0.5"))

MEMORY_HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("MEMORY_HDBSCAN_MIN_CLUSTER_SIZE", "5"))
MEMORY_HDBSCAN_MIN_SAMPLES = int(os.getenv("MEMORY_HDBSCAN_MIN_SAMPLES", "3"))

MEMORY_L4_MIN_IMPORTANCE = float(os.getenv("MEMORY_L4_MIN_IMPORTANCE", "4"))
MEMORY_L4_MIN_AGE_DAYS = int(os.getenv("MEMORY_L4_MIN_AGE_DAYS", "14"))
MEMORY_L4_MIN_ACCESS_COUNT = int(os.getenv("MEMORY_L4_MIN_ACCESS_COUNT", "3"))

MEMORY_L2_PRUNE_AGE_DAYS = int(os.getenv("MEMORY_L2_PRUNE_AGE_DAYS", "180"))
MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD = float(os.getenv("MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD", "0.5"))

MEMORY_L3_RANK_BOOST = float(os.getenv("MEMORY_L3_RANK_BOOST", "1.2"))
MEMORY_L4_MAX_ENTRIES = int(os.getenv("MEMORY_L4_MAX_ENTRIES", "20"))
MEMORY_L4_WARN_TOKENS = int(os.getenv("MEMORY_L4_WARN_TOKENS", "1500"))
```

- [ ] **Step 2: Re-export from `services/agent/selene_agent/utils/config.py`**

Follow the existing pattern (the file re-exports `shared_config` attributes). If the agent config is a plain re-import, nothing to add; if it explicitly names the exports, append each new name. Verify with:

```bash
docker compose exec -T agent python -c "from selene_agent.utils import config; print(config.MEMORY_HALF_LIFE_DAYS)"
```

Expected: `60.0`

- [ ] **Step 3: Document in `.env.example`**

Append:

```
# ---- v2 memory consolidation ----
AUTONOMY_MEMORY_REVIEW_CRON=0 3 * * *
AUTONOMY_MEMORY_MAX_SCAN=5000
AUTONOMY_MEMORY_LLM_CALL_CAP=20

MEMORY_HALF_LIFE_DAYS=60
MEMORY_ACCESS_COEF=0.5

MEMORY_HDBSCAN_MIN_CLUSTER_SIZE=5
MEMORY_HDBSCAN_MIN_SAMPLES=3

MEMORY_L4_MIN_IMPORTANCE=4
MEMORY_L4_MIN_AGE_DAYS=14
MEMORY_L4_MIN_ACCESS_COUNT=3

MEMORY_L2_PRUNE_AGE_DAYS=180
MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD=0.5

MEMORY_L3_RANK_BOOST=1.2
MEMORY_L4_MAX_ENTRIES=20
MEMORY_L4_WARN_TOKENS=1500
```

- [ ] **Step 4: Commit**

```bash
git add services/agent/selene_agent/utils/config.py shared/configs/shared_config.py .env.example
git commit -m "feat(agent): add v2 memory consolidation env vars"
```

---

## Task 3: Add Python dependencies

**Files:**
- Modify: `services/agent/pyproject.toml`
- Modify: `services/agent/modules/mcp_qdrant_tools/requirements.txt` (only if numpy not already present)

- [ ] **Step 1: Add `hdbscan` and `numpy` to agent deps**

In `services/agent/pyproject.toml`, add to the main `dependencies` list:

```toml
"hdbscan>=0.8.33",
"numpy>=1.24",
```

(If `numpy` already appears, skip it.)

- [ ] **Step 2: Rebuild agent image**

```bash
docker compose build agent
docker compose up -d agent
```

- [ ] **Step 3: Verify import works**

```bash
docker compose exec -T agent python -c "import hdbscan, numpy; print(hdbscan.__version__, numpy.__version__)"
```

Expected: version strings print without error.

- [ ] **Step 4: Commit**

```bash
git add services/agent/pyproject.toml
git commit -m "chore(agent): add hdbscan and numpy for memory consolidation"
```

---

## Task 4: Extend Qdrant payload on `create_memory` + backward-compat reads

**Files:**
- Modify: `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py:187-198, 286-303`

- [ ] **Step 1: Write failing test `tests/test_qdrant_search.py::test_payload_has_v2_fields`**

This test stubs the Qdrant client and asserts the payload written by `create_memory` includes v2 fields with correct defaults.

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py::test_payload_has_v2_fields -v
```

Expected: FAIL — payload is missing `access_count`, etc.

- [ ] **Step 3: Extend the `_create_memory` payload**

In `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py`, replace the `payload = {...}` block (near line 187) with:

```python
payload = {
    "text": text,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "importance": importance,
    "tags": tags,
    "source": "mcp_server",
    # Memory tiering: new rows are L2. source_ids links consolidated
    # (L3/L4) entries back to originating L2 rows.
    "tier": "L2",
    "source_ids": [],
    # v2 access tracking + importance dynamics.
    "access_count": 0,
    "last_accessed_at": None,
    "importance_effective": importance,
    # v2 L4 proposal queue.
    "pending_l4_approval": False,
    "proposed_at": None,
    "proposal_rationale": None,
}
```

- [ ] **Step 4: Extend the `_search_memories` result formatter**

In the same file, locate the `memory = {...}` block inside `_search_memories` (near line 287) and extend it:

```python
memory = {
    "id": str(result.id),
    "text": result.payload.get("text", ""),
    "timestamp": result.payload.get("timestamp", ""),
    "importance": result.payload.get("importance", 0),
    "tags": result.payload.get("tags", []),
    "tier": result.payload.get("tier", "L2"),
    "source_ids": result.payload.get("source_ids", []),
    # v2 fields with backward-compat defaults for pre-v2 rows.
    "access_count": result.payload.get("access_count", 0),
    "last_accessed_at": result.payload.get("last_accessed_at"),
    "importance_effective": result.payload.get(
        "importance_effective", result.payload.get("importance", 0)
    ),
    "relevance_score": float(result.score),
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py::test_payload_has_v2_fields -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py services/agent/tests/test_qdrant_search.py
git commit -m "feat(memory): stamp v2 payload fields on create_memory, backward-compat reads"
```

---

## Task 5: Initialize Qdrant payload indexes at startup

**Files:**
- Modify: `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py:56-69`

- [ ] **Step 1: Add a `_init_payload_indexes` method and call it from `_init_collection`**

At the top of the file, ensure this import is present:

```python
from qdrant_client.models import PayloadSchemaType
```

(Add it to the existing `from qdrant_client.models import ...` line.)

Then add this method to `QdrantMCPServer` (place it just after `_init_collection`):

```python
def _init_payload_indexes(self) -> None:
    """Idempotently create payload indexes required for v2 scroll/filter queries."""
    indexes = [
        ("tier", PayloadSchemaType.KEYWORD),
        ("pending_l4_approval", PayloadSchemaType.BOOL),
        ("importance_effective", PayloadSchemaType.FLOAT),
    ]
    for field_name, schema in indexes:
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
            )
            logger.info(f"Payload index created or already existed: {field_name}")
        except Exception as e:
            # Qdrant returns an error on re-create; log and continue.
            logger.debug(f"Payload index {field_name}: {e}")
```

At the end of `_init_collection`, add:

```python
self._init_payload_indexes()
```

- [ ] **Step 2: Verify by restarting the agent**

```bash
docker compose restart agent
docker compose logs --tail 50 agent | grep -i "payload index"
```

Expected: three lines, one per indexed field.

- [ ] **Step 3: Commit**

```bash
git add services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py
git commit -m "feat(memory): create Qdrant payload indexes for tier/approval/importance"
```

---

## Task 6: Tier-weighted ranking and L4 exclusion in `search_memories`

**Files:**
- Modify: `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py:240-282`
- Modify: `services/agent/tests/test_qdrant_search.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_qdrant_search.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py -v
```

Expected: both new tests FAIL.

- [ ] **Step 3: Add the `MatchValue` import and ranking constants**

In the Qdrant module imports, change:

```python
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, DatetimeRange,
    Filter, FieldCondition, PayloadSchemaType
)
```

to also include `MatchValue`:

```python
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, DatetimeRange,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)
```

- [ ] **Step 4: Add L4 exclusion to the filter and re-rank results by tier weight**

In `_search_memories`, inside the filter-building block, add to `must_not_conditions` unconditionally:

```python
# v2: L4 entries are injected into every system prompt already — exclude
# them from semantic retrieval to avoid wasting token budget.
must_not_conditions.append(
    FieldCondition(key="tier", match=MatchValue(value="L4"))
)
```

Then, replace the result-formatting loop with one that computes an adjusted score and re-sorts before returning. The final part of `_search_memories` becomes:

```python
# Search in Qdrant
results = self.client.query_points(
    collection_name=self.collection_name,
    query=query_embedding,
    query_filter=search_filter,
    limit=limit * 2,  # over-fetch slightly so tier re-ranking has room
    with_payload=True
).points

from selene_agent.utils import config as cfg
TIER_WEIGHT = {"L2": 1.0, "L3": cfg.MEMORY_L3_RANK_BOOST, "L4": 1.0}

scored = []
for result in results:
    tier = result.payload.get("tier", "L2")
    weight = TIER_WEIGHT.get(tier, 1.0)
    adjusted = float(result.score) * weight
    scored.append((adjusted, result))
scored.sort(key=lambda t: t[0], reverse=True)
scored = scored[:limit]

memories = []
for adjusted, result in scored:
    memory = {
        "id": str(result.id),
        "text": result.payload.get("text", ""),
        "timestamp": result.payload.get("timestamp", ""),
        "importance": result.payload.get("importance", 0),
        "tags": result.payload.get("tags", []),
        "tier": result.payload.get("tier", "L2"),
        "source_ids": result.payload.get("source_ids", []),
        "access_count": result.payload.get("access_count", 0),
        "last_accessed_at": result.payload.get("last_accessed_at"),
        "importance_effective": result.payload.get(
            "importance_effective", result.payload.get("importance", 0)
        ),
        "relevance_score": float(result.score),
        "adjusted_score": adjusted,
    }
    if "expires" in result.payload:
        memory["expires"] = result.payload["expires"]
    memories.append(memory)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py -v
```

Expected: all four tests PASS.

- [ ] **Step 6: Commit**

```bash
git add services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py services/agent/tests/test_qdrant_search.py
git commit -m "feat(memory): tier-weighted ranking and L4 exclusion in search_memories"
```

---

## Task 7: Async access tracking on `search_memories`

**Files:**
- Modify: `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py`
- Modify: `services/agent/tests/test_qdrant_search.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_qdrant_search.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py::test_search_fires_access_update -v
```

Expected: FAIL — `set_payload` never called.

- [ ] **Step 3: Implement `_record_accesses` and schedule it**

Add this method to `QdrantMCPServer`:

```python
async def _record_accesses(self, ids: List[str]) -> None:
    """Fire-and-forget bump of access_count + last_accessed_at for the given ids.

    Increment is approximate: Qdrant's set_payload is not atomic-increment.
    We read current counts and write back count+1. Concurrent retrievals may
    drop ticks — acceptable because consolidation applies log(1+access_count)
    which dampens counting noise.
    """
    if not ids:
        return
    try:
        current = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        by_id = {str(p.id): (p.payload or {}).get("access_count", 0) for p in current}
        # Group ids by their new count so we can issue one set_payload per group.
        from collections import defaultdict
        groups = defaultdict(list)
        for pid in ids:
            groups[by_id.get(pid, 0) + 1].append(pid)
        for new_count, group_ids in groups.items():
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "access_count": new_count,
                    "last_accessed_at": now_iso,
                },
                points=group_ids,
            )
    except Exception as e:
        logger.warning(f"_record_accesses failed (non-fatal): {e}")
```

Then, immediately before the `return {...}` at the end of `_search_memories`, add:

```python
# Fire-and-forget: do NOT await; retrieval must not wait on this.
if memories:
    asyncio.create_task(self._record_accesses([m["id"] for m in memories]))
```

- [ ] **Step 4: Run all qdrant tests**

```bash
docker compose exec -T agent pytest tests/test_qdrant_search.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py services/agent/tests/test_qdrant_search.py
git commit -m "feat(memory): async access_count tracking on search_memories"
```

---

## Task 8: `memory_math` — importance decay/boost pure functions

**Files:**
- Create: `services/agent/selene_agent/autonomy/memory_math.py`
- Create: `services/agent/tests/test_memory_math.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for the pure-function decay/boost math."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from selene_agent.autonomy import memory_math


def test_importance_effective_fresh_entry_returns_near_base():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now  # zero age
    got = memory_math.compute_importance_effective(
        base_importance=3,
        created_at=created,
        access_count=0,
        now=now,
        half_life_days=60,
        access_coef=0.5,
    )
    # decay=1, boost=0 -> 3.0
    assert got == pytest.approx(3.0, abs=0.01)


def test_importance_effective_halves_after_half_life():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now - timedelta(days=60)
    got = memory_math.compute_importance_effective(
        base_importance=4,
        created_at=created,
        access_count=0,
        now=now,
        half_life_days=60,
        access_coef=0.5,
    )
    # 4 * e^(-60/60) ~= 4 * 0.3679 ~= 1.47
    assert got == pytest.approx(4 * 2.718281828 ** -1, abs=0.01)


def test_importance_effective_access_boost_is_logarithmic():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now
    base = memory_math.compute_importance_effective(
        base_importance=2, created_at=created, access_count=0,
        now=now, half_life_days=60, access_coef=0.5,
    )
    more = memory_math.compute_importance_effective(
        base_importance=2, created_at=created, access_count=10,
        now=now, half_life_days=60, access_coef=0.5,
    )
    # log(1+10) * 0.5 ~= 1.2
    import math
    assert (more - base) == pytest.approx(0.5 * math.log(1 + 10), abs=0.01)


def test_importance_effective_clamps_to_zero_ten():
    now = datetime(2026, 4, 13, tzinfo=timezone.utc)
    created = now - timedelta(days=3650)  # very old
    got = memory_math.compute_importance_effective(
        base_importance=1, created_at=created, access_count=0,
        now=now, half_life_days=60, access_coef=0.5,
    )
    assert got >= 0.0
    got_hi = memory_math.compute_importance_effective(
        base_importance=5, created_at=now, access_count=10**8,
        now=now, half_life_days=60, access_coef=0.5,
    )
    assert got_hi <= 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_math.py -v
```

Expected: FAIL — `memory_math` module does not exist.

- [ ] **Step 3: Implement `memory_math.py`**

```python
"""Pure functions for memory importance dynamics.

No Qdrant, no I/O. Unit-testable in isolation.
"""
from __future__ import annotations

import math
from datetime import datetime


def compute_importance_effective(
    *,
    base_importance: float,
    created_at: datetime,
    access_count: int,
    now: datetime,
    half_life_days: float,
    access_coef: float,
) -> float:
    """Effective importance = base * exp(-age / half_life) + coef * log(1 + accesses).

    Clamped to [0, 10]. Half-life controls decay; access_coef controls boost
    per order-of-magnitude of retrievals.
    """
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    decay = math.exp(-age_days / half_life_days) if half_life_days > 0 else 1.0
    boost = access_coef * math.log(1 + max(0, access_count))
    value = base_importance * decay + boost
    return max(0.0, min(10.0, value))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec -T agent pytest tests/test_memory_math.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/memory_math.py services/agent/tests/test_memory_math.py
git commit -m "feat(memory): importance decay/boost math (pure functions)"
```

---

## Task 9: `memory_clustering` — HDBSCAN wrapper + LLM cluster summarizer

**Files:**
- Create: `services/agent/selene_agent/autonomy/memory_clustering.py`
- Create: `services/agent/tests/test_memory_clustering.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for clustering wrapper and cluster summarization."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from selene_agent.autonomy import memory_clustering


def _planted_vectors(n_per_cluster=6, n_clusters=3, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 5
    vecs, truth = [], []
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            vecs.append(centers[c] + rng.normal(size=dim) * 0.3)
            truth.append(c)
    return np.array(vecs), truth


def test_hdbscan_finds_planted_clusters():
    vecs, truth = _planted_vectors()
    labels = memory_clustering.cluster_vectors(
        vecs, min_cluster_size=3, min_samples=2,
    )
    # At least 2 distinct non-noise labels should map to our 3 planted clusters.
    non_noise = [l for l in labels if l != -1]
    assert len(set(non_noise)) >= 2


def test_hdbscan_returns_all_noise_when_too_few_points():
    vecs = np.random.default_rng(0).normal(size=(2, 16))
    labels = memory_clustering.cluster_vectors(
        vecs, min_cluster_size=5, min_samples=3,
    )
    assert all(l == -1 for l in labels)


@pytest.mark.asyncio
async def test_summarize_cluster_happy_path():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "summary": "Matt prefers oat milk in coffee.",
            "tags": ["coffee", "preferences", "core_fact"],
            "rationale": "Four independent mentions of oat milk preference.",
        })))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["oat milk pls", "no dairy in coffee", "oat milk again", "oat milk"],
    )
    assert out is not None
    assert "oat milk" in out["summary"].lower()
    assert "coffee" in out["tags"]


@pytest.mark.asyncio
async def test_summarize_cluster_null_pattern_returns_none():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content=json.dumps({
            "summary": None, "tags": [], "rationale": "No coherent pattern."
        })))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["a", "b", "c"],
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_cluster_malformed_json_returns_none():
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="sorry I can't help"))
    ]
    out = await memory_clustering.summarize_cluster(
        client=client,
        model_name="gpt-3.5-turbo",
        member_texts=["a", "b"],
    )
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_clustering.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `memory_clustering.py`**

```python
"""HDBSCAN clustering + LLM cluster summarization for memory_review.

`cluster_vectors` is a thin, testable wrapper. `summarize_cluster` performs
one LLM call per cluster and normalizes the structured output.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np


CLUSTER_SUMMARIZER_SYSTEM = (
    "You consolidate related memories into a single durable summary. "
    "You will receive N short memory texts that have been clustered by "
    "semantic similarity. Produce ONE consolidated summary capturing the "
    "stable pattern across them, plus up to 3 tags. If the texts do not "
    "share a coherent pattern, return null as the summary. "
    "Respond with ONE JSON object and nothing else: "
    '{"summary": string|null, "tags": array of <=3 strings, "rationale": string}. '
    "No prose, no code fences, no preamble."
)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def cluster_vectors(
    vectors: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> List[int]:
    """Run HDBSCAN with cosine metric. Returns a label per input row.

    Label -1 indicates noise (not clustered). Returns all-noise if there are
    fewer input rows than ``min_cluster_size``.
    """
    if len(vectors) < min_cluster_size:
        return [-1] * len(vectors)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",  # HDBSCAN doesn't support cosine natively
    )
    # Normalize vectors so euclidean ≈ cosine.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = np.where(norms > 0, vectors / norms, vectors)
    labels = clusterer.fit_predict(normed)
    return [int(l) for l in labels]


async def summarize_cluster(
    *,
    client,
    model_name: str,
    member_texts: List[str],
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """Call the LLM once to summarize a cluster. Returns normalized dict or None.

    None is returned when the LLM says the pattern is not coherent, or when
    the LLM output cannot be parsed as the expected JSON shape.
    """
    user_prompt = "Memory texts to consolidate:\n\n" + "\n".join(
        f"- {t}" for t in member_texts
    ) + "\n\nOutput the JSON object only."
    resp = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": CLUSTER_SUMMARIZER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    parsed = _extract_json(content)
    if parsed is None:
        return None
    summary = parsed.get("summary")
    if summary is None or not isinstance(summary, str) or not summary.strip():
        return None
    tags = parsed.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t)[:64] for t in tags[:3]]
    rationale = str(parsed.get("rationale") or "")[:280]
    return {"summary": summary.strip(), "tags": tags, "rationale": rationale}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec -T agent pytest tests/test_memory_clustering.py -v
```

Expected: all five tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/memory_clustering.py services/agent/tests/test_memory_clustering.py
git commit -m "feat(memory): HDBSCAN wrapper and LLM cluster summarizer"
```

---

## Task 10: `memory_review` handler — Steps 1+2 (scan + decay/boost)

**Files:**
- Create: `services/agent/selene_agent/autonomy/handlers/memory_review.py`
- Create: `services/agent/tests/test_memory_review_handler.py`

- [ ] **Step 1: Write failing tests for the scan+decay pass**

```python
"""Tests for the memory_review handler pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


def _stub_point(pid, text, importance, tier="L2", created="2026-04-13T00:00:00+00:00",
                access_count=0, source_ids=None, importance_effective=None):
    p = MagicMock()
    p.id = pid
    p.payload = {
        "text": text,
        "timestamp": created,
        "importance": importance,
        "importance_effective": importance_effective
            if importance_effective is not None else importance,
        "access_count": access_count,
        "tier": tier,
        "source_ids": source_ids or [],
        "tags": [],
    }
    p.vector = [0.0] * 8
    return p


@pytest.fixture
def qdrant_stub():
    c = MagicMock()
    c.scroll = MagicMock()
    c.set_payload = MagicMock()
    c.upsert = MagicMock()
    c.delete = MagicMock()
    return c


@pytest.mark.asyncio
async def test_scan_and_decay_updates_importance_effective(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # Two L2 points: one fresh, one 60d old, same base importance.
    pts = [
        _stub_point("fresh", "x", importance=4,
                    created="2026-04-13T00:00:00+00:00"),
        _stub_point("old", "x", importance=4,
                    created="2026-02-12T00:00:00+00:00"),
    ]
    qdrant_stub.scroll.return_value = (pts, None)

    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l2_scanned": 0, "importance_adjusted": 0}
    await memory_review._scan_and_decay(qdrant_stub, stats)
    assert stats["l2_scanned"] == 2
    assert stats["importance_adjusted"] == 2
    # set_payload was called with per-id updates; collect values.
    calls = qdrant_stub.set_payload.call_args_list
    got = {}
    for call in calls:
        payload = call.kwargs["payload"]
        for pid in call.kwargs["points"]:
            got[pid] = payload["importance_effective"]
    assert got["fresh"] == pytest.approx(4.0, abs=0.01)
    # 60 days -> half-life decay ~= 4 * 1/e ~= 1.47
    assert got["old"] == pytest.approx(4 * 2.718281828 ** -1, abs=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `handlers/memory_review.py` with Steps 1+2**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/handlers/memory_review.py services/agent/tests/test_memory_review_handler.py
git commit -m "feat(memory): memory_review handler — scan + decay/boost (steps 1/2)"
```

---

## Task 11: `memory_review` — Step 3 (cluster new L2 → create L3)

**Files:**
- Modify: `services/agent/selene_agent/autonomy/handlers/memory_review.py`
- Modify: `services/agent/tests/test_memory_review_handler.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_memory_review_handler.py`:

```python
@pytest.mark.asyncio
async def test_cluster_step_creates_l3_with_source_ids(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review
    from selene_agent.autonomy import memory_clustering

    # 6 L2 points, all recent. Pretend HDBSCAN clusters them into label 0.
    pts = [_stub_point(f"e{i}", "text", 3) for i in range(6)]
    for p in pts:
        p.vector = [float(i) for i in range(8)]
    qdrant_stub.scroll.return_value = (pts, None)

    monkeypatch.setattr(memory_clustering, "cluster_vectors", lambda v, **k: [0] * len(v))
    async def _summarize(**kw):
        return {"summary": "unified topic", "tags": ["t1", "t2"], "rationale": "because"}
    monkeypatch.setattr(memory_clustering, "summarize_cluster", _summarize)

    # Embedding service stub.
    monkeypatch.setattr(memory_review, "_embed", lambda text: [0.0] * 1024)

    stats = {"l3_created": 0, "clusters_found": 0, "noise_points": 0, "llm_calls": 0}
    since = datetime(2026, 4, 1, tzinfo=timezone.utc)
    await memory_review._cluster_to_l3(
        qdrant_stub, stats,
        since=since, llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l3_created"] == 1
    assert stats["clusters_found"] == 1
    # upsert was called once with a PointStruct whose payload has tier=L3 + source_ids.
    upsert_call = qdrant_stub.upsert.call_args
    point = upsert_call.kwargs["points"][0]
    assert point.payload["tier"] == "L3"
    assert set(point.payload["source_ids"]) == {f"e{i}" for i in range(6)}


@pytest.mark.asyncio
async def test_cluster_step_skips_when_too_few_new_points(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    qdrant_stub.scroll.return_value = ([_stub_point("a", "x", 3)], None)
    stats = {"l3_created": 0, "clusters_found": 0, "noise_points": 0, "llm_calls": 0}
    await memory_review._cluster_to_l3(
        qdrant_stub, stats,
        since=datetime(2026, 4, 1, tzinfo=timezone.utc),
        llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l3_created"] == 0
    qdrant_stub.upsert.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: FAIL — `_cluster_to_l3` not defined.

- [ ] **Step 3: Implement Step 3 in `memory_review.py`**

Add these imports at the top (near existing imports):

```python
import os
import uuid
from typing import Optional

import numpy as np
import requests

from selene_agent.autonomy import memory_clustering
```

Add this helper:

```python
def _embed(text: str) -> List[float]:
    """Get a single embedding via the TEI service (same host as mcp_qdrant_tools)."""
    url = os.getenv("EMBEDDINGS_URL", "http://embeddings:3000")
    r = requests.post(f"{url}/embed", json={"inputs": text}, timeout=30)
    r.raise_for_status()
    return r.json()[0]
```

Add the Step 3 function:

```python
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
```

Update `handle()` to call Step 3 after Step 1/2:

```python
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
```

- [ ] **Step 4: Run tests**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/handlers/memory_review.py services/agent/tests/test_memory_review_handler.py
git commit -m "feat(memory): memory_review step 3 — cluster new L2 → L3"
```

---

## Task 12: `memory_review` — Steps 4+5 (propose L4, prune L2 with source protection)

**Files:**
- Modify: `services/agent/selene_agent/autonomy/handlers/memory_review.py`
- Modify: `services/agent/tests/test_memory_review_handler.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_memory_review_handler.py`:

```python
@pytest.mark.asyncio
async def test_propose_l4_flags_eligible_l3(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # L3 candidate: old enough, important enough, accessed enough.
    old_enough = "2026-03-15T00:00:00+00:00"
    candidate = _stub_point(
        "l3a", "core preference", importance=5,
        tier="L3", created=old_enough, access_count=5,
        importance_effective=5.0,
    )
    qdrant_stub.scroll.return_value = ([candidate], None)
    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l4_proposed": 0, "llm_calls": 0}
    await memory_review._propose_l4(
        qdrant_stub, stats,
        llm_client=MagicMock(), model_name="gpt-3.5-turbo",
    )
    assert stats["l4_proposed"] == 1
    call = qdrant_stub.set_payload.call_args
    payload = call.kwargs["payload"]
    assert payload["pending_l4_approval"] is True
    assert payload["proposed_at"] is not None


@pytest.mark.asyncio
async def test_prune_respects_source_protection(qdrant_stub, monkeypatch):
    from selene_agent.autonomy.handlers import memory_review

    # L3 references "protected". Both "protected" and "unprotected" are
    # stale+low-importance L2, but only "unprotected" should be deleted.
    l3 = _stub_point("l3", "x", 3, tier="L3", source_ids=["protected"])
    l2_protected = _stub_point(
        "protected", "p", importance=1, tier="L2",
        created="2025-09-01T00:00:00+00:00",
        importance_effective=0.1,
    )
    l2_free = _stub_point(
        "unprotected", "u", importance=1, tier="L2",
        created="2025-09-01T00:00:00+00:00",
        importance_effective=0.1,
    )

    def _scroll_side_effect(**kw):
        flt = kw.get("scroll_filter")
        # Detect tier being filtered by stringifying the filter.
        s = str(flt)
        if "'L3'" in s:
            return ([l3], None)
        return ([l2_protected, l2_free], None)

    qdrant_stub.scroll.side_effect = _scroll_side_effect
    monkeypatch.setattr(memory_review, "_now", lambda: datetime(2026, 4, 13, tzinfo=timezone.utc))

    stats = {"l2_pruned": 0}
    await memory_review._prune_l2(qdrant_stub, stats)
    assert stats["l2_pruned"] == 1
    delete_call = qdrant_stub.delete.call_args
    ids = delete_call.kwargs["points_selector"].points
    assert ids == ["unprotected"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: FAIL for the two new tests.

- [ ] **Step 3: Implement Steps 4 and 5**

Add to `memory_review.py`:

```python
async def _propose_l4(
    client,
    stats: Dict[str, Any],
    *,
    llm_client,
    model_name: str,
) -> None:
    """Step 4: flag eligible L3 entries as pending_l4_approval."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, Range,
    )
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

    flt = Filter(
        must=[
            FieldCondition(key="tier", match=MatchValue(value="L3")),
            FieldCondition(key="pending_l4_approval", match=MatchValue(value=False)),
            FieldCondition(
                key="importance_effective",
                range=Range(gte=config.MEMORY_L4_MIN_IMPORTANCE),
            ),
        ]
    )
    candidates = _scroll_all(
        client, flt=flt, collection=COLLECTION_NAME, cap=500,
    )
    now = _now()
    now_iso = now.isoformat()
    for p in candidates:
        payload = p.payload or {}
        created = _parse_ts(payload.get("timestamp", ""))
        age_days = max(0, (now - created).days)
        if age_days < config.MEMORY_L4_MIN_AGE_DAYS:
            continue
        access_ok = (
            int(payload.get("access_count", 0) or 0) >= config.MEMORY_L4_MIN_ACCESS_COUNT
            or "core_fact" in (payload.get("tags") or [])
        )
        if not access_ok:
            continue
        # Short LLM-authored rationale. If this fails, fall back to static text.
        rationale = (
            "Consolidated high-importance memory has aged and been retrieved "
            "enough to warrant persistent context."
        )
        try:
            resp = await llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content":
                     "Write ONE short sentence (<=120 chars) justifying why "
                     "this consolidated memory should be promoted to the "
                     "always-in-context tier. No prose beyond the sentence."},
                    {"role": "user", "content": str(payload.get("text", ""))[:500]},
                ],
                max_tokens=80,
                temperature=0.2,
            )
            candidate_text = (resp.choices[0].message.content or "").strip()
            if candidate_text:
                rationale = candidate_text[:240]
            stats["llm_calls"] = stats.get("llm_calls", 0) + 1
        except Exception as e:
            logger.warning(f"[memory_review] rationale LLM failed: {e}")

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "pending_l4_approval": True,
                "proposed_at": now_iso,
                "proposal_rationale": rationale,
            },
            points=[str(p.id)],
        )
        stats["l4_proposed"] = stats.get("l4_proposed", 0) + 1


async def _prune_l2(client, stats: Dict[str, Any]) -> None:
    """Step 5: delete stale low-importance L2 entries not referenced by any L3."""
    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue, Range, PointIdsList,
    )
    from selene_agent.modules.mcp_qdrant_tools.qdrant_mcp_server import COLLECTION_NAME

    # Gather all L3 source_ids first (protection set).
    l3_flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    l3s = _scroll_all(client, flt=l3_flt, collection=COLLECTION_NAME, cap=5000)
    protected: set[str] = set()
    for p in l3s:
        for sid in (p.payload or {}).get("source_ids") or []:
            protected.add(str(sid))

    # Scroll candidates: L2, importance_effective below threshold.
    cand_flt = Filter(
        must=[
            FieldCondition(key="tier", match=MatchValue(value="L2")),
            FieldCondition(
                key="importance_effective",
                range=Range(lt=config.MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD),
            ),
        ]
    )
    candidates = _scroll_all(
        client, flt=cand_flt, collection=COLLECTION_NAME, cap=5000,
    )
    now = _now()
    to_delete: List[str] = []
    for p in candidates:
        payload = p.payload or {}
        created = _parse_ts(payload.get("timestamp", ""))
        age_days = max(0, (now - created).days)
        if age_days < config.MEMORY_L2_PRUNE_AGE_DAYS:
            continue
        pid = str(p.id)
        if pid in protected:
            continue
        to_delete.append(pid)

    if to_delete:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=to_delete),
        )
    stats["l2_pruned"] = len(to_delete)
```

Then wire them into `handle()` after the cluster step:

```python
    try:
        await _propose_l4(qc, stats, llm_client=client, model_name=model_name)
    except Exception as e:
        logger.error(f"[memory_review] step 4 failed: {e}")

    try:
        await _prune_l2(qc, stats)
    except Exception as e:
        logger.error(f"[memory_review] step 5 failed: {e}")
```

- [ ] **Step 4: Run all memory_review tests**

```bash
docker compose exec -T agent pytest tests/test_memory_review_handler.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/handlers/memory_review.py services/agent/tests/test_memory_review_handler.py
git commit -m "feat(memory): memory_review steps 4+5 — L4 proposals + source-protected prune"
```

---

## Task 13: Engine wiring — dispatch + default agenda seed

**Files:**
- Modify: `services/agent/selene_agent/autonomy/engine.py:180-183`
- Modify: `services/agent/selene_agent/autonomy/db.py:82-103`

- [ ] **Step 1: Register the new handler in the engine's dispatch table**

In `services/agent/selene_agent/autonomy/engine.py`, find the existing block:

```python
handler = {
    "briefing": briefing_handler.handle,
    "anomaly_sweep": anomaly_handler.handle,
}.get(kind)
```

Change to:

```python
from selene_agent.autonomy.handlers import memory_review as memory_review_handler

handler = {
    "briefing": briefing_handler.handle,
    "anomaly_sweep": anomaly_handler.handle,
    "memory_review": memory_review_handler.handle,
}.get(kind)
```

(Move the new import to the top of the file alongside the other handler imports.)

- [ ] **Step 2: Add the default agenda row in `db.py::ensure_default_agenda`**

In the `defaults = [...]` list in `services/agent/selene_agent/autonomy/db.py`, append a third entry:

```python
{
    "kind": "memory_review",
    "cron": config.AUTONOMY_MEMORY_REVIEW_CRON,
    "autonomy_level": "observe",
    "cfg": {
        "max_scan": config.AUTONOMY_MEMORY_MAX_SCAN,
        "llm_call_cap": config.AUTONOMY_MEMORY_LLM_CALL_CAP,
    },
},
```

- [ ] **Step 3: Restart agent and confirm default row exists**

```bash
docker compose restart agent
sleep 5
docker compose exec -T postgres psql -U postgres -d havencore -c \
  "SELECT kind, schedule_cron, autonomy_level, enabled FROM agenda_items WHERE kind='memory_review';"
```

Expected: one row, `schedule_cron='0 3 * * *'`, `autonomy_level='observe'`, `enabled=t`.

- [ ] **Step 4: Trigger the run manually to confirm end-to-end wiring**

```bash
ITEM_ID=$(docker compose exec -T postgres psql -U postgres -d havencore -t -A -c \
  "SELECT id FROM agenda_items WHERE kind='memory_review' LIMIT 1;")
curl -s -X POST "http://localhost:6002/api/autonomy/trigger/$ITEM_ID" | jq
docker compose exec -T postgres psql -U postgres -d havencore -c \
  "SELECT kind, status, summary FROM autonomy_runs WHERE kind='memory_review' ORDER BY triggered_at DESC LIMIT 1;"
```

Expected: a new row in `autonomy_runs` with `kind='memory_review'` and `status='ok'`.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/autonomy/engine.py services/agent/selene_agent/autonomy/db.py
git commit -m "feat(memory): wire memory_review handler into engine + default agenda seed"
```

---

## Task 14: `l4_context` — builder + in-memory cache

**Files:**
- Create: `services/agent/selene_agent/utils/l4_context.py`
- Create: `services/agent/tests/test_l4_context.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_l4_context.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `utils/l4_context.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```bash
docker compose exec -T agent pytest tests/test_l4_context.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/selene_agent/utils/l4_context.py services/agent/tests/test_l4_context.py
git commit -m "feat(memory): L4 context block builder with invalidation cache"
```

---

## Task 15: L4 injection into user and autonomous system prompts

**Files:**
- Modify: `services/agent/selene_agent/orchestrator.py:89-93`
- Modify: `services/agent/selene_agent/autonomy/turn.py:61-73`

- [ ] **Step 1: Inject into user-facing orchestrator**

Locate the system-prompt initialization in `orchestrator.py` (near line 91). Replace:

```python
system_prompt = config.SYSTEM_PROMPT
self.messages = [{"role": "system", "content": system_prompt}]
```

with:

```python
system_prompt = config.SYSTEM_PROMPT
# v2: prepend L4 persistent-memory block when populated.
try:
    from selene_agent.utils.l4_context import build_l4_block
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    if loop.is_running():
        # Build synchronously via run_coroutine_threadsafe would deadlock;
        # instead, skip for synchronous init path. Callers that run in an
        # async context should call AgentOrchestrator.prepare() below.
        self._l4_pending = True
    else:
        block = loop.run_until_complete(build_l4_block())
        if block:
            system_prompt = block + "\n\n" + system_prompt
        self._l4_pending = False
except Exception:
    self._l4_pending = False
self.messages = [{"role": "system", "content": system_prompt}]
```

And add a `prepare()` async method on `AgentOrchestrator` that lazily prepends the block when `_l4_pending` is true. Place it near the other instance methods:

```python
async def prepare(self) -> None:
    """Lazily prepend L4 block if __init__ ran in a running-loop context."""
    if not getattr(self, "_l4_pending", False):
        return
    from selene_agent.utils.l4_context import build_l4_block
    block = await build_l4_block()
    if block and self.messages and self.messages[0].get("role") == "system":
        self.messages[0]["content"] = block + "\n\n" + self.messages[0]["content"]
    self._l4_pending = False
```

Then update `AgentOrchestrator.run()` (the public generator) to call `await self.prepare()` as its first line.

- [ ] **Step 2: Inject into autonomous turns**

In `services/agent/selene_agent/autonomy/turn.py::run`, replace:

```python
orch.messages = [{"role": "system", "content": self.system_prompt}]
```

with:

```python
from selene_agent.utils.l4_context import build_l4_block
_l4 = await build_l4_block()
_sys = (_l4 + "\n\n" + self.system_prompt) if _l4 else self.system_prompt
orch.messages = [{"role": "system", "content": _sys}]
# Autonomous turn already handled L4 injection — skip prepare()'s path.
orch._l4_pending = False
```

- [ ] **Step 3: Smoke-test via the chat endpoint with a seeded L4 entry**

```bash
# Seed one L4 entry directly via Qdrant for the smoke test.
docker compose exec -T agent python - <<'PY'
import asyncio, os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import requests, uuid
client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=6333)
r = requests.post("http://embeddings:3000/embed", json={"inputs": "test L4 entry"})
vec = r.json()[0]
client.upsert(
    collection_name=os.getenv("QDRANT_COLLECTION", "user_data"),
    points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload={
        "text": "SMOKE: Matt is testing L4 injection.",
        "tier": "L4", "importance": 5, "importance_effective": 5.0,
        "timestamp": "2026-04-13T00:00:00+00:00",
        "pending_l4_approval": False, "source_ids": [], "tags": ["smoke"],
        "access_count": 0, "last_accessed_at": None,
        "proposed_at": None, "proposal_rationale": None,
    })],
)
PY
docker compose restart agent
sleep 5
# Hit the chat and inspect agent logs for the system prompt build.
curl -s http://localhost:6002/api/chat -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}' > /dev/null
docker compose logs --tail 100 agent | grep -i "persistent_memories" || \
  docker compose exec -T agent python -c \
  "import asyncio; from selene_agent.utils.l4_context import invalidate_cache, build_l4_block; \
   invalidate_cache(); print(asyncio.run(build_l4_block()))"
```

Expected: the final python command prints a block containing `SMOKE: Matt is testing L4 injection.`

- [ ] **Step 4: Commit**

```bash
git add services/agent/selene_agent/orchestrator.py services/agent/selene_agent/autonomy/turn.py
git commit -m "feat(memory): inject L4 persistent-memory block into user + autonomous prompts"
```

---

## Task 16: Memory API — L4 CRUD and proposal queue

**Files:**
- Create: `services/agent/selene_agent/api/memory.py`
- Modify: `services/agent/selene_agent/selene_agent.py:244-253`
- Create: `services/agent/tests/test_memory_api.py`

- [ ] **Step 1: Write failing tests for the L4 and proposal endpoints**

```python
"""Tests for /api/memory/* routes — L4 CRUD and proposal queue."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from selene_agent.api.memory import router
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_list_l4_returns_active_entries(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        p = MagicMock()
        p.id, p.payload = "x", {
            "text": "core", "importance": 5, "importance_effective": 5.0,
            "tier": "L4", "timestamp": "2026-04-01T00:00:00+00:00",
            "tags": [], "pending_l4_approval": False,
        }
        c.scroll.return_value = ([p], None)
        qc.return_value = c
        r = client.get("/api/memory/l4")
    assert r.status_code == 200
    body = r.json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["id"] == "x"


def test_approve_promotes_to_l4_and_invalidates_cache(client):
    from selene_agent.utils import l4_context
    with patch("selene_agent.api.memory._qdrant_client") as qc, \
         patch.object(l4_context, "invalidate_cache") as inv:
        c = MagicMock()
        qc.return_value = c
        r = client.post(f"/api/memory/l4/proposals/{uuid.uuid4()}/approve")
    assert r.status_code == 200
    c.set_payload.assert_called_once()
    payload = c.set_payload.call_args.kwargs["payload"]
    assert payload["tier"] == "L4"
    assert payload["pending_l4_approval"] is False
    inv.assert_called_once()


def test_reject_clears_flag_without_promoting(client):
    from selene_agent.utils import l4_context
    with patch("selene_agent.api.memory._qdrant_client") as qc, \
         patch.object(l4_context, "invalidate_cache") as inv:
        c = MagicMock()
        qc.return_value = c
        r = client.post(f"/api/memory/l4/proposals/{uuid.uuid4()}/reject")
    assert r.status_code == 200
    payload = c.set_payload.call_args.kwargs["payload"]
    assert "tier" not in payload  # stays L3
    assert payload["pending_l4_approval"] is False
    inv.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_api.py -v
```

Expected: FAIL — router does not exist.

- [ ] **Step 3: Implement `api/memory.py` — L4 + proposals portion**

```python
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
```

- [ ] **Step 4: Mount the router in `selene_agent.py`**

Near the existing `app.include_router(autonomy_router, prefix="/api")` line, add:

```python
from selene_agent.api.memory import router as memory_router
app.include_router(memory_router, prefix="/api")
```

(Place the import with the other router imports at the top of the file.)

- [ ] **Step 5: Run tests**

```bash
docker compose exec -T agent pytest tests/test_memory_api.py -v
```

Expected: all three tests PASS.

- [ ] **Step 6: Commit**

```bash
git add services/agent/selene_agent/api/memory.py services/agent/selene_agent/selene_agent.py services/agent/tests/test_memory_api.py
git commit -m "feat(memory): REST endpoints for L4 CRUD and proposal queue"
```

---

## Task 17: Memory API — L3 browse, sources, delete; runs; stats; health

**Files:**
- Modify: `services/agent/selene_agent/api/memory.py`
- Modify: `services/agent/selene_agent/api/status.py` (or wherever `/health` lives — grep if unsure)
- Modify: `services/agent/tests/test_memory_api.py`

- [ ] **Step 1: Locate the `/health` implementation**

```bash
grep -rn "def .*health\|@.*get.*/health" /home/matt/code/havencore/services/agent/selene_agent
```

Note the file that owns the route — likely `api/status.py`. Use that path in Step 4 below.

- [ ] **Step 2: Append failing tests for the remaining endpoints**

In `tests/test_memory_api.py`, append:

```python
def test_list_l3_paginates(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        pts = []
        for i in range(3):
            p = MagicMock()
            p.id = f"l{i}"
            p.payload = {
                "text": f"t{i}", "importance": 3, "importance_effective": 3.0,
                "tier": "L3", "timestamp": "2026-04-01T00:00:00+00:00",
                "tags": [], "source_ids": [f"s{i}a", f"s{i}b"],
            }
            pts.append(p)
        c.scroll.return_value = (pts, None)
        qc.return_value = c
        r = client.get("/api/memory/l3?limit=2&offset=1")
    assert r.status_code == 200
    assert len(r.json()["entries"]) == 2


def test_l3_sources_returns_source_l2_entries(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        l3 = MagicMock()
        l3.id, l3.payload = "l3x", {"source_ids": ["a", "b"], "tier": "L3"}
        c.retrieve.side_effect = [[l3], [MagicMock(id="a", payload={"text": "A", "tier": "L2"}),
                                           MagicMock(id="b", payload={"text": "B", "tier": "L2"})]]
        qc.return_value = c
        r = client.get("/api/memory/l3/l3x/sources")
    assert r.status_code == 200
    texts = [s["text"] for s in r.json()["sources"]]
    assert set(texts) == {"A", "B"}


def test_delete_l3_removes_consolidated_entry(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        qc.return_value = c
        r = client.delete("/api/memory/l3/abc")
    assert r.status_code == 200
    c.delete.assert_called_once()


def test_stats_returns_tier_counts(client):
    with patch("selene_agent.api.memory._qdrant_client") as qc:
        c = MagicMock()
        c.count.side_effect = [MagicMock(count=n) for n in (100, 10, 2, 3)]
        qc.return_value = c
        r = client.get("/api/memory/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["l2_count"] == 100
    assert body["l3_count"] == 10
    assert body["l4_count"] == 2
    assert body["pending_proposals"] == 3
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
docker compose exec -T agent pytest tests/test_memory_api.py -v
```

Expected: the four new tests FAIL.

- [ ] **Step 4: Extend `api/memory.py` with L3, runs, stats endpoints**

Append these to `api/memory.py`:

```python
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
    src_ids = (l3s[0].payload or {}).get("source_ids") or []
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


@router.get("/memory/stats")
def stats():
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
    import asyncio
    try:
        block = asyncio.get_event_loop().run_until_complete(
            __import__("selene_agent.utils.l4_context", fromlist=["build_l4_block"]).build_l4_block()
        )
        l4_est_tokens = max(0, len(block) // 4)
    except Exception:
        l4_est_tokens = 0

    return {
        "l2_count": l2,
        "l3_count": l3,
        "l4_count": l4,
        "pending_proposals": pending,
        "l4_est_tokens": l4_est_tokens,
    }
```

- [ ] **Step 5: Add `memory_stats` to the `/health` response**

Open the file that owns the `/health` route (found in Step 1). Locate the autonomy block inside the health payload and add `memory_stats` alongside it:

```python
# inside the existing /health handler, alongside the autonomy block:
from qdrant_client.models import Filter, FieldCondition, MatchValue
from selene_agent.api.memory import _qdrant_client, _collection
try:
    qc = _qdrant_client()
    def _c(flt):
        return qc.count(collection_name=_collection(), count_filter=flt, exact=True).count
    memory_stats = {
        "l2": _c(Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L2"))])),
        "l3": _c(Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])),
        "l4": _c(Filter(must=[
            FieldCondition(key="tier", match=MatchValue(value="L4")),
            FieldCondition(key="pending_l4_approval", match=MatchValue(value=False)),
        ])),
        "pending": _c(Filter(must=[
            FieldCondition(key="tier", match=MatchValue(value="L3")),
            FieldCondition(key="pending_l4_approval", match=MatchValue(value=True)),
        ])),
    }
except Exception:
    memory_stats = {"error": "unavailable"}
```

Then merge `memory_stats` into the returned dict.

- [ ] **Step 6: Run all memory API tests and smoke-check the API**

```bash
docker compose exec -T agent pytest tests/test_memory_api.py -v
docker compose restart agent
sleep 5
curl -s http://localhost:6002/api/memory/stats | jq
curl -s http://localhost:6002/api/memory/l4 | jq
curl -s http://localhost:6002/api/memory/runs | jq
curl -s http://localhost/health | jq '.autonomy, .memory_stats'
```

Expected: endpoints return valid JSON; `/health` includes a `memory_stats` block.

- [ ] **Step 7: Commit**

```bash
git add services/agent/selene_agent/api/memory.py services/agent/tests/test_memory_api.py services/agent/selene_agent/api/status.py
git commit -m "feat(memory): L3 browse/sources/delete, runs, stats, /health memory_stats"
```

---

## Task 18: SvelteKit `/memory` page — stats header, L4 section, proposals

**Files:**
- Create: `services/agent/frontend/src/routes/memory/+page.svelte`
- Create: `services/agent/frontend/src/routes/memory/+page.ts` (if the project uses load functions; inspect `/history` to match pattern)

- [ ] **Step 1: Inspect the existing `/history` page structure to match conventions**

```bash
ls /home/matt/code/havencore/services/agent/frontend/src/routes/history/
cat /home/matt/code/havencore/services/agent/frontend/src/routes/history/+page.svelte | head -60
```

Note: conventions used for fetch, Tailwind classes, loading/error states. Follow them.

- [ ] **Step 2: Create the `/memory` page with the header + L4 + proposals sections**

```svelte
<script lang="ts">
  import { onMount } from 'svelte';

  type L4Entry = {
    id: string;
    text: string;
    importance: number;
    importance_effective: number;
    timestamp: string;
    tags: string[];
    pending_l4_approval: boolean;
    proposed_at?: string;
    proposal_rationale?: string;
    source_ids: string[];
  };

  type Stats = {
    l2_count: number; l3_count: number; l4_count: number;
    pending_proposals: number; l4_est_tokens: number;
  };

  let stats: Stats | null = null;
  let l4: L4Entry[] = [];
  let proposals: L4Entry[] = [];
  let runs: any[] = [];
  let loading = true;
  let error = '';
  let lastRunTime: string | null = null;
  let triggering = false;
  let newL4Text = '';
  let newL4Importance = 5;

  async function refresh() {
    loading = true;
    error = '';
    try {
      const [s, l, p, r] = await Promise.all([
        fetch('/api/memory/stats').then(r => r.json()),
        fetch('/api/memory/l4').then(r => r.json()),
        fetch('/api/memory/l4/proposals').then(r => r.json()),
        fetch('/api/memory/runs?limit=20').then(r => r.json()),
      ]);
      stats = s;
      l4 = l.entries ?? [];
      proposals = p.proposals ?? [];
      runs = r.runs ?? [];
      lastRunTime = runs.length ? runs[0].triggered_at : null;
    } catch (e: any) {
      error = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  async function approve(id: string) {
    await fetch(`/api/memory/l4/proposals/${id}/approve`, { method: 'POST' });
    await refresh();
  }
  async function reject(id: string) {
    await fetch(`/api/memory/l4/proposals/${id}/reject`, { method: 'POST' });
    await refresh();
  }
  async function deleteL4(id: string) {
    if (!confirm('Remove this L4 entry? It will be demoted to L3, not deleted.')) return;
    await fetch(`/api/memory/l4/${id}`, { method: 'DELETE' });
    await refresh();
  }
  async function addL4() {
    if (!newL4Text.trim()) return;
    await fetch('/api/memory/l4', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text: newL4Text, importance: newL4Importance, tags: [] }),
    });
    newL4Text = '';
    await refresh();
  }
  async function runNow() {
    triggering = true;
    try {
      await fetch('/api/memory/runs/trigger', { method: 'POST' });
      // Poll for completion.
      for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 3000));
        await refresh();
        if (runs[0]?.triggered_at !== lastRunTime) break;
      }
    } finally {
      triggering = false;
    }
  }

  onMount(refresh);
</script>

<svelte:head><title>Memory — Selene</title></svelte:head>

<div class="p-6 max-w-5xl mx-auto">
  <h1 class="text-2xl font-bold mb-4">Memory</h1>

  {#if error}
    <div class="mb-4 p-3 bg-red-900/30 border border-red-700 rounded">{error}</div>
  {/if}

  <!-- Header stats -->
  <section class="mb-6 grid grid-cols-5 gap-4 text-sm">
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L2 episodic</div>
      <div class="text-xl">{stats?.l2_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L3 consolidated</div>
      <div class="text-xl">{stats?.l3_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L4 persistent</div>
      <div class="text-xl">{stats?.l4_count ?? '—'}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">Pending proposals</div>
      <div class="text-xl {proposals.length ? 'text-amber-400' : ''}">{stats?.pending_proposals ?? 0}</div>
    </div>
    <div class="p-3 bg-gray-800 rounded">
      <div class="text-gray-400">L4 ~tokens</div>
      <div class="text-xl {stats && stats.l4_est_tokens > 1500 ? 'text-red-400' : ''}">
        {stats?.l4_est_tokens ?? 0}
      </div>
    </div>
  </section>

  <div class="mb-6 flex items-center gap-3">
    <button
      class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50"
      on:click={runNow}
      disabled={triggering}
    >{triggering ? 'Running…' : 'Run consolidation now'}</button>
    <span class="text-gray-400 text-sm">
      Last run: {lastRunTime ?? 'never'}
    </span>
  </div>

  <!-- L4 section -->
  <section class="mb-8">
    <h2 class="text-xl font-semibold mb-3">L4 — Persistent context</h2>

    <div class="mb-4 p-3 bg-gray-800 rounded flex gap-2 items-start">
      <textarea
        class="flex-1 bg-gray-900 rounded p-2"
        rows="2"
        placeholder="Add an L4 entry (injected into every system prompt)"
        bind:value={newL4Text}
      ></textarea>
      <div class="flex flex-col gap-2">
        <label class="text-xs text-gray-400">Importance
          <input type="number" min="1" max="5" bind:value={newL4Importance}
                 class="w-16 bg-gray-900 rounded px-2 py-1" />
        </label>
        <button class="px-3 py-1 bg-green-600 hover:bg-green-500 rounded"
                on:click={addL4}>Add</button>
      </div>
    </div>

    {#if l4.length === 0}
      <div class="text-gray-400 italic">No L4 entries yet.</div>
    {:else}
      <table class="w-full text-sm">
        <thead class="text-left text-gray-400 border-b border-gray-700">
          <tr><th class="py-2">Text</th><th>Importance</th><th>Age</th><th></th></tr>
        </thead>
        <tbody>
          {#each l4 as e}
            <tr class="border-b border-gray-800">
              <td class="py-2">{e.text}</td>
              <td>{e.importance}</td>
              <td>{e.timestamp?.slice(0, 10)}</td>
              <td class="text-right">
                <button class="text-red-400 hover:text-red-200"
                        on:click={() => deleteL4(e.id)}>Remove</button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}

    <h3 class="text-lg font-semibold mt-6 mb-2">Pending proposals</h3>
    {#if proposals.length === 0}
      <div class="text-gray-400 italic">No pending proposals.</div>
    {:else}
      <div class="space-y-2">
        {#each proposals as p}
          <div class="p-3 bg-gray-800 rounded">
            <div class="font-medium">{p.text}</div>
            {#if p.proposal_rationale}
              <div class="text-sm text-gray-400 italic mt-1">{p.proposal_rationale}</div>
            {/if}
            <div class="text-xs text-gray-500 mt-1">
              sources: {p.source_ids.length} · importance: {p.importance}
            </div>
            <div class="mt-2 flex gap-2">
              <button class="px-3 py-1 bg-green-600 hover:bg-green-500 rounded"
                      on:click={() => approve(p.id)}>Approve</button>
              <button class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded"
                      on:click={() => reject(p.id)}>Reject</button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </section>
</div>
```

- [ ] **Step 3: Rebuild the frontend and verify page loads**

```bash
cd /home/matt/code/havencore/services/agent/frontend && npm run build && cd -
docker compose restart agent
sleep 5
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:6002/memory
```

Expected: 200.

Also open `http://localhost:6002/memory` in a browser and confirm:
- Header stats render (L2/L3/L4 counts even if zero)
- "Run consolidation now" button visible
- Adding an L4 entry directly works and it appears in the list immediately after
- Deleting an L4 entry demotes it (removes from list)

- [ ] **Step 4: Commit**

```bash
git add services/agent/frontend/src/routes/memory/
git commit -m "feat(frontend): add /memory page — stats, L4 editor, proposal queue"
```

---

## Task 19: SvelteKit `/memory` page — L3 browser and run history

**Files:**
- Modify: `services/agent/frontend/src/routes/memory/+page.svelte`

- [ ] **Step 1: Add L3 browser + run history sections below the L4 section**

Before the closing `</div>` of the page's outer `<div>`, append:

```svelte
  <!-- L3 browser -->
  <section class="mb-8">
    <h2 class="text-xl font-semibold mb-3">L3 — Consolidated memories</h2>
    {#if l3Entries.length === 0}
      <div class="text-gray-400 italic">No L3 entries yet.</div>
    {:else}
      <table class="w-full text-sm">
        <thead class="text-left text-gray-400 border-b border-gray-700">
          <tr><th class="py-2">Text</th><th>Importance (eff)</th><th>Sources</th><th>Age</th><th></th></tr>
        </thead>
        <tbody>
          {#each l3Entries as e}
            <tr class="border-b border-gray-800">
              <td class="py-2">{e.text}</td>
              <td>{e.importance_effective?.toFixed(2)}</td>
              <td>
                <button class="text-blue-400 hover:text-blue-200 underline"
                        on:click={() => showSources(e.id)}>
                  {e.source_ids.length}
                </button>
              </td>
              <td>{e.timestamp?.slice(0, 10)}</td>
              <td class="text-right">
                <button class="text-red-400 hover:text-red-200"
                        on:click={() => deleteL3(e.id)}>Delete</button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
      <div class="mt-2 flex gap-2 text-sm">
        <button class="px-2 py-1 bg-gray-700 rounded disabled:opacity-50"
                disabled={l3Offset === 0}
                on:click={() => { l3Offset = Math.max(0, l3Offset - 50); refreshL3(); }}>Prev</button>
        <button class="px-2 py-1 bg-gray-700 rounded disabled:opacity-50"
                disabled={!l3HasMore}
                on:click={() => { l3Offset += 50; refreshL3(); }}>Next</button>
      </div>
    {/if}
  </section>

  {#if sourcesModal}
    <div class="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
         on:click={() => (sourcesModal = null)}>
      <div class="bg-gray-900 p-6 rounded max-w-2xl max-h-[80vh] overflow-auto"
           on:click|stopPropagation>
        <h3 class="text-lg font-semibold mb-3">Source L2 entries</h3>
        {#each sourcesModal as src}
          <div class="p-2 border-b border-gray-800 text-sm">
            <div>{src.text}</div>
            <div class="text-xs text-gray-500">{src.timestamp}</div>
          </div>
        {/each}
        <button class="mt-3 px-3 py-1 bg-gray-700 rounded"
                on:click={() => (sourcesModal = null)}>Close</button>
      </div>
    </div>
  {/if}

  <!-- Run history -->
  <section>
    <h2 class="text-xl font-semibold mb-3">Consolidation runs</h2>
    {#if runs.length === 0}
      <div class="text-gray-400 italic">No runs yet.</div>
    {:else}
      <table class="w-full text-sm">
        <thead class="text-left text-gray-400 border-b border-gray-700">
          <tr><th class="py-2">When</th><th>Status</th><th>Summary</th><th>Stats</th></tr>
        </thead>
        <tbody>
          {#each runs as r}
            <tr class="border-b border-gray-800 align-top"
                class:text-red-400={r.status === 'error'}>
              <td class="py-2">{r.triggered_at}</td>
              <td>{r.status}</td>
              <td>{r.summary}</td>
              <td class="text-xs">
                L3+{r.metrics?.l3_created ?? 0}
                · L4?{r.metrics?.l4_proposed ?? 0}
                · pruned {r.metrics?.l2_pruned ?? 0}
                · {r.metrics?.total_ms ?? 0}ms
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
```

- [ ] **Step 2: Add state + fetch functions to the `<script>` block**

Near the existing state declarations inside `<script>`:

```ts
  let l3Entries: L4Entry[] = [];
  let l3Offset = 0;
  let l3HasMore = false;
  let sourcesModal: any[] | null = null;

  async function refreshL3() {
    const r = await fetch(`/api/memory/l3?limit=50&offset=${l3Offset}`).then(r => r.json());
    l3Entries = r.entries ?? [];
    l3HasMore = !!r.has_more;
  }
  async function showSources(id: string) {
    const r = await fetch(`/api/memory/l3/${id}/sources`).then(r => r.json());
    sourcesModal = r.sources ?? [];
  }
  async function deleteL3(id: string) {
    if (!confirm('Delete this L3 entry? Source L2 entries will remain untouched.')) return;
    await fetch(`/api/memory/l3/${id}`, { method: 'DELETE' });
    await refreshL3();
  }
```

Update the `refresh()` function to also call `refreshL3()`:

```ts
  async function refresh() {
    loading = true; error = '';
    try {
      const [s, l, p, r] = await Promise.all([
        fetch('/api/memory/stats').then(r => r.json()),
        fetch('/api/memory/l4').then(r => r.json()),
        fetch('/api/memory/l4/proposals').then(r => r.json()),
        fetch('/api/memory/runs?limit=20').then(r => r.json()),
      ]);
      stats = s;
      l4 = l.entries ?? [];
      proposals = p.proposals ?? [];
      runs = r.runs ?? [];
      lastRunTime = runs.length ? runs[0].triggered_at : null;
      await refreshL3();
    } catch (e: any) {
      error = e?.message ?? String(e);
    } finally {
      loading = false;
    }
  }
```

- [ ] **Step 3: Rebuild frontend and smoke-test**

```bash
cd /home/matt/code/havencore/services/agent/frontend && npm run build && cd -
docker compose restart agent
```

In a browser, reload `/memory` and confirm:
- L3 section shows entries (if any exist), click source count opens modal with source L2 texts
- Run history table shows the manual-trigger run from Task 13
- Pagination Prev/Next on L3 works (test with >50 entries if needed)

- [ ] **Step 4: Commit**

```bash
git add services/agent/frontend/src/routes/memory/+page.svelte
git commit -m "feat(frontend): /memory page — L3 browser + run history"
```

---

## Task 20: End-to-end integration test + docs

**Files:**
- Create: `services/agent/tests/test_integration_memory_review.py`
- Create: `docs/services/agent/autonomy/memory/README.md`

- [ ] **Step 1: Write the integration test**

This test requires a live Qdrant — run via `docker compose exec`.

```python
"""End-to-end test: consolidation run against live Qdrant.

Seeds ~20 synthetic L2 entries across two themes, triggers memory_review
via the engine's manual path, and asserts L3 entries appear with
populated source_ids and correct run metrics.
"""
from __future__ import annotations

import os
import uuid
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
                "timestamp": "2026-04-10T00:00:00+00:00",
                "tags": [], "source_ids": [], "access_count": 0,
                "last_accessed_at": None, "pending_l4_approval": False,
                "proposed_at": None, "proposal_rationale": None,
            },
        )])

    # Trigger the run via the engine.
    from selene_agent.autonomy import db as autonomy_db
    items = await autonomy_db.list_all_items()
    target = next(i for i in items if i["kind"] == "memory_review")
    from selene_agent.selene_agent import app
    engine = app.state.autonomy_engine
    result = await engine.trigger(target["id"])
    assert result["status"] == "ok"

    # Assert at least one L3 was created with source_ids pointing into our seeded set.
    flt = Filter(must=[FieldCondition(key="tier", match=MatchValue(value="L3"))])
    pts, _ = client.scroll(collection_name=coll, scroll_filter=flt,
                           limit=50, with_payload=True)
    seeded_set = set(ids)
    assert any(
        bool(set((p.payload or {}).get("source_ids") or []) & seeded_set)
        for p in pts
    ), "expected at least one L3 with source_ids overlapping seeded L2 set"

    # Cleanup — delete seeded L2 points.
    client.delete(collection_name=coll, points_selector=PointIdsList(points=ids))
```

- [ ] **Step 2: Run the integration test**

```bash
docker compose exec -T agent pytest tests/test_integration_memory_review.py -v -s
```

Expected: PASS. (If the LLM judges the clusters incoherent and no L3 is created, the test will fail — in that rare case, inspect the handler's LLM output via `autonomy_runs.messages` and adjust the prompt.)

- [ ] **Step 3: Create the memory docs**

```markdown
# Memory consolidation (AutonomyEngine v2)

## Tiers
- **L1 Ephemeral** — in-turn context.
- **L2 Episodic** — raw Qdrant entries (`tier='L2'`).
- **L3 Consolidated** — summarized clusters of L2 entries with `source_ids` back-links (`tier='L3'`).
- **L4 Persistent context** — small high-importance set, approved by user, injected into every system prompt (`tier='L4'`).

## Nightly job
Runs at `AUTONOMY_MEMORY_REVIEW_CRON` (default `0 3 * * *`). Handler: `selene_agent.autonomy.handlers.memory_review`. Pipeline:

1. Scan L2 → compute `importance_effective` via `exp(-age/halflife) + coef*log(1+accesses)`.
2. Cluster new L2 (since last run) with HDBSCAN; LLM summarizes each cluster into one L3 entry with `source_ids`.
3. Flag eligible L3 as `pending_l4_approval=true` (age ≥ 14d, importance_effective ≥ 4, access_count ≥ 3 OR tag `core_fact`).
4. Prune stale low-importance L2 entries; entries referenced by any L3's `source_ids` are always protected.

## Retrieval
`search_memories` excludes L4 (already in prompt), applies `MEMORY_L3_RANK_BOOST` to L3 scores, and fires an async `access_count` update.

## Dashboard
`/memory` — stats, L4 view/editor, proposal queue, L3 browser with source drill-down, run history, manual trigger.

## Tuning knobs (env)
See `.env.example` — `MEMORY_HALF_LIFE_DAYS`, `MEMORY_ACCESS_COEF`, `MEMORY_L3_RANK_BOOST`, `MEMORY_L4_MAX_ENTRIES`, proposal thresholds, prune thresholds.

## Operations
- Trigger on demand: dashboard "Run consolidation now" or `POST /api/memory/runs/trigger`.
- Health: `GET /health` includes `memory_stats: {l2,l3,l4,pending}`.
- Promotion is **always user-gated** — the pipeline never writes `tier='L4'`.
```

- [ ] **Step 4: Run the full test suite one last time**

```bash
docker compose exec -T agent pytest -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add services/agent/tests/test_integration_memory_review.py docs/services/agent/autonomy/memory/
git commit -m "test(memory): end-to-end integration test + memory docs"
```

---

## Self-Review Summary

**Spec coverage map:**
- Qdrant payload extensions → Tasks 4, 5
- Retrieval changes (ranking, L4 exclusion, access tracking) → Tasks 6, 7
- Pipeline step 1 (scan) → Task 10
- Pipeline step 2 (decay/boost) → Task 10 (+ Task 8 math)
- Pipeline step 3 (cluster → L3) → Task 11 (+ Task 9 clustering)
- Pipeline step 4 (propose L4) → Task 12
- Pipeline step 5 (prune L2 with source protection) → Task 12
- LLM call cap → Task 11 (enforced in `_cluster_to_l3`)
- Engine dispatch + default agenda → Task 13
- L4 injection (user + autonomous) → Task 15 (+ Task 14 builder)
- L4 cache invalidation hooks → Tasks 14 (define), 16 (L4 mutations call it)
- REST endpoints (12 routes) → Tasks 16, 17
- Health `memory_stats` → Task 17
- SvelteKit `/memory` page (4 sections) → Tasks 18, 19
- End-to-end verification → Task 20
- Docs → Task 20

**Placeholder scan:** none. Every step contains the exact code or command it requires.

**Type consistency:** `L4Entry` type shape in the SvelteKit file matches the `_point_out` response from the Python API (both contain the same keys). Pipeline step function names (`_scan_and_decay`, `_cluster_to_l3`, `_propose_l4`, `_prune_l2`) are consistent across the tasks that reference them. `invalidate_cache` is called from exactly the endpoints that mutate L4 or its proposal state.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-13-autonomy-v2-memory.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
