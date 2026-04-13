# HavenCore AutonomyEngine v2 — Memory Consolidation Design

**Status:** Spec (design-approved, not yet planned)
**Date:** 2026-04-13
**Predecessor:** `~/.claude/plans/harmonic-percolating-tower.md` (v1 AutonomyEngine — shipped)
**Scope:** v2 = memory consolidation + memory dashboard. Everything else deferred to v3.

## Context

v1 shipped an event-driven AutonomyEngine with two agenda kinds (`briefing`, `anomaly_sweep`) and groundwork for memory tiering (every new Qdrant point now stamped `tier='L2'` and `source_ids=[]`). The v1 doc stubbed a `<persistent_memories>` section in the autonomous system prompt with a no-op populator, awaiting v2.

v2 turns that groundwork into a working tiered-memory system modeled on hippocampal consolidation: raw episodic entries (L2) are periodically clustered and summarized into consolidated memories (L3), and the most stable high-importance L3 entries get promoted (with user approval) into a small persistent context set (L4) that is injected into every system prompt.

The user has explicitly rejected LoRA / fine-tuning as a long-term-memory mechanism. Rationale (preserved from the auto-memory record): fine-tuning learns style/patterns not facts, 72B AWQ has a painful fine-tune feedback loop, catastrophic forgetting is real, and retrieval modeling maps cleanly to plasticity-via-importance-weighting. Only revisit fine-tuning if retrieval provably can't solve a specific behavior.

## Scope

### In scope (v2)
- New agenda kind `memory_review` — nightly consolidation pipeline.
- L2 → L3 clustering (HDBSCAN) + per-cluster LLM summarization.
- Importance decay/boost with access-count tracking on `search_memories`.
- L3 → L4 promotion as a user-gated proposal queue (never automatic).
- L2 pruning with source protection.
- `search_memories` behavior changes: L3 rank boost, L4 exclusion.
- L4 injection into system prompts for both user-facing and autonomous turns.
- New SvelteKit `/memory` dashboard page (L4 view/editor, proposal queue, L3 browser, run history, manual trigger).
- Small REST surface `/api/memory/*` backing the dashboard.

### Out of scope (deferred to v3)
- Additional agenda kinds (`reminder`, `watch`, `routine`).
- Reactive event sources (HA webhook receiver, MQTT subscribe).
- Full `/autonomy` dashboard (agenda CRUD, live feed).
- Quiet-hours enforcement.
- Dashboard memory nice-to-haves: L2 browser, cluster inspector, importance timeline.
- Explicit `get_memory_sources(memory_id)` MCP tool (v2 relies on the LLM following `source_ids` via freeform `search_memories` follow-ups; revisit if clumsy).
- `speak` and `act` autonomy tiers.
- Smaller gate model for anomaly triage.
- Pattern learning.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ agent service                                                    │
│                                                                  │
│  AutonomyEngine (from v1)                                        │
│     └─► new agenda kind: 'memory_review' (nightly ~3am default)  │
│          └─► MemoryReviewHandler (deterministic pipeline)        │
│                ├─ 1. scan L2                                     │
│                ├─ 2. apply decay/boost → importance_effective    │
│                ├─ 3. cluster new L2 → L3 (HDBSCAN + LLM)         │
│                ├─ 4. propose L3 → L4 (flag only, never promote)  │
│                └─ 5. prune stale L2 (respecting source refs)     │
│                                                                  │
│  MCP qdrant tools (modified)                                     │
│     ├─ search_memories: tier-weighted ranking + access tracking  │
│     └─ create_memory: unchanged (v1 already stamps tier='L2')    │
│                                                                  │
│  System-prompt builder (modified)                                │
│     └─ build_l4_block() injected into both user + autonomous     │
│       system prompts                                             │
│                                                                  │
│  REST: /api/memory/* (dashboard backend)                         │
│  SvelteKit: /memory page                                         │
└──────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
   Postgres                                  Qdrant
   (autonomy_runs:                       (payload gains: access_count,
    one row per nightly run,              last_accessed_at,
    stats in metrics JSON —               importance_effective,
    no new tables)                        pending_l4_approval,
                                          proposed_at,
                                          proposal_rationale;
                                          tier transitions L2→L3→L4)
```

### Invariants

1. **Session isolation (from v1)** — nothing in this system mutates user conversation state. L4 injection modifies the rendered system prompt string only.
2. **User-gated L4** — the consolidation job proposes; it never writes `tier='L4'` directly. Only the dashboard approve endpoint does that.
3. **L4 excluded from retrieval** — L4 entries are in every prompt already; `search_memories` filters them out to avoid wasting tokens.
4. **Source protection on prune** — L2 entries referenced by any L3's `source_ids` are never deleted by the prune step, regardless of age/importance.

## Data model

### Qdrant payload — new/modified fields

On the `user_data` collection:

| Field                  | Type              | Default     | Origin |
|------------------------|-------------------|-------------|--------|
| `tier`                 | `'L2'|'L3'|'L4'`  | `'L2'`      | v1 (already stamped) |
| `source_ids`           | `list[str]`       | `[]`        | v1 (already stamped) |
| `access_count`         | `int`             | `0`         | v2 |
| `last_accessed_at`     | `iso8601 str`     | `null`      | v2 |
| `importance_effective` | `float`           | `null`      | v2 (computed nightly) |
| `pending_l4_approval`  | `bool`            | `false`     | v2 |
| `proposed_at`          | `iso8601 str`     | `null`      | v2 |
| `proposal_rationale`   | `str`             | `null`      | v2 |

### Qdrant payload indexes

Add at collection init (idempotent via `create_payload_index`):
- `tier` (keyword) — required for efficient `scroll` during consolidation.
- `pending_l4_approval` (bool) — for dashboard queue queries.
- `importance_effective` (float) — for ranking / prune candidate queries.

### Backward compatibility

Existing L2 entries without v2 fields read with sensible defaults (`access_count=0`, `importance_effective=importance` on first consolidation pass). No migration script — the nightly job backfills naturally as it touches entries. Search-path already defaults missing `tier` to `'L2'` (v1 behavior).

### Postgres

No new tables. The consolidation job writes one `autonomy_runs` row per execution. Run statistics live in the existing `metrics` jsonb:

```json
{
  "l2_scanned": 412,
  "l3_created": 7,
  "l3_updated": 2,
  "l4_proposed": 1,
  "l2_pruned": 18,
  "importance_adjusted": 412,
  "clusters_found": 7,
  "noise_points": 34,
  "llm_calls": 7,
  "total_ms": 38421
}
```

`summary` receives a one-liner like `"7 new L3 from 412 L2 scanned, 1 L4 proposal, 18 pruned"`.

**Why no Postgres table for L4 proposals:** the proposal *is* the L3 Qdrant entry with `pending_l4_approval=true`. Duplicating into Postgres creates two sources of truth.

## Consolidation handler pipeline

Location: `services/agent/selene_agent/autonomy/handlers/memory_review.py` (new).

The handler runs as a plain async function, not an `AutonomousTurn`. This is a deterministic pipeline with scoped LLM calls, not an agentic loop — it doesn't need tool gating or a fresh orchestrator. It calls the LLM directly via the existing `AsyncOpenAI` client that v1's engine already holds.

### Step 1 — Scan L2

Qdrant scroll with filter `tier='L2'`, no time window (decay must reach old entries). Pull `id`, `text`, `vector`, `importance`, `timestamp`, `tags`, `access_count`, `last_accessed_at`. Cap at `AUTONOMY_MEMORY_MAX_SCAN` (default 5000) as a runaway guard.

### Step 2 — Apply decay/boost

For every scanned L2 entry:

```
age_days = (now - timestamp).days
decay    = exp(-age_days / MEMORY_HALF_LIFE_DAYS)     # default 60
boost    = MEMORY_ACCESS_COEF * log(1 + access_count) # default 0.5
importance_effective = clamp(importance * decay + boost, 0, 10)
```

Batch-update via Qdrant `set_payload`. This is the "neurons you use get stronger, ones you don't atrophy" mechanism.

### Step 3 — Cluster L2 → L3 (incremental)

Take only L2 entries created since the last successful `memory_review` run (use the agenda item's `last_fired_at`). Rationale: L3 is incremental; re-clustering the full corpus nightly is expensive and produces churn.

- If fewer than `MEMORY_HDBSCAN_MIN_CLUSTER_SIZE` (default 5) new entries, skip clustering entirely. Log and continue.
- Otherwise run HDBSCAN on the new entries' vectors (`min_cluster_size=5`, `min_samples=3`, metric `cosine`).
- For each resulting cluster (HDBSCAN noise points are left as plain L2):
  - Single LLM call with the cluster member texts and a structured prompt: *"These memories cluster together. Produce one consolidated summary capturing the stable pattern across them, and a 3-tag list. If there is no coherent pattern, respond with `null`. Output JSON: `{summary: str|null, tags: list[str], rationale: str}`."*
  - If `summary` is non-null: create a new Qdrant point with `tier='L3'`, `text=summary`, `source_ids=[cluster member ids]`, `importance=median(members.importance)`, `tags=<LLM-chosen>`, `timestamp=now`.
  - If `null`: skip (LLM judged the cluster incoherent) — log to run metrics.

LLM calls are hard-capped at `AUTONOMY_MEMORY_LLM_CALL_CAP` (default 20) per run.

### Step 4 — Propose L3 → L4

Scroll `tier='L3' AND pending_l4_approval=false AND importance_effective >= MEMORY_L4_MIN_IMPORTANCE` (default 4). For each:

- Check age ≥ `MEMORY_L4_MIN_AGE_DAYS` (default 14) from the L3 entry's own timestamp.
- Check access condition: `access_count >= MEMORY_L4_MIN_ACCESS_COUNT` (default 3) OR `tags` includes `core_fact`.
- Both pass → set `pending_l4_approval=true`, `proposed_at=now`, `proposal_rationale=<short LLM-authored one-liner>`.

### Step 5 — Prune stale L2

Scroll `tier='L2' AND importance_effective < MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD` (default 0.5). For each candidate, check:
- age > `MEMORY_L2_PRUNE_AGE_DAYS` (default 180), AND
- the entry's id is not in any L3's `source_ids`.

Source-protection query: one scroll with `tier='L3'` pulling just `source_ids`, flatten into a set, in-memory check per candidate. Thousands of ids max — fine without indexing gymnastics.

Delete matching points.

### Error handling

Any step that raises is caught, logged to the run's `error` field, but the pipeline continues to the next step. Partial runs are safe — the pipeline is idempotent and the next execution will finish what was missed.

### Concurrency

The handler holds no Qdrant-level lock. `create_memory` may fire during a run (user chatting while consolidation runs) — that's safe; new L2 entries get picked up next night.

## Retrieval changes

Two surgical changes to `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py`.

### 1. Tier-weighted ranking in `search_memories`

After Qdrant returns results, apply a scoring multiplier before sorting:

```python
TIER_WEIGHT = {"L2": 1.0, "L3": MEMORY_L3_RANK_BOOST, "L4": 1.0}  # L4 excluded anyway
adjusted_score = raw_score * TIER_WEIGHT[tier]
```

Sort by `adjusted_score`, truncate to `limit`. Response includes both `relevance_score` (raw) and `tier`. `source_ids` is already in the payload, so the LLM can follow a hit down to its originating L2 entries via a targeted follow-up `search_memories` call if it wants.

`MEMORY_L3_RANK_BOOST` default: `1.2`. Env-configurable.

**L4 exclusion** — add `must_not` filter `tier='L4'` to the search filter. One-line change.

### 2. Access tracking

On every `search_memories` call where results come back, fire-and-forget update each returned point's `access_count` (+1) and `last_accessed_at` (now):

```python
# inside _search_memories, after building `memories`, before return
asyncio.create_task(self._record_accesses([m["id"] for m in memories]))
```

`_record_accesses` issues a single batched `set_payload` with a filter matching the returned ids. One write per search, not per result.

**Failure semantics:** if the access update fails, log and swallow. Access counts are a best-effort signal; retrieval latency and correctness take precedence.

**Counting drift is acceptable** — concurrent `set_payload` updates can drop a tick. The consolidation job applies `log(1 + access_count)`, which dampens counting noise by design.

## L4 injection

### Builder — `services/agent/selene_agent/utils/l4_context.py` (new)

```python
async def build_l4_block() -> str:
    # Qdrant scroll: tier='L4' AND pending_l4_approval=false
    # Sort by importance desc, cap at MEMORY_L4_MAX_ENTRIES (default 20)
    # Render as plain-text block with a stable header
```

Rendered output:

```
<persistent_memories>
- Household: Matt lives alone; no other household members.
- Safety: Never actuate HA devices unprompted in autonomous turns.
- Preferences: Voice responses must avoid emojis and special characters.
- Home: Two-bedroom apartment with Sony Bravia (primary TV) and Plex LAN server.
</persistent_memories>
```

Empty L4 set → return empty string; block omitted entirely from prompts (no dangling tags).

### Caching

L4 mutates only via approve/reject/edit/delete API calls — rare. Cache the rendered block in memory with a version counter; invalidate on any `/api/memory/l4/*` mutation. Avoids a Qdrant round-trip per turn.

### Injection sites

1. **User-facing orchestrator** (`AgentOrchestrator.run()`) — prepend the L4 block above the existing `config.SYSTEM_PROMPT` body. Every user turn now sees L4 context.
2. **Autonomous turns** — v1 wired a no-op `<persistent_memories>` populator. v2 replaces it with `build_l4_block()`. Both anomaly sweeps and briefings get L4 context automatically.

### Token budget guardrails

- Hard cap: `MEMORY_L4_MAX_ENTRIES` (default 20).
- Soft warn: log at WARN if rendered block exceeds `MEMORY_L4_WARN_TOKENS` (default 1500). Sustained warns indicate tier mis-classification.
- Dashboard shows approximate L4 token count so you can see it before approving another entry.

### Ordering

Importance descending, then age descending. Signal over scan-ability.

## Dashboard + API

### REST — `services/agent/selene_agent/api/memory.py` (new router)

```
GET    /api/memory/l4                         list active L4 entries
POST   /api/memory/l4                         create L4 entry directly (editor)
PATCH  /api/memory/l4/{id}                    edit text/importance
DELETE /api/memory/l4/{id}                    remove from L4 (demotes to L3)

GET    /api/memory/l4/proposals               list pending_l4_approval=true
POST   /api/memory/l4/proposals/{id}/approve  set tier=L4, clear flag
POST   /api/memory/l4/proposals/{id}/reject   clear flag (stays L3)

GET    /api/memory/l3?limit=50&offset=0       paginated L3 browse
GET    /api/memory/l3/{id}/sources            resolve source_ids → L2 texts
DELETE /api/memory/l3/{id}                    delete consolidated entry (sources untouched)

GET    /api/memory/runs?limit=20              autonomy_runs WHERE kind='memory_review'
POST   /api/memory/runs/trigger               fire memory_review agenda item manually

GET    /api/memory/stats                      {l2_count, l3_count, l4_count,
                                                pending_proposals, l4_est_tokens,
                                                last_run_at}
```

Auth pattern matches existing `/api/*` routes (same-origin dashboard; no strong auth today).

### SvelteKit — `services/agent/frontend/src/routes/memory/+page.svelte` (new)

Four stacked sections on one page. Reuses existing Tailwind / component patterns from `/history` and `/metrics`. No new component library.

**1. Header stats bar** — L2/L3/L4 counts, pending proposal badge count, last run time, `Run now` button, estimated L4 prompt-token count (warn-color near limit).

**2. L4 section** (top — highest stakes)
- Active entries table: text, importance, age, `[edit]` `[delete]`.
- `+ Add L4 entry` inline editor (text, importance 1–5, optional tags).
- Pending proposals sub-list: approve/reject per row, proposal rationale, source count.

**3. L3 section**
- Paginated table: text, `importance_effective`, age, source count (clickable).
- Click row → modal showing source L2 entries with text + timestamp.
- Per-row `[delete]` with confirm (sources remain as L2).

**4. Run history section**
- Last 20 `memory_review` runs: timestamp, duration, status, summary, stats pills (`L3+7 | L4?1 | pruned 18`).
- Click row → full metrics JSON + error field if any.

### Manual trigger flow

`POST /api/memory/runs/trigger` calls `AutonomyEngine.trigger(memory_review_item_id)` from v1's existing manual-trigger plumbing. Returns the new run id; dashboard polls `/api/memory/runs` for completion (typical run 30s–2min, no WebSocket needed).

## Configuration

Add to `shared/configs/shared_config.py` and `.env.example`:

```
# Schedule
AUTONOMY_MEMORY_REVIEW_CRON=0 3 * * *       # nightly 3am local

# Scan + guardrails
AUTONOMY_MEMORY_MAX_SCAN=5000
AUTONOMY_MEMORY_LLM_CALL_CAP=20

# Importance dynamics
MEMORY_HALF_LIFE_DAYS=60
MEMORY_ACCESS_COEF=0.5

# Clustering
MEMORY_HDBSCAN_MIN_CLUSTER_SIZE=5
MEMORY_HDBSCAN_MIN_SAMPLES=3

# L3 → L4 proposal thresholds
MEMORY_L4_MIN_IMPORTANCE=4
MEMORY_L4_MIN_AGE_DAYS=14
MEMORY_L4_MIN_ACCESS_COUNT=3

# L2 pruning
MEMORY_L2_PRUNE_AGE_DAYS=180
MEMORY_L2_PRUNE_IMPORTANCE_THRESHOLD=0.5

# Retrieval + injection
MEMORY_L3_RANK_BOOST=1.2
MEMORY_L4_MAX_ENTRIES=20
MEMORY_L4_WARN_TOKENS=1500
```

## Engine wiring — modifications to v1 code

- `services/agent/selene_agent/autonomy/engine.py` — add `"memory_review": memory_review_handler.handle` to the dispatch table.
- `services/agent/selene_agent/autonomy/db.py::ensure_default_agenda` — add a third default row for `memory_review` with `autonomy_level='observe'` (mutates Qdrant state; does not send notifications).
- `services/agent/selene_agent/autonomy/tool_gating.py` — no changes (memory_review doesn't construct an `AutonomousTurn`).
- `services/agent/selene_agent/utils/config.py` — new env vars listed above.
- `services/agent/pyproject.toml` — add `hdbscan` (or switch to `scikit-learn>=1.3` if simpler; decide at plan phase).

## Critical files

### New
- `services/agent/selene_agent/autonomy/handlers/memory_review.py`
- `services/agent/selene_agent/utils/l4_context.py`
- `services/agent/selene_agent/api/memory.py`
- `services/agent/frontend/src/routes/memory/+page.svelte`
- Tests — unit + integration (see verification plan)

### Modified
- `services/agent/selene_agent/autonomy/engine.py` — dispatch table
- `services/agent/selene_agent/autonomy/db.py` — default agenda seed
- `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py` — ranking, L4 exclusion, access tracking, payload index init
- `services/agent/selene_agent/orchestrator.py` — prepend L4 block to system prompt (user-facing turns)
- `services/agent/selene_agent/autonomy/turn.py` — swap v1's no-op `<persistent_memories>` populator for `build_l4_block()`
- `services/agent/selene_agent/utils/config.py` — new env vars
- `shared/configs/shared_config.py` — passthrough
- `services/agent/pyproject.toml` — add `hdbscan` (or `scikit-learn`)
- `.env.example` — document new vars
- `docs/` — per `havencore-docs` skill, new doc under `docs/services/agent/autonomy/memory/`

### Reused (do not fork)
- `AutonomyEngine` dispatch/run plumbing (v1 `engine.py`).
- `autonomy_runs` persistence (v1 `db.py`).
- `AgentOrchestrator` system-prompt construction (prepend-only edit).
- `AsyncOpenAI` client held by the engine (for consolidation LLM calls).
- Existing Qdrant `_get_embedding` and `create_memory` path.

## Verification plan

### Unit
- `compute_importance_effective` — decay curve math at representative ages × access counts.
- HDBSCAN wrapper with a fixture of ~20 synthetic embeddings across 3 planted clusters → expect 3 clusters + some noise.
- Cluster summarizer with mocked LLM returning (a) valid JSON, (b) `summary=null`, (c) malformed JSON → graceful handling each case.
- Source-protection logic: prune candidate appearing in an L3's `source_ids` must survive.
- Tier-weighted ranking: constructed fake Qdrant result set with L2/L3 mixed → assert correct ordering after boost.
- L4 block builder: 0 entries → empty string; > max entries → truncated by importance; stable ordering.

### Integration (requires live Qdrant — run via `docker compose exec agent`)
- Seed Qdrant with ~50 synthetic L2 entries across 3 themes → trigger `memory_review` → assert L3 rows created with `source_ids` populated, `autonomy_runs` row stats match.
- Approve an L4 proposal via API → assert payload transition (`tier='L4'`, flag cleared); assert `build_l4_block()` now includes it; assert user-turn system prompt contains `<persistent_memories>`.
- Delete L4 entry → demotes to L3, cache invalidated, prompt no longer contains it.
- Call `search_memories` on a topic covered by both an L3 and its L2 sources → L3 ranks above L2 sources (boost effect).
- Run `memory_review` twice consecutively → second run creates no duplicate L3s (incremental-only clustering).

### Manual end-to-end
1. Run the system for a week with `AUTONOMY_MEMORY_REVIEW_CRON=*/30 * * * *` (test cadence) and normal conversation. Inspect dashboard nightly; confirm clustering produces meaningful L3s.
2. Force-age: temporarily set `MEMORY_HALF_LIFE_DAYS=1`. Next run should push `importance_effective` down on stale entries.
3. Approve a proposal → restart agent → new user turn's Loki log confirms L4 block prepended.

### Health check
`/health` autonomy block (from v1) gains `memory_stats: {l2, l3, l4, pending}` — cheap counts from a single scroll each.

## Open design notes (non-blocking)

- **HDBSCAN vs scikit-learn HDBSCAN.** Both work. Decide at plan phase based on existing transitive deps; no behavioral difference.
- **LLM call cap interaction.** If `AUTONOMY_MEMORY_LLM_CALL_CAP` is hit mid-run, remaining clusters are deferred to the next night. Log a warning; surface in run metrics as `llm_call_cap_hit=true`.
- **Timezones.** Cron expressions interpreted in `config.CURRENT_TIMEZONE` before conversion to UTC for storage — matches v1 convention.
- **Proposal staleness.** If a proposal sits un-acted-upon for N days, the consolidation job could revisit and either withdraw or re-affirm it. Deferred to v3; for v2, pending proposals stay pending until the user acts.
- **L3 → L3 re-consolidation.** If two L3 entries on the same topic drift together over time, re-clustering them into a new L3 makes sense in principle. Deferred to v3 — v2 handles L2 → L3 only, and L3 entries are treated as stable once created.
