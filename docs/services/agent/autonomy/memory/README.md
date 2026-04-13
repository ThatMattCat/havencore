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
