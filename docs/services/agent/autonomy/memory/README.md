# Memory consolidation

## Tiers
- **L1 Ephemeral** — in-turn context.
- **L2 Episodic** — raw Qdrant entries (`tier='L2'`).
- **L3 Consolidated** — summarized clusters of L2 entries with `source_ids` back-links (`tier='L3'`).
- **L4 Persistent context** — small high-importance set, approved by user, injected into every system prompt (`tier='L4'`).

## Nightly job
Runs at `AUTONOMY_MEMORY_REVIEW_CRON` (default `0 3 * * *`). Handler: `selene_agent.autonomy.handlers.memory_review`. Pipeline:

1. Scan L2 → compute `importance_effective` via `exp(-age/halflife) + coef*log(1+accesses)`.
2. Cluster new L2 (since last run) with HDBSCAN; LLM summarizes each cluster into one L3 entry that **absorbs its sources**: the L3 payload stores `source_texts: [{id, text, timestamp, importance}]` alongside `source_ids`, and the originating L2 points are hard-deleted once the L3 upsert verifies. If the verification read fails, the L2s are left in place for the next run to retry.
3. Flag eligible L3 as `pending_l4_approval=true` (age ≥ 14d, importance_effective ≥ 4, access_count ≥ 3 OR tag `core_fact`).
4. Prune stale low-importance L2 entries. No source-protection set is needed — any L2 that got clustered into an L3 is already gone.

## Retrieval
`search_memories` excludes L4 (already in prompt), applies `MEMORY_L3_RANK_BOOST` to L3 scores, and fires an async `access_count` update.

### Per-turn injection (pool-backed chats)
On every user turn routed through `SessionOrchestratorPool` (i.e. `/api/chat` + `/ws/chat`), the orchestrator embeds the incoming message, queries Qdrant for top-K `L2|L3` hits, and injects a synthetic `system` message `<retrieved_memories>…</retrieved_memories>` into the request sent to the LLM. The synthetic message is **not persisted** to `self.messages`, so it can't pollute cold-resume or compound across turns.

- Top-K depends on the current [agent phase](#agent-phases): 5 in learning, 3 in operating.
- Hits scoring below `MEMORY_RETRIEVAL_MIN_SCORE` (default 0.3) are dropped; if nothing clears the bar, no block is injected.
- `/v1/chat/completions` and the autonomy engine both set `retrieval_enabled=False` on their ephemeral orchestrators — external OpenAI callers manage their own context, and autonomy turns construct messages programmatically.
- Toggle the whole feature with `MEMORY_RETRIEVAL_ENABLED`.

## Agent phases
Two operational phases, persisted in the Postgres `agent_state` table (`key='agent_phase'`):

| Phase | Retrieval K | System-prompt addendum |
|-------|-------------|------------------------|
| `learning` (default on fresh install) | 5 | "Lean into memory creation. When the user shares a preference, routine, relationship, constraint, or goal, call `create_memory`. Use `search_memories` liberally; use `delete_memory` when the user asks to forget something." |
| `operating` | 3 | "Create memories when new durable facts emerge. Search when a response would benefit from past context. Use `delete_memory` when the user asks to forget something." |

Flip via `POST /api/agent/phase` (see [API reference](../../../../api-reference.md#agent-dashboard-apis)) or the phase selector at the top of the `/memory` dashboard. Changing phase triggers `SessionOrchestratorPool.rebuild_system_prompts()` which non-blockingly refreshes `messages[0]` on every idle session — mid-turn sessions pick up the change on their next turn.

## Dashboard
`/memory` — stats, L4 view/editor, proposal queue, L3 browser with source drill-down, run history, manual trigger, and **semantic search** across any combination of L2/L3/L4 tiers (text input + per-tier checkboxes; results show similarity score and tier; L3 hits link to the same source modal).

The main `/` Dashboard surfaces a Memory summary card: per-tier counts, an L4 token-budget meter (warns above ~1500 chars/4 tokens), pending-proposal call-to-action, and last consolidation timestamp.

## REST surface
All endpoints live under `/api/memory/*` — see [API Reference](../../../../api-reference.md#agent-dashboard-apis) for the full table. Notable shapes:
- `POST /api/memory/search` — body `{q, tiers: ["L2","L3","L4"], limit}` returns scored payloads; embeds the query via the same `embeddings` service the agent uses for retrieval.
- `DELETE /api/memory/l4/{id}` is a **demotion** (sets `tier='L3'`), not a hard delete — preserves the underlying memory.
- `GET /api/memory/l2?limit=&offset=` + `DELETE /api/memory/l2/{id}` — paginated L2 browser and hard-delete, mirrored on the dashboard as a collapsible table above the L3 card.
- `GET /api/memory/l3/{id}/sources` — returns the L3's `source_texts` from the payload when present (new L3s after absorption). Falls back to `retrieve(source_ids)` against Qdrant for pre-absorption L3s; the response shape (`{sources: [{text, importance, timestamp}]}`) is unchanged.
- `POST /api/memory/admin/purge` — body `{tier: "L2"|"L3"|"all", source: "<tag>"}` OR `{ids: [...]}`. Hygiene tool: deletes points tagged with a given `source` string (e.g. `"test_integration"`). Requires a `source` or explicit `ids` list; a bare tier filter is rejected to prevent accidental wipes.

## Tuning knobs (env)
See `.env.example` — `MEMORY_HALF_LIFE_DAYS`, `MEMORY_ACCESS_COEF`, `MEMORY_L3_RANK_BOOST`, `MEMORY_L4_MAX_ENTRIES`, proposal thresholds, prune thresholds, plus retrieval injection (`MEMORY_RETRIEVAL_ENABLED`, `MEMORY_RETRIEVAL_TOPK_LEARNING`, `MEMORY_RETRIEVAL_TOPK_OPERATING`, `MEMORY_RETRIEVAL_MIN_SCORE`) and the default phase (`AGENT_PHASE_DEFAULT`).

## Operations
- Trigger on demand: dashboard "Run consolidation now" or `POST /api/memory/runs/trigger`.
- Health: `GET /health` includes `memory_stats: {l2,l3,l4,pending}`.
- Promotion is **always user-gated** — the pipeline never writes `tier='L4'`.
