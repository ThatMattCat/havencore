# MCP Server: Qdrant (`mcp_qdrant_tools`)

Reference doc for the semantic-memory MCP server. Wraps the Qdrant
vector database plus the HuggingFace text-embeddings-inference service
to give the agent a persistent memory layer.

## Overview

| | |
|---|---|
| Module path | `services/agent/selene_agent/modules/mcp_qdrant_tools/` |
| Entry point | `python -m selene_agent.modules.mcp_qdrant_tools` (wraps `qdrant_mcp_server.py`) |
| Transport | MCP stdio |
| Server name | `qdrant-server` |
| Vector backend | Qdrant, default collection `user_data`, cosine distance |
| Embeddings backend | `embeddings` service (HuggingFace text-embeddings-inference) serving `BAAI/bge-large-en-v1.5` (1024-dim) |
| Tool count | 3 |

On startup the server opens a Qdrant client and ensures the target
collection exists (creates it with the configured dimension + cosine
distance if absent). Embeddings are requested synchronously via the TEI
`POST /embed` endpoint — the first element of the returned batch is used
as the vector.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `create_memory(text, importance?, tags?, expires_in_days?)` | Embed `text` and upsert a new point into the collection. `importance` is `1..5` (default `3`). `tags` is a list of strings. `expires_in_days` adds a future `expires` timestamp used by the search-time filter. Returns the generated UUID. |
| `search_memories(query, limit?, days_back?)` | Semantic search for the most similar memories. `limit` is `1..20` (default `5`). `days_back` restricts to entries created within the last N days. Expired entries (where `expires` has passed) are always excluded. Returns `id`, `text`, `timestamp`, `importance`, `tags`, and `relevance_score`. |
| `delete_memory(memory_id)` | Hard-delete a stored point by UUID. The expected flow is `search_memories` → read the `id` of the matching hit → `delete_memory(id)`. If the deleted point's tier was `L4`, the server also invalidates the L4 block cache so the next prompt rebuild drops it. Returns `{success, memory_id, tier_deleted}`. |

## Point payload schema

```json
{
  "text": "the stored content",
  "timestamp": "2026-04-12T12:34:56+00:00",
  "importance": 3,
  "tags": ["user", "preference"],
  "tier": "L2",
  "source": "mcp_server",
  "expires": "2026-07-12T12:34:56+00:00"
}
```

`expires` is only written when `expires_in_days` is passed. The search
filter uses `must_not` on `expires.lte=<now>` to exclude past entries —
points without an `expires` field are never excluded by that filter.

`tier` defaults to `L2` for MCP-created points. The nightly memory-review
job writes `L3` clusters and the dashboard writes `L4` promotions; those
tiers also carry extra fields:

```json
// L3 (consolidated)
{
  "tier": "L3",
  "source_ids": ["uuid1", "uuid2"],
  "source_texts": [
    {"id": "uuid1", "text": "...", "timestamp": "...", "importance": 3}
  ]
}
```

`source_texts` is populated during consolidation so that the originating
L2 points can be hard-deleted immediately after the L3 upsert verifies —
the L3 row archives the text itself. The dashboard's "view sources"
modal reads `source_texts` when present and falls back to a
`retrieve(source_ids)` against Qdrant for L3 rows written before the
absorption change.

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `QDRANT_HOST` | `qdrant` | Hostname of the Qdrant service (in-cluster default is the service name). |
| `QDRANT_PORT` | `6333` | HTTP API port. |
| `EMBEDDINGS_URL` | `http://embeddings:3000` | TEI endpoint. The server calls `<url>/embed`. |
| `EMBEDDING_DIM` | `1024` | Vector dimension — must match the embedding model. `bge-large-en-v1.5` is 1024. |
| `QDRANT_COLLECTION` | `user_data` | Target collection name. The server creates it on startup if missing. |

The agent spawns the server via `MCP_SERVERS` in `.env`:

```json
{
  "name": "qdrant",
  "command": "python",
  "args": ["-m", "selene_agent.modules.mcp_qdrant_tools"],
  "enabled": true
}
```

## Internals worth knowing

- **Collection is auto-created.** `_init_collection` tries `get_collection`
  and catches the not-found exception to create it. The agent does not
  need to seed Qdrant manually.
- **Embeddings are fetched synchronously.** `_get_embedding` uses
  `requests.post` — this blocks the asyncio loop briefly. Fine for a
  single tool call at a time, but don't expect concurrent `create_memory`
  calls to overlap cleanly.
- **Expired points are filtered at search time, not deleted.** The server
  never prunes expired points on its own. If the collection grows big
  enough to matter, schedule a Qdrant-side cleanup externally.
- **`days_back` filters on `timestamp`, not relevance.** Combined with the
  expiry filter this gives you "recent non-expired memories". Both
  filters compose via Qdrant's `must` / `must_not`.

## Usage patterns from the system prompt

The agent's system prompt (in `selene_agent/utils/config.py`) tells the
LLM to use `create_memory`, `search_memories`, and `delete_memory` for
anything about the user, preferences, or the house. The prompt is
augmented with a phase-specific addendum — see
[autonomy/memory → Agent phases](../autonomy/memory/README.md#agent-phases).
Typical flows:

1. **Remember.** User says something worth remembering ("I prefer the
   bedroom at 68 degrees at night") → agent calls `create_memory`.
2. **Recall.** On later turns the agent can call `search_memories`
   explicitly. In pool-backed sessions (`/api/chat`, `/ws/chat`) the
   orchestrator *also* injects a top-K retrieval block before the LLM
   call, so even without a tool call the model sees relevant memories —
   see [autonomy/memory → Per-turn injection](../autonomy/memory/README.md#per-turn-injection-pool-backed-chats).
3. **Forget.** User asks to correct or delete a stored fact → agent
   calls `search_memories` to find the matching `id`, then
   `delete_memory(id)`. The system prompt is explicit about this flow
   because models otherwise tend to "remember" the delete request
   instead of acting on it.

Keep this in mind when tuning prompts or debugging retrieval: hits come
back in `relevance_score` order and low scores are often still returned.
The agent is expected to judge relevance itself rather than trusting the
top-k blindly.

## Troubleshooting

### Tools error with a connection refused

`QDRANT_HOST` / `QDRANT_PORT` unreachable from the agent container, or
the Qdrant service hasn't finished starting. Check:

```bash
docker compose ps qdrant
docker compose exec agent curl -I http://qdrant:6333/healthz
```

### `create_memory` fails with `Failed to get embedding`

The embeddings service is down or serving a different model / dimension
than configured. Verify:

```bash
docker compose logs embeddings
curl -X POST http://localhost:3000/embed \
  -H "Content-Type: application/json" \
  -d '{"inputs":"test"}'
```

If the returned vector length doesn't match `EMBEDDING_DIM`, either fix
the env var or swap the embedding model to match.

### Search returns nothing useful

- The collection is empty. Confirm with the Qdrant REST API:
  ```bash
  curl http://localhost:6333/collections/user_data | jq .result.points_count
  ```
- The query phrasing is too different from the stored text. Bi-encoders
  like `bge-large-en-v1.5` are reasonably forgiving but still semantic —
  pronouns and vague wording hurt recall.

### Dimension mismatch on create

```
Vector dimension … does not match collection dimension …
```

The collection was created at a different `EMBEDDING_DIM`. Either change
`QDRANT_COLLECTION` to a fresh name or drop the existing collection in
Qdrant:

```bash
curl -X DELETE http://localhost:6333/collections/user_data
```

(Then restart the agent — the server will recreate it with the new
dimension.)

## Related files

- `services/agent/selene_agent/modules/mcp_qdrant_tools/qdrant_mcp_server.py`
  — implementation.
- `services/agent/selene_agent/modules/mcp_qdrant_tools/__main__.py`
  — entrypoint.
- `shared/configs/shared_config.py` — Qdrant / embeddings config surface.

## See also

- [Qdrant service](../../qdrant/README.md) — raw Qdrant REST API.
- [Embeddings service](../../embeddings/README.md) — the TEI container.
