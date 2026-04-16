# qdrant

Vector database backing the semantic-memory tiers (L1–L4). The agent
stores embedded user statements, preferences, and autonomy
observations, then retrieves them by semantic similarity.

| | |
|---|---|
| **Ports** | `6333` (REST), `6334` (gRPC) |
| **Health** | `curl -f http://localhost:6333/readyz` (wired into compose) |
| **Image** | `qdrant/qdrant:latest` (rollback digest pinned in compose.yaml) |

## Collections

- `user_data` — canonical memory collection. Points carry `tier`
  (L1–L4), `importance`, `approval` status, `access_count`, and source
  metadata. Indexes for `tier`, `approval`, `importance` are created by
  the agent on first connect.

## Key env

- `QDRANT__LOG_LEVEL`, `QDRANT__SERVICE__GRPC_PORT` — service-side
- From the agent (`.env`): `QDRANT_HOST`, `QDRANT_PORT`, `EMBEDDING_DIM`

## Volumes

- `./volumes/qdrant_storage` — persistent storage (data + indexes)

## More

- Deep dive: [../../docs/services/qdrant/README.md](../../docs/services/qdrant/README.md)
- Memory tier architecture:
  [../../docs/services/agent/autonomy/memory/README.md](../../docs/services/agent/autonomy/memory/README.md)
- MCP tool surface (`create_memory`, `search_memories`):
  `services/agent/selene_agent/modules/mcp_qdrant_tools/`
