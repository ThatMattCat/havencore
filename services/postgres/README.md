# postgres

Stores conversation history, per-turn metrics, and autonomy run logs.

| | |
|---|---|
| **Port** | `5432` |
| **Health** | `pg_isready -U havencore -d havencore` (wired into compose) |
| **Image** | `postgres:15-alpine` |

## Schema

Initialized from [`init.sql`](./init.sql) on first boot. Tables:

- `conversation_histories` — archived conversations flushed by the
  `SessionOrchestratorPool` on one of three triggers: idle timeout
  (default 180s; see `CONVERSATION_TIMEOUT`), LRU eviction when the pool
  hits its 64-session cap, or shutdown flush on agent restart/stop.
  `metadata.reset_reason` records which fired. Rows are keyed by an
  externally-stable `session_id` (dashboard tab UUID, puck mac-hash), so
  one logical device accumulates multiple rows over time — and any row
  can be cold-resumed into the live pool via
  `POST /api/conversations/{session_id}/resume`
- `turn_metrics` — one row per agent turn, fed by the orchestrator's
  `METRIC` event; powers the `/metrics` dashboard page
- autonomy audit tables (runs, decisions, confirmations)

## Key env

From `.env`:

- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` — used both by this
  container (via `env_file`) and by the agent for connection strings

## Volumes

- `postgres_data` (named volume) — database files
- `./init.sql` — mounted read-only into the init hook

## More

- Deep dive: [../../docs/services/postgres/README.md](../../docs/services/postgres/README.md)
- Conversation DB access layer:
  `services/agent/selene_agent/utils/conversation_db.py`
