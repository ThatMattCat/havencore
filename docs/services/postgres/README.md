# PostgreSQL Database

Conversation history storage, per-turn metrics, and other agent-side persistent state.

## Purpose

- Conversation history storage
- User session persistence
- Per-turn metrics (the `turn_metrics` table)
- Configuration and analytics data

## Database schema

Both tables are created by `services/postgres/init.sql` when the container
first starts. The canonical source of truth is that SQL file.

### `conversation_histories`

```sql
CREATE TABLE conversation_histories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    conversation_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

See the agent's [Conversation history](../agent/conversation-history.md)
doc for how rows get populated.

### `turn_metrics`

Per-turn LLM / tool-call timings written by the orchestrator. The
dashboard's `/metrics` page and the `/api/metrics/*` endpoints read
directly from this table.

```sql
CREATE TABLE turn_metrics (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    llm_ms INTEGER NOT NULL,
    tool_ms_total INTEGER NOT NULL,
    total_ms INTEGER NOT NULL,
    iterations INTEGER NOT NULL,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    device_name TEXT,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0
);
```

`tool_calls` is a JSONB array of `{name, duration_ms, ok}` entries — one
per tool call in the turn. `device_name` denormalizes the satellite/client
label set via `X-Device-Name` (or the WS `device_name` session field) so
metrics views can group by device without joining `conversation_histories`;
`NULL` for turns where the client didn't send a label or for the stateless
`/v1/chat/completions` path.

`cache_read_tokens` and `cache_creation_tokens` hold the Anthropic
prompt-cache counters accumulated across every LLM call in the turn
(summed across tool iterations). They are `0` for turns served by vLLM
and for legacy rows that predate the columns — only the Anthropic
provider populates non-zero values. The Metrics page derives a
`cache_hit_rate = read / (read + create)` aggregate from these two
columns.

`device_name`, `cache_read_tokens`, and `cache_creation_tokens` are
added on existing deployments via idempotent `ALTER TABLE ... ADD
COLUMN IF NOT EXISTS` statements that run at agent startup
(`MetricsDB.ensure_schema`).

## Configuration

```bash
POSTGRES_HOST="postgres"
POSTGRES_PORT=5432
POSTGRES_DB="havencore"
POSTGRES_USER="havencore"
POSTGRES_PASSWORD="havencore_password"
```

## Data storage patterns

### Conversation storage

The `SessionOrchestratorPool` flushes a session to Postgres on one of
three triggers (see the agent's
[Conversation history](../agent/conversation-history.md) doc for detail):

- **Idle sweep (summarize-and-continue)** — a 30s background task persists sessions whose
  `last_query_time` is older than their effective idle window (per-session
  override if set, else `CONVERSATION_TIMEOUT`, default 90s), then runs a
  one-shot LLM summary and reinitializes `messages` to
  `[system, summary, last N exchanges]` with the same `session_id`.
  `reset_reason` = `idle_timeout_summarize`.
- **LRU eviction** — when the pool hits `max_size` (64) and a new
  session is admitted, the least-recently-used entry is flushed and
  dropped. `reset_reason` = `lru_eviction`.
- **Shutdown flush** — on agent restart/stop/SIGTERM every non-empty
  session is persisted. `reset_reason` = `shutdown_flush`.

Sessions with only the system prompt (`messages` length ≤ 1) are skipped.

### Metadata structure

```json
{
  "reset_reason": "idle_timeout_summarize",
  "message_count": 5,
  "last_query_time": 1705315800.123,
  "agent_name": "Selene",
  "idle_timeout_override": 45,
  "device_name": "Kitchen Speaker",
  "rolling_summary": "User asked about weather and turned on the bedroom lamp...",
  "tail_exchanges_kept": 2,
  "idle_seconds": 47,
  "timeout_seconds": 45
}
```

`idle_timeout_override` and `device_name` are written on every flush
regardless of trigger (`null` if the client never set them).
`rolling_summary`, `tail_exchanges_kept`, `idle_seconds`, and
`timeout_seconds` are written on `idle_timeout_summarize` rows only.
`rolling_summary` may be `null` if the summary LLM call timed out or
errored (the reset still happened, with the tail preserved).

## Database operations

```bash
# Connect to database
docker compose exec postgres psql -U havencore -d havencore

# Query recent conversations
SELECT session_id, created_at, metadata->>'message_count'
FROM conversation_histories
ORDER BY created_at DESC LIMIT 10;

# View conversation content
SELECT jsonb_pretty(conversation_data)
FROM conversation_histories
WHERE session_id = 'your-session-id';
```

## Backup and recovery

```bash
# Create backup
docker compose exec postgres pg_dump -U havencore havencore > backup.sql

# Restore backup
docker compose exec -T postgres psql -U havencore -d havencore < backup.sql

# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker compose exec postgres pg_dump -U havencore havencore | gzip > "backup_${TIMESTAMP}.sql.gz"
```

## Performance monitoring

```bash
# Check database size
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT pg_size_pretty(pg_database_size('havencore'));
"

# Monitor connections
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT count(*) as connections FROM pg_stat_activity;
"

# Check table sizes
docker compose exec postgres psql -U havencore -d havencore -c "
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables WHERE schemaname='public';
"
```
