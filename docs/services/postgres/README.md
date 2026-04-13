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
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb
);
```

`tool_calls` is a JSONB array of `{name, duration_ms, ok}` entries — one
per tool call in the turn.

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

Triggered automatically when:

- A new query is received after 3+ minutes of inactivity
- The existing conversation has multiple messages
- A session timeout occurs

### Metadata structure

```json
{
  "reset_reason": "timeout_3_minutes",
  "message_count": 5,
  "last_query_timestamp": "2024-01-15T10:30:00Z",
  "agent_name": "Selene",
  "trace_id": "abc123"
}
```

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
