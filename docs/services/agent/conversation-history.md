# Conversation History Storage

## Overview

HavenCore stores conversation histories to PostgreSQL whenever a pooled session's messages would otherwise be lost — on idle timeout, LRU eviction, or agent shutdown. Rows are keyed by `session_id`, so a single external session can produce multiple stored rows over its lifetime (one per reset).

## Features

- **Automatic Storage**: The agent flushes conversations to PostgreSQL before they are dropped or reset. The `metadata.reset_reason` field records which trigger fired.
- **Rich Metadata**: Each stored conversation includes metadata such as:
  - Reset reason — one of `timeout_<seconds>_seconds` (idle sweep), `lru_eviction` (pool at capacity), or `shutdown_flush` (agent stopping)
  - Message count
  - Last query timestamp
  - Agent name
  - Trace ID for debugging
  - Storage timestamp

## Session identity

The `session_id` column is set by the client, not minted per turn:

- Dashboard tabs pass a persisted UUID via the `X-Session-Id` header (REST) or the first WS frame.
- Selene pucks pass a device-stable id (e.g. mac-hash), so a given physical device always writes to the same `session_id` across turns — even across idle resets and restarts.
- Callers that omit an id get a freshly-minted UUID returned in the response.

Because the id is externally stable, `get_conversation_history(session_id)` can return multiple rows for one logical device: each row is a self-contained conversation bounded by a reset event.

## Database Schema

The conversation histories are stored in the `conversation_histories` table with the following structure:

```sql
CREATE TABLE conversation_histories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    conversation_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);
```

## Configuration

The PostgreSQL connection is configured via environment variables:

- `POSTGRES_HOST` - Database host (default: "postgres")
- `POSTGRES_PORT` - Database port (default: 5432)
- `POSTGRES_DB` - Database name (default: "havencore")
- `POSTGRES_USER` - Database user (default: "havencore")
- `POSTGRES_PASSWORD` - Database password (default: "havencore_password")

## Storage Triggers

A session is flushed to PostgreSQL whenever its messages would be lost. Three triggers can fire:

1. **Idle sweep** — a 30-second background task in `SessionOrchestratorPool` looks for sessions whose `last_query_time` is older than `CONVERSATION_TIMEOUT` (default 180s) and calls the orchestrator's `_check_session_timeout()` under the per-session lock. That flushes the messages and re-initializes the session in place (same `session_id`, empty messages, L4 block reapplied). Busy sessions are skipped, not blocked.
2. **LRU eviction** — when the pool hits `max_size` (default 64) and a new session is admitted, the least-recently-used entry is flushed and removed from memory. Its `session_id` persists in the DB and can be cold-resumed later.
3. **Shutdown flush** — on agent shutdown (restart, stop, SIGTERM), the pool iterates every live session and flushes it before the process exits. Every non-empty session is guaranteed to be persisted across restarts.

In all three cases, sessions with only the system prompt (no real turns) are skipped — the threshold is `> 1` message.

### Cold resume

A stored `session_id` can be reloaded into the live pool via `POST /api/conversations/{session_id}/resume`. The dashboard's "Resume" button on `/history` calls this endpoint, sets the returned id in `sessionStorage`, and navigates back to `/chat`. The pool rehydrates the orchestrator from the latest stored row, re-prepends the L4 memory block via `prepare()` (not `initialize()`, which would clobber the restored messages), and the next turn continues the conversation.

## Implementation Details

- Uses asyncio and asyncpg for async database operations
- Connection pooling for efficient database connections
- Comprehensive error handling and logging
- Stores conversation data as JSONB for flexible querying
- Automatic retries on database connection failures

## Querying Stored Conversations

You can query stored conversations using standard PostgreSQL queries:

```sql
-- Get recent conversations
SELECT session_id, created_at, metadata->>'message_count' as message_count 
FROM conversation_histories 
ORDER BY created_at DESC 
LIMIT 10;

-- Get conversations by reset reason
SELECT * FROM conversation_histories 
WHERE metadata->>'reset_reason' = 'timeout_3_minutes';

-- Get conversation messages
SELECT jsonb_pretty(conversation_data) 
FROM conversation_histories 
WHERE session_id = 'your-session-id';
```

## Monitoring

The system logs important events:
- Database connection initialization
- Successful conversation storage
- Database connection errors
- Storage failures

All logs include trace IDs for debugging purposes.