# Conversation History Storage

## Overview

HavenCore stores conversation histories to PostgreSQL whenever a pooled session's messages would otherwise be lost — on idle timeout (summarize-and-continue), LRU eviction, or agent shutdown. Rows are keyed by `session_id`, so a single external session can produce multiple stored rows over its lifetime (one per reset).

## Features

- **Automatic Storage**: The agent flushes conversations to PostgreSQL before they are dropped or reset. The `metadata.reset_reason` field records which trigger fired.
- **Rich Metadata**: Each stored conversation includes metadata such as:
  - Reset reason — one of `idle_timeout_summarize` (idle sweep; see below), `lru_eviction` (pool at capacity), or `shutdown_flush` (agent stopping)
  - Message count
  - Last query timestamp
  - Agent name
  - `idle_timeout_override` — the per-session window in seconds, if the client set one; `null` otherwise
  - `rolling_summary` — compact recap written by the summarize-on-timeout path (idle sweeps only; `null` on LRU/shutdown flushes or when the summary LLM call fails)
  - `tail_exchanges_kept` — how many user/assistant pairs were carried forward
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

1. **Idle sweep (summarize-and-continue)** — a 30-second background task in `SessionOrchestratorPool` looks for sessions whose `last_query_time` is older than their effective idle window (`orch.effective_timeout()` — the per-session override, or `CONVERSATION_TIMEOUT` default `90s`). For each expired session it acquires the per-session lock and runs `_summarize_and_reset(reason="idle_timeout_summarize")`:
   - persist the full prior history to Postgres (with `rolling_summary` and `idle_timeout_override` in metadata),
   - reinitialize `messages` to `[system prompt (+ L4 block), summary as a system message, last N user/assistant exchanges verbatim]` (N = `SESSION_SUMMARY_TAIL_EXCHANGES`, default 2),
   - preserve the same `session_id` so the thread continues.

   The summary is a one-shot LLM call capped by `SESSION_SUMMARY_MAX_TOKENS` and `SESSION_SUMMARY_LLM_TIMEOUT_SEC`. On timeout or error the reset falls back to keep-tail-only (no summary injected) and logs a structured `session_summarize_reset` line with `summary_ok=false`. Busy sessions are skipped (non-blocking lock try-acquire), not blocked.
2. **LRU eviction** — when the pool hits `max_size` (default 64) and a new session is admitted, the least-recently-used entry is flushed and removed from memory. Its `session_id` persists in the DB and can be cold-resumed later.
3. **Shutdown flush** — on agent shutdown (restart, stop, SIGTERM), the pool iterates every live session and flushes it before the process exits. Every non-empty session is guaranteed to be persisted across restarts.

In all three cases, sessions with only the system prompt (no real turns) are skipped — the threshold is `> 1` message.

### Per-session idle timeout override

Clients can widen or tighten the idle window for a single session without touching the global default:

- REST: `POST /api/chat` accepts an optional `X-Idle-Timeout: <seconds>` header.
- WebSocket: `/ws/chat` accepts an optional `idle_timeout` integer on any `{"type":"session", ...}` frame (first frame or mid-stream).

The value is clamped to `[CONVERSATION_TIMEOUT_MIN, CONVERSATION_TIMEOUT_MAX]` (defaults 10 and 3600). Bad values are logged and ignored. The override is stored on the live orchestrator and persisted into `metadata.idle_timeout_override` so that cold-resume via `POST /api/conversations/{session_id}/resume` rehydrates the same window.

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

-- Get conversations closed by the idle sweep (summarize-and-continue)
SELECT session_id, created_at,
       metadata->>'rolling_summary' AS summary,
       metadata->>'idle_timeout_override' AS override
FROM conversation_histories
WHERE metadata->>'reset_reason' = 'idle_timeout_summarize'
ORDER BY created_at DESC
LIMIT 20;

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