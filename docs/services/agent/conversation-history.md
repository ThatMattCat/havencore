# Conversation History Storage

## Overview

HavenCore stores conversation histories to PostgreSQL whenever a pooled session's messages would otherwise be lost — on idle timeout (summarize-and-continue), LRU eviction, or agent shutdown. Rows are keyed by `session_id`, so a single external session can produce multiple stored rows over its lifetime (one per reset).

## Features

- **Automatic Storage**: The agent flushes conversations to PostgreSQL before they are dropped or reset. The `metadata.reset_reason` field records which trigger fired.
- **Rich Metadata**: Each stored conversation includes metadata such as:
  - Reset reason — one of `idle_timeout_summarize` (idle sweep; see below), `context_size_summarize` (size sweep; see below), `lru_eviction` (pool at capacity), `lru_eviction_size` (oversized eviction routed through summarize-and-reset), or `shutdown_flush` (agent stopping)
  - Message count
  - Last query timestamp
  - Agent name
  - `idle_timeout_override` — the per-session window in seconds, if the client set one; `null` otherwise
  - `device_name` — human-readable label of the satellite/client driving the session (e.g. `"Kitchen Speaker"`); `null` if the client never sent one
  - `rolling_summary` — compact recap written by any summarize-and-reset path (idle sweep, size sweep, or oversized-eviction bypass; `null` on plain LRU/shutdown flushes or when the summary LLM call fails)
  - `tail_exchanges_kept` — how many user/assistant pairs were carried forward
  - Trace ID for debugging
  - Storage timestamp

## Session identity

The `session_id` column is set by the client, not minted per turn:

- Dashboard tabs pass a persisted UUID via the `X-Session-Id` header (REST) or the first WS frame.
- Selene pucks pass a device-stable id (e.g. mac-hash), so a given physical device always writes to the same `session_id` across turns — even across idle resets and restarts.
- Callers that omit an id get a freshly-minted UUID returned in the response.

Because the id is externally stable, `get_conversation_history(session_id)` can return multiple rows for one logical device: each row is a self-contained conversation bounded by a reset event.

### Fetching a specific flush

`GET /api/conversations/{session_id}` returns every stored flush for that `session_id` newest-first. When the caller already has a specific row in mind (e.g. the dashboard's `/history` list, which shows one entry per flush), append `?id=<flush_id>` — using the `id` primary-key exposed on each list row — to scope the response to that single snapshot. The DB query still filters by `session_id` under the hood, so a mismatched `session_id`/`id` pair 404s. The dashboard uses this to keep each row's detail view aligned with its own header metadata (same `device_name`, same `message_count`, same `created_at`).

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

A session is flushed to PostgreSQL whenever its messages would be lost. Four triggers can fire:

1. **Idle sweep (summarize-and-continue)** — a 30-second background task in `SessionOrchestratorPool` looks for sessions whose `last_query_time` is older than their effective idle window (`orch.effective_timeout()` — the per-session override, or `CONVERSATION_TIMEOUT` default `90s`). For each expired session it acquires the per-session lock and runs `_summarize_and_reset(reason="idle_timeout_summarize")`:
   - persist the full prior history to Postgres (with `rolling_summary` and `idle_timeout_override` in metadata),
   - reinitialize `messages` to `[system prompt (+ L4 block), summary as a system message, last N user/assistant exchanges verbatim]` (N = `SESSION_SUMMARY_TAIL_EXCHANGES`, default 2),
   - preserve the same `session_id` so the thread continues.

   The summary is a one-shot LLM call capped by `SESSION_SUMMARY_MAX_TOKENS` and `SESSION_SUMMARY_LLM_TIMEOUT_SEC`. On timeout or error the reset falls back to keep-tail-only (no summary injected) and logs a structured `session_summarize_reset` line with `summary_ok=false`. Busy sessions are skipped (non-blocking lock try-acquire), not blocked.

   When a sweep-driven compaction fires, the pool publishes a `{"type":"summary_reset","reason":"idle_timeout_summarize","summary":...}` frame over its per-session pub/sub queue, which any connected `/ws/chat` client drains alongside its normal receive loop. Compactions detected at turn start (via `_check_session_timeout`) are surfaced the same way, emitted inline as a `SUMMARY_RESET` event at the top of `run()`. Both paths carry identical shape; the Chat UI renders an expandable "Conversation summarized" marker so the user sees when (and what) context was compressed instead of a silent history rewrite.

   A session is only eligible for summarize-reset once per user-active period: the sweep gates on whether a user turn has happened since the last reset. Without that gate, the post-reset `last_query_time` would still appear expired on the very next sweep tick and the session would be re-summarized every interval forever.
2. **Context-size sweep (summarize-and-continue)** — the same sweep + per-turn check that handles idle also gates on token budget. The threshold is computed per-orchestrator from the active provider's `max_model_len` (vLLM reports it via `/v1/models`; Anthropic uses a static prefix-keyed map; see [`configuration.md`](../../configuration.md#agent-runtime-tuning) for the env vars). When a session's serialized message bytes exceed `CONVERSATION_CONTEXT_LIMIT_FRACTION × max_model_len` — or the absolute `CONVERSATION_CONTEXT_LIMIT_TOKENS` override when that is > 0 — `_summarize_and_reset(reason="context_size_summarize")` runs the same persist-and-reinitialize path as the idle trigger. The corresponding `summary_reset` frame carries `reason: "context_size_summarize"`.

   Size and idle are independent axes. A dashboard session running with `idle_timeout=-1` ("never auto-summarize") will never qualify on idle but **will** size-summarize once it crosses the budget — unbounded message growth would otherwise blow the context window regardless of the idle policy. The size check is skipped on autonomy turns (single-shot orchestrators flip `context_size_check_enabled = False`) and silently no-ops when the active provider can't report a `max_model_len` and no absolute override is set.
3. **LRU eviction** — when the pool hits `max_size` (default 64) and a new session is admitted, the least-recently-used entry is flushed and removed from memory. Its `session_id` persists in the DB and can be cold-resumed later. If the evicted session is over the size budget at flush time, `_flush_one` routes it through `_summarize_and_reset(reason="lru_eviction_size")` instead of writing the raw bloated buffer — otherwise cold-resume would just replay the bloat on the next visit.
4. **Shutdown flush** — on agent shutdown (restart, stop, SIGTERM), the pool iterates every live session and flushes it before the process exits. Every non-empty session is guaranteed to be persisted across restarts. The same oversized-buffer bypass applies — an oversized session at shutdown is summarized rather than written raw.

In all four cases, sessions with only the system prompt (no real turns) are skipped — the threshold is `> 1` message.

### Per-session idle timeout override

Clients can widen or tighten the idle window for a single session without touching the global default:

- REST: `POST /api/chat` accepts an optional `X-Idle-Timeout: <seconds>` header.
- WebSocket: `/ws/chat` accepts an optional `idle_timeout` integer on any `{"type":"session", ...}` frame (first frame or mid-stream).

The value is clamped to `[CONVERSATION_TIMEOUT_MIN, CONVERSATION_TIMEOUT_MAX]` (defaults 10 and 3600). Bad values are logged and ignored. The override is stored on the live orchestrator and persisted into `metadata.idle_timeout_override` so that cold-resume via `POST /api/conversations/{session_id}/resume` rehydrates the same window.

The sentinel value `-1` means "never auto-summarize on idle" — it bypasses the clamp and causes both the pool's idle sweep and `AgentOrchestrator._check_session_timeout()` (the turn-start check) to skip this session. The dashboard sends `-1` on every WS open so an interactive tab lives until the user hits "New Chat" or the pool LRU-evicts it; satellites/pucks omit the field and get the global default. Note that `-1` only opts out of the **idle** axis — the context-size sweep still applies, so an unattended dashboard tab that grows past the token budget will still be summarized (with `reset_reason = "context_size_summarize"`). A `-1` session that stays under the size budget for its whole lifetime flushes via LRU eviction or shutdown without a `rolling_summary`.

### Device attribution

Clients can label which satellite/tab/puck is driving the session so the dashboard can render "Kitchen Speaker asked X" instead of an opaque session id:

- REST: `POST /api/chat` accepts an optional `X-Device-Name: <label>` header.
- WebSocket: `/ws/chat` accepts an optional `device_name` string on any `{"type":"session", ...}` frame (first frame or mid-stream).

Validation: trimmed, ASCII control characters stripped, capped at 64 characters (truncate-and-warn). Unicode and emoji are allowed — this is a UI label, never spoken. Empty/whitespace-only values are no-ops, so a frame omitting the field never clobbers a previously set name. Non-string values are logged and ignored. Renames are last-write-wins; each flushed history row preserves whatever the value was at flush time, so renaming a puck mid-conversation surfaces correctly in the timeline.

The label is persisted to `metadata.device_name` on every flush and denormalized onto each `turn_metrics` row (so the metrics view can group by device without joining). Cold-resume via `POST /api/conversations/{session_id}/resume` rehydrates the most recent value. `/v1/chat/completions` is stateless and unattributed.

The dashboard tab is itself a client and self-identifies the same way. The user sets a name from **System → "This Browser"**; it's persisted in `localStorage` under `haven.device_name` (per-browser, durable across tabs and reloads) and sent on the WS open frame, with mid-stream updates pushed when the user edits the name with a chat session live. Naming is opt-in — leave it blank and dashboard-driven turns flush with `device_name = null`, the same as any unlabelled client. When set, the label surfaces in the chat-page session badge, as a chip on each row of `/history`, and in both the per-row "Device" column and the "Per-device activity" card on `/metrics`.

### Cold resume

A stored `session_id` can be reloaded into the live pool via `POST /api/conversations/{session_id}/resume`. The pool rehydrates the orchestrator from the latest stored row, re-prepends the L4 memory block via `prepare()` (not `initialize()`, which would clobber the restored messages), and the next turn continues the conversation.

The endpoint returns the post-hydrate orchestrator messages alongside the session id:

```json
{
  "session_id": "…",
  "resumed": true,
  "message_count": 23,
  "messages": [ … ]
}
```

The leading base system prompt is filtered out of `messages` before the response is sent (it's not user-facing); a `[Prior conversation summary]` system message — the rolling summary the model sees on subsequent turns — is preserved. The dashboard's "Resume" button on `/history` consumes this payload directly: it clears the chat transcript, populates the messages store with the filtered list (rolling summary rendered as a collapsible "Conversation summarized" card, tail user/assistant pairs as normal bubbles), persists the returned `session_id`, and navigates to `/chat`. The user lands on a transcript that mirrors what the model will see on the very next turn.

### History detail view

`/history`'s detail panel renders a stored flush's `messages`. When `metadata.rolling_summary` is set (any of the summarize-and-reset paths), the panel defaults to a **summary view** — a single rolling-summary card representing what the LLM saw on subsequent turns — and offers a "Show raw transcript" toggle to reveal the pre-reset message buffer for debugging. When `rolling_summary` is null (plain `lru_eviction` or `shutdown_flush`), no toggle appears and the panel renders the buffer directly as before.

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

-- Get conversations closed by any summarize-and-continue path
SELECT session_id, created_at,
       metadata->>'reset_reason'         AS reason,
       metadata->>'rolling_summary'      AS summary,
       metadata->>'idle_timeout_override' AS override
FROM conversation_histories
WHERE metadata->>'reset_reason' IN (
        'idle_timeout_summarize',
        'context_size_summarize',
        'lru_eviction_size'
      )
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