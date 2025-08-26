# Conversation History Storage

## Overview

HavenCore now automatically stores conversation histories to a PostgreSQL database when conversations are reset due to inactivity timeouts.

## Features

- **Automatic Storage**: When a conversation is reset after 3 minutes of inactivity, the full conversation history is stored to PostgreSQL before the reset occurs.
- **Rich Metadata**: Each stored conversation includes metadata such as:
  - Reset reason (e.g., "timeout_3_minutes")
  - Message count
  - Last query timestamp
  - Agent name
  - Trace ID for debugging
  - Storage timestamp

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

## Storage Trigger

Conversation storage is automatically triggered when:
1. A new query is received
2. The last query was more than 180 seconds (3 minutes) ago
3. There are existing messages in the conversation (more than just the system prompt)

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