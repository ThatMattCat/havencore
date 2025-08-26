-- Initialize HavenCore database schema
-- This script is automatically run when the PostgreSQL container starts

-- Create conversation_histories table
CREATE TABLE IF NOT EXISTS conversation_histories (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255),
    conversation_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create index on session_id for faster queries
CREATE INDEX IF NOT EXISTS idx_conversation_histories_session_id ON conversation_histories(session_id);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_conversation_histories_created_at ON conversation_histories(created_at);

-- Create index on metadata for flexible queries
CREATE INDEX IF NOT EXISTS idx_conversation_histories_metadata ON conversation_histories USING GIN(metadata);