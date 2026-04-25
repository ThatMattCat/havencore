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

-- Per-turn agent metrics (LLM + tool-call timings)
CREATE TABLE IF NOT EXISTS turn_metrics (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    llm_ms INTEGER NOT NULL,
    tool_ms_total INTEGER NOT NULL,
    total_ms INTEGER NOT NULL,
    iterations INTEGER NOT NULL,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    device_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_turn_metrics_created_at ON turn_metrics (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_turn_metrics_session ON turn_metrics (session_id);

-- Face recognition: identities, enrolled images, and detection history.
-- The face-recognition service runs the same DDL as an idempotent migration
-- on startup, so existing deployments don't need a DB rebuild — both paths
-- must stay in sync.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS people (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    access_level TEXT NOT NULL DEFAULT 'unknown',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    qdrant_point_id UUID NOT NULL,
    is_primary BOOLEAN DEFAULT false,
    source TEXT NOT NULL,
    quality_score REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_face_images_person ON face_images(person_id);

CREATE TABLE IF NOT EXISTS face_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL,
    camera TEXT NOT NULL,
    captured_at TIMESTAMPTZ DEFAULT now(),
    person_id UUID REFERENCES people(id) ON DELETE SET NULL,
    confidence REAL,
    quality_score REAL,
    snapshot_path TEXT NOT NULL,
    review_state TEXT NOT NULL DEFAULT 'auto',
    embedding_contributed BOOLEAN DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_face_detections_captured ON face_detections(captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_face_detections_unknown
    ON face_detections(review_state) WHERE person_id IS NULL;