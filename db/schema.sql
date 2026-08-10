-- Enable pgvector (idempotent: safe to run repeatedly)
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document (e.g., a PDF manual)
CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,          -- auto-incrementing unique id
    title      TEXT NOT NULL,                  -- human-readable name
    source     TEXT NOT NULL,                  -- file path or URL it came from
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()  -- when we ingested it
);

-- One row per chunk carved out of a document
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL                 -- which document this belongs to
        REFERENCES documents(id) ON DELETE CASCADE,  -- delete doc => delete its chunks
    chunk_index INT NOT NULL,                   -- order of the chunk within the doc
    content     TEXT NOT NULL,                  -- the actual chunk text
    metadata    JSONB NOT NULL DEFAULT '{}',    -- flexible extras: page, section...
    embedding   vector(384),                    -- the 384-dim dense embedding
    UNIQUE (document_id, chunk_index)           -- no duplicate chunk positions
);