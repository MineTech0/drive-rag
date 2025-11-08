-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    path TEXT,
    mime_type TEXT NOT NULL,
    revision TEXT,
    modified_time TIMESTAMPTZ,
    drive_link TEXT NOT NULL,
    content_sha256 TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_documents_file_id ON documents(file_id);
CREATE INDEX idx_documents_content_sha256 ON documents(content_sha256);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    page_or_heading TEXT,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks(document_id, chunk_index);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(768),
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast vector similarity search
CREATE INDEX idx_embeddings_vector ON embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search table
CREATE TABLE IF NOT EXISTS documents_fts (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    tsv tsvector NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create GIN index for full-text search
CREATE INDEX idx_documents_fts_tsv ON documents_fts USING gin(tsv);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event TEXT NOT NULL,
    entity_id UUID,
    payload JSONB,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_event ON audit_logs(event);
CREATE INDEX idx_audit_logs_ts ON audit_logs(ts);
CREATE INDEX idx_audit_logs_entity_id ON audit_logs(entity_id);

-- Ingest jobs table (for tracking background jobs)
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    root_folder_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed')),
    full_reindex BOOLEAN DEFAULT FALSE,
    processed INTEGER DEFAULT 0,
    indexed INTEGER DEFAULT 0,
    errors JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ingest_jobs_state ON ingest_jobs(state);
CREATE INDEX idx_ingest_jobs_created_at ON ingest_jobs(created_at);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ingest_jobs_updated_at BEFORE UPDATE ON ingest_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
