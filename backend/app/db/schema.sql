PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    original_path TEXT,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_nodes (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    content TEXT,
    start_sec REAL,
    end_sec REAL,
    speaker TEXT,
    page_number INTEGER,
    bbox_json TEXT,
    frame_path TEXT,
    entities_json TEXT NOT NULL,
    confidence REAL,
    provenance_json TEXT NOT NULL,
    text_vector_id TEXT,
    visual_vector_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE INDEX IF NOT EXISTS idx_evidence_source_id ON evidence_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_modality ON evidence_nodes(modality);
CREATE INDEX IF NOT EXISTS idx_evidence_speaker ON evidence_nodes(speaker);
CREATE INDEX IF NOT EXISTS idx_evidence_page_number ON evidence_nodes(page_number);
