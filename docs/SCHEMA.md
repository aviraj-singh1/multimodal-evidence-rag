# Evidence Schema

`Source` records the original filename, media type, path, hash, creation time, and source metadata.

`EvidenceNode` preserves modality, raw content, source ID, timestamps, speaker, page, bounding box, frame path, entities, confidence, provenance, and vector references.

`EvidenceEdge` stores canonical endpoints, relationship type, confidence, signal breakdown, and a human-readable explanation. P0 relationships are deterministic temporal overlap and entity overlap.

SQLite tables are defined in `backend/app/db/schema.sql`; API payloads return canonical nodes without rewriting raw OCR or transcript content.
