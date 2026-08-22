"""SQLite-backed canonical corpus discovery for evidence-ready sources and nodes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from backend.app.evidence.models import EvidenceNode, Source


class EvidenceRepository:
    """Read the complete canonical corpus without fixture-specific source lists."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.is_file():
            raise FileNotFoundError(f"canonical SQLite database does not exist: {self.database_path}")
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def list_sources(self) -> list[Source]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT id, filename, media_type, original_path, sha256, created_at, metadata_json
                   FROM sources ORDER BY id"""
            ).fetchall()
        return [Source(
            id=row[0], filename=row[1], media_type=row[2], original_path=row[3], sha256=row[4],
            created_at=row[5], metadata=json.loads(row[6]),
        ) for row in rows]

    def list_evidence_nodes(self) -> list[EvidenceNode]:
        return self._load_nodes()

    def list_evidence_nodes_for_source(self, source_id: str) -> list[EvidenceNode]:
        return self._load_nodes(source_id)

    def _load_nodes(self, source_id: str | None = None) -> list[EvidenceNode]:
        query = """SELECT id, source_id, modality, content, start_sec, end_sec, speaker,
                          page_number, bbox_json, frame_path, entities_json, confidence,
                          provenance_json, text_vector_id, visual_vector_id
                   FROM evidence_nodes"""
        parameters: tuple[str, ...] = ()
        if source_id is not None:
            query += " WHERE source_id = ?"
            parameters = (source_id,)
        # Preserve canonical insertion/build order when mirroring nodes back to evidence.json.
        query += " ORDER BY rowid"
        with closing(self._connect()) as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [EvidenceNode(
            id=row[0], source_id=row[1], modality=row[2], content=row[3], start_sec=row[4],
            end_sec=row[5], speaker=row[6], page_number=row[7],
            bbox=json.loads(row[8]) if row[8] is not None else None, frame_path=row[9],
            entities=json.loads(row[10]), confidence=row[11], provenance=json.loads(row[12]),
            text_vector_id=row[13], visual_vector_id=row[14],
        ) for row in rows]
