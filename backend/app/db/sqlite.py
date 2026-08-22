"""Small transactional SQLite store for sources and evidence nodes."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from backend.app.evidence.models import EvidenceNode, Source


class EvidenceStore:
    """Replace one source's evidence atomically using deterministic node IDs."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.schema_path = Path(__file__).with_name("schema.sql")

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            with connection:
                connection.executescript(self.schema_path.read_text(encoding="utf-8"))

    def replace_source(self, source: Source, nodes: list[EvidenceNode]) -> None:
        if any(node.source_id != source.id for node in nodes):
            raise ValueError("all evidence nodes must belong to the supplied source")
        self.initialize()
        built_at = datetime.now(timezone.utc).isoformat()
        with closing(self.connect()) as connection:
            with connection:
                connection.execute(
                """INSERT INTO sources
                   (id, filename, media_type, original_path, sha256, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     filename=excluded.filename, media_type=excluded.media_type,
                     original_path=excluded.original_path, sha256=excluded.sha256,
                     created_at=excluded.created_at, metadata_json=excluded.metadata_json""",
                (source.id, source.filename, source.media_type, source.original_path,
                 source.sha256, source.created_at, _json(source.metadata)),
            )
                connection.execute("DELETE FROM evidence_nodes WHERE source_id = ?", (source.id,))
                connection.executemany(
                """INSERT INTO evidence_nodes
                   (id, source_id, modality, content, start_sec, end_sec, speaker, page_number,
                    bbox_json, frame_path, entities_json, confidence, provenance_json,
                    text_vector_id, visual_vector_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (node.id, node.source_id, node.modality, node.content, node.start_sec,
                     node.end_sec, node.speaker, node.page_number,
                     _json(node.bbox) if node.bbox is not None else None, node.frame_path,
                     _json(node.entities), node.confidence, _json(node.provenance),
                     node.text_vector_id, node.visual_vector_id, built_at)
                    for node in nodes
                ],
                )

    def source_count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])

    def evidence_count(self, source_id: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM evidence_nodes"
        parameters: tuple[str, ...] = ()
        if source_id is not None:
            query += " WHERE source_id = ?"
            parameters = (source_id,)
        with closing(self.connect()) as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def evidence_ids(self, source_id: str) -> list[str]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id FROM evidence_nodes WHERE source_id = ? ORDER BY id", (source_id,)
            ).fetchall()
        return [row[0] for row in rows]

    def modality_counts(self) -> Counter[str]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT modality, COUNT(*) FROM evidence_nodes GROUP BY modality"
            ).fetchall()
        return Counter({modality: count for modality, count in rows})

    def foreign_key_violations(self) -> list[tuple[object, ...]]:
        with closing(self.connect()) as connection:
            return connection.execute("PRAGMA foreign_key_check").fetchall()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
