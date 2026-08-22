"""A new SQLite source enters relationships, indexes, and query loading without source lists."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.api.query import ROOT, load_nodes
from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.entities import enrich_entities, load_entity_config
from backend.app.evidence.models import EvidenceNode, Source
from backend.app.evidence.relationships import build_relationships
from backend.app.evidence.repository import EvidenceRepository
from backend.app.vector.chroma import eligible_text_nodes, eligible_visual_nodes


class CorpusGeneralizationTests(unittest.TestCase):
    def test_synthetic_fifth_source_flows_without_source_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "app.db"
            source_id = "src_synthetic_new"
            source = Source(
                id=source_id, filename="synthetic.mp4", media_type="video",
                original_path="synthetic.mp4", sha256="a" * 64,
                created_at="2026-01-01T00:00:00+00:00", metadata={},
            )
            shared = dict(
                source_id=source_id, speaker=None, page_number=None, bbox=None,
                entities=[], confidence=None, text_vector_id=None, visual_vector_id=None,
            )
            nodes = [
                EvidenceNode(
                    id="ev_synthetic_new_speech_0001", modality="speech",
                    content="The rate limiter handles requests.", start_sec=0.0, end_sec=10.0,
                    frame_path=None, speaker="SPEAKER_00",
                    provenance={"file": "synthetic.mp4", "source_id": source_id,
                                "locator_type": "time_range", "start_sec": 0.0, "end_sec": 10.0},
                    **{key: value for key, value in shared.items() if key != "speaker"},
                ),
                EvidenceNode(
                    id="ev_synthetic_new_frame_0001", modality="video_frame",
                    content=None, start_sec=5.0, end_sec=5.0, frame_path="synthetic-frame.jpg",
                    provenance={"file": "synthetic.mp4", "source_id": source_id,
                                "locator_type": "video_frame", "frame_id": "frame_0001",
                                "timestamp_sec": 5.0}, **shared,
                ),
                EvidenceNode(
                    id="ev_synthetic_new_ocr_frame_0001_001", modality="ocr_text",
                    content="Rate limiter", start_sec=5.0, end_sec=5.0,
                    frame_path="synthetic-frame.jpg", bbox=[0.0, 0.0, 10.0, 10.0],
                    provenance={"file": "synthetic.mp4", "source_id": source_id,
                                "locator_type": "frame_ocr_region", "frame_id": "frame_0001",
                                "timestamp_sec": 5.0, "ocr_region_id": "ocr_001"},
                    **{key: value for key, value in shared.items() if key not in {"bbox", "frame_path"}},
                ),
            ]
            EvidenceStore(database).replace_source(source, nodes)
            repository = EvidenceRepository(database)
            self.assertIn(source_id, {item.id for item in repository.list_sources()})
            discovered = repository.list_evidence_nodes()
            self.assertEqual({node.id for node in discovered}, {node.id for node in nodes})
            self.assertEqual({node.id for node in load_nodes(database)}, {node.id for node in nodes})
            config = load_entity_config(ROOT / "config")
            enriched = enrich_entities(discovered, config)
            edges, _ = build_relationships(enriched, config.common_entity_threshold)
            synthetic_node_ids = {node.id for node in enriched}
            self.assertTrue(any(edge.source_node_id in synthetic_node_ids and
                                edge.target_node_id in synthetic_node_ids for edge in edges))
            self.assertEqual(len(eligible_text_nodes(enriched)), 2)
            self.assertEqual(len(eligible_visual_nodes(enriched)), 1)
