"""Temporal, entity, edge-model, and real-corpus relationship tests."""

from __future__ import annotations

import unittest

from backend.app.evidence.entities import enrich_entities, load_entity_config
from backend.app.evidence.relationships import (
    build_relationships,
    deterministic_edge_id,
    entity_score,
    temporal_score,
)
from test_entities import ROOT, load_nodes


class RelationshipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_entity_config(ROOT / "config")
        cls.nodes = enrich_entities(load_nodes(), cls.config)
        cls.edges, cls.excluded = build_relationships(cls.nodes, cls.config.common_entity_threshold)
        cls.by_id = {node.id: node for node in cls.nodes}

    def test_temporal_scoring_and_threshold(self) -> None:
        self.assertEqual(temporal_score(10.0, 20.0, 15.0), 1.0)
        self.assertAlmostEqual(temporal_score(10.0, 20.0, 25.0), 2 / 3)
        self.assertEqual(temporal_score(10.0, 20.0, 36.0), 0.0)
        self.assertEqual(temporal_score(10.0, 20.0, 27.5), 0.5)
        self.assertLess(temporal_score(10.0, 20.0, 27.6), 0.5)

    def test_entity_jaccard_and_empty_intersection(self) -> None:
        self.assertAlmostEqual(entity_score({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(entity_score({"a"}, {"b"}), 0.0)
        self.assertEqual(entity_score({"server"}, {"server"}, {"server"}), 0.0)

    def test_deterministic_unique_edges_and_valid_signals(self) -> None:
        self.assertEqual(deterministic_edge_id("b", "a"), deterministic_edge_id("a", "b"))
        pairs = [(edge.source_node_id, edge.target_node_id) for edge in self.edges]
        self.assertEqual(len(pairs), len(set(pairs)))
        for edge in self.edges:
            self.assertGreaterEqual(edge.confidence, 0.0)
            self.assertLessEqual(edge.confidence, 1.0)
            self.assertTrue(edge.explanation)
            self.assertIsNone(edge.signal_breakdown.semantic)

    def test_actual_known_speech_frame_relationship_exists(self) -> None:
        found = []
        for edge in self.edges:
            left, right = self.by_id[edge.source_node_id], self.by_id[edge.target_node_id]
            nodes = (left, right)
            if {node.modality for node in nodes} == {"speech", "video_frame"}:
                frame = next(node for node in nodes if node.modality == "video_frame")
                speech = next(node for node in nodes if node.modality == "speech")
                if frame.start_sec == 720.0 and speech.start_sec <= 720.0 <= speech.end_sec:
                    found.append(edge)
        self.assertTrue(found)
        self.assertEqual(found[0].relationship_type, "TEMPORAL_OVERLAP")

    def test_actual_cross_source_entity_relationship_exists(self) -> None:
        found = []
        for edge in self.edges:
            left, right = self.by_id[edge.source_node_id], self.by_id[edge.target_node_id]
            if left.source_id != right.source_id and "pdf_text" in {left.modality, right.modality}:
                if edge.relationship_type == "ENTITY_OVERLAP":
                    found.append(edge)
        self.assertTrue(found)
        self.assertTrue(any({self.by_id[e.source_node_id].modality, self.by_id[e.target_node_id].modality}
                            == {"ocr_text", "pdf_text"} for e in found))

    def test_no_semantic_edges_and_vector_references_are_deterministic(self) -> None:
        self.assertTrue(all(edge.relationship_type in {"TEMPORAL_OVERLAP", "ENTITY_OVERLAP"} for edge in self.edges))
        self.assertTrue(all(node.text_vector_id in {None, f"text:{node.id}"} for node in self.nodes))
        self.assertTrue(all(node.visual_vector_id in {None, f"visual:{node.id}"} for node in self.nodes))
