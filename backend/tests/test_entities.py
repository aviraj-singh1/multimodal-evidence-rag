"""Deterministic entity normalization, extraction, and aggregation tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.app.evidence.entities import enrich_entities, extract_entities, load_entity_config, normalize_text
from backend.app.evidence.models import EvidenceNode


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
IDS = ("src_1347fcb63e98", "src_9899772d9cc4", "src_66bf6ebf0dbc", "src_00b5ee2f232b")


def load_nodes() -> list[EvidenceNode]:
    nodes: list[EvidenceNode] = []
    for source_id in IDS:
        payload = json.loads((CACHE / source_id / "evidence.json").read_text(encoding="utf-8"))
        nodes.extend(EvidenceNode(**node) for node in payload["nodes"])
    return nodes


class EntityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_entity_config(ROOT / "config")

    def test_normalization_aliases_duplicates_and_stopwords(self) -> None:
        self.assertEqual(normalize_text("  Rules—Cache!!! "), "rules cache")
        entities = extract_entities("API rate limiter, API rate limiter; rules-cache and system.", self.config)
        self.assertEqual(entities, ["rate limiter", "rules cache"])
        self.assertNotIn("system", entities)
        self.assertEqual(extract_entities("Postgres and PostgreSQL", self.config), ["postgresql"])

    def test_raw_content_immutable_and_repeated_enrichment_idempotent(self) -> None:
        nodes = load_nodes()
        raw = {node.id: node.content for node in nodes}
        first = enrich_entities(nodes, self.config)
        second = enrich_entities(first, self.config)
        self.assertEqual([node.entities for node in first], [node.entities for node in second])
        self.assertEqual(raw, {node.id: node.content for node in first})
        noisy = next(node for node in first if node.content == "AI Rate limiter")
        self.assertEqual(noisy.content, "AI Rate limiter")
        self.assertIn("rate limiter", noisy.entities)
        self.assertNotIn("api", noisy.entities)
        code = next(node for node in first if node.content and "529" in node.content)
        self.assertNotIn("http 429", code.entities)

    def test_frame_and_image_entities_aggregate_from_ocr(self) -> None:
        nodes = enrich_entities(load_nodes(), self.config)
        frame = next(node for node in nodes if node.modality == "video_frame" and node.start_sec == 840.0)
        self.assertTrue({"rate limiter", "rules cache", "rules engine"}.issubset(frame.entities))
        image = next(node for node in nodes if node.id == "ev_66bf6ebf0dbc_image")
        children = [node for node in nodes if node.source_id == image.source_id and node.modality == "ocr_text"]
        expected = sorted({entity for child in children for entity in child.entities})
        self.assertEqual(image.entities, expected)
        self.assertIsNone(frame.content)
        self.assertIsNone(image.content)
