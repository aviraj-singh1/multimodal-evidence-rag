"""Canonical query-entity retrieval tests."""

import unittest

from backend.app.api.query import load_nodes
from backend.app.retrieval.entity_search import entity_search


class EntityRetrievalTests(unittest.TestCase):
    def test_known_and_unknown_entities(self) -> None:
        nodes = load_nodes()
        hits = entity_search(nodes, ["rate limiter"], 12)
        by_id = {node.id: node for node in nodes}
        self.assertTrue(hits)
        self.assertTrue(all("rate limiter" in by_id[str(hit["evidence_id"])].entities for hit in hits))
        self.assertEqual(entity_search(nodes, [], 12), [])
        self.assertEqual(entity_search(nodes, ["unknown concept"], 12), [])
