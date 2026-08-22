"""Persisted graph traversal bounds and value-add tests."""

import unittest

from backend.app.api.query import get_query_service
from backend.app.retrieval.graph_expansion import expand_graph


class GraphExpansionTests(unittest.TestCase):
    def test_threshold_hops_cap_and_real_nonseed(self) -> None:
        service = get_query_service()
        seed = [{"evidence_id": "ev_1347fcb63e98_speech_0088", "seed_score": 0.9}]
        none, _ = expand_graph(service.graph, seed, 0)
        self.assertEqual(none, [])
        expanded, _ = expand_graph(service.graph, seed, 1, 0.55, 3)
        self.assertLessEqual(len(expanded), 3)
        self.assertTrue(expanded)
        self.assertNotIn(seed[0]["evidence_id"], {item["evidence_id"] for item in expanded})
        strict, _ = expand_graph(service.graph, seed, 1, 1.01, 25)
        self.assertEqual(strict, [])
