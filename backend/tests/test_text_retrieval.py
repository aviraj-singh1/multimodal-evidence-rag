"""MiniLM text retrieval tests."""

import unittest

from backend.app.api.query import ROOT, load_nodes
from backend.app.retrieval.text_search import text_search
from backend.app.vector.chroma import ChromaStore


class TextRetrievalTests(unittest.TestCase):
    def test_known_query_scores_and_ids(self) -> None:
        hits = text_search(ChromaStore(ROOT / "data" / "chroma"), "high throughput cache", 8)
        valid = {node.id for node in load_nodes()}
        self.assertTrue(hits)
        self.assertTrue(all(0 <= float(hit["normalized_score"]) <= 1 for hit in hits))
        self.assertTrue(all(hit["evidence_id"] in valid for hit in hits))
        self.assertTrue(any("speech_0088" in str(hit["evidence_id"]) or "pdf_" in str(hit["evidence_id"]) for hit in hits))
