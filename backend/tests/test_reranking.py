"""Reranking score and provenance tests."""

import unittest

from backend.app.api.query import load_nodes
from backend.app.retrieval.reranker import provenance_quality, rerank


class RerankingTests(unittest.TestCase):
    def test_scores_bounded_connected_evidence_and_provenance(self) -> None:
        by_id = {node.id: node for node in load_nodes()}
        seed_id = "ev_1347fcb63e98_speech_0088"
        expanded_id = "ev_1347fcb63e98_frame_0009"
        ranked = rerank([{"evidence_id": seed_id, "seed_score": 0.9, "reasons": []}],
                        [{"evidence_id": expanded_id, "origin_seed_id": seed_id,
                          "propagated_relevance": 0.9, "connection_score": 1.0, "reason": "test"}], by_id)
        self.assertTrue(all(0 <= item["final_score"] <= 1 for item in ranked))
        self.assertGreaterEqual(next(x for x in ranked if x["evidence_id"] == expanded_id)["final_score"], 0.9)
        for modality in ("speech", "video_frame", "pdf_text", "ocr_text"):
            self.assertEqual(provenance_quality(next(node for node in by_id.values() if node.modality == modality)), 1.0)
