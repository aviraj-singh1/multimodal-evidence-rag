"""Max-score fusion behavior tests."""

import unittest

from backend.app.retrieval.fusion import fuse_candidates


class FusionTests(unittest.TestCase):
    def test_duplicates_merge_without_missing_channel_penalty(self) -> None:
        fused = fuse_candidates([
            [{"evidence_id": "ev_a", "channel": "text", "normalized_score": 0.8}],
            [{"evidence_id": "ev_a", "channel": "entity", "normalized_score": 0.6},
             {"evidence_id": "ev_b", "channel": "visual", "normalized_score": 0.9}],
        ])
        self.assertEqual(len(fused), 2)
        a = next(item for item in fused if item["evidence_id"] == "ev_a")
        self.assertEqual(a["seed_score"], 0.8)
        self.assertEqual(a["channel_scores"], {"text": 0.8, "entity": 0.6})
