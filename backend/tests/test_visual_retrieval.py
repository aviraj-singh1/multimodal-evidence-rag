"""CLIP visual retrieval tests."""

import unittest

from backend.app.api.query import ROOT, load_nodes
from backend.app.retrieval.visual_search import visual_search
from backend.app.vector.chroma import ChromaStore


class VisualRetrievalTests(unittest.TestCase):
    def test_architecture_query_returns_known_visual_nodes(self) -> None:
        hits = visual_search(ChromaStore(ROOT / "data" / "chroma"),
                             "architecture diagram showing rate limiter components", 5)
        nodes = {node.id: node for node in load_nodes()}
        self.assertTrue(hits)
        self.assertTrue(all(nodes[str(hit["evidence_id"])].modality in {"video_frame", "image"} for hit in hits))
        architecture_ids = {"ev_1347fcb63e98_frame_0009", "ev_1347fcb63e98_frame_0010",
                            "ev_1347fcb63e98_frame_0011", "ev_1347fcb63e98_frame_0012",
                            "ev_9899772d9cc4_image", "ev_66bf6ebf0dbc_image"}
        self.assertTrue(any(hit["evidence_id"] in architecture_ids for hit in hits))
