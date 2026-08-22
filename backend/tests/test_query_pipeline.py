"""Complete deterministic pipeline and FastAPI endpoint tests."""

import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.api.query import QueryRequest, get_query_service
from backend.app.main import app


class QueryPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = get_query_service()

    def test_validation_complete_multimodal_graph_expanded_pipeline(self) -> None:
        with self.assertRaises(ValidationError):
            QueryRequest(question="   ")
        result = self.service.query("What architecture was discussed, who explained it, and where was it shown?", 8, 1)
        self.assertTrue(result["evidence_bundle"])
        self.assertEqual(result["generation_mode"], "deterministic")
        modalities = {item["node"]["modality"] for item in result["evidence_bundle"]}
        self.assertGreaterEqual(len(modalities), 2)
        self.assertTrue(result["expanded_evidence"])
        self.assertTrue(any(not item["direct"] for item in result["evidence_bundle"]))
        self.assertIn("Evidence indicates", result["answer"])

    def test_raw_ocr_unchanged_and_no_external_llm(self) -> None:
        noisy = next(node for node in self.service.nodes if node.content == "AI Rate limiter")
        self.assertEqual(noisy.content, "AI Rate limiter")
        self.assertTrue(any(node.content == "failure 529 HTTP code" for node in self.service.nodes))

    def test_api_endpoints(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/api/health").status_code, 200)
        query = client.post("/api/query", json={"question": "How is high throughput handled?", "top_k": 5, "graph_hops": 1})
        self.assertEqual(query.status_code, 200)
        self.assertTrue(query.json()["evidence_bundle"])
        listing = client.get("/api/evidence", params={"modality": "speech"})
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["count"], 168)
        detail = client.get("/api/evidence/ev_1347fcb63e98_frame_0009")
        self.assertEqual(detail.status_code, 200)
        self.assertTrue(detail.json()["connected_edges"])
