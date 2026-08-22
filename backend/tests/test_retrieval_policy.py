"""Generic retrieval-policy behavior and evaluation-leakage tests."""

import json
import unittest
from pathlib import Path

import networkx as nx

from backend.app.retrieval.evidence_bundle import build_bundle, select_bundle
from backend.app.retrieval.graph_expansion import expand_graph
from backend.app.retrieval.policy import RetrievalPolicy


ROOT = Path(__file__).resolve().parents[2]


def candidate(evidence_id: str, modality: str = "speech", score: float = 0.8,
              origin: str = "direct", content: str | None = None, start: float = 0.0,
              source: str = "source", page: int | None = None, frame: str | None = None,
              **extra: object) -> dict[str, object]:
    item = {
        "evidence_id": evidence_id, "origin": origin, "seed_score": score,
        "final_score": score, "node": {"id": evidence_id, "modality": modality,
        "content": content or evidence_id, "entities": [], "start_sec": start,
        "source_id": source, "page_number": page, "frame_path": frame, "provenance": {}},
    }
    item.update(extra)
    return item


class RetrievalPolicyTests(unittest.TestCase):
    def test_strong_direct_candidate_survives(self) -> None:
        ranked = [candidate("graph", origin="graph_expanded", score=0.99), candidate("direct", score=0.95)]
        self.assertIn("direct", {x["evidence_id"] for x in build_bundle(ranked, 1)})

    def test_weak_direct_candidate_can_be_displaced(self) -> None:
        ranked = [candidate("graph", origin="graph_expanded", score=0.9), candidate("direct", score=0.2)]
        self.assertEqual(build_bundle(ranked, 1)[0]["evidence_id"], "graph")

    def test_expanded_node_retains_origin_metadata(self) -> None:
        graph = nx.Graph(); graph.add_edge("seed", "other", confidence=0.9,
            relationship_type="ENTITY_OVERLAP", signal_breakdown={}, explanation="test")
        expanded, _ = expand_graph(graph, [{"evidence_id": "seed", "seed_score": 0.8}], 1)
        item = expanded[0]
        self.assertEqual(item["origin"], "graph_expanded")
        self.assertEqual((item["origin_seed_id"], item["origin_seed_score"], item["hop_count"]), ("seed", 0.8, 1))
        self.assertEqual(item["edge_confidence"], 0.9)

    def test_expanded_relevance_decreases_with_weaker_edge(self) -> None:
        def score(confidence: float) -> float:
            graph = nx.Graph(); graph.add_edge("seed", "other", confidence=confidence,
                relationship_type="ENTITY_OVERLAP", signal_breakdown={}, explanation="test")
            return float(expand_graph(graph, [{"evidence_id": "seed", "seed_score": 0.9}], 1, 0.5)[0][0]["propagated_relevance"])
        self.assertGreater(score(0.9), score(0.6))

    def test_same_frame_ocr_is_bounded(self) -> None:
        ranked = [candidate(str(i), "ocr_text", 0.9 - i * .01, frame="frame.png") for i in range(4)]
        self.assertLessEqual(len(build_bundle(ranked, 4)), RetrievalPolicy().max_ocr_per_frame)

    def test_adjacent_duplicate_speech_does_not_dominate(self) -> None:
        text = "the same repeated local passage about request counters"
        ranked = [candidate(str(i), content=text, start=i * 3.0, score=0.9 - i * .01) for i in range(4)]
        self.assertEqual(len(build_bundle(ranked, 4)), 2)  # two protected anchors; remaining duplicates are suppressed

    def test_strong_graph_complement_can_enter(self) -> None:
        anchor = candidate("direct", "speech", 0.9)
        complement = candidate("visual", "image", 0.7, "graph_expanded",
            origin_seed_id="direct", propagated_relevance=0.6, edge_confidence=0.9)
        bundle = build_bundle([anchor, complement], 2)
        self.assertEqual(bundle[1]["selection_role"], "graph_complement")

    def test_simple_text_query_is_not_forced_visual(self) -> None:
        ranked = [candidate("text", "speech", 0.9), candidate("more-text", "pdf_text", 0.8),
                  candidate("weak-visual", "image", 0.1, "graph_expanded",
                            origin_seed_id="text", propagated_relevance=0.1, edge_confidence=0.2)]
        self.assertNotIn("image", {x["node"]["modality"] for x in build_bundle(ranked, 2)})

    def test_bundle_never_exceeds_requested_size(self) -> None:
        bundle, _ = select_bundle([candidate(str(i), score=0.9 - i * .01, start=i * 100) for i in range(10)], 5)
        self.assertLessEqual(len(bundle), 5)

    def test_production_retrieval_has_no_evaluation_leakage(self) -> None:
        suite = json.loads((ROOT / "config" / "evaluation_questions.json").read_text(encoding="utf-8"))
        forbidden = [q["question"] for q in suite["questions"]]
        forbidden += [eid for q in suite["questions"] for group in q["required_groups"]
                      for eid in group["acceptable_evidence_ids"]]
        production = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "backend" / "app").rglob("*.py"))
        self.assertFalse([value for value in forbidden if value in production])
