"""Internal multimodal query service and POST /api/query route."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.entities import extract_entities, load_entity_config
from backend.app.evidence.models import EvidenceNode
from backend.app.evidence.repository import EvidenceRepository
from backend.app.generation.deterministic import format_answer
from backend.app.graph.loader import load_graph
from backend.app.retrieval.entity_search import entity_search
from backend.app.retrieval.evidence_bundle import build_bundle
from backend.app.retrieval.fusion import fuse_candidates, select_seeds
from backend.app.retrieval.graph_expansion import expand_graph
from backend.app.retrieval.reranker import rerank
from backend.app.retrieval.text_search import text_search
from backend.app.retrieval.visual_search import visual_search
from backend.app.vector.chroma import ChromaStore


ROOT = Path(__file__).resolve().parents[3]
def load_nodes(database_path: Path | None = None) -> list[EvidenceNode]:
    return EvidenceRepository(database_path or ROOT / "data" / "app.db").list_evidence_nodes()


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=12)
    graph_hops: int = Field(default=1, ge=0, le=2)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class QueryService:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.nodes = load_nodes(root / "data" / "app.db")
        self.by_id = {node.id: node for node in self.nodes}
        self.store = EvidenceStore(root / "data" / "app.db")
        self.chroma = ChromaStore(root / "data" / "chroma")
        self.entity_config = load_entity_config(root / "config")
        self.graph = load_graph(self.nodes, self.store.load_edges())

    def query(self, question: str, top_k: int = 8, graph_hops: int = 1) -> dict[str, object]:
        request = QueryRequest(question=question, top_k=top_k, graph_hops=graph_hops)
        started = time.perf_counter()
        query_entities = extract_entities(request.question, self.entity_config)
        mark = time.perf_counter(); text_hits = text_search(self.chroma, request.question, 8); text_ms = (time.perf_counter()-mark)*1000
        mark = time.perf_counter(); visual_hits = visual_search(self.chroma, request.question, 5); visual_ms = (time.perf_counter()-mark)*1000
        entity_hits = entity_search(self.nodes, query_entities, 12)
        fused = fuse_candidates([text_hits, visual_hits, entity_hits])
        seeds = select_seeds(fused, self.by_id, request.top_k)
        mark = time.perf_counter(); expanded, connections = expand_graph(self.graph, seeds, request.graph_hops); graph_ms = (time.perf_counter()-mark)*1000
        ranked = rerank(seeds, expanded, self.by_id)
        bundle = build_bundle(ranked, 8)
        return {
            "question": request.question, "query_entities": query_entities,
            "channels": {"text": text_hits, "visual": visual_hits, "entity": entity_hits},
            "seed_evidence": [{**seed, "modality": self.by_id[str(seed["evidence_id"])].modality}
                              for seed in seeds],
            "expanded_evidence": [{**item, "modality": self.by_id[str(item["evidence_id"])].modality}
                                  for item in expanded],
            "connections": connections, "evidence_bundle": bundle,
            "answer": format_answer(bundle), "generation_mode": "deterministic",
            "latency_ms": round((time.perf_counter()-started)*1000, 3),
            "timings_ms": {"text": round(text_ms, 3), "visual": round(visual_ms, 3),
                           "graph": round(graph_ms, 3)},
        }


@lru_cache(maxsize=1)
def get_query_service() -> QueryService:
    return QueryService()


router = APIRouter()


@router.post("/api/query")
def query_endpoint(request: QueryRequest) -> dict[str, object]:
    return get_query_service().query(request.question, request.top_k, request.graph_hops)
