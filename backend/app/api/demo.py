"""Cached demo-corpus status for the judge-facing frontend."""

from collections import Counter
import json
from fastapi import APIRouter
from backend.app.api.query import ROOT, get_query_service
from backend.app.evidence.repository import EvidenceRepository

router = APIRouter()

@router.get("/api/demo/load")
def load_demo() -> dict[str, object]:
    service = get_query_service()
    sources = EvidenceRepository(ROOT / "data" / "app.db").list_sources()
    per_source = Counter(node.source_id for node in service.nodes)
    modalities = Counter(node.modality for node in service.nodes)
    evaluation = json.loads((ROOT / "data" / "evaluation" / "results_v1_5.json").read_text(encoding="utf-8"))
    return {
        "status": "ready", "processing_mode": "cached", "active_retrieval_policy": "V1.5",
        "source_count": len(sources), "evidence_count": len(service.nodes),
        "edge_count": service.graph.number_of_edges(), "modality_counts": dict(sorted(modalities.items())),
        "evaluation_version": "V1.5", "evaluation_summary": evaluation["summary"],
        "sources": [{**source.model_dump(mode="json"), "evidence_count": per_source[source.id],
                     "processing_status": "ready"} for source in sources],
    }
