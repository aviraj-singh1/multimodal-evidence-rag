"""Evidence inspection API with provenance and graph neighbors."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.query import get_query_service


router = APIRouter()


@router.get("/api/evidence")
def list_evidence(
    source_id: str | None = None,
    modality: str | None = None,
    entity: str | None = None,
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, object]:
    nodes = get_query_service().nodes
    filtered = [node for node in nodes if
                (source_id is None or node.source_id == source_id) and
                (modality is None or node.modality == modality) and
                (entity is None or entity in node.entities) and
                (min_confidence is None or (node.confidence is not None and node.confidence >= min_confidence))]
    return {"count": len(filtered), "nodes": [node.model_dump(mode="json") for node in filtered]}


@router.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str) -> dict[str, object]:
    service = get_query_service()
    node = service.by_id.get(evidence_id)
    if node is None:
        raise HTTPException(status_code=404, detail="evidence not found")
    edges = []
    neighbors = []
    for neighbor_id in service.graph.neighbors(evidence_id):
        attributes = service.graph.edges[evidence_id, neighbor_id]
        edges.append({"neighbor_id": neighbor_id, **attributes})
        neighbors.append(service.by_id[neighbor_id].model_dump(mode="json"))
    return {"node": node.model_dump(mode="json"), "connected_edges": edges, "neighbors": neighbors}
