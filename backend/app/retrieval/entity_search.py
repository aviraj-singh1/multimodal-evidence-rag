"""Deterministic query-entity retrieval over canonical node metadata."""

from __future__ import annotations

from backend.app.evidence.models import EvidenceNode


def entity_search(nodes: list[EvidenceNode], query_entities: list[str], top_k: int = 12) -> list[dict[str, object]]:
    query = set(query_entities)
    if not query:
        return []
    candidates = []
    for node in nodes:
        shared = sorted(query.intersection(node.entities))
        if shared:
            score = len(shared) / len(query)
            candidates.append({"evidence_id": node.id, "channel": "entity",
                               "raw_distance_or_score": score,
                               "normalized_score": round(score, 6), "matched_entities": shared})
    return sorted(candidates, key=lambda item: (-float(item["normalized_score"]), str(item["evidence_id"])))[:top_k]
