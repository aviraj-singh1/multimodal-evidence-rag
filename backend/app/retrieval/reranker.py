"""Transparent deterministic evidence reranking."""

from __future__ import annotations

from backend.app.evidence.models import EvidenceNode


def provenance_quality(node: EvidenceNode) -> float:
    provenance = node.provenance
    if not provenance.get("file"):
        return 0.0
    if node.modality == "speech":
        return 1.0 if node.start_sec is not None and node.end_sec is not None else 0.5
    if node.modality == "video_frame":
        return 1.0 if node.start_sec is not None and node.frame_path else 0.5
    if node.modality == "pdf_text":
        return 1.0 if node.page_number is not None and node.bbox else 0.5
    if node.modality == "ocr_text":
        return 1.0 if node.bbox and (node.start_sec is not None or node.frame_path) else 0.75
    if node.modality == "image":
        return 1.0 if node.frame_path else 0.5
    return 0.5


def rerank(seeds: list[dict[str, object]], expanded: list[dict[str, object]], nodes_by_id: dict[str, EvidenceNode]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for seed in seeds:
        node = nodes_by_id[str(seed["evidence_id"])]
        direct = float(seed["seed_score"])
        final = 0.55 * direct + 0.10 * provenance_quality(node) + 0.05
        results.append({**seed, "direct": True, "connection_score": 0.0,
                        "final_score": round(min(1.0, final), 6), "node": node.model_dump(mode="json")})
    for item in expanded:
        node = nodes_by_id[str(item["evidence_id"])]
        relevance = float(item["propagated_relevance"])
        connection = float(item["connection_score"])
        final = 0.55 * relevance + 0.30 * connection + 0.10 * provenance_quality(node) + 0.05
        results.append({**item, "direct": False, "final_score": round(min(1.0, final), 6),
                        "reasons": [item["reason"]], "node": node.model_dump(mode="json")})
    return sorted(results, key=lambda item: (-float(item["final_score"]), str(item["evidence_id"])))
