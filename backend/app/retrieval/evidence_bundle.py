"""Anchor-preserving, relevance-first evidence bundle selection."""

from __future__ import annotations

import re

from backend.app.retrieval.policy import RetrievalPolicy


def _tokens(item: dict[str, object]) -> set[str]:
    node = item["node"]
    content = str(node.get("content") or "").lower()
    return set(re.findall(r"[a-z0-9]+", content)) | {str(x).lower() for x in node.get("entities", [])}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _is_duplicate(item: dict[str, object], selected: list[dict[str, object]], policy: RetrievalPolicy) -> bool:
    node = item["node"]
    modality = str(node["modality"])
    same_modality = [prior for prior in selected if prior["node"]["modality"] == modality]
    if modality == "ocr_text":
        frame = node.get("frame_path") or node.get("provenance", {}).get("source_evidence_id")
        local = [prior for prior in same_modality
                 if (prior["node"].get("frame_path") or prior["node"].get("provenance", {}).get("source_evidence_id")) == frame]
        if len(local) >= policy.max_ocr_per_frame:
            return True
    elif modality == "pdf_text":
        key = (node.get("source_id"), node.get("page_number"))
        local = [prior for prior in same_modality
                 if (prior["node"].get("source_id"), prior["node"].get("page_number")) == key]
        if len(local) >= policy.max_pdf_blocks_per_page:
            return True
    elif modality == "speech":
        start = float(node.get("start_sec") or 0.0)
        for prior in same_modality:
            prior_node = prior["node"]
            if prior_node.get("source_id") != node.get("source_id"):
                continue
            if abs(start - float(prior_node.get("start_sec") or 0.0)) <= policy.speech_duplicate_window_seconds:
                if _jaccard(_tokens(item), _tokens(prior)) >= policy.near_duplicate_jaccard:
                    return True
    return any(str(prior["evidence_id"]) == str(item["evidence_id"]) for prior in selected)


def select_bundle(ranked: list[dict[str, object]], maximum: int = 8,
                  policy: RetrievalPolicy | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
    policy = policy or RetrievalPolicy()
    selected: list[dict[str, object]] = []
    suppressed: list[str] = []
    anchors = sorted(
        (item for item in ranked if item.get("origin") == "direct"
         and float(item.get("seed_score", 0.0)) >= policy.direct_anchor_min_score),
        key=lambda item: (-float(item.get("seed_score", 0.0)), str(item["evidence_id"])),
    )[:policy.direct_anchor_count]
    for item in anchors[:maximum]:
        selected.append({**item, "selection_role": "direct_anchor"})

    anchor_ids = {str(item["evidence_id"]) for item in selected}
    anchor_modalities = {item["node"]["modality"] for item in selected}
    complements = [item for item in ranked
                   if item.get("origin") == "graph_expanded"
                   and float(item.get("edge_confidence", item.get("connection_score", 0.0))) >= policy.complementary_min_edge_confidence
                   and float(item.get("propagated_relevance", 0.0)) >= policy.complementary_min_propagated_relevance
                   and item["node"]["modality"] not in anchor_modalities]
    if complements and len(selected) < maximum and not _is_duplicate(complements[0], selected, policy):
        selected.append({**complements[0], "selection_role": "graph_complement"})

    for item in ranked:
        if len(selected) >= maximum:
            break
        if any(str(existing["evidence_id"]) == str(item["evidence_id"]) for existing in selected):
            continue
        if _is_duplicate(item, selected, policy):
            suppressed.append(str(item["evidence_id"]))
            continue
        selected.append({**item, "selection_role": "ranked_fill"})
    return selected, {"protected_anchor_ids": sorted(anchor_ids), "suppressed_duplicate_ids": suppressed}


def build_bundle(ranked: list[dict[str, object]], maximum: int = 8,
                 policy: RetrievalPolicy | None = None) -> list[dict[str, object]]:
    return select_bundle(ranked, maximum, policy)[0]
