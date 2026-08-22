"""Deterministic temporal and entity relationship generation."""

from __future__ import annotations

import hashlib
from collections import Counter

from backend.app.evidence.models import EvidenceEdge, EvidenceNode, SignalBreakdown


TEMPORAL_WEIGHT = 0.60
ENTITY_WEIGHT = 0.40
TEMPORAL_THRESHOLD = 0.50
ENTITY_THRESHOLD = 0.20


def temporal_score(interval_start: float, interval_end: float, point: float) -> float:
    if interval_start <= point <= interval_end:
        return 1.0
    distance = interval_start - point if point < interval_start else point - interval_end
    return max(0.0, 1.0 - distance / 15.0) if distance <= 15.0 else 0.0


def entity_score(left: set[str], right: set[str], excluded: set[str] | None = None) -> float:
    excluded = excluded or set()
    a, b = left - excluded, right - excluded
    if not a or not b or not a.intersection(b):
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def deterministic_edge_id(left_id: str, right_id: str) -> str:
    source, target = sorted((left_id, right_id))
    digest = hashlib.sha256(f"{source}|{target}".encode("utf-8")).hexdigest()[:20]
    return f"edge_{digest}"


def excluded_common_entities(nodes: list[EvidenceNode], threshold: float) -> set[str]:
    bearing = [node for node in nodes if node.entities]
    frequencies = Counter(entity for node in bearing for entity in set(node.entities))
    return {entity for entity, count in frequencies.items() if bearing and count / len(bearing) > threshold}


def build_relationships(nodes: list[EvidenceNode], common_threshold: float) -> tuple[list[EvidenceEdge], set[str]]:
    excluded = excluded_common_entities(nodes, common_threshold)
    candidates: dict[tuple[str, str], tuple[EvidenceNode, EvidenceNode, float | None, float | None]] = {}
    speech = [node for node in nodes if node.modality == "speech"]
    timed_visual = [node for node in nodes if node.modality == "video_frame" or (
        node.modality == "ocr_text" and node.provenance.get("locator_type") == "frame_ocr_region"
    )]
    for utterance in speech:
        assert utterance.start_sec is not None and utterance.end_sec is not None
        for visual in timed_visual:
            assert visual.start_sec is not None
            score = temporal_score(utterance.start_sec, utterance.end_sec, visual.start_sec)
            if score >= TEMPORAL_THRESHOLD:
                key = tuple(sorted((utterance.id, visual.id)))
                candidates[key] = (utterance, visual, score, None)
    entity_nodes = [node for node in nodes if node.entities]
    for index, left in enumerate(entity_nodes):
        for right in entity_nodes[index + 1:]:
            if not _useful_entity_pair(left, right):
                continue
            score = entity_score(set(left.entities), set(right.entities), excluded)
            if score < ENTITY_THRESHOLD:
                continue
            key = tuple(sorted((left.id, right.id)))
            if key in candidates:
                old_left, old_right, temporal, _ = candidates[key]
                candidates[key] = (old_left, old_right, temporal, score)
            else:
                candidates[key] = (left, right, None, score)
    edges = [_make_edge(*candidate, excluded=excluded) for _, candidate in sorted(candidates.items())]
    if len({(edge.source_node_id, edge.target_node_id) for edge in edges}) != len(edges):
        raise ValueError("duplicate node pair generated")
    return edges, excluded


def _useful_entity_pair(left: EvidenceNode, right: EvidenceNode) -> bool:
    if left.id == right.id:
        return False
    if left.modality == right.modality and left.source_id == right.source_id:
        return False
    if left.modality == right.modality and left.modality in {"ocr_text", "pdf_text", "speech"}:
        return False
    if {left.modality, right.modality} == {"video_frame", "ocr_text"} and left.source_id == right.source_id:
        return False
    return left.source_id != right.source_id or left.modality != right.modality


def _make_edge(
    left: EvidenceNode,
    right: EvidenceNode,
    temporal: float | None,
    entity: float | None,
    excluded: set[str],
) -> EvidenceEdge:
    source, target = sorted((left, right), key=lambda node: node.id)
    if temporal is not None:
        relationship_type = "TEMPORAL_OVERLAP"
    elif entity is not None:
        relationship_type = "ENTITY_OVERLAP"
    else:
        raise ValueError("edge requires at least one signal")
    if temporal is not None and entity is not None:
        confidence = TEMPORAL_WEIGHT * temporal + ENTITY_WEIGHT * entity
    else:
        confidence = temporal if temporal is not None else entity
    shared = sorted(set(left.entities).intersection(right.entities) - excluded)
    explanation = _explanation(left, right, temporal, shared)
    return EvidenceEdge(
        id=deterministic_edge_id(source.id, target.id),
        source_node_id=source.id, target_node_id=target.id,
        relationship_type=relationship_type, confidence=round(float(confidence), 6),
        signal_breakdown=SignalBreakdown(
            temporal=round(temporal, 6) if temporal is not None else None,
            entity=round(entity, 6) if entity is not None else None,
            semantic=None,
        ),
        explanation=explanation,
    )


def _explanation(left: EvidenceNode, right: EvidenceNode, temporal: float | None, shared: list[str]) -> str:
    if temporal is not None:
        speech = left if left.modality == "speech" else right
        point = right if speech is left else left
        if shared:
            return (f"Evidence at {point.start_sec:.2f}s occurs during or near speech at "
                    f"{speech.start_sec:.2f}-{speech.end_sec:.2f}s and both reference: {', '.join(shared)}.")
        return (f"Evidence at {point.start_sec:.2f}s occurs during or near speech at "
                f"{speech.start_sec:.2f}-{speech.end_sec:.2f}s.")
    return f"Evidence shares entities: {', '.join(shared)}."
