"""Exact group-recall, completeness, and response-visible provenance metrics."""

from __future__ import annotations

from backend.app.evaluation.models import EvaluationQuestion
from backend.app.evidence.models import EvidenceNode


def group_metrics(question: EvaluationQuestion, represented: list[set[str]]) -> tuple[dict[str, bool], float, bool]:
    hits = {group.name: any(set(group.acceptable_evidence_ids).intersection(ids) for ids in represented)
            for group in question.required_groups}
    recall = sum(hits.values()) / len(hits)
    return hits, recall, all(hits.values())


def expected_provenance_fields(node: EvidenceNode) -> list[str]:
    locator = node.provenance.get("locator_type")
    if node.modality == "speech": return ["source", "start_sec", "end_sec", "speaker"]
    if node.modality == "video_frame": return ["source", "timestamp", "frame_path"]
    if node.modality == "pdf_text": return ["source", "page_number", "bbox"]
    if node.modality == "image": return ["source", "frame_path"]
    if locator == "frame_ocr_region": return ["source", "timestamp", "bbox"]
    return ["source", "bbox"]


def graph_provenance(item: dict[str, object], node: EvidenceNode) -> float:
    data = item["node"]
    present = {"source": bool(data["provenance"].get("file")), "start_sec": data.get("start_sec") is not None,
               "end_sec": data.get("end_sec") is not None, "speaker": bool(data.get("speaker")),
               "timestamp": data.get("start_sec") is not None, "frame_path": bool(data.get("frame_path")),
               "page_number": data.get("page_number") is not None, "bbox": bool(data.get("bbox"))}
    expected=expected_provenance_fields(node); return sum(present[field] for field in expected)/len(expected)


def baseline_provenance(item: dict[str, object], node: EvidenceNode) -> float:
    present={field:False for field in expected_provenance_fields(node)}
    present["source"]=bool(item.get("source_filename"))
    expected=expected_provenance_fields(node); return sum(present[field] for field in expected)/len(expected)
