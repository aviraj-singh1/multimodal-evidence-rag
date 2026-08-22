"""Explainable max-score fusion with bounded modality-diverse seeds."""

from __future__ import annotations

from backend.app.evidence.models import EvidenceNode


def fuse_candidates(channels: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    fused: dict[str, dict[str, object]] = {}
    for candidates in channels:
        for candidate in candidates:
            evidence_id = str(candidate["evidence_id"])
            item = fused.setdefault(evidence_id, {"evidence_id": evidence_id, "channel_scores": {}, "reasons": []})
            channel = str(candidate["channel"])
            score = float(candidate["normalized_score"])
            item["channel_scores"][channel] = max(score, item["channel_scores"].get(channel, 0.0))
            reason = f"DIRECT {channel}: {score:.3f}"
            if candidate.get("matched_entities"):
                reason += f" ({', '.join(candidate['matched_entities'])})"
            item["reasons"].append(reason)
    for item in fused.values():
        item["seed_score"] = max(item["channel_scores"].values())
    return sorted(fused.values(), key=lambda item: (-float(item["seed_score"]), str(item["evidence_id"])))


def select_seeds(fused: list[dict[str, object]], nodes_by_id: dict[str, EvidenceNode], top_k: int = 8) -> list[dict[str, object]]:
    selected = list(fused[:top_k])
    selected_ids = {str(item["evidence_id"]) for item in selected}
    represented = {nodes_by_id[str(item["evidence_id"])].modality for item in selected}
    for family in ({"video_frame", "image"}, {"pdf_text"}, {"speech"}):
        if represented.intersection(family):
            continue
        alternative = next((item for item in fused if nodes_by_id[str(item["evidence_id"])].modality in family
                            and float(item["seed_score"]) >= 0.20), None)
        if alternative and str(alternative["evidence_id"]) not in selected_ids:
            if len(selected) >= top_k:
                selected[-1] = alternative
            else:
                selected.append(alternative)
            selected_ids = {str(item["evidence_id"]) for item in selected}
            represented = {nodes_by_id[str(item["evidence_id"])].modality for item in selected}
    return sorted(selected, key=lambda item: (-float(item["seed_score"]), str(item["evidence_id"])))
