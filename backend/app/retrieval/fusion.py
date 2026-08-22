"""Explainable max-score fusion and relevance-first seed selection."""

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
    del nodes_by_id  # retained in the public signature for compatibility
    return list(fused[:top_k])
