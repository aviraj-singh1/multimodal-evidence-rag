"""Configuration for the post-Milestone-7 retrieval selection policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalPolicy:
    direct_anchor_count: int = 2
    direct_anchor_min_score: float = 0.25
    hop_decay: float = 0.80
    complementary_min_edge_confidence: float = 0.70
    complementary_min_propagated_relevance: float = 0.15
    near_duplicate_jaccard: float = 0.65
    speech_duplicate_window_seconds: float = 20.0
    max_ocr_per_frame: int = 2
    max_pdf_blocks_per_page: int = 2
    direct_relevance_weight: float = 0.80
    expanded_relevance_weight: float = 0.65
    expanded_edge_weight: float = 0.10
    provenance_weight: float = 0.10
    direct_origin_bonus: float = 0.05


def load_retrieval_policy(config_dir: Path) -> RetrievalPolicy:
    path = config_dir / "retrieval_policy.json"
    if not path.exists():
        return RetrievalPolicy()
    values = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(values) - set(RetrievalPolicy.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown retrieval policy keys: {sorted(unknown)}")
    return RetrievalPolicy(**values)
