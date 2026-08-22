"""Compact complementary evidence bundle selection."""

from __future__ import annotations


def build_bundle(ranked: list[dict[str, object]], maximum: int = 8) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    modalities: set[str] = set()
    for item in ranked:
        modality = str(item["node"]["modality"])
        if modality not in modalities:
            selected.append(item); modalities.add(modality)
        if len(selected) >= maximum:
            return selected
    for item in ranked:
        if item not in selected:
            selected.append(item)
        if len(selected) >= maximum:
            break
    return selected
