"""Deterministic, configurable entity extraction and parent aggregation."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from backend.app.evidence.models import EvidenceNode


@dataclass(frozen=True)
class EntityConfig:
    vocabulary: tuple[str, ...]
    aliases: dict[str, str]
    stop_entities: frozenset[str]
    common_entity_threshold: float


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*[-/]\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?()[]{}\"'")


def load_entity_config(config_dir: Path) -> EntityConfig:
    aliases_payload = json.loads((config_dir / "entity_aliases.json").read_text(encoding="utf-8"))
    stop_payload = json.loads((config_dir / "entity_stopwords.json").read_text(encoding="utf-8"))
    aliases = {normalize_text(key): normalize_text(value) for key, value in aliases_payload["aliases"].items()}
    vocabulary = tuple(sorted({normalize_text(value) for value in aliases_payload["vocabulary"]}))
    stop = frozenset(normalize_text(value) for value in stop_payload["stop_entities"])
    return EntityConfig(vocabulary, aliases, stop, float(aliases_payload["common_entity_threshold"]))


def extract_entities(content: str | None, config: EntityConfig) -> list[str]:
    if not content:
        return []
    text = normalize_text(content)
    candidates = {term: term for term in config.vocabulary}
    candidates.update(config.aliases)
    matches: list[tuple[int, int, str]] = []
    for phrase, canonical in candidates.items():
        pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            matches.append((match.start(), match.end(), canonical))
    # Prefer the longest phrase at overlapping spans so "rules cache" does not also emit "cache".
    selected: list[tuple[int, int, str]] = []
    for start, end, canonical in sorted(matches, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        if any(start < chosen_end and end > chosen_start for chosen_start, chosen_end, _ in selected):
            continue
        selected.append((start, end, canonical))
    return sorted({canonical for _, _, canonical in selected if canonical not in config.stop_entities})


def enrich_entities(nodes: list[EvidenceNode], config: EntityConfig) -> list[EvidenceNode]:
    enriched: list[EvidenceNode] = []
    for node in nodes:
        entities = extract_entities(node.content, config) if node.modality in {"speech", "ocr_text", "pdf_text"} else []
        enriched.append(node.model_copy(update={"entities": entities}))
    by_id = {node.id: node for node in enriched}
    frame_entities: dict[tuple[str, str], set[str]] = {}
    image_entities: dict[str, set[str]] = {}
    for node in enriched:
        locator = node.provenance.get("locator_type")
        if locator == "frame_ocr_region":
            key = (node.source_id, str(node.provenance["frame_id"]))
            frame_entities.setdefault(key, set()).update(node.entities)
        elif locator == "image_ocr_region":
            image_entities.setdefault(node.source_id, set()).update(node.entities)
    for node in enriched:
        if node.modality == "video_frame":
            key = (node.source_id, str(node.provenance["frame_id"]))
            by_id[node.id] = node.model_copy(update={"entities": sorted(frame_entities.get(key, set()))})
        elif node.modality == "image":
            by_id[node.id] = node.model_copy(update={"entities": sorted(image_entities.get(node.source_id, set()))})
    return [by_id[node.id] for node in enriched]
