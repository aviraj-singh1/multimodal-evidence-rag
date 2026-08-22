"""Build canonical EvidenceNodes from immutable extraction caches."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.app.cache.manager import PIPELINE_VERSION, read_json
from backend.app.evidence.models import EvidenceNode, Source


def source_short(source_id: str) -> str:
    if not re.fullmatch(r"src_[0-9a-f]{12}", source_id):
        raise ValueError(f"invalid deterministic source ID: {source_id}")
    return source_id.removeprefix("src_")


def evidence_id(source_id: str, kind: str, *indices: int) -> str:
    """Return a stable, human-debuggable ID from a source and locator."""
    short = source_short(source_id)
    if kind == "image":
        return f"ev_{short}_image"
    formats = {
        "speech": ("speech", (4,)),
        "frame": ("frame", (4,)),
        "ocr_frame": ("ocr_frame", (4, 3)),
        "ocr_image": ("ocr_image", (3,)),
        "pdf": ("pdf_p", (3, 3)),
    }
    try:
        label, widths = formats[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported evidence ID kind: {kind}") from exc
    if len(indices) != len(widths) or any(index < 1 for index in indices):
        raise ValueError(f"{kind} requires {len(widths)} positive indices")
    if kind == "pdf":
        return f"ev_{short}_{label}{indices[0]:03d}_b{indices[1]:03d}"
    suffix = "_".join(f"{value:0{width}d}" for value, width in zip(indices, widths))
    return f"ev_{short}_{label}_{suffix}"


def discover_evidence_sources(cache_root: Path) -> list[str]:
    """Find caches whose required Milestone 1-3 derived artifacts exist."""
    required = {
        "video": ("transcript.json", "frames.json", "ocr.json"),
        "image": ("ocr.json",),
        "pdf": ("pdf_blocks.json",),
    }
    discovered: list[str] = []
    for directory in sorted(cache_root.iterdir() if cache_root.is_dir() else []):
        source_path = directory / "source.json"
        if not directory.is_dir() or not source_path.is_file():
            continue
        source = read_json(source_path)
        names = required.get(source.get("source_type"))
        if names and all((directory / name).is_file() for name in names):
            discovered.append(directory.name)
    return discovered


def build_source(cache_dir: Path, repo_root: Path, dataset_dir: Path) -> Source:
    raw = read_json(cache_dir / "source.json")
    source_id = raw["source_id"]
    if source_id != cache_dir.name:
        raise ValueError(f"source ID does not match cache directory: {cache_dir}")
    source_file = dataset_dir / raw["filename"]
    original_path = _portable_path(source_file, repo_root) if source_file.is_file() else None
    canonical = {"source_id", "filename", "source_type", "sha256", "created_at"}
    return Source(
        id=source_id,
        filename=raw["filename"],
        media_type=raw["source_type"],
        original_path=original_path,
        sha256=raw["sha256"],
        created_at=raw["created_at"],
        metadata={key: value for key, value in raw.items() if key not in canonical},
    )


def build_evidence(cache_dir: Path, repo_root: Path, dataset_dir: Path) -> tuple[Source, list[EvidenceNode]]:
    source = build_source(cache_dir, repo_root, dataset_dir)
    if source.media_type == "video":
        nodes = _build_video(source, cache_dir)
    elif source.media_type == "image":
        nodes = _build_image(source, cache_dir)
    elif source.media_type == "pdf":
        nodes = _build_pdf(source, cache_dir)
    else:
        raise ValueError(f"no evidence builder for source type {source.media_type!r}")
    validate_evidence(source, nodes, cache_dir, repo_root)
    return source, nodes


def evidence_payload(source_id: str, nodes: list[EvidenceNode]) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "pipeline_version": PIPELINE_VERSION,
        "evidence_count": len(nodes),
        "nodes": [node.model_dump(mode="json") for node in nodes],
    }


def modality_counts(nodes: list[EvidenceNode]) -> Counter[str]:
    return Counter(node.modality for node in nodes)


def _node(source: Source, **values: Any) -> EvidenceNode:
    return EvidenceNode(
        source_id=source.id,
        entities=[],
        text_vector_id=None,
        visual_vector_id=None,
        **values,
    )


def _build_video(source: Source, cache_dir: Path) -> list[EvidenceNode]:
    transcript = read_json(cache_dir / "transcript.json")
    frames = read_json(cache_dir / "frames.json")
    ocr = read_json(cache_dir / "ocr.json")
    nodes: list[EvidenceNode] = []
    for index, segment in enumerate(transcript["segments"], start=1):
        provenance = {
            "file": source.filename,
            "source_id": source.id,
            "locator_type": "time_range",
            "start_sec": segment["start_sec"],
            "end_sec": segment["end_sec"],
            "transcript_segment_id": segment["segment_id"],
            "multi_speaker": segment.get("multi_speaker", False),
        }
        if "speaker_alignment" in segment:
            provenance["speaker_alignment"] = segment["speaker_alignment"]
        nodes.append(_node(
            source,
            id=evidence_id(source.id, "speech", index), modality="speech",
            content=segment["text"], start_sec=segment["start_sec"], end_sec=segment["end_sec"],
            speaker=segment["speaker"], page_number=None, bbox=None, frame_path=None,
            confidence=segment.get("confidence"), provenance=provenance,
        ))
    for index, frame in enumerate(frames["frames"], start=1):
        frame_path = _cache_relative_path(source.id, frame["path"])
        timestamp = frame["timestamp_sec"]
        nodes.append(_node(
            source,
            id=evidence_id(source.id, "frame", index), modality="video_frame",
            content=None, start_sec=timestamp, end_sec=timestamp, speaker=None,
            page_number=None, bbox=None, frame_path=frame_path, confidence=None,
            provenance={"file": source.filename, "source_id": source.id,
                        "locator_type": "video_frame", "frame_id": frame["frame_id"],
                        "timestamp_sec": timestamp},
        ))
    for frame_index, frame in enumerate(ocr["frames"], start=1):
        frame_path = _cache_relative_path(source.id, frame["frame_path"])
        timestamp = frame["timestamp_sec"]
        for region_index, region in enumerate(frame["regions"], start=1):
            nodes.append(_node(
                source,
                id=evidence_id(source.id, "ocr_frame", frame_index, region_index), modality="ocr_text",
                content=region["text"], start_sec=timestamp, end_sec=timestamp, speaker=None,
                page_number=None, bbox=region["bbox"], frame_path=frame_path,
                confidence=region["confidence"],
                provenance={"file": source.filename, "source_id": source.id,
                            "locator_type": "frame_ocr_region", "frame_id": frame["frame_id"],
                            "timestamp_sec": timestamp, "ocr_region_id": region["region_id"],
                            "ocr_engine": ocr["engine"]},
            ))
    return nodes


def _build_image(source: Source, cache_dir: Path) -> list[EvidenceNode]:
    ocr = read_json(cache_dir / "ocr.json")
    image_path = source.original_path
    nodes = [_node(
        source,
        id=evidence_id(source.id, "image"), modality="image", content=None,
        start_sec=None, end_sec=None, speaker=None, page_number=None, bbox=None,
        frame_path=image_path, confidence=None,
        provenance={"file": source.filename, "source_id": source.id, "locator_type": "image"},
    )]
    for index, region in enumerate(ocr["regions"], start=1):
        nodes.append(_node(
            source,
            id=evidence_id(source.id, "ocr_image", index), modality="ocr_text",
            content=region["text"], start_sec=None, end_sec=None, speaker=None,
            page_number=None, bbox=region["bbox"], frame_path=image_path,
            confidence=region["confidence"],
            provenance={"file": source.filename, "source_id": source.id,
                        "locator_type": "image_ocr_region", "ocr_region_id": region["region_id"],
                        "ocr_engine": ocr["engine"]},
        ))
    return nodes


def _build_pdf(source: Source, cache_dir: Path) -> list[EvidenceNode]:
    pdf = read_json(cache_dir / "pdf_blocks.json")
    nodes: list[EvidenceNode] = []
    for page in pdf["pages"]:
        page_number = page["page_number"]
        for block_index, block in enumerate(page["blocks"], start=1):
            nodes.append(_node(
                source,
                id=evidence_id(source.id, "pdf", page_number, block_index), modality="pdf_text",
                content=block["text"], start_sec=None, end_sec=None, speaker=None,
                page_number=page_number, bbox=block["bbox"], frame_path=None, confidence=None,
                provenance={"file": source.filename, "source_id": source.id,
                            "locator_type": "pdf_block", "page_number": page_number,
                            "block_id": block["block_id"]},
            ))
    return nodes


def validate_evidence(source: Source, nodes: list[EvidenceNode], cache_dir: Path, repo_root: Path) -> None:
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate evidence IDs for {source.id}")
    expected_locators = {
        "speech": "time_range", "video_frame": "video_frame", "image": "image",
        "pdf_text": "pdf_block",
    }
    for node in nodes:
        if node.source_id != source.id or node.provenance.get("source_id") != source.id:
            raise ValueError(f"ambiguous source provenance on {node.id}")
        if node.provenance.get("file") != source.filename:
            raise ValueError(f"ambiguous file provenance on {node.id}")
        locator = node.provenance.get("locator_type")
        required = expected_locators.get(node.modality)
        if required and locator != required:
            raise ValueError(f"invalid locator for {node.id}")
        if node.modality == "ocr_text" and locator not in {"frame_ocr_region", "image_ocr_region"}:
            raise ValueError(f"invalid OCR locator for {node.id}")
        if node.modality == "video_frame" or locator == "frame_ocr_region":
            if not node.frame_path or not (repo_root / node.frame_path).is_file():
                raise ValueError(f"missing frame referenced by {node.id}")
        if node.entities or node.text_vector_id is not None or node.visual_vector_id is not None:
            raise ValueError(f"future indexing data present on {node.id}")
        # Extraction success is not confidence. Only OCR supplies a normalized confidence here.
        if node.modality != "ocr_text" and node.confidence is not None:
            raise ValueError(f"artificial confidence present on {node.id}")


def _cache_relative_path(source_id: str, path: str) -> str:
    return (Path("data") / "cache" / source_id / path).as_posix()


def _portable_path(path: Path, repo_root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), repo_root.resolve())).as_posix()
