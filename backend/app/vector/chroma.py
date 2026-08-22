"""Chroma persistent collections and deterministic corpus indexing."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from backend.app.evidence.models import EvidenceNode
from backend.app.vector.embeddings import embed_clip_images, embed_texts


TEXT_COLLECTION = "text_evidence"
VISUAL_COLLECTION = "visual_evidence"


class ChromaStore:
    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )

    def collection(self, name: str):
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def rebuild(self, nodes: list[EvidenceNode], repo_root: Path) -> tuple[list[EvidenceNode], int, int]:
        text_nodes = eligible_text_nodes(nodes)
        visual_nodes = eligible_visual_nodes(nodes)
        text_vectors = embed_texts([node.content or "" for node in text_nodes])
        visual_paths = [(repo_root / str(node.frame_path)).resolve() for node in visual_nodes]
        if not all(path.is_file() for path in visual_paths):
            raise FileNotFoundError("one or more visual evidence files are missing")
        visual_vectors = embed_clip_images(visual_paths)
        for name in (TEXT_COLLECTION, VISUAL_COLLECTION):
            try:
                self.client.delete_collection(name)
            except ValueError:
                pass
        text_collection = self.collection(TEXT_COLLECTION)
        visual_collection = self.collection(VISUAL_COLLECTION)
        text_collection.add(
            ids=[node.id for node in text_nodes], embeddings=text_vectors,
            documents=[node.content or "" for node in text_nodes],
            metadatas=[_metadata(node) for node in text_nodes],
        )
        visual_collection.add(
            ids=[node.id for node in visual_nodes], embeddings=visual_vectors,
            metadatas=[_metadata(node) for node in visual_nodes],
        )
        text_ids = {node.id for node in text_nodes}
        visual_ids = {node.id for node in visual_nodes}
        updated = [node.model_copy(update={
            "text_vector_id": f"text:{node.id}" if node.id in text_ids else None,
            "visual_vector_id": f"visual:{node.id}" if node.id in visual_ids else None,
        }) for node in nodes]
        return updated, len(text_nodes), len(visual_nodes)


def _metadata(node: EvidenceNode) -> dict[str, str | int | float]:
    metadata: dict[str, str | int | float] = {
        "evidence_id": node.id, "source_id": node.source_id, "modality": node.modality,
    }
    for key, value in (("speaker", node.speaker), ("page_number", node.page_number),
                       ("start_sec", node.start_sec), ("end_sec", node.end_sec)):
        if value is not None:
            metadata[key] = value
    return metadata


def eligible_text_nodes(nodes: list[EvidenceNode]) -> list[EvidenceNode]:
    return [node for node in nodes if node.modality in {"speech", "ocr_text", "pdf_text"} and node.content]


def eligible_visual_nodes(nodes: list[EvidenceNode]) -> list[EvidenceNode]:
    return [node for node in nodes if node.modality in {"video_frame", "image"} and node.frame_path]
