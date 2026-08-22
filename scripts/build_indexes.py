"""Build deterministic persistent MiniLM and CLIP Chroma indexes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.cache.manager import PIPELINE_VERSION, write_json
from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.models import EvidenceNode
from backend.app.evidence.repository import EvidenceRepository
from backend.app.vector.chroma import ChromaStore, TEXT_COLLECTION, VISUAL_COLLECTION
from backend.app.vector.embeddings import CLIP_DIMENSION, CLIP_MODEL, MINILM_DIMENSION, MINILM_MODEL


def main() -> int:
    started = time.perf_counter()
    nodes = EvidenceRepository(ROOT / "data" / "app.db").list_evidence_nodes()
    immutable = {node.id: (node.content, node.entities, node.provenance, node.confidence) for node in nodes}
    chroma = ChromaStore(ROOT / "data" / "chroma")
    updated, text_count, visual_count = chroma.rebuild(nodes, ROOT)
    if any(immutable[node.id] != (node.content, node.entities, node.provenance, node.confidence) for node in updated):
        raise RuntimeError("indexing altered canonical evidence")
    EvidenceStore(ROOT / "data" / "app.db").update_vector_references(updated)
    by_source: dict[str, list[EvidenceNode]] = {}
    for node in updated:
        by_source.setdefault(node.source_id, []).append(node)
    for source_id, source_nodes in by_source.items():
        write_json(ROOT / "data" / "cache" / source_id / "evidence.json", {
            "source_id": source_id, "pipeline_version": PIPELINE_VERSION,
            "evidence_count": len(source_nodes), "nodes": [node.model_dump(mode="json") for node in source_nodes],
        })
    runtime = time.perf_counter() - started
    print("VECTOR INDEX BUILD\n")
    print(f"Text evidence: {text_count}")
    print(f"Visual evidence: {visual_count}")
    print(f"MiniLM: PASS ({MINILM_MODEL}, {MINILM_DIMENSION}d)")
    print(f"CLIP: PASS ({CLIP_MODEL}, {CLIP_DIMENSION}d)")
    print(f"Chroma: PASS ({TEXT_COLLECTION}={chroma.collection(TEXT_COLLECTION).count()}, "
          f"{VISUAL_COLLECTION}={chroma.collection(VISUAL_COLLECTION).count()})")
    print("SQLite vector references: PASS")
    print(f"Runtime: {runtime:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
