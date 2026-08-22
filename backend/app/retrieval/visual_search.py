"""CLIP text-to-image search over frame and standalone-image evidence."""

from __future__ import annotations

from backend.app.vector.chroma import ChromaStore, VISUAL_COLLECTION
from backend.app.vector.embeddings import embed_clip_query


def visual_search(store: ChromaStore, question: str, top_k: int = 5) -> list[dict[str, object]]:
    collection = store.collection(VISUAL_COLLECTION)
    if collection.count() == 0:
        return []
    result = collection.query(query_embeddings=[embed_clip_query(question)], n_results=min(top_k, collection.count()))
    return [{"evidence_id": evidence_id, "channel": "visual",
             "raw_distance_or_score": float(distance),
             "normalized_score": round(max(0.0, min(1.0, 1.0 - float(distance))), 6)}
            for evidence_id, distance in zip(result["ids"][0], result["distances"][0])]
