"""MiniLM cosine search over raw textual evidence."""

from __future__ import annotations

from backend.app.vector.chroma import ChromaStore, TEXT_COLLECTION
from backend.app.vector.embeddings import embed_texts


def text_search(store: ChromaStore, question: str, top_k: int = 8) -> list[dict[str, object]]:
    collection = store.collection(TEXT_COLLECTION)
    if collection.count() == 0:
        return []
    result = collection.query(query_embeddings=[embed_texts([question])[0]], n_results=min(top_k, collection.count()))
    candidates = []
    for evidence_id, distance in zip(result["ids"][0], result["distances"][0]):
        # Chroma cosine distance is 1-cosine_similarity; clamp negative similarities to zero relevance.
        candidates.append({"evidence_id": evidence_id, "channel": "text",
                           "raw_distance_or_score": float(distance),
                           "normalized_score": round(max(0.0, min(1.0, 1.0 - float(distance))), 6)})
    return candidates
