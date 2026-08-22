"""MiniLM cosine retrieval over the isolated baseline_text collection."""

from __future__ import annotations

from backend.app.baseline.builder import BaselineChunk
from backend.app.vector.chroma import ChromaStore
from backend.app.vector.embeddings import embed_texts


BASELINE_COLLECTION = "baseline_text"


def baseline_search(store: ChromaStore, question: str, chunks: dict[str, BaselineChunk],
                    top_k: int = 5) -> list[dict[str, object]]:
    collection = store.collection(BASELINE_COLLECTION)
    result = collection.query(query_embeddings=[embed_texts([question])[0]],
                              n_results=min(top_k, collection.count()))
    output = []
    for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
        chunk = chunks[chunk_id]
        output.append({**chunk.to_dict(), "score": round(max(0.0, min(1.0, 1.0-float(distance))), 6),
                       "raw_distance": float(distance)})
    return output
