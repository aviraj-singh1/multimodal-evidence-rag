"""Build the isolated fixed-chunk baseline_text Chroma index."""

from __future__ import annotations

import sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from backend.app.baseline.builder import CHUNK_OVERLAP, CHUNK_SIZE, build_chunks
from backend.app.baseline.search import BASELINE_COLLECTION
from backend.app.cache.manager import PIPELINE_VERSION, write_json
from backend.app.evidence.repository import EvidenceRepository
from backend.app.vector.chroma import ChromaStore
from backend.app.vector.embeddings import embed_texts


def main() -> int:
    started=time.perf_counter(); repository=EvidenceRepository(ROOT/'data'/'app.db')
    chunks=build_chunks(repository.list_evidence_nodes(),repository.list_sources())
    store=ChromaStore(ROOT/'data'/'chroma')
    try: store.client.delete_collection(BASELINE_COLLECTION)
    except ValueError: pass
    collection=store.collection(BASELINE_COLLECTION)
    collection.add(ids=[c.chunk_id for c in chunks],documents=[c.text for c in chunks],
                   embeddings=embed_texts([c.text for c in chunks]),
                   metadatas=[{"chunk_id":c.chunk_id,"source_id":c.source_id,
                               "source_filename":c.source_filename,"chunk_index":c.chunk_index} for c in chunks])
    write_json(ROOT/'data'/'cache'/'baseline'/'chunks.json',
               {"pipeline_version":PIPELINE_VERSION,"chunk_size":CHUNK_SIZE,"chunk_overlap":CHUNK_OVERLAP,
                "chunk_count":len(chunks),"chunks":[c.to_dict() for c in chunks]})
    print(f"BASELINE BUILD\n\nChunks: {len(chunks)}\nCollection: {collection.count()}\nChunk size: {CHUNK_SIZE}\nOverlap: {CHUNK_OVERLAP}\nRuntime: {time.perf_counter()-started:.3f}s")
    return 0

if __name__=='__main__': raise SystemExit(main())
