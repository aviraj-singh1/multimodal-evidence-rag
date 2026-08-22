"""Normal text-only baseline service with deterministic formatting."""

from __future__ import annotations

import json
import time
from pathlib import Path

from backend.app.baseline.builder import BaselineChunk
from backend.app.baseline.search import baseline_search
from backend.app.vector.chroma import ChromaStore


class BaselineService:
    def __init__(self, root: Path) -> None:
        self.store = ChromaStore(root / "data" / "chroma")
        payload = json.loads((root / "data" / "cache" / "baseline" / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = {item["chunk_id"]: BaselineChunk(**item) for item in payload["chunks"]}

    def query(self, question: str, top_k: int = 5) -> dict[str, object]:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.perf_counter()
        results = baseline_search(self.store, question.strip(), self.chunks, top_k)
        lines = ["Relevant baseline text includes:"]
        for item in results:
            lines.append(f"- {str(item['text']).replace(chr(10), ' ')[:200]}")
        lines.append("Sources: " + ", ".join(sorted({str(item["source_filename"]) for item in results})))
        return {"question": question.strip(), "results": results, "answer": "\n".join(lines),
                "generation_mode": "deterministic", "latency_ms": round((time.perf_counter()-started)*1000, 3)}
