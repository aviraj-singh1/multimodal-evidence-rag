"""Minimal FastAPI application for Milestone 6 retrieval."""

from fastapi import FastAPI

from backend.app.api.evidence import router as evidence_router
from backend.app.api.baseline import router as baseline_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.query import router as query_router


app = FastAPI(title="EvidenceGraph RAG", version="0.1")
app.include_router(query_router)
app.include_router(evidence_router)
app.include_router(baseline_router)
app.include_router(evaluation_router)


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ready", "external_llm": False}
