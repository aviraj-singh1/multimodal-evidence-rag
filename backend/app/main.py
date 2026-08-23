"""Minimal FastAPI application for Milestone 6 retrieval."""

from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.evidence import router as evidence_router
from backend.app.api.baseline import router as baseline_router
from backend.app.api.evaluation import router as evaluation_router
from backend.app.api.query import router as query_router
from backend.app.api.query import ROOT
from backend.app.api.demo import router as demo_router


app = FastAPI(title="EvidenceGraph RAG", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)
app.include_router(query_router)
app.include_router(evidence_router)
app.include_router(baseline_router)
app.include_router(evaluation_router)
app.include_router(demo_router)
app.mount("/media", StaticFiles(directory=ROOT / "data"), name="media")
app.mount("/demo-media", StaticFiles(directory=ROOT / "demo_test"), name="demo-media")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"status": "ready", "external_llm": False,
            "clip_enabled": os.getenv("DISABLE_CLIP", "").lower() not in {"1", "true", "yes"}}
