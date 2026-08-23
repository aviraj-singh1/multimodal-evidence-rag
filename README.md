# EvidenceGraph RAG

Hackathon MVP for retrieving connected, provenance-preserving evidence across video, audio, images, OCR, and PDF documents.

The prepared demo uses FastAPI, SQLite, Chroma, NetworkX, MiniLM, optional CLIP, and a React frontend. Expensive extraction outputs and indexes are cached. The active retrieval policy is V1.5; V1.6 remains preserved as a documented experiment.

## Run the demo

1. Start the API from the repository root: `.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`
2. Start the UI from `frontend`: `npm run dev`
3. Open `http://localhost:3000`

See `docs/DEMO.md` for the judge walkthrough and `docs/ARCHITECTURE.md` for the evidence flow.
