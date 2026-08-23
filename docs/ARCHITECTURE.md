# Architecture

`media → cached extraction → EvidenceNode → deterministic EvidenceEdge → multimodal retrieval → graph expansion → evidence bundle → deterministic answer`

- FFmpeg prepares audio and timestamped frames.
- faster-whisper and pyannote provide timestamped, speaker-attributed speech.
- PaddleOCR and PyMuPDF preserve regions, blocks, bounding boxes, and pages.
- SQLite is the canonical evidence/edge store.
- MiniLM indexes raw text; CLIP optionally indexes real images.
- NetworkX expands direct seeds through `TEMPORAL_OVERLAP` and `ENTITY_OVERLAP` edges.
- FastAPI returns evidence, provenance, scores, and relationship explanations.
- React presents the EvidenceGraph result beside the isolated text-only baseline.

CLIP can be disabled with `DISABLE_CLIP=1`; text/entity retrieval and graph expansion remain operational.
