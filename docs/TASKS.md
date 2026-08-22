# Build Progress

> Rule: Do not start the next milestone until the current milestone's acceptance checks pass.

---

## Milestone 0 — Dependency Smoke Tests

Status: COMPLETE

- [x] FFmpeg available
- [x] FFprobe available
- [x] Python venv working
- [x] faster-whisper import + tiny transcription test
- [x] pyannote import + HF authentication test
- [x] PaddleOCR import + one image OCR test
- [x] PyMuPDF import + PDF extraction test
- [x] sentence-transformers MiniLM loads
- [x] CLIP loads OR mark CLIP disabled
- [x] Record fallback decisions in TASKS.md

### Smoke Test Notes

- Python: 3.11.9 from the existing `.venv`; `pip check` reports no broken requirements.
- Device: AMD Ryzen 7 5800H CPU. NVIDIA GeForce RTX 3050 Laptop GPU (4 GB) is present, but the installed PyTorch 2.5.1 build is CPU-only, so smoke-test inference ran in CPU mode.
- FFmpeg / FFprobe: 9.0; FFprobe successfully read `meeting_demo.mp4` from the external demo dataset without modifying it.
- PyMuPDF: 1.24.14; extracted page 1 text from `rate_limiter_design.pdf` (2 pages).
- Text embedding: `sentence-transformers/all-MiniLM-L6-v2`; produced a 384-dimensional embedding.
- Speech: `faster-whisper` 1.1.0 with the `base` model, CPU `int8`; transcribed only a 12-second WAV copied/extracted into `data/smoke/` and exposed timestamped segments.
- Diarization: `pyannote.audio` 3.3.2 loaded `pyannote/speaker-diarization-3.1` (including gated dependency `pyannote/segmentation-3.0`) using `HF_TOKEN` from the repo-root `.env`. A 12-second smoke WAV returned 4 speaker-labelled turns; the first was `SPEAKER_00` at 0.03–2.02 seconds. The token value was never printed.
- OCR: PaddlePaddle 2.6.2 + PaddleOCR 2.9.1; loaded English detection/recognition models and returned OCR result structures and text from `data/smoke/ocr_sample.png`.
- Visual embedding: `openai/clip-vit-base-patch32`; produced matching 512-dimensional text and image embeddings.
- Failures/manual blockers: none.
- Fallbacks: no fallback was activated. Tesseract is not needed because PaddleOCR passed; cached precomputed diarization is not needed because pyannote passed; CLIP remains enabled because it passed.

### Acceptance
- [x] Every P0 dependency either works or has documented fallback

---

## Milestone 1 — Media Preprocessing

Status: COMPLETE

- [x] MP4 → 16 kHz mono WAV
- [x] MP3/WAV → normalized 16 kHz mono WAV
- [x] MP4 → timestamped representative frames
- [x] Limit frame count to sensible amount
- [x] Generate `frames.json`
- [x] Generate source metadata
- [x] FFmpeg / FFprobe error handling
- [x] Create source-specific cache directory
- [x] CLI smoke test

### Output

data/cache/{source_id}/
- audio.wav
- frames/
- frames.json
- source.json

### Preprocessing Notes

- Demo fixture: external read-only `../demo_dataset/meeting_demo.mp4` (H.264, 1280×720, 25 FPS, 1354.0 seconds; AAC stereo 44.1 kHz source audio).
- Deterministic source ID: `src_1347fcb63e98`, derived from the first 12 characters of the source SHA-256.
- Audio output: PCM signed 16-bit little-endian WAV, mono, 16 kHz; FFprobe validation and a decode test passed.
- Direct audio: external MP3 and a direct WAV input both normalized successfully without overwriting their inputs.
- Frames: 15 JPEGs with original-video timestamps, bounded below the hard maximum of 30.
- Selection: scene threshold 0.35 plus 120-second interval supplementation because scene-only extraction left a 485.64-second coverage gap. Manual contact-sheet inspection confirmed both speakers and multiple meaningful completed architecture-diagram states were captured.
- Cache: source hash, pipeline version, settings, manifests, and expected outputs are validated. An exact rerun reported `REUSED`.
- Tests: 7 focused tests passed, covering deterministic IDs, metadata, MP4/audio outputs, direct WAV normalization, frame structure/files/timestamps/count, cache reuse, and invalid inputs.

### Acceptance
- [x] Demo MP4 produces playable WAV
- [x] Extracted frames have correct timestamps
- [x] Rerunning same source can reuse cache

---

## Milestone 2 — Transcription + Speaker Attribution

Status: COMPLETE

- [x] faster-whisper transcription
- [x] Preserve segment start/end timestamps
- [x] Generate `transcript.json`
- [x] pyannote speaker diarization
- [x] Generate `diarization.json`
- [x] Match speaker intervals to transcript segments
- [x] Support cached/precomputed diarization fallback
- [x] Tests

### Speech Processing Notes

- Input: existing full-length `data/cache/src_1347fcb63e98/audio.wav`; no trimming or Milestone 1 reprocessing was performed.
- Transcription: faster-whisper 1.1.0, `base`, CPU, int8, beam size 5, VAD enabled. Generated 168 native Whisper segments in 121.079 seconds; detected language `en`. Segment confidence remains `null` because `avg_logprob` was not misrepresented as a 0–1 confidence.
- Diarization: `pyannote/speaker-diarization-3.1`, CPU. Generated 463 sorted speaker turns in 1091.575 seconds and detected 6 speaker clusters.
- Alignment: maximum summed interval overlap per speaker; 168/168 transcript segments attributed, 0 unattributed, 12 flagged `multi_speaker`; alignment took approximately 0.003 seconds.
- Timeline: transcript, diarization, and frame timestamps all use the unchanged original 0–1354 second `meeting_demo.mp4` coordinate system.
- Cache: transcription and diarization cache validity includes source ID/hash, pipeline version, model, configuration, and output schema. Exact second run reported both stages `REUSED`.
- Tests: 10 focused tests passed; `pip check` reports no broken requirements.
- Manual QA: transcript text is understandable and aligns plausibly with representative frames (for example, the 720-second frame aligns with the 713.90–724.94 rate-limiter-cache discussion). Speaker changes at the opening interviewer/candidate handoff are plausible.
- Known quality issue: the video visibly contains two primary participants, while unconstrained pyannote produced 6 speaker clusters. This is likely speaker over-clustering; labels remain usable for deterministic provenance but should be reviewed before relying on exact speaker-count semantics. No identity recognition or silent model/configuration change was applied.

### Acceptance
- [x] Transcript text is correct enough for demo
- [x] Every demo speech segment has timestamp
- [x] Important demo segments have speaker labels

---

## Milestone 3 — OCR + PDF Extraction

Status: COMPLETE

### Image / Frames
- [x] PaddleOCR integration
- [x] OCR text
- [x] OCR confidence
- [x] OCR bounding boxes
- [ ] Tesseract fallback if PaddleOCR fails

### PDF
- [x] PyMuPDF integration
- [x] Page-level extraction
- [x] Block-level extraction
- [x] Preserve page number
- [x] Preserve bbox where available

### Content Extraction Notes

- Video OCR: PaddleOCR processed all 15 existing representative frames only; 10 frames contained text and 64 non-empty OCR regions were preserved with confidence, bbox, frame ID/path, and exact `frames.json` timestamp.
- Strong architecture frames: 840, 960, and 1080 seconds. Recovered labels include `Rules Engine`, `Rules cache`, `API rate limiter`, `client(s)`, `Load balancer`, `Api/web servers`, `failure 429 HTTP code`, `Cache(high throughput)`, `Logging mechanism`, and `Long term storage`.
- Frame-by-frame QA: 10.56 `Exponent`; 120 `Exponent`; 148.2 none; 240 `Exponent`; 360 none; 475.48 none; 480 none; 600 `client`; 720 partial diagram; 840 architecture diagram; 960 architecture diagram plus requirements; 1080 architecture diagram plus requirements; 1200 none; 1250.48 `Exponent`; 1320 `Exponent`.
- OCR quality limitations were retained unmodified: the 720-second frame misread visible `429` as `529`; the 840-second frame read `AI Rate limiter`; `architecture_02.png` produced a stray `AP'` and `*`, and read `Long term storagee`. Arrows and diagram relationships are visual structure rather than OCR text and were not encoded in this milestone.
- Standalone images: `architecture_01.png` → `src_9899772d9cc4` with 12 regions; `architecture_02.png` → `src_66bf6ebf0dbc` with 20 regions. Both preserve filename, SHA-256, dimensions, confidence, and bbox provenance. `architecture_02.png` is the strongest complete architecture fixture.
- PDF: `rate_limiter_design.pdf` → `src_00b5ee2f232b`; 2 pages and 52 text blocks (page 1: 31, page 2: 21), with 1-based page numbers, page dimensions, page text, block text, and bbox.
- PDF QA: page 1 extracts `Rate Limiter Architecture`, server-side middleware, rules engine, high-throughput cache, HTTP 429, logging, and the request path. Page 2 extracts `Request Lifecycle and Rules-Cache Optimization`, the rules-cache optimization, and architecture concepts. The intended demo design content is sufficiently extractable.
- Cache: exact second CLI run reported `REUSED` for video OCR, both standalone images, and PDF extraction; PaddleOCR is lazily initialized only for uncached work.
- Tests: 9 focused OCR/PDF tests passed; `pip check` reports no broken requirements.
- Fallback: Tesseract was not invoked or silently substituted because PaddleOCR passed.

### Acceptance
- [x] Architecture diagram text is recovered
- [x] Important PDF information is extracted with correct page number

---

## Milestone 4 — Evidence Nodes + Storage + Cache

Status: NOT STARTED

- [ ] Define `Source`
- [ ] Define `ProcessingJob`
- [ ] Define `EvidenceNode`
- [ ] SQLite schema
- [ ] Persist sources
- [ ] Persist evidence nodes
- [ ] Build speech nodes
- [ ] Build frame nodes
- [ ] Build image nodes
- [ ] Build OCR nodes/data
- [ ] Build PDF nodes
- [ ] Preserve provenance
- [ ] Cache `evidence.json`
- [ ] Unit tests

### Required EvidenceNode fields

- id
- source_id
- modality
- content
- start_sec
- end_sec
- speaker
- page_number
- bbox
- frame_path
- entities
- confidence
- provenance
- text_vector_id
- visual_vector_id

### Acceptance
- [ ] Speech node has timestamp + speaker
- [ ] Frame node has timestamp + image
- [ ] PDF node has page number
- [ ] Every node has source + provenance

---

## Milestone 5 — Entities + Evidence Relationships

Status: NOT STARTED

### Entity Layer
- [ ] Entity extraction
- [ ] Lowercase / punctuation normalization
- [ ] Alias map
- [ ] Ignore generic stop entities

### Relationships
- [ ] Define `EvidenceEdge`
- [ ] Temporal overlap scoring
- [ ] Entity overlap scoring
- [ ] Combined confidence calculation
- [ ] Persist edges in SQLite
- [ ] Store `signal_breakdown`
- [ ] Store human-readable explanation
- [ ] Load graph into NetworkX
- [ ] Cache `edges.json`
- [ ] Unit tests

### Acceptance
- [ ] Known speech ↔ frame pair creates TEMPORAL_OVERLAP
- [ ] Known speech/frame ↔ PDF pair creates ENTITY_OVERLAP
- [ ] Every edge exposes confidence + reason

---

## Milestone 6 — Multimodal Retrieval + Graph Expansion

Status: NOT STARTED

### Vector Setup
- [ ] Chroma persistent storage
- [ ] `text_evidence` collection
- [ ] `visual_evidence` collection
- [ ] MiniLM text embeddings
- [ ] CLIP image embeddings if enabled

### Retrieval
- [ ] Query MiniLM embedding
- [ ] Query CLIP text embedding
- [ ] Query entity extraction
- [ ] Text retrieval
- [ ] Visual retrieval
- [ ] Entity retrieval
- [ ] Candidate fusion
- [ ] NetworkX graph expansion
- [ ] Reranking
- [ ] Evidence bundle creation

### Answer
- [ ] Deterministic answer formatter
- [ ] External LLM adapter only if time permits

### API
- [ ] POST `/api/query`
- [ ] GET `/api/evidence`
- [ ] GET `/api/evidence/{id}`

### Acceptance
- [ ] Query retrieves 2+ modalities
- [ ] Graph expansion returns evidence not directly found by vector search
- [ ] Results retain timestamps/pages/frames
- [ ] Query works with external LLM disabled

---

## Milestone 7 — Baseline RAG + Evaluation

Status: NOT STARTED

### Baseline
- [ ] Flatten transcript + OCR + PDF text
- [ ] Fixed text chunks
- [ ] MiniLM embeddings
- [ ] `baseline_text` Chroma collection
- [ ] POST `/api/baseline/query`

### Evaluation
- [ ] Create 6+ evaluation questions
- [ ] Ground-truth required evidence
- [ ] Evidence Recall@5
- [ ] Complete Evidence Rate
- [ ] Provenance Completeness
- [ ] Run baseline
- [ ] Run EvidenceGraph
- [ ] Export JSON
- [ ] Export CSV

### Acceptance
- [ ] Same questions run against both systems
- [ ] Metrics are based on actual results
- [ ] No fabricated comparison numbers

---

## Milestone 8 — Frontend

Status: NOT STARTED

### Dataset
- [ ] Source list
- [ ] Processing status
- [ ] Load Demo Dataset button

### Ask
- [ ] Query input
- [ ] EvidenceGraph query
- [ ] Baseline toggle
- [ ] Final answer
- [ ] Evidence cards

### Evidence Explorer
- [ ] Speech evidence card
- [ ] Frame/image evidence card
- [ ] PDF evidence card
- [ ] Connected evidence
- [ ] Edge type
- [ ] Confidence
- [ ] Signal breakdown
- [ ] "Why these connect"

### Acceptance
- [ ] Judge can understand the evidence chain without seeing backend logs
- [ ] Flagship query is demo-ready

---

## Milestone 9 — Demo Reliability + Documentation

Status: NOT STARTED

### Cache
- [ ] Final demo corpus completely processed
- [ ] transcript.json
- [ ] diarization.json
- [ ] frames.json
- [ ] ocr.json
- [ ] pdf_blocks.json
- [ ] evidence.json
- [ ] edges.json
- [ ] embeddings persisted
- [ ] `/api/demo/load`

### Reliability
- [ ] Restart backend and reload demo from cache
- [ ] Test with external LLM disabled
- [ ] Test with CLIP disabled
- [ ] Test flagship query repeatedly

### Docs
- [ ] README.md
- [ ] ARCHITECTURE.md
- [ ] SCHEMA.md
- [ ] FUTURE.md
- [ ] DEMO.md

### Final Demo Acceptance
- [ ] Video/audio/image/PDF represented
- [ ] Cross-modal relationship visible
- [ ] Query requires multiple modalities
- [ ] Speaker shown
- [ ] Timestamp shown
- [ ] Frame shown
- [ ] PDF page shown
- [ ] Baseline comparison shown
- [ ] Evaluation table shown

---

# Final Build Freeze

Do not add new features after Milestone 9 begins.

Priority:

Evidence → Relationships → Retrieval → Evaluation → UI
