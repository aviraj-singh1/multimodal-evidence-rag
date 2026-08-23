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

- Demo fixture: repository-local `demo_test/meeting_demo.mp4` (H.264, 1280×720, 25 FPS, 1354.0 seconds; AAC stereo 44.1 kHz source audio).
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

Status: COMPLETE

- [x] Define `Source`
- [ ] Define `ProcessingJob`
- [x] Define `EvidenceNode`
- [x] SQLite schema
- [x] Persist sources
- [x] Persist evidence nodes
- [x] Build speech nodes
- [x] Build frame nodes
- [x] Build image nodes
- [x] Build OCR nodes/data
- [x] Build PDF nodes
- [x] Preserve provenance
- [x] Cache `evidence.json`
- [x] Unit tests

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
- [x] Speech node has timestamp + speaker
- [x] Frame node has timestamp + image
- [x] PDF node has page number
- [x] Every node has source + provenance

### Evidence Storage Notes

- Four evidence-ready sources were persisted to `data/app.db`: one video, two standalone images, and one PDF. The audio-only preprocessing cache was not treated as evidence-ready because it has no transcript artifact.
- Built 333 deterministic EvidenceNodes: 168 speech, 15 video frames, 96 OCR regions, 2 image parents, and 52 PDF text blocks. Per-source counts are 247, 13, 21, and 52 respectively.
- Raw extraction artifacts remain the source of truth. Speech text/timestamps/speaker clusters, frame timestamps/paths, OCR text/confidence/bboxes, and PDF text/page/bboxes are copied without correction or invented confidence.
- Known OCR errors remain unchanged, including `529`, `AI Rate limiter`, and `Long term storagee`; the corresponding correct PDF observations remain separate.
- Every node has deterministic source/locator provenance. All `entities` are empty and both vector IDs are null; no Milestone 5+ indexing or relationships were created.
- Rebuilds use an atomic per-source transaction: source UPSERT, delete that source's nodes, then reinsert deterministic nodes. Idempotency and rollback after an intentional constraint failure passed.
- Each source has `evidence.json`; cache counts and IDs agree with SQLite. Foreign-key validation passed.
- Tests: 28 passed, 1 pre-existing FFmpeg-dependent test class skipped because FFmpeg was not on the test process PATH. The new evidence model/builder/storage tests all passed.
- `ProcessingJob` remains unchecked because the explicit Milestone 4 implementation brief required only `sources` and `evidence_nodes` tables and prohibited expanding this storage milestone.

---

## Milestone 5 — Entities + Evidence Relationships

Status: COMPLETE

### Entity Layer
- [x] Entity extraction
- [x] Lowercase / punctuation normalization
- [x] Alias map
- [x] Ignore generic stop entities

### Relationships
- [x] Define `EvidenceEdge`
- [x] Temporal overlap scoring
- [x] Entity overlap scoring
- [x] Combined confidence calculation
- [x] Persist edges in SQLite
- [x] Store `signal_breakdown`
- [x] Store human-readable explanation
- [x] Load graph into NetworkX
- [x] Cache `edges.json`
- [x] Unit tests

### Acceptance
- [x] Known speech ↔ frame pair creates TEMPORAL_OVERLAP
- [x] Known speech/frame ↔ PDF pair creates ENTITY_OVERLAP
- [x] Every edge exposes confidence + reason

### Entity and Relationship Notes

- Deterministic configurable phrase/alias extraction enriched 191 of 333 EvidenceNodes with 19 canonical entities. Frame and standalone-image parents aggregate entity metadata from their OCR children while retaining null content.
- Raw evidence is immutable: OCR `AI Rate limiter` remains unchanged but safely yields `rate limiter`; `failure 529 HTTP code` remains unchanged and does not yield `http 429`. No source content, provenance, extraction confidence, evidence ID, or vector ID changed.
- The corpus graph contains 451 canonical undirected edges: 213 temporal-primary and 238 entity-primary. Two temporal edges contain both temporal and entity signals; 450 edges are cross-modal and 190 are cross-source.
- Graph-explosion control excludes entity signals appearing in more than 8% of entity-bearing nodes. Excluded from edge generation only: `cache`, `client`, `logging`, `rate limiter`, `request`, `rules engine`, `server`, and `web server`; these remain stored on nodes.
- Graph quality: average degree 2.708709, maximum degree 32, and 178 isolated nodes. Ten real temporal/cross-source/manual examples were inspected and were coherent connection signals.
- Flagship temporal evidence is present: the 720-second frame connects to speech at 713.90-724.94 seconds with temporal score 1.0; shared `throughput` contributes entity score 0.5 and combined confidence 0.8.
- Cross-source OCR/PDF evidence is present, including frame OCR `throughput)` linked to the page-1 PDF throughput block. Speech/PDF and standalone-image/PDF entity connections also exist.
- SQLite, the four enriched `evidence.json` files, global `data/cache/graph/edges.json`, and the NetworkX graph agree. Foreign keys, atomic rollback, and exact rerun idempotency passed.
- Tests: 39 passed; one pre-existing FFmpeg-dependent test class was skipped because FFmpeg was not on the test process PATH. `pip check` reports no broken requirements.

---

## Milestone 6 — Multimodal Retrieval + Graph Expansion

Status: COMPLETE

### Vector Setup
- [x] Chroma persistent storage
- [x] `text_evidence` collection
- [x] `visual_evidence` collection
- [x] MiniLM text embeddings
- [x] CLIP image embeddings if enabled

### Retrieval
- [x] Query MiniLM embedding
- [x] Query CLIP text embedding
- [x] Query entity extraction
- [x] Text retrieval
- [x] Visual retrieval
- [x] Entity retrieval
- [x] Candidate fusion
- [x] NetworkX graph expansion
- [x] Reranking
- [x] Evidence bundle creation

### Answer
- [x] Deterministic answer formatter
- [ ] External LLM adapter only if time permits

### API
- [x] POST `/api/query`
- [x] GET `/api/evidence`
- [x] GET `/api/evidence/{id}`

### Acceptance
- [x] Query retrieves 2+ modalities
- [x] Graph expansion returns evidence not directly found by vector search
- [x] Results retain timestamps/pages/frames
- [x] Query works with external LLM disabled

### Retrieval and Index Notes

- Persistent ChromaDB contains `text_evidence` with 316 raw-content MiniLM vectors and `visual_evidence` with 17 actual-image CLIP vectors. Evidence IDs are Chroma IDs; deterministic `text:<id>` and `visual:<id>` references agree between SQLite and all four `evidence.json` caches.
- Models: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and `openai/clip-vit-base-patch32` (512 dimensions), CPU. The first model download/index build took 301.130 seconds; an exact cached rebuild took 3.542 seconds and retained the same collection counts and IDs.
- ChromaDB 0.4.24 was selected after 0.5.23 produced frozen-stack conflicts with Transformers tokenizers and PaddlePaddle protobuf. Compatible pins restore `pip check`: ChromaDB 0.4.24, tokenizers 0.21.4, protobuf 3.20.2, and PostHog 3.7.0.
- Retrieval uses cosine relevance `clamp(1 - cosine_distance, 0, 1)`, deterministic query entities, max-score channel fusion, bounded diverse seeds, persisted NetworkX expansion (1 hop default, 0.55 minimum confidence, 25-node cap), transparent reranking, and an eight-item complementary evidence bundle.
- Q1-Q6 all returned at least four final modalities. Q6 returned speech, OCR, frame, standalone image, and PDF evidence across all four sources, including speaker/timestamp, frame path, and PDF page provenance.
- Required graph value-add is verified: direct retrieval seeds introduced non-seed evidence through persisted edges. For Q1, graph expansion added the 960-second frame through a temporal connection to retrieved speech; for throughput/rules-cache queries it added cross-source images, OCR, speech, and PDF evidence.
- HTTP 429 retrieval ranks correct speech and correct 429 OCR; the noisy `failure 529 HTTP code` remains unchanged and was not returned as a 429 match. `AI Rate limiter` also remains raw.
- Q1-Q6 average full-pipeline latency in one process was 209.964 ms including the 1046.984 ms cold first query; warmed queries were approximately 40-47 ms. Models are cached once per process.
- Manual QA: Q1, Q2, Q4, and Q5 were directly strong. Q3 and Q6 have weaker top text matches because their wording contains no configured canonical entity, but CLIP, modality-diverse fusion, and graph expansion recover coherent visuals, speech, and PDF evidence. No Milestone 5 graph retuning was performed.
- FastAPI `/api/health`, `/api/query`, `/api/evidence`, and `/api/evidence/{id}` pass. Generation is deterministic and external-LLM independent.
- Tests: 49 passed with one pre-existing FFmpeg PATH-dependent class skipped. No baseline index, evaluation, frontend, semantic edges, or external LLM integration was created.

---

## Milestone 7 — Baseline RAG + Evaluation

Status: COMPLETE

### Baseline
- [x] Flatten transcript + OCR + PDF text
- [x] Fixed text chunks
- [x] MiniLM embeddings
- [x] `baseline_text` Chroma collection
- [x] POST `/api/baseline/query`

### Evaluation
- [x] Create 6+ evaluation questions
- [x] Ground-truth required evidence
- [x] Evidence Recall@5
- [x] Complete Evidence Rate
- [x] Provenance Completeness
- [x] Run baseline
- [x] Run EvidenceGraph
- [x] Export JSON
- [x] Export CSV

### Acceptance
- [x] Same questions run against both systems
- [x] Metrics are based on actual results
- [x] No fabricated comparison numbers

### Baseline and Evaluation Notes

- Fair baseline: 54 deterministic 500-character chunks with 100-character overlap, built only from the same 316 raw speech/OCR/PDF EvidenceNodes. `member_evidence_ids` are retained for audit scoring only and do not affect MiniLM cosine ranking. Visual parents, CLIP, entities, edges, and graph expansion are excluded.
- Eight manually grounded questions were defined before evaluation, spanning two text/speech sanity questions and speech+visual, speech+document, visual+document, three-family flagship, rules-cache cross-modal, and speaker/timestamp/document provenance requirements.
- Both normal services received identical question strings with top-k 5. Baseline uses MiniLM only; EvidenceGraph uses its unchanged Milestone 6 text/visual/entity fusion, one-hop persisted graph expansion, reranking, and bundle construction.
- Actual macro results — Baseline: Evidence Recall@5 0.541667, Complete Evidence Rate 0.25, Provenance Completeness 0.25, average warmed latency 18.207125 ms. EvidenceGraph: Recall 0.291667, Complete Rate 0.0, Provenance 0.625, warmed latency 50.230375 ms.
- EvidenceGraph did not win headline retrieval metrics. It tied baseline recall on Q5-Q8, won Q3, and lost Q1, Q2, and Q4. Its advantage was structured provenance and visual evidence; its weakness was final top-five ranking/bundle selection failing to retain several exact speech/document groups.
- Flagship Q6: both systems hit one of three groups (0.333333 recall); baseline hit the PDF group, while EvidenceGraph hit the visual group with complete visual provenance. Neither returned a complete speech+visual+document top-five bundle under the manual ground truth.
- HTTP 429 safety passed: correct speech/OCR/PDF evidence ranks for the query. The noisy 529 text remains raw and is not an acceptable ground-truth ID; one baseline chunk containing both genuine 429 and noisy 529 is credited only through its separately genuine member evidence.
- Exports: `data/evaluation/results.json` and `data/evaluation/results.csv`, generated by the runner and not manually edited. Production leakage search found no Q1-Q8 strings, IDs, or ground-truth node IDs in retrieval ranking logic.
- Tests: 54 passed with one pre-existing FFmpeg PATH-dependent class skipped; `pip check` reports no broken requirements.

---

## Milestone 8 — Frontend

Status: COMPLETE

### Dataset
- [x] Source list
- [x] Processing status
- [x] Load Demo Dataset button

### Ask
- [x] Query input
- [x] EvidenceGraph query
- [x] Baseline toggle
- [x] Final answer
- [x] Evidence cards

### Evidence Explorer
- [x] Speech evidence card
- [x] Frame/image evidence card
- [x] PDF evidence card
- [x] Connected evidence
- [x] Edge type
- [x] Confidence
- [x] Signal breakdown
- [x] "Why these connect"

### Acceptance
- [x] Judge can understand the evidence chain without seeing backend logs
- [x] Flagship query is demo-ready

### Frontend Notes

- One responsive React screen exposes the four cached sources, processing readiness, the flagship query, EvidenceGraph/text-baseline switching, deterministic answers, and the frozen V1.5 comparison metrics.
- Evidence cards retain modality, raw content, source filename, speaker, timestamps, PDF pages, frame imagery, direct/expanded origin, and provenance. Relationship rows expose edge type, confidence, and the persisted explanation.
- FastAPI serves cached frame/image assets under `/media` and allows only local development origins. The frontend displays the first five items while leaving the exact active V1.5 retrieval behavior unchanged.
- Production frontend build passed; backend integration and media loading passed.

---

## Milestone 9 — Demo Reliability + Documentation

Status: COMPLETE

### Cache
- [x] Final demo corpus completely processed
- [x] transcript.json
- [x] diarization.json
- [x] frames.json
- [x] ocr.json
- [x] pdf_blocks.json
- [x] evidence.json
- [x] edges.json
- [x] embeddings persisted
- [x] `/api/demo/load`

### Reliability
- [x] Restart backend and reload demo from cache
- [x] Test with external LLM disabled
- [x] Test with CLIP disabled
- [x] Test flagship query repeatedly

### Docs
- [x] README.md
- [x] ARCHITECTURE.md
- [x] SCHEMA.md
- [x] FUTURE.md
- [x] DEMO.md

### Final Demo Acceptance
- [x] Video/audio/image/PDF represented
- [x] Cross-modal relationship visible
- [x] Query requires multiple modalities
- [x] Speaker shown
- [x] Timestamp shown
- [x] Frame shown
- [x] PDF page shown
- [x] Baseline comparison shown
- [x] Evaluation table shown

### Demo Reliability Notes

- `/api/demo/load` restores the prepared corpus from SQLite/Chroma/NetworkX state without rerunning extraction models and reports 4 sources, 333 evidence nodes, 451 edges, active policy V1.5, and frozen V1.5 metrics.
- Backend restart-from-cache passed. Three repeated flagship queries returned identical top-five evidence IDs; warmed runs were approximately 46–48 ms after the cold model load.
- External LLM remains disabled. `DISABLE_CLIP=1` was tested: visual candidates become empty while text/entity retrieval, graph expansion, and evidence-bundle generation continue successfully.
- Final verification: 65 backend tests passed, one local FFmpeg-PATH-dependent class skipped, `pip check` clean, and the React production build passed.

---

# Final Build Freeze

Do not add new features after Milestone 9 begins.

Priority:

Evidence → Relationships → Retrieval → Evaluation → UI
