# EvidenceGraph RAG

EvidenceGraph RAG is a provenance-first multimodal retrieval system for finding connected evidence across video, audio, images, OCR text, and PDF documents. It retrieves canonical `EvidenceNode` records, expands relevant graph relationships, and optionally uses a local Ollama model to synthesize a grounded answer from only the final evidence bundle.

The prepared judge demo contains 4 sources, 333 evidence nodes, 451 graph edges, 316 MiniLM text vectors, and 17 CLIP visual vectors. The active retrieval policy is **V1.5** and the default local synthesis model is **`qwen3:4b-instruct`**.

## Highlights

- Multimodal ingestion for MP4, MP3/WAV, PNG/JPEG, and text-readable PDFs
- Cached transcription, speaker diarization, OCR, frame extraction, and PDF extraction
- Stable evidence IDs and backend-owned provenance
- Hybrid MiniLM text, CLIP visual, and entity retrieval
- Graph expansion through temporal, source, entity, and cross-modal relationships
- Fixed V1.5 reranking and evidence selection
- Evidence inspection with source, timestamp, page, region, confidence, and graph-neighbor details
- Optional local Ollama synthesis with strict `E1`, `E2`, ... evidence labels
- Automatic deterministic fallback if Ollama is offline or returns an invalid answer
- Text-only baseline and reproducible baseline-versus-graph evaluation

## Architecture

```text
Video / Audio / Images / PDF
            |
            v
  Cached extraction pipeline
  (FFmpeg, Whisper, pyannote, PaddleOCR, PyMuPDF)
            |
            v
 Canonical EvidenceNodes + EvidenceEdges
       |                    |
       v                    v
 SQLite provenance      NetworkX graph
       |
       +-------> Chroma indexes
                 (MiniLM text + CLIP visual)
                         |
                         v
         V1.5 retrieval and reranking
                         |
                         v
       Final evidence bundle + connections
                  /                 \
                 v                   v
      Local Ollama synthesis   Deterministic fallback
                 \                   /
                  v                 v
            Answer + E-label citations
```

Ollama is synthesis only. It does not search the corpus, modify evidence, create graph edges, alter ranking, or own provenance. The backend supplies the final retrieved evidence and connections, validates cited evidence labels, and falls back safely if generation cannot be used.

## Repository layout

```text
multimodal-evidence-rag/
├── backend/                 FastAPI API and core pipeline modules
│   ├── app/api/             Query, evidence, demo, baseline, evaluation routes
│   ├── app/evidence/        Canonical evidence and relationship logic
│   ├── app/generation/      Ollama adapter and deterministic formatter
│   ├── app/retrieval/       V1.5 retrieval, expansion, reranking, selection
│   └── tests/               Backend tests
├── config/                  Entities, policies, and evaluation configuration
├── data/                    Prepared SQLite database and evaluation artifacts
├── demo_test/               Local demo corpus (ignored by Git)
├── docs/                    Demo and architecture notes
├── frontend/                React 19 + vinext judge-facing UI
├── scripts/                 Pipeline, query, evaluation, and smoke-test CLIs
├── .env.example             Safe environment template
└── requirements.txt         Pinned Python dependencies
```

## Prerequisites

For the prepared demo:

- Python 3.11
- Node.js 22.13 or newer
- npm
- Ollama, if local synthesized answers are wanted
- Ollama model `qwen3:4b-instruct`

For rebuilding extraction caches, also install FFmpeg and ensure `ffmpeg` and `ffprobe` are on `PATH`. A GPU is optional; the default configuration runs on CPU.

## Demo dataset

Place the local demo files under `demo_test/` at the repository root:

```text
demo_test/
├── architecture_01.png
├── architecture_02.png
├── meeting_audio.mp3
├── meeting_demo.mp4
└── rate_limiter_design.pdf
```

`demo_test/` is intentionally ignored by Git because demo media may be large or private. Prepared SQLite records use repository-relative `demo_test/...` paths, and the backend serves the files through `/demo-media`.

## Quick start

Run commands from the repository root unless stated otherwise.

### 1. Create the Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned ML dependencies are large, so the first installation can take several minutes.

### 2. Configure the environment

```powershell
Copy-Item .env.example .env
```

On macOS or Linux, use `cp .env.example .env`.

Default local-generation settings:

```dotenv
GENERATION_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
```

No Ollama API key is required. Keep `HF_TOKEN` empty unless authenticated Hugging Face access is needed for speaker diarization.

### 3. Verify Ollama

```powershell
ollama --version
ollama list
Invoke-RestMethod http://localhost:11434/api/tags
```

Confirm `qwen3:4b-instruct` is listed. If it is not installed, pull it manually:

```powershell
ollama pull qwen3:4b-instruct
```

The application never installs Ollama or pulls a model automatically.

### 4. Install frontend dependencies

```powershell
Set-Location frontend
npm install
Set-Location ..
```

### 5. Start the backend

```powershell
.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

- Health: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 6. Start the frontend

Open a second terminal:

```powershell
Set-Location frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The prepared database, extraction caches, graph, indexes, and evaluation results are already included. Do not rebuild them just to run the demo.

## Query and grounding guarantees

For each query, the backend:

1. Runs MiniLM text search, optional CLIP visual search, and entity matching.
2. Fuses candidates and selects seed evidence.
3. Expands connected evidence through the graph.
4. Applies the fixed V1.5 reranker and evidence-bundle selector.
5. Assigns backend-owned labels such as `E1`, `E2`, and `E3` to the final bundle.
6. Sends only that bundle and its retrieved connections to local Ollama.
7. Accepts only citations that refer to supplied labels.
8. Uses the deterministic formatter whenever Ollama is offline, times out, cites an unknown label, or otherwise fails validation.

A query never fails solely because Ollama is unavailable. Relevant response fields are:

- `generation_mode`: `ollama` or `deterministic`
- `generation_model`: local model used for synthesis
- `cited_evidence_labels`: validated `E#` labels
- `generation_latency_ms`: synthesis time
- `generation_fallback_reason`: included when fallback was used

To run without CLIP visual search:

```powershell
$env:DISABLE_CLIP="true"
.venv\Scripts\python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

To run deterministic answer formatting intentionally, set `GENERATION_PROVIDER` to a value other than `ollama`. Retrieval remains unchanged.

## API reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Backend readiness and CLIP state |
| `GET` | `/api/demo/load` | Corpus counts, sources, policy, and evaluation summary |
| `POST` | `/api/query` | Full multimodal EvidenceGraph query |
| `GET` | `/api/evidence` | Filter evidence by source, modality, entity, or confidence |
| `GET` | `/api/evidence/{evidence_id}` | Inspect a node, its edges, and neighbors |
| `POST` | `/api/baseline/query` | Isolated text-only baseline query |
| `POST` | `/api/evaluate` | Run and export the comparison evaluation |
| `GET` | `/demo-media/{filename}` | Serve original files from `demo_test/` |
| `GET` | `/media/{path}` | Serve generated media and cache artifacts from `data/` |

Example query in PowerShell:

```powershell
$body = @{
  question = "How does the rate limiter architecture work?"
  top_k = 8
  graph_hops = 1
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/query `
  -ContentType application/json `
  -Body $body
```

Request constraints:

- `question`: required, non-empty string
- `top_k`: 1-12; default 8
- `graph_hops`: 0-2; default 1

## Judge demo flow

1. Open the frontend and confirm 4 sources, 333 evidence nodes, 451 edges, and V1.5.
2. Ask: **“How does the rate limiter architecture work?”**
3. Point out the answer's `E#` citations and open the corresponding evidence cards.
4. Show the multimodal evidence and its page, timestamp, frame, region, source, and confidence metadata.
5. Expand a card to show graph-connected evidence.
6. Stop Ollama or select an unavailable model, repeat the query, and show the safe deterministic fallback.
7. Use the baseline comparison to show the benefit of connected multimodal retrieval.

Other useful questions:

- “What rules and storage are used by the rate limiter?”
- “What happens when a request exceeds the allowed rate?”
- “Which components connect the API gateway, rules engine, and cache?”

## Common commands

Command-line query:

```powershell
.venv\Scripts\python scripts\query_evidence.py "How does the rate limiter architecture work?"
```

Backend tests:

```powershell
.venv\Scripts\python -m pytest backend\tests -q
```

Frontend production build and rendered-HTML tests:

```powershell
Set-Location frontend
npm test
```

Dependency smoke test:

```powershell
.venv\Scripts\python scripts\smoke_test_dependencies.py
```

Evaluation:

```powershell
.venv\Scripts\python scripts\evaluate_systems.py
```

Expected prepared-corpus invariants:

```text
Sources:         4
Evidence:        333
Edges:           451
Text vectors:    316
Visual vectors:   17
Policy:         V1.5
```

## Rebuilding the corpus

Rebuilding is optional and can be expensive. Extraction is cache-aware, but speech, OCR, and embedding models may require substantial downloads, memory, and processing time.

Pipeline stages:

1. Preprocess audio/video and extract representative frames.
2. Transcribe media and optionally diarize speakers.
3. Run OCR on images/video frames and extract PDF text blocks.
4. Build canonical evidence.
5. Build deterministic relationships.
6. Build MiniLM and CLIP indexes.
7. Build the isolated text baseline.
8. Run the evaluation.

Extraction examples:

```powershell
.venv\Scripts\python scripts\preprocess_media.py demo_test\meeting_demo.mp4
.venv\Scripts\python scripts\preprocess_media.py demo_test\meeting_audio.mp3
.venv\Scripts\python scripts\extract_content.py --image demo_test\architecture_01.png --image demo_test\architecture_02.png --pdf demo_test\rate_limiter_design.pdf
```

Preprocessing prints deterministic `src_...` IDs. Use the relevant IDs for transcription and video-frame OCR:

```powershell
.venv\Scripts\python scripts\transcribe_media.py src_your_source_id
.venv\Scripts\python scripts\extract_content.py --video-source src_your_video_source_id
```

After all source caches are evidence-ready, build canonical and derived stores in this order:

```powershell
.venv\Scripts\python scripts\build_evidence.py --all
.venv\Scripts\python scripts\build_relationships.py
.venv\Scripts\python scripts\build_indexes.py
.venv\Scripts\python scripts\build_baseline.py
.venv\Scripts\python scripts\evaluate_systems.py
```

If rebuilding an existing database after changing source files, back up `data/app.db` first. Evidence replacement is protected by foreign keys to existing graph edges; use a clean derived store or intentionally regenerate dependent edges rather than deleting data casually.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Data and artifact directory |
| `SQLITE_PATH` | `./data/app.db` | Canonical SQLite store |
| `CHROMA_PATH` | `./data/chroma` | Persistent vector store |
| `WHISPER_MODEL` | `base` | faster-whisper model |
| `DEVICE` | `cpu` | ML inference device |
| `HF_TOKEN` | empty | Optional Hugging Face token for diarization |
| `ENABLE_DIARIZATION` | `true` | Enable speaker diarization during extraction |
| `ENABLE_PADDLE_OCR` | `true` | Enable PaddleOCR extraction |
| `ENABLE_CLIP` | `true` | Enable CLIP indexing |
| `DISABLE_CLIP` | unset | Runtime switch to skip visual query search |
| `ENABLE_SEMANTIC_EDGES` | `false` | Keep graph construction deterministic by default |
| `GENERATION_PROVIDER` | `ollama` | Select local Ollama; another value uses fallback |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama service URL |
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Local synthesis model |
| `OLLAMA_TIMEOUT_SEC` | `30` | Request timeout before fallback |
| `OLLAMA_TEMPERATURE` | `0.1` | Low-temperature grounded generation |
| `OLLAMA_NUM_PREDICT` | `350` | Maximum generated tokens |

Legacy external-LLM variables remain disabled and are not part of the local Ollama path. Never commit `.env` or put secrets in `.env.example`.

## Troubleshooting

### Backend does not start

- Run the command from the repository root.
- Confirm Python 3.11 is active with `python --version`.
- Reinstall pinned packages with `pip install -r requirements.txt`.
- Check whether port 8000 is already in use.

### Frontend does not start

- Confirm Node.js is at least 22.13 with `node --version`.
- Run `npm install` inside `frontend/`.
- Check whether port 3000 is already in use.
- Confirm the backend is available at `http://127.0.0.1:8000`.

### Ollama falls back to deterministic mode

- Run `ollama list` and verify `qwen3:4b-instruct` exists.
- Verify `http://localhost:11434/api/tags` responds.
- Confirm `OLLAMA_BASE_URL` has no `/api/chat` suffix.
- Confirm `OLLAMA_MODEL` exactly matches the installed model name.
- Inspect `generation_fallback_reason` in the response.

Fallback is expected and safe when Ollama is offline.

### Demo media returns 404

- Confirm the five files exist in the repository-root `demo_test/` directory.
- Keep the filenames shown above.
- Restart the backend after moving the dataset.

### FFmpeg or extraction fails

- Run `ffmpeg -version` and `ffprobe -version`.
- Confirm both programs are on `PATH`.
- Verify the input file exists and is readable.
- Re-run the stage; completed valid cache artifacts are reused.

### CLIP is slow or unavailable

Set `DISABLE_CLIP=true` before starting the backend. Text, entity, and graph retrieval continue to work.

## Design boundaries

- V1.5 is the active fixed retrieval policy; V1.6 is retained only as an inactive experiment.
- Evidence, edges, vectors, graph data, evaluation data, retrieval ranking, and provenance are backend-owned.
- Ollama receives only the final retrieved evidence bundle and its connections.
- Ollama cannot add evidence or valid citations that were not supplied by the backend.
- Speaker labels such as `SPEAKER_00` are diarization clusters unless separately identified by evidence.
- OCR and transcription confidence reflect extractor output, not factual certainty.

## Additional documentation

- [Judge demo walkthrough](docs/DEMO.md)
- [Architecture notes](docs/ARCHITECTURE.md)
- [V1.5 retrieval policy](docs/RETRIEVAL_POLICY_V1_5.md)
- [V1.6 inactive experiment](docs/RETRIEVAL_POLICY_V1_6.md)
- [Evidence schema](docs/SCHEMA.md)
