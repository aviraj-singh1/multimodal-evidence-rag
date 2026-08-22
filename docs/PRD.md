# Final PRD — Multimodal Evidence Graph for RAG

**Version:** 1.0 — Build Freeze  
**Purpose:** 13-hour hackathon MVP that can evolve into a production product

---

## 0. Build Rule

This document is the single source of truth for Codex, Antigravity, and human contributors.

Do not change the stack, data model, API contracts, or P0/P1/P2 priorities without an explicit team decision.

The product is **NOT**:

`multimedia → text → chunks → vector DB → LLM`

The product **IS**:

`multimodal observations → structured evidence → explicit relationships → modality-aware retrieval → graph expansion → provenance-preserving answer`

**The evidence graph is the product.**

---

# 1. Product Summary

## Working Name

**EvidenceGraph RAG**

Temporary working name only. Branding is not part of the MVP.

## Problem

Important knowledge is spread across:

- videos;
- audio recordings;
- screenshots;
- diagrams;
- PDFs;
- documents.

Traditional RAG converts most of this into text and retrieves similar chunks.

This loses important context such as:

- who said something;
- when it was said;
- what was displayed at the same time;
- how the visual relates to the speech;
- how another document supports the same claim;
- where the evidence originally came from.

## Goal

Build a system that:

1. ingests multimodal information;
2. turns it into structured evidence;
3. creates explicit cross-modal relationships;
4. retrieves connected evidence;
5. answers a question with exact provenance.

## Flagship Question

> What architecture was proposed to reduce database load, who explained it, and where was the corresponding diagram shown?

A correct answer may require:

- transcript evidence;
- speaker;
- timestamp;
- video frame;
- OCR from that frame;
- PDF evidence.

---

# 2. Hackathon Success Definition

The project succeeds only if all of the following work end-to-end:

1. Video, audio, image, and PDF/document information can be ingested.
2. Useful information is extracted while retaining source location.
3. Extracted information becomes structured `EvidenceNode` objects.
4. Real cross-modal `EvidenceEdge` relationships are generated.
5. Temporal relationships work.
6. Entity relationships work.
7. Queries retrieve evidence using more than one retrieval channel.
8. Retrieved seeds expand through the evidence graph.
9. Results retain timestamp/page/frame/source provenance.
10. A text-only RAG baseline exists.
11. Both systems can be evaluated using the same questions.
12. At least one demo query requires multiple modalities.
13. The final demo can run completely from cached/preprocessed data.

---

# 3. Primary User

A:

- engineer;
- researcher;
- analyst;
- investigator;
- compliance professional;
- knowledge worker;

whose information is spread across meetings, diagrams, screenshots, and documents.

## Primary User Story

> As a user, I want to ask a question across all of my information and receive not only the answer, but also the exact pieces of evidence supporting it and where each piece originally came from.

---

# 4. Product Principles

## 4.1 Preserve modality

Speech stays speech.

A video frame stays a visual object.

A PDF block stays attached to its page.

Do not flatten everything before building the knowledge layer.

## 4.2 Preserve provenance

Every evidence item must know where it came from.

Examples:

```text
meeting.mp4
04:12–04:24
```

```text
meeting.mp4
Frame @ 04:17
```

```text
architecture.pdf
Page 8
```

## 4.3 Use semantic evidence units

Do not make arbitrary 30-second video clips the central knowledge unit.

Use:

```text
Speech → utterance / transcript segment

Video → meaningful keyframe

PDF → page/block

Image → image + OCR
```

## 4.4 Relationships are explicit

This:

```text
Speech
  ↓ TEMPORAL_OVERLAP
Frame
```

must exist as stored data.

It should not merely be an assumption made by an LLM during answer generation.

## 4.5 Deterministic P0 indexing

The core relationship engine must not require an LLM call for every evidence item.

P0 relationships are based on:

```text
timestamps
+
entities
```

## 4.6 Modality-aware retrieval

Do not trust one vector representation for everything.

Use separate channels:

```text
Text retrieval

Visual retrieval

Entity retrieval
```

and fuse their results.

## 4.7 Explainability

The system must be capable of showing:

> Why are these two pieces of evidence connected?

Example:

```text
92% relationship confidence

Temporal overlap: 1.0

Shared entities:
Redis
PostgreSQL
```

---

# 5. Priority Scope

## P0 — Must Work

These are non-negotiable:

- video ingestion;
- audio ingestion;
- image ingestion;
- PDF ingestion;
- FFmpeg processing;
- timestamped transcription;
- speaker attribution for demo corpus;
- video keyframes;
- OCR;
- PDF extraction;
- EvidenceNode schema;
- EvidenceEdge schema;
- entity normalization;
- temporal relationships;
- entity relationships;
- text retrieval;
- entity retrieval;
- graph expansion;
- provenance;
- baseline RAG;
- 6+ evaluation questions;
- evaluation metrics;
- minimal frontend;
- cached demo.

## P1 — Only After P0 Works

- CLIP visual search if not already stable;
- semantic relationship edges;
- external LLM answer generation;
- relationship explanation captions;
- confidence visualization;
- clickable timestamp seeking;
- derived semantic events.

## P2 — Do Not Build Unless Everything Else Is Finished

- Neo4j;
- advanced temporal knowledge graphs;
- custom models;
- custom OCR;
- custom ASR;
- learned relationship classifiers;
- sophisticated entity linking;
- authentication;
- cloud storage;
- permissions;
- distributed workers;
- multi-tenancy.

---

# 6. Frozen Technology Stack

## Frontend

```text
React
Vite
Tailwind CSS
Axios
```

## Backend

```text
Python 3.11
FastAPI
Pydantic
```

## Media

```text
FFmpeg
```

## Speech

```text
faster-whisper
pyannote.audio
```

## OCR

Primary:

```text
PaddleOCR
```

Fallback:

```text
Tesseract / pytesseract
```

## PDFs

```text
PyMuPDF
```

## Text Embeddings

```text
sentence-transformers
all-MiniLM-L6-v2
```

## Visual Embeddings

```text
CLIP
```

CLIP is used only for the **visual retrieval channel**.

It is not our universal knowledge representation.

## Structured Storage

```text
SQLite
```

Tables:

```text
sources
processing_jobs
evidence_nodes
evidence_edges
```

## Vector Store

```text
ChromaDB
```

Collections:

```text
text_evidence

visual_evidence

baseline_text
```

## Graph Traversal

```text
NetworkX
```

## Files

```text
Local filesystem
```

## Final Answer

P0:

```text
Deterministic evidence formatter
```

P1:

```text
One optional external LLM provider
```

The core project must function with:

```text
ENABLE_EXTERNAL_LLM=false
```

---

# 7. Manual Setup Required

Some things cannot simply be delegated to Codex.

## Local software

Install and verify:

```text
Python 3.11

Node.js

Git

FFmpeg
```

## Models

Pre-download/test where hackathon rules allow:

```text
faster-whisper small

pyannote model

PaddleOCR weights

CLIP

all-MiniLM-L6-v2
```

## Hugging Face

A human must:

1. sign in;
2. accept any required pyannote conditions;
3. generate an HF token;
4. place it inside `.env`;
5. verify diarization once.

```text
HF_TOKEN=...
```

## Demo Dataset

Prepare:

```text
meeting.mp4

architecture.png

architecture.pdf
```

Optional:

```text
meeting.wav
```

Meeting requirements:

```text
5–7 minutes

2+ speakers

one clear technical decision

architecture diagram shown while discussing it
```

The same concept must occur in:

```text
speech
+
visual
+
document
```

---

# 8. Repository Structure

```text
multimodal-evidence-rag/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │
│   │   ├── api/
│   │   │   ├── ingest.py
│   │   │   ├── jobs.py
│   │   │   ├── evidence.py
│   │   │   ├── query.py
│   │   │   └── evaluation.py
│   │
│   │   ├── ingestion/
│   │   │   ├── video.py
│   │   │   ├── audio.py
│   │   │   ├── image.py
│   │   │   └── pdf.py
│   │
│   │   ├── extraction/
│   │   │   ├── transcription.py
│   │   │   ├── diarization.py
│   │   │   ├── frames.py
│   │   │   ├── ocr.py
│   │   │   ├── pdf_blocks.py
│   │   │   └── entities.py
│   │
│   │   ├── evidence/
│   │   │   ├── models.py
│   │   │   ├── builder.py
│   │   │   ├── normalization.py
│   │   │   └── relationships.py
│   │
│   │   ├── retrieval/
│   │   │   ├── text_search.py
│   │   │   ├── visual_search.py
│   │   │   ├── entity_search.py
│   │   │   ├── fusion.py
│   │   │   ├── graph_expansion.py
│   │   │   └── evidence_bundle.py
│   │
│   │   ├── baseline/
│   │   │   └── text_rag.py
│   │
│   │   ├── generation/
│   │   │   ├── interface.py
│   │   │   ├── deterministic.py
│   │   │   └── external_llm.py
│   │
│   │   ├── db/
│   │   │   ├── sqlite.py
│   │   │   ├── chroma.py
│   │   │   └── schema.sql
│   │
│   │   ├── cache/
│   │   │   └── manager.py
│   │
│   │   └── config.py
│   │
│   └── tests/
│
├── frontend/
│
├── data/
│   ├── uploads/
│   ├── audio/
│   ├── frames/
│   ├── cache/
│   ├── chroma/
│   └── demo/
│
├── evaluation/
│   ├── questions.json
│   └── results/
│
├── config/
│   └── entity_aliases.json
│
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── SCHEMA.md
│   ├── TASKS.md
│   ├── DEMO.md
│   └── FUTURE.md
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 9. Environment Variables

```text
HF_TOKEN=

DATA_DIR=./data

SQLITE_PATH=./data/app.db

CHROMA_PATH=./data/chroma

WHISPER_MODEL=small

DEVICE=cpu

ENABLE_DIARIZATION=true

ENABLE_PADDLE_OCR=true

ENABLE_CLIP=true

ENABLE_SEMANTIC_EDGES=false

ENABLE_EXTERNAL_LLM=false

EXTERNAL_LLM_API_KEY=

EXTERNAL_LLM_MODEL=
```

---

# 10. Data Models

## 10.1 Source

```json
{
  "id": "src_001",
  "filename": "meeting.mp4",
  "media_type": "video",
  "original_path": "data/uploads/meeting.mp4",
  "sha256": "...",
  "created_at": "...",
  "metadata": {
    "duration_sec": 420.3
  }
}
```

---

## 10.2 EvidenceNode

```json
{
  "id": "ev_101",
  "source_id": "src_001",
  "modality": "speech",
  "content": "We should place Redis in front of PostgreSQL.",
  "start_sec": 252.4,
  "end_sec": 264.1,
  "speaker": "SPEAKER_02",
  "page_number": null,
  "bbox": null,
  "frame_path": null,
  "entities": [
    "redis",
    "postgresql"
  ],
  "confidence": 0.94,
  "provenance": {
    "file": "meeting.mp4",
    "locator_type": "time_range",
    "start_sec": 252.4,
    "end_sec": 264.1
  },
  "text_vector_id": "txt_ev_101",
  "visual_vector_id": null
}
```

Allowed modality values:

```text
speech

audio_segment

video_frame

image

ocr_text

pdf_text
```

---

## 10.3 EvidenceEdge

```json
{
  "id": "edge_001",
  "source_node_id": "ev_101",
  "target_node_id": "ev_150",
  "relationship_type": "TEMPORAL_OVERLAP",
  "confidence": 0.96,
  "signal_breakdown": {
    "temporal": 1.0,
    "entity": 0.8,
    "semantic": null
  },
  "explanation": "Frame occurs during the speech segment and both mention Redis."
}
```

P0 relationships:

```text
TEMPORAL_OVERLAP

ENTITY_OVERLAP
```

P1:

```text
SEMANTIC_SIMILARITY
```

---

# 11. Evidence Units

## Speech

Do not use arbitrary 30-second windows.

Use:

```text
Whisper segment
```

or merge adjacent segments when:

```text
same speaker

gap <= 1.5 seconds
```

## Video

Extract meaningful keyframes.

Target for 5–7 minute video:

```text
5–20 frames
```

Every frame stores:

```text
timestamp

image path

OCR

entities

visual embedding
```

## PDF

One node per:

```text
page block
```

or meaningful paragraph group.

Store:

```text
page

bbox

text
```

## Image

Store:

```text
image evidence

OCR

entities

visual vector
```

---

# 12. Extraction Workflow

## Video

```text
meeting.mp4
      ↓
    FFmpeg
      ↓
┌─────┴─────────┐
↓               ↓
audio.wav     keyframes
                ↓
             frames.json
```

Audio should be:

```text
mono

16 kHz
```

---

# 13. Speech Pipeline

```text
audio.wav
    ↓
faster-whisper
    ↓
timestamped transcript
    ↓
pyannote
    ↓
speaker intervals
    ↓
interval matching
    ↓
speaker-attributed speech evidence
```

Cache:

```text
transcript.json

diarization.json
```

---

# 14. Functional Requirements

## FR-01 — Multimodal Upload

Support:

```text
MP4

WAV / MP3

PNG / JPEG

PDF
```

Acceptance:

- supported file creates `Source`;
- unsupported file returns clear error.

---

## FR-02 — Video Decomposition

System extracts:

```text
audio

keyframes

timestamps
```

from video.

Acceptance:

```text
audio.wav exists

frames.json exists

each frame has timestamp
```

---

## FR-03 — Speech Transcription

System generates timestamped speech segments.

Each contains:

```text
start

end

text

confidence if available
```

---

## FR-04 — Speaker Attribution

Prepared demo corpus must contain speaker-attributed transcript evidence.

Acceptance:

```text
EvidenceNode.speaker != null
```

for demo speech segments.

---

## FR-05 — OCR

System extracts visible text from:

```text
images

video frames
```

and preserves source/frame provenance.

---

## FR-06 — PDF Extraction

System extracts PDF text blocks.

Acceptance:

Every PDF evidence item contains:

```text
page number
```

and bbox when available.

---

## FR-07 — Evidence Representation

Every meaningful extracted observation becomes an EvidenceNode.

Every node preserves:

```text
content

modality

source

confidence

provenance
```

---

## FR-08 — Entity Normalization

System normalizes entities.

Example:

```text
Redis Cache

redis cache

caching layer
```

may resolve to:

```text
redis
```

---

## FR-09 — Temporal Relationships

Known speech/frame pairs must connect through time.

Example:

```text
speech:
252–264 sec

frame:
257 sec

→ TEMPORAL_OVERLAP
```

---

## FR-10 — Entity Relationships

Evidence items sharing meaningful normalized entities can connect.

Example:

```text
speech:
Redis, PostgreSQL

PDF:
Redis, PostgreSQL

→ ENTITY_OVERLAP
```

---

## FR-11 — Multi-Channel Retrieval

Query system supports:

```text
text retrieval

visual retrieval

entity retrieval
```

At least two channels must work during final demo.

---

## FR-12 — Graph Expansion

Retrieved seed evidence must pull related evidence through graph edges.

Example:

```text
Speech seed
    ↓
Frame
    ↓
PDF
```

---

## FR-13 — Provenance

Every result remains traceable.

Speech:

```text
file + timestamp
```

Frame:

```text
file + timestamp
```

PDF:

```text
file + page
```

Image:

```text
file
```

---

## FR-14 — Baseline Comparison

Implement normal text RAG.

Same query must be runnable against:

```text
Baseline

EvidenceGraph
```

---

## FR-15 — Explainable Connections

Every displayed relationship shows:

```text
relationship type

confidence

signal breakdown
```

---

## FR-16 — Cached Demo

Processed demo data can be loaded without rerunning:

```text
Whisper

pyannote

OCR

embeddings
```

---

## FR-17 — LLM Independence

If external LLM is disabled:

```text
query still works
```

and evidence is still returned.

---

## FR-18 — Evaluation Export

Save evaluation results as:

```text
JSON

CSV
```

---

# 15. Non-Goals

We explicitly will **not** build:

- custom ASR model;
- custom OCR model;
- custom vision model;
- custom embedding model;
- custom vector database;
- production authentication;
- permissions;
- organizations;
- billing;
- cloud object storage;
- production queues;
- distributed processing;
- Neo4j;
- every file type;
- scanned-PDF OCR;
- human voice identification;
- production-scale entity resolution;
- large-scale indexing;
- polished SaaS dashboard;
- mobile app.

---

# 16. Entity Extraction

P0 must not depend on an LLM.

Use:

```text
spaCy where useful

regex

technical tokens

domain dictionary

alias map
```

Normalization:

```text
lowercase

strip whitespace

remove punctuation

standardize abbreviation

apply aliases
```

Example alias configuration:

```json
{
  "redis cache": "redis",
  "caching layer": "redis",
  "postgres": "postgresql",
  "db": "database"
}
```

---

# 17. Relationship Engine

## Temporal Score

If frame occurs inside speech interval:

```text
temporal_score = 1
```

If outside but <=15 seconds away:

```text
temporal_score =
max(0, 1 - distance / 15)
```

Otherwise:

```text
0
```

Create temporal edge if:

```text
score >= 0.5
```

---

## Entity Score

Use Jaccard similarity:

```text
intersection(A,B)
-----------------
union(A,B)
```

Edge requirements:

```text
at least one meaningful shared entity

AND

entity_score >= 0.20
```

---

## P0 Relationship Confidence

Weights:

```text
Temporal = 0.60

Entity = 0.40
```

Only use signals that are applicable.

Renormalize available weights.

---

## P1 Semantic Relationship

Optional:

```text
Temporal = 0.50

Entity = 0.30

Semantic = 0.20
```

Do not build until P0 works.

---

# 18. Retrieval

## Step 1 — Query Preparation

Create:

```text
MiniLM query vector

CLIP text vector

normalized query entities
```

---

## Step 2 — Independent Retrieval

### Text Channel

Search:

```text
speech

OCR

PDF text
```

through MiniLM.

### Visual Channel

Search:

```text
frames

images
```

using CLIP.

### Entity Channel

Search SQLite for normalized entity overlap.

---

## Step 3 — Candidate Fusion

For each node:

```text
seed_score =
max(
  text_score,
  visual_score,
  entity_score
)
```

Do not average a visual item downward because no text score exists.

---

## Step 4 — Graph Expansion

Use NetworkX.

Default:

```text
1 hop

max 2 hops

minimum edge confidence = 0.55
```

---

## Step 5 — Reranking

Suggested:

```text
0.55 × seed relevance

0.30 × connection confidence

0.10 × provenance quality

0.05 × modality coverage
```

---

## Step 6 — Evidence Bundle

Prefer:

```text
one useful speech result

one useful visual

one useful document
```

over:

```text
five almost identical transcript chunks
```

---

# 19. Baseline RAG

```text
Transcript
+
OCR
+
PDF text
    ↓
Flatten
    ↓
Fixed chunks
    ↓
MiniLM
    ↓
Vector search
```

No:

```text
relationships

graph expansion

speaker-aware ranking

frame-aware ranking
```

---

# 20. Evaluation

Create at least:

```text
6 questions
```

Recommended:

```text
1 single-modality

2 two-modality

2 three-modality

1 provenance/speaker adversarial
```

## Metrics

### Evidence Recall@5

How much required evidence appears in the first five results?

### Complete Evidence Rate

Did the system retrieve **all** evidence required by the question?

### Provenance Completeness

Do returned results contain usable source locators?

---

# 21. API

## GET

```text
/api/health
```

## POST

```text
/api/ingest
```

## GET

```text
/api/jobs/{job_id}
```

## GET

```text
/api/sources
```

## GET

```text
/api/evidence
```

Filters:

```text
source_id

modality

entity

min_confidence
```

## GET

```text
/api/evidence/{id}
```

Returns:

```text
node

edges

neighbors
```

## POST

```text
/api/query
```

Request:

```json
{
  "question": "...",
  "top_k": 5,
  "graph_hops": 1
}
```

## POST

```text
/api/baseline/query
```

## POST

```text
/api/evaluate
```

## POST

```text
/api/demo/load
```

---

# 22. Frontend

Do not make four complex pages.

Build one application with three sections.

## Dataset

Show:

```text
sources

processing state

Load Demo Dataset
```

## Ask

```text
Question input

Run EvidenceGraph

Baseline toggle
```

Answer shows:

### Speech

```text
Speaker

Transcript

Timestamp

Source
```

### Visual

```text
Image

OCR

Timestamp

Source
```

### Document

```text
Excerpt

Page

Source
```

## Evidence Explorer

Selecting a result shows:

```text
node

connected nodes

relationship type

confidence

signal breakdown

why they connect
```

---

# 23. Reliability

Cache all expensive outputs.

Structure:

```text
data/cache/{source_id}/

source.json

audio.wav

transcript.json

diarization.json

frames.json

frames/

ocr.json

pdf_blocks.json

evidence.json

edges.json

manifest.json
```

`manifest.json` contains:

```text
source hash

pipeline version

models used

completed stages
```

---

# 24. Failure Fallbacks

## pyannote

Fallback:

```text
cached precomputed diarization
```

Be transparent that cached fixture metadata is being used.

## PaddleOCR

Fallback:

```text
Tesseract
```

## CLIP

Disable visual-vector retrieval.

Still retain:

```text
OCR retrieval

entity relationships

temporal links
```

## External LLM

Use deterministic evidence response.

## Chroma

Fallback:

```text
NumPy cosine similarity
```

for the small demo corpus.

## Video Pipeline

Load cache.

Keep a tiny 20–30 second file for optional live processing demonstration.

---

# 25. Demo Dataset

Example technical decision:

```text
Speaker:

"We should place Redis in front of PostgreSQL as a read-through caching layer."
```

At the same moment:

```text
Client
 ↓
API
 ↓
Redis
 ↓
PostgreSQL
```

PDF contains:

```text
Read-through caching architecture
```

---

# 26. Flagship Demo

Question:

> What architecture was proposed to reduce database load, who explained it, and where was the design shown?

Expected:

```text
Answer:
Redis read-through caching.

Speech:
SPEAKER_02
04:12–04:24

Frame:
meeting.mp4
04:17

PDF:
architecture.pdf
Page 8
```

Relationships:

```text
Speech
  │
  ├─ TEMPORAL_OVERLAP
  ↓
Frame
  │
  ├─ ENTITY_OVERLAP
  ↓
PDF
```

---

# 27. Why This Stands Out

## Semantic Evidence Units

We do not build the core graph around arbitrary fixed-length video chunks.

## Separate Retrieval Channels

We do not expect one embedding model to understand all modalities perfectly.

## Explicit Evidence Graph

Relationships are data.

They are not invented at answer time.

## Deterministic Indexing

Core graph construction does not require costly LLM extraction.

## Inspectable Confidence

Judges can see why evidence was linked.

## Provenance by Design

Timestamp/page/frame provenance exists from ingestion onward.

## Actual Evaluation

We compare against normal RAG instead of simply claiming ours is better.

---

# 28. 13-Hour Plan

## Hour 0–1

Smoke test:

```text
FFmpeg

Whisper

pyannote

OCR

PyMuPDF

MiniLM

CLIP
```

## Hour 1–3

Build extraction.

## Hour 3–4

Build EvidenceNode + SQLite + cache.

## Hour 4–6

Build relationships.

## Hour 6–8

Build retrieval + graph expansion.

## Hour 8–9

Build baseline.

## Hour 9–10

Build evaluation.

## Hour 10–11

Build frontend.

## Hour 11–11:30

Freeze unstable features.

Cache final corpus.

## Hour 11:30–12:15

README + architecture + future improvements.

## Hour 12:15–13

Demo rehearsal.

No new features.

---

# 29. Definition of Done

Before judging:

### Ingestion

```text
✓ MP4 → audio + frames

✓ Audio → transcript

✓ Image → OCR

✓ PDF → page evidence
```

### Representation

```text
✓ every node has source

✓ every node has modality

✓ provenance populated

✓ timestamps/pages preserved
```

### Relationships

```text
✓ speech ↔ frame temporal edge

✓ speech/frame ↔ PDF entity edge

✓ confidence visible
```

### Retrieval

```text
✓ query retrieves multiple modalities

✓ graph expansion retrieves evidence that vector search alone did not directly return
```

### Baseline

```text
✓ same questions run through text RAG

✓ metrics generated
```

### Reliability

```text
✓ demo works from cache

✓ works without external LLM

✓ works if CLIP is disabled
```

---

# 30. Deliverables

Before submitting:

- working repo;
- README;
- multimodal ingestion pipeline;
- EvidenceNode representation;
- EvidenceEdge representation;
- retrieval system;
- evidence query UI;
- 3+ modality demo;
- baseline RAG;
- evaluation JSON/CSV;
- architecture explanation;
- key design decisions;
- future improvements;
- cached demo outputs.

---

# 31. Production Roadmap

After the hackathon:

## Storage

```text
SQLite → PostgreSQL

local files → object storage

local Chroma → pgvector / managed vector DB
```

## Processing

Add:

```text
queue

workers

incremental ingestion

observability

deduplication
```

## Entity Intelligence

Replace alias maps with:

```text
entity linking

entity clustering

organization dictionaries
```

## Relationships

Add:

```text
semantic

support

contradiction

decision

causal

change-over-time
```

## Retrieval

Add:

```text
query decomposition

learned reranking

confidence calibration

evidence diversity
```

## Target Markets

Do not initially position this as another generic meeting-note product.

Focus on use cases where provenance matters:

```text
engineering architecture decisions

compliance

audit

incident investigations

legal/deposition review

research evidence tracking
```

---

# 32. Agent Build Contract

Codex and Antigravity must follow:

1. Read this PRD before coding.
2. Work on one module at a time.
3. Do not replace the frozen stack.
4. Do not add cloud infrastructure.
5. Do not build authentication.
6. Do not start P1 while P0 tests are failing.
7. Add a test/smoke script for each module.
8. Cache all expensive outputs.
9. Keep provenance structured.
10. Never make the LLM the retrieval engine.
11. Do not silently change API contracts.
12. Use documented fallbacks instead of redesigning the project.
13. A feature is not complete until its acceptance test passes.

---

# 33. First Four Codex / Antigravity Tasks

## TASK 01 — Project Skeleton + FFmpeg

Implement only:

```text
FastAPI

config

/api/health

video upload

audio extraction

keyframe extraction

cache directory

tests
```

Do **not** implement:

```text
Whisper

OCR

DB

RAG

frontend
```

---

## TASK 02 — Speech

Implement:

```text
faster-whisper

pyannote

transcript.json

diarization.json

speaker merge

tests
```

---

## TASK 03 — Visual + PDF + Evidence

Implement:

```text
OCR

PDF extraction

EvidenceNode

SQLite

provenance

tests
```

---

## TASK 04 — Relationships

Implement:

```text
entity normalization

temporal edges

entity edges

confidence

signal_breakdown

NetworkX graph

tests
```

**Do not start retrieval until Task 04 works.**

---

# 34. Final Product Principle

A judge should be able to ask:

> What architecture was proposed to reduce database load, who explained it, and where was the design shown?

and immediately receive:

```text
correct speech

speaker

timestamp

frame

PDF page

relationship between them

confidence

why they connect

original provenance
```

while you can immediately switch to the text-only baseline and demonstrate what context it failed to recover.

**Evidence first.  
Relationships second.  
Retrieval third.  
UI last.**