# Retrieval Policy V1.6

V1.6 is the final, frozen upstream candidate-recall pass. The V1.5 anchor,
propagation, duplicate-suppression, and final-selection policy is unchanged.

## Candidate generation

- MiniLM text candidates: 20
- CLIP visual candidates: 10 of the 17 available visual nodes
- Entity candidates: 20
- Fused seed candidates considered for graph expansion: 12
- Final bundle size: the caller's requested `top_k`

Candidate overfetch is independent of final result count. It increases the
evidence inspected upstream without exposing more results than requested.

## Retrieval-only context

MiniLM indexes each speech segment with its immediate previous and next speech
segments from the same source. PDF blocks use their immediate previous and next
blocks from the same source and page. OCR remains region-level and is not
globally concatenated. Chroma stores this context only as its retrieval document;
the canonical `EvidenceNode.content` and returned evidence remain raw.

No lexical layer, query rewriting, evaluation-specific vocabulary, or external
LLM was added.
