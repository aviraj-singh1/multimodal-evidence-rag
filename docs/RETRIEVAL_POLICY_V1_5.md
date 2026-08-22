# Retrieval Policy V1.5

This post-Milestone-7 policy is frozen. It was selected using generic development
queries and policy tests before the Q1-Q8 regression measurement.

- Preserve at most 2 direct candidates with normalized relevance at least 0.25.
- Propagate graph relevance as `seed score * edge confidence * 0.8^hop_count`.
- Admit one complementary graph modality only when edge confidence is at least
  0.70 and propagated relevance is at least 0.15.
- Rank direct evidence primarily by relevance; graph evidence cannot receive a
  large provenance-only boost. Modality diversity is secondary and is expressed
  by the single qualified complementary slot rather than forced family coverage.
- Suppress final-selection duplicates deterministically: at most 2 OCR regions
  per frame, at most 2 PDF blocks per page, and near-identical speech within a
  20-second window at token Jaccard similarity of at least 0.65.

The machine-readable configuration is `config/retrieval_policy.json`.
