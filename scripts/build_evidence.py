"""Build and persist canonical evidence for extracted source caches."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.cache.manager import write_json
from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.builder import (
    build_evidence,
    discover_evidence_sources,
    evidence_payload,
    modality_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="build every evidence-ready cache")
    selection.add_argument("--source", help="build one deterministic source ID")
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "data" / "cache")
    parser.add_argument("--database", type=Path, default=REPO_ROOT / "data" / "app.db")
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT.parent / "demo_dataset")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_root = args.cache_root.resolve()
    source_ids = discover_evidence_sources(cache_root) if args.all else [args.source]
    if not source_ids:
        raise RuntimeError("no evidence-ready sources found")
    store = EvidenceStore(args.database.resolve())
    total_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    idempotent = True
    json_consistent = True
    for source_id in source_ids:
        cache_dir = cache_root / source_id
        if source_id not in discover_evidence_sources(cache_root):
            raise RuntimeError(f"source is not evidence-ready: {source_id}")
        source, nodes = build_evidence(cache_dir, REPO_ROOT, args.dataset_dir.resolve())
        store.replace_source(source, nodes)
        first_ids = store.evidence_ids(source.id)
        store.replace_source(source, nodes)
        idempotent &= first_ids == store.evidence_ids(source.id) and len(first_ids) == len(nodes)
        payload = evidence_payload(source.id, nodes)
        write_json(cache_dir / "evidence.json", payload)
        json_consistent &= payload["evidence_count"] == store.evidence_count(source.id)
        counts = modality_counts(nodes)
        source_counts[source.id] = counts
        total_counts.update(counts)
    if store.foreign_key_violations():
        raise RuntimeError("SQLite foreign-key validation failed")
    print("EVIDENCE BUILD\n")
    print(f"Sources: {len(source_ids)}")
    print("\nEvidence:")
    for modality in ("speech", "video_frame", "ocr_text", "image", "pdf_text", "audio_segment"):
        if total_counts[modality]:
            print(f"  {modality}: {total_counts[modality]}")
    print(f"\nTotal: {sum(total_counts.values())}")
    print("\nPer source:")
    for source_id, counts in source_counts.items():
        details = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"  {source_id}: {sum(counts.values())} ({details})")
    print(f"\nSQLite: {'PASS' if not store.foreign_key_violations() else 'FAIL'}")
    print(f"evidence.json: {'PASS' if json_consistent else 'FAIL'}")
    print(f"Idempotency: {'PASS' if idempotent else 'FAIL'}")
    return 0 if json_consistent and idempotent else 1


if __name__ == "__main__":
    raise SystemExit(main())
