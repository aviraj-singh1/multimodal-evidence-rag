"""Run the complete deterministic evidence query pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.api.query import QueryService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--graph-hops", type=int, default=1)
    args = parser.parse_args()
    result = QueryService().query(args.question, args.top_k, args.graph_hops)
    print(f"QUERY\n\n{result['question']}\n\nENTITIES\n\n{', '.join(result['query_entities']) or '(none)'}")
    for name in ("text", "visual", "entity"):
        print(f"\n{name.upper()} HITS")
        for hit in result["channels"][name][:5]:
            print(f"  {hit['evidence_id']} {hit['normalized_score']:.3f}")
    print("\nSEEDS")
    for item in result["seed_evidence"]:
        print(f"  {item['evidence_id']} [{item['modality']}] {item['seed_score']:.3f}")
    print("\nGRAPH EXPANSION")
    for item in result["expanded_evidence"][:10]:
        print(f"  {item['evidence_id']} [{item['modality']}] {item['reason']}")
    print("\nFINAL EVIDENCE")
    for item in result["evidence_bundle"]:
        node = item["node"]
        print(f"  {node['id']} [{node['modality']}] score={item['final_score']:.3f} {item['reasons'][0]}")
    print(f"\nANSWER\n\n{result['answer']}\n\nLATENCY\n\n{result['latency_ms']:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
