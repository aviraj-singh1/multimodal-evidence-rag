"""Enrich evidence entities and build the deterministic corpus graph."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.cache.manager import PIPELINE_VERSION, write_json
from backend.app.db.sqlite import EvidenceStore
from backend.app.evidence.entities import enrich_entities, load_entity_config
from backend.app.evidence.models import EvidenceNode
from backend.app.evidence.repository import EvidenceRepository
from backend.app.evidence.relationships import build_relationships
from backend.app.graph.loader import load_graph


CACHE_ROOT = REPO_ROOT / "data" / "cache"
def main() -> int:
    original = EvidenceRepository(REPO_ROOT / "data" / "app.db").list_evidence_nodes()
    immutable = {node.id: (node.content, node.provenance, node.confidence, node.text_vector_id, node.visual_vector_id)
                 for node in original}
    config = load_entity_config(REPO_ROOT / "config")
    nodes = enrich_entities(original, config)
    edges, excluded = build_relationships(nodes, config.common_entity_threshold)
    for node in nodes:
        if immutable[node.id] != (node.content, node.provenance, node.confidence, node.text_vector_id, node.visual_vector_id):
            raise RuntimeError(f"entity enrichment altered immutable evidence: {node.id}")
    store = EvidenceStore(REPO_ROOT / "data" / "app.db")
    store.replace_graph(nodes, edges)
    first_ids = store.edge_ids()
    store.replace_graph(nodes, edges)
    idempotent = first_ids == store.edge_ids() == sorted(edge.id for edge in edges)
    by_source: dict[str, list[EvidenceNode]] = {}
    for node in nodes:
        by_source.setdefault(node.source_id, []).append(node)
    for source_id, source_nodes in by_source.items():
        write_json(CACHE_ROOT / source_id / "evidence.json", {
            "source_id": source_id, "pipeline_version": PIPELINE_VERSION,
            "evidence_count": len(source_nodes),
            "nodes": [node.model_dump(mode="json") for node in source_nodes],
        })
    graph_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "node_count": len(nodes), "edge_count": len(edges),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "edges": [edge.model_dump(mode="json") for edge in edges],
    }
    write_json(CACHE_ROOT / "graph" / "edges.json", graph_payload)
    graph = load_graph(nodes, store.load_edges())
    entity_frequency = Counter(entity for node in nodes for entity in set(node.entities))
    primary = Counter(edge.relationship_type for edge in edges)
    both = sum(edge.signal_breakdown.temporal is not None and edge.signal_breakdown.entity is not None for edge in edges)
    node_by_id = {node.id: node for node in nodes}
    cross_modal = sum(node_by_id[e.source_node_id].modality != node_by_id[e.target_node_id].modality for e in edges)
    cross_source = sum(node_by_id[e.source_node_id].source_id != node_by_id[e.target_node_id].source_id for e in edges)
    print("EVIDENCE GRAPH BUILD\n")
    print(f"EvidenceNodes: {len(nodes)}")
    print(f"Nodes with entities: {sum(bool(node.entities) for node in nodes)}")
    print(f"Canonical entities: {len(entity_frequency)}")
    print(f"Excluded common entities: {', '.join(sorted(excluded)) or 'none'}")
    print(f"Edges: {len(edges)}")
    print(f"TEMPORAL_OVERLAP: {primary['TEMPORAL_OVERLAP']}")
    print(f"ENTITY_OVERLAP: {primary['ENTITY_OVERLAP']}")
    print(f"Both signals: {both}")
    print(f"Cross-modal: {cross_modal}")
    print(f"Cross-source: {cross_source}")
    print(f"NetworkX: {'PASS' if graph.number_of_edges() == len(edges) else 'FAIL'}")
    print(f"SQLite: {'PASS' if store.edge_count() == len(edges) and not store.foreign_key_violations() else 'FAIL'}")
    print(f"edges.json: {'PASS' if graph_payload['edge_count'] == len(edges) else 'FAIL'}")
    print(f"Idempotency: {'PASS' if idempotent else 'FAIL'}")
    return 0 if idempotent else 1


if __name__ == "__main__":
    raise SystemExit(main())
