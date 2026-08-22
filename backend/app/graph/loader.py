"""Load canonical evidence into an undirected NetworkX graph."""

from __future__ import annotations

import networkx as nx

from backend.app.evidence.models import EvidenceEdge, EvidenceNode


def load_graph(nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id, modality=node.modality, source_id=node.source_id)
    for edge in edges:
        if edge.source_node_id not in graph or edge.target_node_id not in graph:
            raise ValueError(f"edge references missing node: {edge.id}")
        graph.add_edge(
            edge.source_node_id, edge.target_node_id,
            relationship_type=edge.relationship_type,
            confidence=edge.confidence,
            signal_breakdown=edge.signal_breakdown.model_dump(),
            explanation=edge.explanation,
            edge_id=edge.id,
        )
    if graph.number_of_edges() != len(edges):
        raise ValueError("NetworkX edge count differs from canonical edge count")
    return graph
