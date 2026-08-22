"""Bounded traversal over persisted evidence relationships."""

from __future__ import annotations

import networkx as nx


def expand_graph(
    graph: nx.Graph,
    seeds: list[dict[str, object]],
    hops: int = 1,
    min_confidence: float = 0.55,
    max_nodes: int = 25,
    hop_decay: float = 0.80,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not 0 <= hops <= 2:
        raise ValueError("graph hops must be between 0 and 2")
    seed_ids = {str(seed["evidence_id"]) for seed in seeds}
    discovered: dict[str, dict[str, object]] = {}
    connections: list[dict[str, object]] = []
    frontier = [(str(seed["evidence_id"]), str(seed["evidence_id"]), float(seed["seed_score"])) for seed in seeds]
    visited = set(seed_ids)
    for hop_index in range(hops):
        hop_count = hop_index + 1
        next_frontier = []
        for current, origin, origin_score in frontier:
            for neighbor in graph.neighbors(current):
                edge = graph.edges[current, neighbor]
                confidence = float(edge["confidence"])
                if confidence < min_confidence:
                    continue
                connection = {"origin_seed_id": origin, "from_id": current, "to_id": neighbor,
                              "relationship_type": edge["relationship_type"],
                              "confidence": confidence, "signal_breakdown": edge["signal_breakdown"],
                              "explanation": edge["explanation"]}
                if neighbor not in seed_ids:
                    propagated = origin_score * confidence * (hop_decay ** hop_count)
                    prior = discovered.get(neighbor)
                    if prior is None or propagated > float(prior["propagated_relevance"]):
                        discovered[neighbor] = {"evidence_id": neighbor, "origin": "graph_expanded",
                                                "origin_seed_id": origin, "origin_seed_score": origin_score,
                                                "propagated_relevance": round(propagated, 6),
                                                "connection_score": confidence, "edge_confidence": confidence,
                                                "hop_count": hop_count,
                                                "reason": f"EXPANDED from {origin}: {edge['relationship_type']} {confidence:.3f}"}
                    connections.append(connection)
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append((neighbor, origin, origin_score))
        frontier = next_frontier
    expanded = sorted(discovered.values(), key=lambda item: (-float(item["propagated_relevance"]), str(item["evidence_id"])))[:max_nodes]
    expanded_ids = {str(item["evidence_id"]) for item in expanded}
    return expanded, [item for item in connections if str(item["to_id"]) in expanded_ids]
