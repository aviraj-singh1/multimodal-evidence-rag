"""Modest, provenance-preserving formatter with no external LLM."""

from __future__ import annotations


def format_answer(bundle: list[dict[str, object]]) -> str:
    if not bundle:
        return "No relevant evidence was found."
    strongest = next((item for item in bundle if item["node"].get("content")), bundle[0])
    content = strongest["node"].get("content") or "a visual observation"
    lines = [f"Evidence indicates: {str(content).replace(chr(10), ' ')[:240]}", "", "Relevant evidence includes:"]
    for item in bundle:
        node = item["node"]
        provenance = node["provenance"]
        location = provenance.get("file", node["source_id"])
        if node.get("start_sec") is not None:
            if node.get("end_sec") != node.get("start_sec"):
                location += f" — {node['start_sec']:.2f}-{node['end_sec']:.2f}s"
            else:
                location += f" — {node['start_sec']:.2f}s"
        elif node.get("page_number") is not None:
            location += f" — page {node['page_number']}"
        speaker = f" {node['speaker']}" if node.get("speaker") else ""
        lines.append(f"- {node['modality']}{speaker}: {location}")
    return "\n".join(lines)
