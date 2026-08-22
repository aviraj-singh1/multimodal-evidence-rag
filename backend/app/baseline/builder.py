"""Deterministically flatten canonical text evidence into fixed character chunks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backend.app.evidence.models import EvidenceNode, Source


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


@dataclass(frozen=True)
class BaselineChunk:
    chunk_id: str
    text: str
    source_id: str
    source_filename: str
    chunk_index: int
    member_evidence_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_chunks(nodes: list[EvidenceNode], sources: list[Source], chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP) -> list[BaselineChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk size must be positive and overlap smaller than chunk size")
    filenames = {source.id: source.filename for source in sources}
    by_source: dict[str, list[EvidenceNode]] = {}
    for node in nodes:
        if node.modality in {"speech", "ocr_text", "pdf_text"} and node.content:
            by_source.setdefault(node.source_id, []).append(node)
    chunks: list[BaselineChunk] = []
    for source_id in sorted(by_source):
        ordered = sorted(by_source[source_id], key=_order_key)
        text_parts: list[str] = []
        spans: list[tuple[int, int, str]] = []
        cursor = 0
        for node in ordered:
            separator = "\n\n" if text_parts else ""
            text_parts.append(separator + (node.content or ""))
            start = cursor + len(separator)
            end = start + len(node.content or "")
            spans.append((start, end, node.id)); cursor = end
        flattened = "".join(text_parts)
        step = chunk_size - overlap
        for index, start in enumerate(range(0, len(flattened), step), start=1):
            end = min(len(flattened), start + chunk_size)
            members = [evidence_id for span_start, span_end, evidence_id in spans
                       if span_start < end and span_end > start]
            chunks.append(BaselineChunk(
                chunk_id=f"baseline_{source_id.removeprefix('src_')}_{index:04d}",
                text=flattened[start:end], source_id=source_id,
                source_filename=filenames[source_id], chunk_index=index,
                member_evidence_ids=members,
            ))
            if end == len(flattened):
                break
    return chunks


def _order_key(node: EvidenceNode) -> tuple[float, int, str]:
    if node.start_sec is not None:
        return (node.start_sec, 0 if node.modality == "speech" else 1, node.id)
    if node.page_number is not None:
        return (1_000_000.0 + node.page_number, 0, node.id)
    return (2_000_000.0, 0, node.id)
