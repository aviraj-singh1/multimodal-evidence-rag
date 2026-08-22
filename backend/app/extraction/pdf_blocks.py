"""Structured text-block extraction from text-readable PDFs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf

from backend.app.cache.manager import (
    PIPELINE_VERSION,
    read_json,
    sha256_file,
    source_id_for_hash,
    write_json,
)
from backend.app.extraction.ocr import ContentExtractionError, ExtractionResult


PDF_ENGINE = "PyMuPDF"
PDF_CONFIGURATION = {"blocks_sort": True, "page_numbers": "1-based"}


def extract_pdf_blocks(pdf_path: Path, cache_root: Path) -> ExtractionResult:
    pdf = pdf_path.expanduser().resolve()
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF does not exist: {pdf}")
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"unsupported PDF extension {pdf.suffix!r}; expected .pdf")
    sha256 = sha256_file(pdf)
    source_id = source_id_for_hash(sha256)
    cache_dir = cache_root.resolve() / source_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        document = pymupdf.open(pdf)
    except Exception as exc:
        raise ContentExtractionError(f"invalid or corrupted PDF {pdf}: {exc}") from exc
    try:
        if document.page_count <= 0:
            raise ContentExtractionError(f"PDF contains no pages: {pdf}")
        source = {
            "source_id": source_id,
            "filename": pdf.name,
            "source_type": "pdf",
            "sha256": sha256,
            "page_count": document.page_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": PIPELINE_VERSION,
        }
        output = cache_dir / "pdf_blocks.json"
        if _pdf_cache_valid(output, source):
            return ExtractionResult(source_id, cache_dir, read_json(output), "REUSED")
        pages: list[dict[str, Any]] = []
        for page_index, page in enumerate(document, start=1):
            blocks: list[dict[str, Any]] = []
            for raw in page.get_text("blocks", sort=True):
                text = str(raw[4]).strip()
                if not text:
                    continue
                bbox = [round(float(value), 3) for value in raw[:4]]
                blocks.append(
                    {
                        "block_id": f"page_{page_index:03d}_block_{len(blocks) + 1:03d}",
                        "bbox": bbox,
                        "text": text,
                    }
                )
            pages.append(
                {
                    "page_number": page_index,
                    "width": round(float(page.rect.width), 3),
                    "height": round(float(page.rect.height), 3),
                    "text": page.get_text("text", sort=True).strip(),
                    "blocks": blocks,
                }
            )
        payload = {
            "source_id": source_id,
            "source_sha256": sha256,
            "filename": pdf.name,
            "source_type": "pdf",
            "pipeline_version": PIPELINE_VERSION,
            "engine": PDF_ENGINE,
            "configuration": PDF_CONFIGURATION,
            "page_count": document.page_count,
            "pages": pages,
        }
        validate_pdf_blocks(payload)
        write_json(cache_dir / "source.json", source)
        write_json(output, payload)
        write_json(
            cache_dir / "manifest.json",
            {
                "source_id": source_id,
                "sha256": sha256,
                "pipeline_version": PIPELINE_VERSION,
                "completed_stages": ["pdf_blocks"],
                "content": {
                    "pdf_blocks": {"engine": PDF_ENGINE, "configuration": PDF_CONFIGURATION}
                },
            },
        )
        return ExtractionResult(source_id, cache_dir, payload, "CREATED")
    finally:
        document.close()


def validate_pdf_blocks(payload: dict[str, Any]) -> None:
    pages = payload.get("pages")
    if not isinstance(pages, list) or len(pages) != payload.get("page_count"):
        raise ContentExtractionError("PDF pages do not match page_count")
    for expected_page, page in enumerate(pages, start=1):
        if page.get("page_number") != expected_page:
            raise ContentExtractionError("PDF page numbers must be sequential and 1-based")
        width, height = page.get("width"), page.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise ContentExtractionError("PDF page dimensions must be numeric")
        for block in page.get("blocks", []):
            if not isinstance(block.get("text"), str) or not block["text"].strip():
                raise ContentExtractionError("PDF blocks must contain non-empty text")
            bbox = block.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ContentExtractionError("PDF block bbox must have four coordinates")
            if not all(isinstance(value, (int, float)) for value in bbox):
                raise ContentExtractionError("PDF block bbox coordinates must be numeric")
            x0, y0, x1, y1 = bbox
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ContentExtractionError(f"PDF block bbox outside page bounds: {bbox}")


def _pdf_cache_valid(path: Path, source: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = read_json(path)
        validate_pdf_blocks(payload)
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        payload.get("source_id") == source.get("source_id")
        and payload.get("source_sha256") == source.get("sha256")
        and payload.get("pipeline_version") == PIPELINE_VERSION
        and payload.get("engine") == PDF_ENGINE
        and payload.get("configuration") == PDF_CONFIGURATION
        and payload.get("page_count") == source.get("page_count")
    )
