"""Focused PDF block schema, provenance, cache, and error tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.cache.manager import read_json
from backend.app.extraction.ocr import ContentExtractionError
from backend.app.extraction.pdf_blocks import extract_pdf_blocks, validate_pdf_blocks


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "data" / "cache"
PDF_CACHE = CACHE_ROOT / "src_00b5ee2f232b"
PDF_FIXTURE = ROOT / "demo_test" / "rate_limiter_design.pdf"


class PDFExtractionTests(unittest.TestCase):
    def test_page_count_numbers_blocks_and_bboxes(self) -> None:
        payload = read_json(PDF_CACHE / "pdf_blocks.json")
        validate_pdf_blocks(payload)
        self.assertEqual(payload["page_count"], 2)
        self.assertEqual([page["page_number"] for page in payload["pages"]], [1, 2])
        for page in payload["pages"]:
            self.assertTrue(page["text"].strip())
            for block in page["blocks"]:
                self.assertTrue(block["text"].strip())
                self.assertTrue(block["block_id"].startswith(f"page_{page['page_number']:03d}_"))
                self.assertEqual(len(block["bbox"]), 4)

    def test_known_demo_content_is_extracted(self) -> None:
        payload = read_json(PDF_CACHE / "pdf_blocks.json")
        text = "\n".join(page["text"] for page in payload["pages"]).lower()
        self.assertIn("rate limiter architecture", text)
        self.assertIn("rules cache", text)
        self.assertIn("high-throughput cache", text)
        self.assertIn("429", text)

    def test_pdf_cache_reuse(self) -> None:
        result = extract_pdf_blocks(PDF_FIXTURE, CACHE_ROOT)
        self.assertEqual(result.cache_status, "REUSED")

    def test_invalid_pdf_is_handled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "corrupt.pdf"
            corrupt.write_bytes(b"not a pdf")
            with self.assertRaisesRegex(ContentExtractionError, "invalid or corrupted PDF"):
                extract_pdf_blocks(corrupt, root / "cache")


if __name__ == "__main__":
    unittest.main()
