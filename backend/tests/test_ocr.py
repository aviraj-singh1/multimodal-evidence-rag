"""Focused schema, provenance, cache, and error tests for OCR outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.app.cache.manager import read_json
from backend.app.extraction.ocr import (
    ContentExtractionError,
    extract_image_ocr,
    extract_video_frame_ocr,
    validate_image_ocr,
    validate_video_ocr,
)


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "data" / "cache"
VIDEO_CACHE = CACHE_ROOT / "src_1347fcb63e98"
IMAGE_IDS = ["src_9899772d9cc4", "src_66bf6ebf0dbc"]


class OCRTests(unittest.TestCase):
    def test_video_ocr_schema_and_exact_frame_provenance(self) -> None:
        payload = read_json(VIDEO_CACHE / "ocr.json")
        frames = read_json(VIDEO_CACHE / "frames.json")
        validate_video_ocr(payload, VIDEO_CACHE, frames)
        manifest_by_id = {frame["frame_id"]: frame for frame in frames["frames"]}
        self.assertEqual(len(payload["frames"]), 15)
        for frame in payload["frames"]:
            expected = manifest_by_id[frame["frame_id"]]
            self.assertEqual(frame["timestamp_sec"], expected["timestamp_sec"])
            self.assertTrue((VIDEO_CACHE / frame["frame_path"]).is_file())

    def test_ocr_regions_have_text_confidence_and_valid_bbox(self) -> None:
        payload = read_json(VIDEO_CACHE / "ocr.json")
        regions = [region for frame in payload["frames"] for region in frame["regions"]]
        self.assertTrue(regions)
        for region in regions:
            self.assertTrue(region["text"].strip())
            self.assertIsInstance(region["confidence"], (int, float))
            self.assertGreaterEqual(region["confidence"], 0)
            self.assertLessEqual(region["confidence"], 1)
            self.assertEqual(len(region["bbox"]), 4)

    def test_standalone_images_preserve_source(self) -> None:
        for source_id in IMAGE_IDS:
            cache = CACHE_ROOT / source_id
            source = read_json(cache / "source.json")
            payload = read_json(cache / "ocr.json")
            validate_image_ocr(payload)
            self.assertEqual(payload["source_id"], source_id)
            self.assertEqual(payload["source_sha256"], source["sha256"])
            self.assertIn(payload["filename"], {"architecture_01.png", "architecture_02.png"})
            self.assertTrue(payload["regions"])

    def test_ocr_cache_reuse_without_engine(self) -> None:
        video = extract_video_frame_ocr(VIDEO_CACHE)
        self.assertEqual(video.cache_status, "REUSED")
        original = ROOT.parent / "demo_dataset" / "architecture_02.png"
        image = extract_image_ocr(original, CACHE_ROOT)
        self.assertEqual(image.cache_status, "REUSED")

    def test_corrupt_and_unsupported_image_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corrupt = root / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            with self.assertRaisesRegex(ContentExtractionError, "invalid or corrupted image"):
                extract_image_ocr(corrupt, root / "cache")
            unsupported = root / "image.gif"
            unsupported.write_bytes(b"GIF89a")
            with self.assertRaisesRegex(ValueError, "unsupported image extension"):
                extract_image_ocr(unsupported, root / "cache")


if __name__ == "__main__":
    unittest.main()

