"""Builders preserve all raw observations, confidence, and provenance."""

from __future__ import annotations

from pathlib import Path
import unittest

from backend.app.cache.manager import read_json
from backend.app.evidence.builder import build_evidence


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / "data" / "cache"
DATASET_DIR = REPO_ROOT / "demo_test"
VIDEO_ID = "src_1347fcb63e98"
IMAGE_IDS = ("src_9899772d9cc4", "src_66bf6ebf0dbc")
PDF_ID = "src_00b5ee2f232b"


def _build(source_id: str):
    return build_evidence(CACHE_ROOT / source_id, REPO_ROOT, DATASET_DIR)


class EvidenceBuilderTests(unittest.TestCase):
    def test_video_build_is_deterministic_and_preserves_speech_and_frames(self) -> None:
        _, first = _build(VIDEO_ID)
        _, second = _build(VIDEO_ID)
        self.assertEqual([node.model_dump() for node in first], [node.model_dump() for node in second])
        transcript = read_json(CACHE_ROOT / VIDEO_ID / "transcript.json")["segments"]
        raw_frames = read_json(CACHE_ROOT / VIDEO_ID / "frames.json")["frames"]
        speech = [node for node in first if node.modality == "speech"]
        frames = [node for node in first if node.modality == "video_frame"]
        self.assertEqual(len(speech), len(transcript))
        self.assertEqual(len(frames), len(raw_frames))
        for node, raw in zip(speech, transcript):
            self.assertEqual((node.content, node.start_sec, node.end_sec, node.speaker),
                             (raw["text"], raw["start_sec"], raw["end_sec"], raw["speaker"]))
            self.assertIsNone(node.confidence)
            self.assertEqual(node.provenance["transcript_segment_id"], raw["segment_id"])
            self.assertEqual(node.provenance["multi_speaker"], raw["multi_speaker"])
        for node, raw in zip(frames, raw_frames):
            self.assertEqual(node.start_sec, raw["timestamp_sec"])
            self.assertEqual(node.end_sec, raw["timestamp_sec"])
            self.assertTrue((REPO_ROOT / node.frame_path).is_file())

    def test_ocr_regions_preserve_raw_content_confidence_bbox_and_errors(self) -> None:
        _, video_nodes = _build(VIDEO_ID)
        raw_video = read_json(CACHE_ROOT / VIDEO_ID / "ocr.json")
        raw_regions = [region for frame in raw_video["frames"] for region in frame["regions"]]
        built_regions = [node for node in video_nodes if node.modality == "ocr_text"]
        self.assertEqual(len(built_regions), len(raw_regions))
        for node, raw in zip(built_regions, raw_regions):
            self.assertEqual(node.content, raw["text"])
            self.assertEqual(node.confidence, raw["confidence"])
            self.assertEqual(node.bbox, raw["bbox"])
        self.assertIn("AI Rate limiter", {node.content for node in built_regions})
        self.assertTrue(any("529" in (node.content or "") for node in built_regions))
        for source_id in IMAGE_IDS:
            _, nodes = _build(source_id)
            raw = read_json(CACHE_ROOT / source_id / "ocr.json")["regions"]
            image_nodes = [node for node in nodes if node.modality == "image"]
            ocr_nodes = [node for node in nodes if node.modality == "ocr_text"]
            self.assertEqual(len(image_nodes), 1)
            self.assertIsNone(image_nodes[0].content)
            self.assertEqual(len(ocr_nodes), len(raw))
            self.assertEqual([(n.content, n.confidence, n.bbox) for n in ocr_nodes],
                             [(r["text"], r["confidence"], r["bbox"]) for r in raw])
        _, noisy_image = _build("src_66bf6ebf0dbc")
        self.assertIn("Long term storagee", {node.content for node in noisy_image})

    def test_pdf_blocks_preserve_page_bbox_and_content(self) -> None:
        _, nodes = _build(PDF_ID)
        raw = read_json(CACHE_ROOT / PDF_ID / "pdf_blocks.json")
        blocks = [(page["page_number"], block) for page in raw["pages"] for block in page["blocks"]]
        self.assertEqual(len(nodes), len(blocks))
        for node, (page_number, block) in zip(nodes, blocks):
            self.assertEqual(node.modality, "pdf_text")
            self.assertEqual(node.content, block["text"])
            self.assertEqual(node.page_number, page_number)
            self.assertEqual(node.bbox, block["bbox"])
            self.assertIsNone(node.confidence)
            self.assertEqual(node.provenance["block_id"], block["block_id"])

    def test_every_node_has_unambiguous_provenance_and_no_future_data(self) -> None:
        for source_id in (VIDEO_ID, *IMAGE_IDS, PDF_ID):
            source, nodes = _build(source_id)
            for node in nodes:
                self.assertEqual(node.source_id, source.id)
                self.assertEqual(node.provenance["source_id"], source.id)
                self.assertEqual(node.provenance["file"], source.filename)
                self.assertTrue(node.provenance["locator_type"])
                self.assertEqual(node.entities, [])
                self.assertIsNone(node.text_vector_id)
                self.assertIsNone(node.visual_vector_id)
