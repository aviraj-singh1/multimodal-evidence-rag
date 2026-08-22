"""Focused validation tests for canonical evidence models and IDs."""

from __future__ import annotations

import unittest
from pydantic import ValidationError

from backend.app.evidence.builder import evidence_id
from backend.app.evidence.models import EvidenceNode, Source


class EvidenceModelTests(unittest.TestCase):
    def test_source_schema_validation(self) -> None:
        source = Source(
            id="src_0123456789ab", filename="meeting.mp4", media_type="video",
            original_path="data/meeting.mp4", sha256="0" * 64,
            created_at="2026-01-01T00:00:00+00:00", metadata={"duration_sec": 1.0},
        )
        self.assertEqual(source.media_type, "video")
        with self.assertRaises(ValidationError):
            Source(**{**source.model_dump(), "media_type": "webpage"})

    def test_evidence_node_schema_and_invalid_modality(self) -> None:
        values = dict(
            id="ev_0123456789ab_speech_0001", source_id="src_0123456789ab",
            modality="speech", content="hello", start_sec=0.0, end_sec=1.0,
            speaker="SPEAKER_00", page_number=None, bbox=None, frame_path=None,
            entities=[], confidence=None,
            provenance={"file": "meeting.mp4", "locator_type": "time_range"},
            text_vector_id=None, visual_vector_id=None,
        )
        self.assertEqual(EvidenceNode(**values).entities, [])
        with self.assertRaises(ValidationError):
            EvidenceNode(**{**values, "modality": "relationship"})
        with self.assertRaises(ValidationError):
            EvidenceNode(**{**values, "confidence": 1.1})

    def test_deterministic_evidence_ids(self) -> None:
        source_id = "src_0123456789ab"
        self.assertEqual(evidence_id(source_id, "speech", 1), evidence_id(source_id, "speech", 1))
        self.assertEqual(evidence_id(source_id, "speech", 1), "ev_0123456789ab_speech_0001")
        self.assertEqual(evidence_id(source_id, "frame", 2), "ev_0123456789ab_frame_0002")
        self.assertEqual(evidence_id(source_id, "ocr_frame", 2, 3), "ev_0123456789ab_ocr_frame_0002_003")
        self.assertEqual(evidence_id(source_id, "image"), "ev_0123456789ab_image")
        self.assertEqual(evidence_id(source_id, "ocr_image", 3), "ev_0123456789ab_ocr_image_003")
        self.assertEqual(evidence_id(source_id, "pdf", 1, 2), "ev_0123456789ab_pdf_p001_b002")
