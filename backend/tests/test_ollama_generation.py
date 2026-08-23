"""Focused tests for local, evidence-only Ollama answer synthesis."""

from __future__ import annotations

import copy
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.generation.ollama import build_evidence_context, generate_answer
from backend.app.main import app


def bundle() -> list[dict[str, object]]:
    return [
        {"evidence_id": "ev_speech", "node": {
            "id": "ev_speech", "source_id": "src_meeting", "modality": "speech",
            "content": "Raw speaker content", "speaker": "SPEAKER_01",
            "start_sec": 509.01, "end_sec": 517.65, "page_number": None,
            "frame_path": None, "entities": ["rate limiter"], "confidence": None,
            "provenance": {"file": "meeting_demo.mp4"},
        }},
        {"evidence_id": "ev_pdf", "node": {
            "id": "ev_pdf", "source_id": "src_pdf", "modality": "pdf_text",
            "content": "Raw PDF content", "speaker": None, "start_sec": None,
            "end_sec": None, "page_number": 2, "frame_path": None,
            "entities": ["rules cache"], "confidence": None,
            "provenance": {"file": "rate_limiter_design.pdf"},
        }},
    ]


def connections() -> list[dict[str, object]]:
    return [
        {"from_id": "ev_speech", "to_id": "ev_pdf", "relationship_type": "ENTITY_OVERLAP",
         "confidence": 1.0, "explanation": "Evidence shares entities: rate limiter."},
        {"from_id": "ev_speech", "to_id": "ev_not_selected", "relationship_type": "ENTITY_OVERLAP",
         "confidence": 0.9, "explanation": "Must not be sent."},
    ]


def multimodal_bundle() -> list[dict[str, object]]:
    items = bundle()
    items.extend([
        {"evidence_id": "ev_frame", "node": {
            "id": "ev_frame", "source_id": "src_meeting", "modality": "video_frame",
            "content": None, "speaker": None, "start_sec": 960.0, "end_sec": 960.0,
            "page_number": None, "frame_path": "frames/frame_0011.jpg",
            "entities": ["rate limiter"], "confidence": None,
            "provenance": {"file": "meeting_demo.mp4", "timestamp_sec": 960.0},
        }},
        {"evidence_id": "ev_ocr", "node": {
            "id": "ev_ocr", "source_id": "src_meeting", "modality": "ocr_text",
            "content": "AI Rate limiter", "speaker": None, "start_sec": 840.0,
            "end_sec": 840.0, "page_number": None, "frame_path": "frames/frame_0010.jpg",
            "entities": ["rate limiter"], "confidence": 0.91,
            "provenance": {"file": "meeting_demo.mp4", "timestamp_sec": 840.0,
                           "ocr_engine": "PaddleOCR"},
        }},
    ])
    return items


class OllamaGenerationTests(unittest.TestCase):
    def test_context_labels_and_preserves_canonical_fields(self) -> None:
        context, labels = build_evidence_context("Question?", bundle(), connections())
        self.assertEqual(labels, {"ev_speech": "E1", "ev_pdf": "E2"})
        for expected in ("Raw speaker content", "SPEAKER_01", "509.01", "517.65",
                         "Raw PDF content", "Page:\n2", "meeting_demo.mp4"):
            self.assertIn(expected, context)
        self.assertIn("CONNECTION E1 <-> E2", context)
        self.assertNotIn("ev_not_selected", context)
        self.assertNotIn("Must not be sent", context)

    def test_frame_and_raw_ocr_fields_are_preserved(self) -> None:
        context, _ = build_evidence_context("Question?", multimodal_bundle(), [])
        for expected in ("960.0", "frames/frame_0011.jpg", "AI Rate limiter",
                         "Confidence:\n0.91", "PaddleOCR"):
            self.assertIn(expected, context)

    def test_only_bundle_is_sent_and_nodes_are_not_mutated(self) -> None:
        selected = bundle()
        original = copy.deepcopy(selected)
        captured: dict[str, object] = {}

        def transport(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
            captured.update(url=url, payload=payload, timeout=timeout)
            return {"message": {"content": 'The evidence says "Raw speaker content" [E1].'}}

        result = generate_answer("Question?", selected, connections(), transport=transport)
        self.assertEqual(result.mode, "ollama")
        self.assertEqual(selected, original)
        payload_text = json.dumps(captured["payload"])
        self.assertIn("Raw speaker content", payload_text)
        self.assertNotIn("ev_not_selected", payload_text)
        self.assertNotIn("api_key", payload_text.lower())

    def test_exception_falls_back_without_secret_or_trace(self) -> None:
        def broken(*_: object) -> dict[str, object]:
            raise RuntimeError("secret diagnostic")

        result = generate_answer("Question?", bundle(), connections(), transport=broken)
        self.assertEqual(result.mode, "deterministic")
        self.assertEqual(result.fallback_reason, "ollama_unavailable")
        self.assertNotIn("secret", result.answer.lower())

    def test_unknown_or_missing_citation_falls_back(self) -> None:
        for answer in ("Unsupported [E9]", "Answer without an evidence label"):
            result = generate_answer(
                "Question?", bundle(), connections(),
                transport=lambda *_args, value=answer: {"message": {"content": value}},
            )
            self.assertEqual(result.mode, "deterministic")
            self.assertEqual(result.fallback_reason, "unsafe_or_empty_response")

    def test_parenthesized_valid_labels_are_normalized(self) -> None:
        result = generate_answer(
            "Question?", bundle(), connections(),
            transport=lambda *_args: {"message": {"content": 'Evidence: "Raw speaker content" (E1, E2)'}},
        )
        self.assertEqual(result.mode, "ollama")
        self.assertEqual(result.answer, 'Evidence: "Raw speaker content" [E1] [E2]')

    def test_grouped_bracket_labels_are_normalized(self) -> None:
        result = generate_answer(
            "Question?", bundle(), connections(),
            transport=lambda *_args: {"message": {"content": "Grounded answer [E1, E2]."}},
        )
        self.assertEqual(result.mode, "ollama")
        self.assertEqual(result.answer, "Grounded answer [E1] [E2].")

    def test_trailing_and_bare_bullet_labels_are_safely_placed(self) -> None:
        answer = "First fact. Second fact. [E1] [E2]\n\nEvidence trail:\n- E1 support"
        result = generate_answer(
            "Question?", bundle(), connections(),
            transport=lambda *_args: {"message": {"content": answer}},
        )
        self.assertEqual(result.mode, "ollama")
        self.assertIn("First fact [E1] [E2].", result.answer)
        self.assertIn("Second fact [E1] [E2].", result.answer)
        self.assertIn("- [E1] support", result.answer)

    def test_trail_labels_are_attached_to_direct_claims(self) -> None:
        answer = "First fact. Second fact.\n\nEvidence trail:\n- [E1] support"
        result = generate_answer(
            "Question?", bundle(), connections(),
            transport=lambda *_args: {"message": {"content": answer}},
        )
        self.assertEqual(result.mode, "ollama")
        self.assertIn("First fact [E1].", result.answer)
        self.assertIn("Second fact [E1].", result.answer)

    def test_invalid_first_response_gets_one_grounded_retry(self) -> None:
        responses = iter([
            {"message": {"content": "Uncited first answer."}},
            {"message": {"content": "Corrected answer [E1]."}},
        ])
        result = generate_answer(
            "Question?", bundle(), connections(), transport=lambda *_args: next(responses),
        )
        self.assertEqual(result.mode, "ollama")
        self.assertEqual(result.answer, "Corrected answer [E1].")

    def test_fake_structured_provenance_triggers_fallback(self) -> None:
        answers = (
            "Alice spoke as SPEAKER_99 [E1].", "See fake.pdf [E2].",
            "The document says page 99 [E2].", "The frame appears at 99:00 [E1].",
        )
        for answer in answers:
            result = generate_answer(
                "Question?", bundle(), connections(),
                transport=lambda *_args, value=answer: {"message": {"content": value}},
            )
            self.assertEqual(result.mode, "deterministic")
            self.assertEqual(result.fallback_reason, "unsafe_or_empty_response")

    def test_inferred_benefit_triggers_retry_then_fallback(self) -> None:
        result = generate_answer(
            "Question?", bundle(), connections(),
            transport=lambda *_args: {"message": {"content": "The cache improves efficiency [E1]."}},
        )
        self.assertEqual(result.mode, "deterministic")
        self.assertEqual(result.fallback_reason, "unsafe_or_empty_response")

    def test_deterministic_provider_skips_transport(self) -> None:
        with patch.dict(os.environ, {"GENERATION_PROVIDER": "deterministic"}):
            result = generate_answer(
                "Question?", bundle(), connections(),
                transport=lambda *_args: self.fail("transport should not run"),
            )
        self.assertEqual(result.mode, "deterministic")
        self.assertEqual(result.fallback_reason, "provider_deterministic")

    def test_timeout_and_invalid_model_fall_back(self) -> None:
        for error in (TimeoutError(), RuntimeError("model not found")):
            result = generate_answer(
                "Question?", bundle(), connections(),
                transport=lambda *_args, value=error: (_ for _ in ()).throw(value),
            )
            self.assertEqual(result.mode, "deterministic")
            self.assertEqual(result.fallback_reason, "ollama_unavailable")

    def test_endpoint_succeeds_when_ollama_is_offline(self) -> None:
        with patch("backend.app.generation.ollama._post_json", side_effect=ConnectionRefusedError()):
            response = TestClient(app).post("/api/query", json={"question": "How is high throughput handled?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["generation_mode"], "deterministic")
        self.assertEqual(data["generation_fallback_reason"], "ollama_unavailable")
        self.assertTrue(data["evidence_bundle"])

    def test_environment_selects_local_url_and_model(self) -> None:
        captured: dict[str, object] = {}

        def transport(url: str, payload: dict[str, object], _timeout: float) -> dict[str, object]:
            captured.update(url=url, payload=payload)
            return {"message": {"content": 'The evidence says "Raw speaker content" [E1].'}}

        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/",
                                     "OLLAMA_MODEL": "qwen3:4b-instruct"}, clear=False):
            result = generate_answer("Question?", bundle(), connections(), transport=transport)
        self.assertEqual(result.mode, "ollama")
        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertEqual(captured["payload"]["model"], "qwen3:4b-instruct")
        self.assertEqual(result.model, "qwen3:4b-instruct")
        self.assertEqual(result.cited_evidence_labels, ("E1",))


if __name__ == "__main__":
    unittest.main()
