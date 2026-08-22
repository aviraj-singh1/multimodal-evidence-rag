"""Unit tests for deterministic speaker overlap alignment."""

from __future__ import annotations

import unittest

from backend.app.extraction.speaker_alignment import align_speakers, interval_overlap


def transcript(start: float, end: float) -> dict:
    return {
        "segments": [
            {
                "segment_id": "seg_0001",
                "start_sec": start,
                "end_sec": end,
                "text": "test",
                "speaker": None,
                "confidence": None,
            }
        ]
    }


class SpeakerAlignmentTests(unittest.TestCase):
    def test_overlap_calculation(self) -> None:
        self.assertEqual(interval_overlap(1, 5, 3, 7), 2)
        self.assertEqual(interval_overlap(1, 2, 3, 4), 0)

    def test_dominant_speaker_assignment(self) -> None:
        turns = {
            "turns": [
                {"start_sec": 0.0, "end_sec": 3.0, "speaker": "SPEAKER_00"},
                {"start_sec": 3.0, "end_sec": 4.0, "speaker": "SPEAKER_01"},
            ]
        }
        result, _ = align_speakers(transcript(0, 4), turns)
        self.assertEqual(result["segments"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(result["segments"][0]["speaker_alignment"]["overlap_sec"], 3.0)

    def test_no_overlap_returns_null(self) -> None:
        turns = {"turns": [{"start_sec": 5.0, "end_sec": 6.0, "speaker": "SPEAKER_00"}]}
        result, _ = align_speakers(transcript(0, 4), turns)
        self.assertIsNone(result["segments"][0]["speaker"])

    def test_multi_speaker_threshold(self) -> None:
        turns = {
            "turns": [
                {"start_sec": 0.0, "end_sec": 2.5, "speaker": "SPEAKER_00"},
                {"start_sec": 2.5, "end_sec": 4.0, "speaker": "SPEAKER_01"},
            ]
        }
        result, _ = align_speakers(transcript(0, 4), turns)
        self.assertTrue(result["segments"][0]["multi_speaker"])


if __name__ == "__main__":
    unittest.main()

