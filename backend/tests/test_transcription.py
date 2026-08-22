"""Schema and prepared-demo checks for Milestone 2 outputs."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from backend.app.cache.manager import read_json, write_json
from backend.app.extraction.diarization import diarize_cached_audio, validate_diarization
from backend.app.extraction.transcription import transcribe_cached_audio, validate_transcript


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "src_1347fcb63e98"


class PreparedDemoSpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        transcript_path = CACHE / "transcript.json"
        diarization_path = CACHE / "diarization.json"
        if not transcript_path.is_file() or not diarization_path.is_file():
            raise unittest.SkipTest("prepared demo speech outputs do not exist yet")
        cls.source = read_json(CACHE / "source.json")
        cls.transcript = read_json(transcript_path)
        cls.diarization = read_json(diarization_path)

    def test_transcript_schema_and_timestamps(self) -> None:
        validate_transcript(self.transcript, float(self.source["duration_sec"]))

    def test_diarization_schema_and_timestamps(self) -> None:
        validate_diarization(self.diarization, float(self.source["duration_sec"]))

    def test_demo_has_attributed_segments_and_two_speakers(self) -> None:
        attributed = [segment for segment in self.transcript["segments"] if segment["speaker"]]
        speakers = {turn["speaker"] for turn in self.diarization["turns"]}
        self.assertTrue(attributed)
        self.assertGreaterEqual(len(speakers), 2)

    def test_original_timeline_invariant(self) -> None:
        duration = float(self.source["duration_sec"])
        frames = read_json(CACHE / "frames.json")["frames"]
        self.assertTrue(all(0 <= frame["timestamp_sec"] <= duration for frame in frames))
        self.assertTrue(
            all(0 <= segment["start_sec"] < segment["end_sec"] <= duration + 0.1 for segment in self.transcript["segments"])
        )
        self.assertEqual(self.transcript["duration_sec"], duration)
        self.assertEqual(self.diarization["duration_sec"], duration)

    def test_expensive_stages_reuse_cache(self) -> None:
        transcription = transcribe_cached_audio(CACHE, ROOT / "data" / "cache" / "models" / "faster-whisper")
        diarization = diarize_cached_audio(CACHE, ROOT / ".env")
        self.assertEqual(transcription.cache_status, "REUSED")
        self.assertEqual(diarization.cache_status, "REUSED")

    def test_missing_audio_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            write_json(
                cache / "source.json",
                {
                    "source_id": cache.name,
                    "sha256": "0" * 64,
                    "duration_sec": 1.0,
                },
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing or empty normalized audio"):
                transcribe_cached_audio(cache, cache / "models")


if __name__ == "__main__":
    unittest.main()
