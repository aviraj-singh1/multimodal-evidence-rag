"""Focused Milestone 1 tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from backend.app.cache.manager import sha256_file, source_id_for_hash
from backend.app.ingestion.audio import MediaPreprocessingError, probe_media
from backend.app.ingestion.video import parse_source_metadata, preprocess_media


class MediaPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.media = cls.root / "fixture.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise unittest.SkipTest("FFmpeg is unavailable")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=10:duration=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=44100:duration=6",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(cls.media),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cls.cache_root = cls.root / "cache"
        cls.first = preprocess_media(
            cls.media,
            cls.cache_root,
            fallback_interval_sec=1,
            target_frames=5,
            min_frames=3,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_sha256_and_source_id_are_stable(self) -> None:
        first_hash = sha256_file(self.media)
        self.assertEqual(first_hash, sha256_file(self.media))
        self.assertEqual(source_id_for_hash(first_hash), self.first.source_id)

    def test_source_metadata_parsing(self) -> None:
        probe = probe_media(self.media)
        metadata = parse_source_metadata(
            self.media, self.first.source_id, sha256_file(self.media), probe, "video"
        )
        self.assertEqual(metadata["width"], 320)
        self.assertEqual(metadata["height"], 180)
        self.assertAlmostEqual(metadata["fps"], 10.0)
        self.assertTrue(metadata["audio"]["present"])

    def test_audio_is_pcm_mono_16khz(self) -> None:
        audio_path = self.first.cache_dir / "audio.wav"
        self.assertTrue(audio_path.is_file())
        probe = probe_media(audio_path)
        stream = next(item for item in probe["streams"] if item["codec_type"] == "audio")
        self.assertEqual(stream["codec_name"], "pcm_s16le")
        self.assertEqual(int(stream["sample_rate"]), 16000)
        self.assertEqual(int(stream["channels"]), 1)

    def test_frames_json_and_files(self) -> None:
        payload = json.loads((self.first.cache_dir / "frames.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["source_id"], self.first.source_id)
        self.assertGreaterEqual(len(payload["frames"]), 1)
        self.assertLessEqual(len(payload["frames"]), 30)
        duration = self.first.source["duration_sec"]
        for frame in payload["frames"]:
            self.assertIsInstance(frame["timestamp_sec"], (int, float))
            self.assertGreaterEqual(frame["timestamp_sec"], 0)
            self.assertLessEqual(frame["timestamp_sec"], duration)
            self.assertTrue((self.first.cache_dir / frame["path"]).is_file())

    def test_exact_source_reuses_cache(self) -> None:
        second = preprocess_media(
            self.media,
            self.cache_root,
            fallback_interval_sec=1,
            target_frames=5,
            min_frames=3,
        )
        self.assertEqual(second.cache_status, "REUSED")
        self.assertEqual(second.source_id, self.first.source_id)

    def test_direct_wav_input_is_normalized(self) -> None:
        wav_input = self.first.cache_dir / "audio.wav"
        result = preprocess_media(wav_input, self.root / "audio-cache")
        self.assertEqual(result.source["source_type"], "audio")
        self.assertIsNone(result.frames)
        self.assertEqual(result.audio["sample_rate"], 16000)
        self.assertEqual(result.audio["channels"], 1)

    def test_invalid_inputs_are_clear(self) -> None:
        with self.assertRaises(FileNotFoundError):
            preprocess_media(self.root / "missing.mp4", self.cache_root)
        unsupported = self.root / "unsupported.txt"
        unsupported.write_text("not media", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported media extension"):
            preprocess_media(unsupported, self.cache_root)
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"not an mp4")
        with self.assertRaises(MediaPreprocessingError):
            preprocess_media(corrupt, self.cache_root)


if __name__ == "__main__":
    unittest.main()
