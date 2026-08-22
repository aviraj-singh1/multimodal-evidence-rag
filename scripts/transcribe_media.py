"""CLI for cached transcription, diarization, and speaker alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.cache.manager import write_json  # noqa: E402
from backend.app.extraction.diarization import (  # noqa: E402
    DIARIZATION_MODEL,
    diarize_cached_audio,
)
from backend.app.extraction.speaker_alignment import align_speakers  # noqa: E402
from backend.app.extraction.transcription import (  # noqa: E402
    SpeechProcessingError,
    TRANSCRIPTION_MODEL,
    transcribe_cached_audio,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe and diarize cached normalized audio.")
    parser.add_argument("source_id", help="Existing deterministic source ID, e.g. src_1347fcb63e98")
    args = parser.parse_args()
    if not args.source_id.startswith("src_") or any(part in args.source_id for part in ("/", "\\", "..")):
        parser.error("source_id must be a deterministic src_ identifier, not a path")
    cache_dir = ROOT / "data" / "cache" / args.source_id
    try:
        transcription = transcribe_cached_audio(
            cache_dir, ROOT / "data" / "cache" / "models" / "faster-whisper"
        )
        diarization = diarize_cached_audio(cache_dir, ROOT / ".env")
        aligned, alignment_runtime = align_speakers(
            transcription.payload, diarization.payload
        )
        write_json(cache_dir / "transcript.json", aligned)
    except (FileNotFoundError, RuntimeError, SpeechProcessingError) as exc:
        print(f"SPEECH PROCESSING FAILED\n\n{exc}", file=sys.stderr)
        return 1

    segments = aligned["segments"]
    turns = diarization.payload["turns"]
    speakers = sorted({turn["speaker"] for turn in turns})
    attributed = sum(segment["speaker"] is not None for segment in segments)
    multi_speaker = sum(bool(segment["multi_speaker"]) for segment in segments)
    overall_cache = (
        "REUSED"
        if transcription.cache_status == diarization.cache_status == "REUSED"
        else "CREATED"
    )
    print("SPEECH PROCESSING\n")
    print(f"Source:\n{args.source_id}")
    print("\nTranscription:")
    print(f"{transcription.cache_status}")
    print(f"Model: faster-whisper/{TRANSCRIPTION_MODEL}")
    print(f"Segments: {len(segments)}")
    print(f"Language: {aligned['language']}")
    print(f"Runtime: {transcription.runtime_sec:.3f} sec")
    print("\nDiarization:")
    print(f"{diarization.cache_status}")
    print(f"Model: {DIARIZATION_MODEL}")
    print(f"Speakers detected: {len(speakers)}")
    print(f"Turns: {len(turns)}")
    print(f"Runtime: {diarization.runtime_sec:.3f} sec")
    print("\nSpeaker alignment:")
    print("PASS")
    print(f"Runtime: {alignment_runtime:.6f} sec")
    print(f"Attributed segments: {attributed} / {len(segments)}")
    print(f"Multi-speaker segments: {multi_speaker}")
    print("\nCache:")
    print(overall_cache)
    print("\nOutputs:")
    print("transcript.json")
    print("diarization.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

