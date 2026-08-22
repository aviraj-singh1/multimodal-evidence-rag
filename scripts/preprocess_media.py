"""CLI smoke tool for Milestone 1 media preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ingestion.audio import MediaPreprocessingError  # noqa: E402
from backend.app.ingestion.video import preprocess_media  # noqa: E402


def relative_output(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize media and extract representative frames.")
    parser.add_argument("source", help="Path to an MP4, MP3, or WAV file")
    parser.add_argument("--scene-threshold", type=float, default=0.35)
    parser.add_argument("--fallback-interval", type=int, default=120)
    parser.add_argument("--target-frames", type=int, default=15)
    args = parser.parse_args()
    try:
        result = preprocess_media(
            args.source,
            ROOT / "data" / "cache",
            scene_threshold=args.scene_threshold,
            fallback_interval_sec=args.fallback_interval,
            target_frames=args.target_frames,
        )
    except (FileNotFoundError, ValueError, MediaPreprocessingError) as exc:
        print(f"MEDIA PREPROCESSING FAILED\n\n{exc}", file=sys.stderr)
        return 1

    print("MEDIA PREPROCESSING\n")
    print(f"Source ID: {result.source_id}")
    print(f"File: {result.source['filename']}")
    print(f"Duration: {result.source['duration_sec']:.3f} sec")
    print("\nAudio:")
    print("PASS")
    print(f"{result.audio['sample_rate'] // 1000} kHz")
    print("mono" if result.audio["channels"] == 1 else f"{result.audio['channels']} channels")
    print("audio.wav")
    print("\nFrames:")
    if result.frames is None:
        print("N/A (audio input)")
    else:
        print("PASS")
        print(f"{len(result.frames['frames'])} representative frames")
        print(f"strategy={result.frames['strategy']['used']}")
    print("\nCache:")
    print(result.cache_status)
    print("\nOutputs:")
    print(relative_output(result.cache_dir / "source.json"))
    print(relative_output(result.cache_dir / "audio.wav"))
    if result.frames is not None:
        print(relative_output(result.cache_dir / "frames.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

