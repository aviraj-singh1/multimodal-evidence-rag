"""Video metadata, audio, frame extraction, and cache orchestration."""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from backend.app.cache.manager import (
    PIPELINE_VERSION,
    cache_is_valid,
    read_json,
    sha256_file,
    source_id_for_hash,
    write_json,
)
from backend.app.ingestion.audio import (
    MediaPreprocessingError,
    normalize_audio,
    probe_media,
    require_executable,
    run_command,
)


SUPPORTED_VIDEO = {".mp4"}
SUPPORTED_AUDIO = {".mp3", ".wav"}
PTS_TIME_PATTERN = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class PreprocessResult:
    source_id: str
    cache_dir: Path
    source: dict[str, Any]
    audio: dict[str, Any]
    frames: dict[str, Any] | None
    cache_status: str


def preprocess_media(
    source_path: str | Path,
    cache_root: str | Path,
    *,
    scene_threshold: float = 0.35,
    fallback_interval_sec: int = 120,
    target_frames: int = 15,
    min_frames: int = 5,
    max_frames: int = 30,
) -> PreprocessResult:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"media file does not exist: {source}")
    extension = source.suffix.lower()
    if extension not in SUPPORTED_VIDEO | SUPPORTED_AUDIO:
        raise ValueError(f"unsupported media extension {extension!r}; expected MP4, MP3, or WAV")
    if not (0.0 < scene_threshold < 1.0):
        raise ValueError("scene_threshold must be between 0 and 1")
    if not (1 <= min_frames <= target_frames <= max_frames <= 30):
        raise ValueError("frame limits must satisfy 1 <= min <= target <= max <= 30")
    require_executable("ffmpeg")
    require_executable("ffprobe")

    source_type = "video" if extension in SUPPORTED_VIDEO else "audio"
    sha256 = sha256_file(source)
    source_id = source_id_for_hash(sha256)
    cache_dir = Path(cache_root).expanduser().resolve() / source_id
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MediaPreprocessingError(f"unable to create output directory {cache_dir}: {exc}") from exc

    settings = (
        {
            "scene_threshold": scene_threshold,
            "fallback_interval_sec": fallback_interval_sec,
            "target_frames": target_frames,
            "min_frames": min_frames,
            "max_frames": max_frames,
        }
        if source_type == "video"
        else None
    )
    if cache_is_valid(cache_dir, sha256, source_type, settings):
        metadata = read_json(cache_dir / "source.json")
        frames = read_json(cache_dir / "frames.json") if source_type == "video" else None
        audio = _normalized_audio_metadata(cache_dir / "audio.wav")
        return PreprocessResult(source_id, cache_dir, metadata, audio, frames, "REUSED")

    probe = probe_media(source)
    metadata = parse_source_metadata(source, source_id, sha256, probe, source_type)
    audio_stream = next((s for s in probe["streams"] if s.get("codec_type") == "audio"), None)
    if audio_stream is None:
        raise MediaPreprocessingError(f"media contains no audio stream: {source}")

    audio = normalize_audio(source, cache_dir / "audio.wav")
    frames_payload = None
    if source_type == "video":
        frames_payload = extract_representative_frames(
            source,
            cache_dir,
            source_id,
            float(metadata["duration_sec"]),
            int(metadata["width"]),
            int(metadata["height"]),
            scene_threshold=scene_threshold,
            fallback_interval_sec=fallback_interval_sec,
            target_frames=target_frames,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    write_json(cache_dir / "source.json", metadata)
    write_json(
        cache_dir / "manifest.json",
        {
            "source_id": source_id,
            "sha256": sha256,
            "pipeline_version": PIPELINE_VERSION,
            "settings": settings,
            "completed_stages": ["metadata", "audio"] + (["frames"] if source_type == "video" else []),
        },
    )
    return PreprocessResult(source_id, cache_dir, metadata, audio, frames_payload, "CREATED")


def parse_source_metadata(
    path: Path,
    source_id: str,
    sha256: str,
    probe: dict[str, Any],
    source_type: str,
) -> dict[str, Any]:
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = _float_or_none(format_info.get("duration"))
    if duration is None or duration <= 0:
        durations = [_float_or_none(stream.get("duration")) for stream in streams]
        duration = max((value for value in durations if value is not None), default=None)
    if duration is None or duration <= 0:
        raise MediaPreprocessingError(f"FFprobe did not provide a positive duration for {path}")
    if source_type == "video" and video is None:
        raise MediaPreprocessingError(f"MP4 contains no video stream: {path}")
    fps = _fraction_or_none((video or {}).get("avg_frame_rate"))
    return {
        "source_id": source_id,
        "filename": path.name,
        "source_type": source_type,
        "sha256": sha256,
        "duration_sec": duration,
        "width": _int_or_none((video or {}).get("width")),
        "height": _int_or_none((video or {}).get("height")),
        "fps": fps,
        "audio": {
            "present": audio is not None,
            "codec": (audio or {}).get("codec_name"),
            "sample_rate": _int_or_none((audio or {}).get("sample_rate")),
            "channels": _int_or_none((audio or {}).get("channels")),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }


def extract_representative_frames(
    source: Path,
    cache_dir: Path,
    source_id: str,
    duration_sec: float,
    width: int,
    height: int,
    *,
    scene_threshold: float,
    fallback_interval_sec: int,
    target_frames: int,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    ffmpeg = require_executable("ffmpeg")
    scan = run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(source),
            "-vf",
            f"select='gt(scene,{scene_threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "NUL",
        ],
        "scene-change scan",
    )
    scene_times = sorted({float(value) for value in PTS_TIME_PATTERN.findall(scan.stderr)})
    scene_times = [value for value in scene_times if 0 <= value <= duration_sec]
    selected = _uniformly_limit(scene_times, target_frames)
    strategy_used = "scene-change"
    timeline_points = [0.0, *scene_times, duration_sec]
    largest_scene_gap = max(
        (right - left for left, right in zip(timeline_points, timeline_points[1:])),
        default=duration_sec,
    )
    if len(selected) < min_frames or largest_scene_gap > fallback_interval_sec * 1.5:
        fallback_times = [
            float(second)
            for second in range(fallback_interval_sec, max(1, math.ceil(duration_sec)), fallback_interval_sec)
        ]
        selected = _prioritize_timeline_coverage(
            scene_times, fallback_times, target_frames, tolerance_sec=1.0
        )
        strategy_used = "scene-change+interval-fallback"
    selected = selected[:max_frames]
    if not selected:
        selected = [0.0]
        strategy_used = "interval-fallback"

    frames_dir = cache_dir / "frames"
    if frames_dir.exists():
        for old_frame in frames_dir.glob("frame_*.jpg"):
            old_frame.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for index, timestamp in enumerate(selected, start=1):
        frame_id = f"frame_{index:04d}"
        output = frames_dir / f"{frame_id}.jpg"
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output),
            ],
            f"frame extraction at {timestamp:.3f}s",
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise MediaPreprocessingError(f"frame extraction did not create {output}")
        frames.append(
            {
                "frame_id": frame_id,
                "path": f"frames/{output.name}",
                "timestamp_sec": round(timestamp, 6),
                "width": width,
                "height": height,
            }
        )
    payload = {
        "source_id": source_id,
        "strategy": {
            "scene_threshold": scene_threshold,
            "fallback_interval_sec": fallback_interval_sec,
            "target_frames": target_frames,
            "max_frames": max_frames,
            "used": strategy_used,
            "scene_candidates": len(scene_times),
            "largest_scene_gap_sec": round(largest_scene_gap, 6),
        },
        "frames": frames,
    }
    write_json(cache_dir / "frames.json", payload)
    return payload


def _uniformly_limit(values: list[float], limit: int) -> list[float]:
    if len(values) <= limit:
        return values.copy()
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def _merge_distinct(first: list[float], second: list[float], tolerance_sec: float) -> list[float]:
    merged = sorted(first + second)
    result: list[float] = []
    for value in merged:
        if not result or value - result[-1] > tolerance_sec:
            result.append(value)
    return result


def _prioritize_timeline_coverage(
    scene_times: list[float],
    interval_times: list[float],
    limit: int,
    tolerance_sec: float,
) -> list[float]:
    intervals = _uniformly_limit(interval_times, min(limit, len(interval_times)))
    remaining = limit - len(intervals)
    candidates = [
        value
        for value in scene_times
        if all(abs(value - interval) > tolerance_sec for interval in intervals)
    ]
    scenes = _uniformly_limit(candidates, remaining) if remaining > 0 else []
    return sorted(_merge_distinct(intervals, scenes, tolerance_sec))


def _normalized_audio_metadata(path: Path) -> dict[str, Any]:
    probe = probe_media(path)
    audio = next((stream for stream in probe["streams"] if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaPreprocessingError(f"cached audio has no audio stream: {path}")
    return {
        "codec": audio.get("codec_name"),
        "sample_rate": _int_or_none(audio.get("sample_rate")),
        "channels": _int_or_none(audio.get("channels")),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fraction_or_none(value: Any) -> float | None:
    try:
        fraction = Fraction(str(value))
        return float(fraction) if fraction.denominator and fraction.numerator else None
    except (ValueError, ZeroDivisionError):
        return None
