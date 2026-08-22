"""Audio normalization through FFmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaPreprocessingError(RuntimeError):
    """Raised when FFmpeg/FFprobe preprocessing fails."""


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise MediaPreprocessingError(f"required executable is not available on PATH: {name}")
    return executable


def run_command(command: list[str], purpose: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except OSError as exc:
        raise MediaPreprocessingError(f"unable to start {purpose}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown command error").strip()
        raise MediaPreprocessingError(f"{purpose} failed: {detail[-2000:]}") from exc


def probe_media(path: Path) -> dict[str, Any]:
    ffprobe = require_executable("ffprobe")
    result = run_command(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        "FFprobe",
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaPreprocessingError(f"FFprobe returned malformed JSON for {path}: {exc}") from exc
    if not isinstance(payload.get("streams"), list) or not isinstance(payload.get("format"), dict):
        raise MediaPreprocessingError(f"FFprobe returned incomplete metadata for {path}")
    return payload


def normalize_audio(source: Path, output: Path) -> dict[str, Any]:
    ffmpeg = require_executable("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(output),
        ],
        "audio normalization",
    )
    if not output.is_file() or output.stat().st_size <= 44:
        raise MediaPreprocessingError(f"audio normalization did not create a valid WAV: {output}")
    payload = probe_media(output)
    audio = next((stream for stream in payload["streams"] if stream.get("codec_type") == "audio"), None)
    if audio is None:
        raise MediaPreprocessingError(f"normalized output has no audio stream: {output}")
    sample_rate = _integer_or_none(audio.get("sample_rate"))
    channels = _integer_or_none(audio.get("channels"))
    if sample_rate != 16000 or channels != 1 or audio.get("codec_name") != "pcm_s16le":
        raise MediaPreprocessingError(
            f"normalized WAV validation failed: codec={audio.get('codec_name')}, "
            f"sample_rate={sample_rate}, channels={channels}"
        )
    return {"codec": audio.get("codec_name"), "sample_rate": sample_rate, "channels": channels}


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

