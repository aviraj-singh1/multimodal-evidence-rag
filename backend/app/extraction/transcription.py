"""Timestamp-preserving faster-whisper transcription with cache reuse."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.cache.manager import PIPELINE_VERSION, read_json, write_json


TRANSCRIPTION_MODEL = "base"
TRANSCRIPTION_CONFIG = {
    "device": "cpu",
    "compute_type": "int8",
    "beam_size": 5,
    "vad_filter": True,
}


class SpeechProcessingError(RuntimeError):
    """Raised when cached speech processing cannot complete safely."""


@dataclass(frozen=True)
class StageResult:
    payload: dict[str, Any]
    cache_status: str
    runtime_sec: float


def transcribe_cached_audio(cache_dir: Path, model_cache: Path) -> StageResult:
    source, audio = load_source_and_audio(cache_dir)
    output = cache_dir / "transcript.json"
    if _transcript_cache_valid(output, source):
        return StageResult(read_json(output), "REUSED", 0.0)

    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(
        TRANSCRIPTION_MODEL,
        device=TRANSCRIPTION_CONFIG["device"],
        compute_type=TRANSCRIPTION_CONFIG["compute_type"],
        download_root=str(model_cache),
    )
    segment_iterator, info = model.transcribe(
        str(audio),
        beam_size=TRANSCRIPTION_CONFIG["beam_size"],
        vad_filter=TRANSCRIPTION_CONFIG["vad_filter"],
    )
    segments: list[dict[str, Any]] = []
    duration = float(source["duration_sec"])
    for raw in segment_iterator:
        text = raw.text.strip()
        start = max(0.0, float(raw.start))
        end = min(duration, float(raw.end))
        if not text or end <= start:
            continue
        segments.append(
            {
                "segment_id": f"seg_{len(segments) + 1:04d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "text": text,
                "speaker": None,
                "confidence": None,
                "speaker_alignment": None,
                "multi_speaker": False,
            }
        )
    runtime = time.perf_counter() - started
    if not segments:
        raise SpeechProcessingError("faster-whisper returned no non-empty transcript segments")
    payload = {
        "source_id": source["source_id"],
        "source_sha256": source["sha256"],
        "pipeline_version": PIPELINE_VERSION,
        "model": TRANSCRIPTION_MODEL,
        "configuration": TRANSCRIPTION_CONFIG,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration_sec": duration,
        "runtime_sec": round(runtime, 6),
        "segments": segments,
    }
    validate_transcript(payload, duration)
    write_json(output, payload)
    update_speech_manifest(cache_dir, "transcription", TRANSCRIPTION_MODEL, TRANSCRIPTION_CONFIG)
    return StageResult(payload, "CREATED", runtime)


def load_source_and_audio(cache_dir: Path) -> tuple[dict[str, Any], Path]:
    source_path = cache_dir / "source.json"
    audio = cache_dir / "audio.wav"
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source metadata: {source_path}")
    if not audio.is_file() or audio.stat().st_size <= 44:
        raise FileNotFoundError(f"missing or empty normalized audio: {audio}")
    source = read_json(source_path)
    if source.get("source_id") != cache_dir.name:
        raise SpeechProcessingError(
            f"cache directory/source ID mismatch: {cache_dir.name} != {source.get('source_id')}"
        )
    if not source.get("sha256") or not isinstance(source.get("duration_sec"), (int, float)):
        raise SpeechProcessingError("source.json lacks sha256 or numeric duration_sec")
    return source, audio


def validate_transcript(payload: dict[str, Any], source_duration: float, tolerance: float = 0.1) -> None:
    if not isinstance(payload.get("segments"), list) or not payload["segments"]:
        raise SpeechProcessingError("transcript must contain a non-empty segments list")
    identifiers: set[str] = set()
    previous_start = -1.0
    for segment in payload["segments"]:
        identifier = segment.get("segment_id")
        start = segment.get("start_sec")
        end = segment.get("end_sec")
        if not isinstance(identifier, str) or identifier in identifiers:
            raise SpeechProcessingError("transcript segment IDs must be unique strings")
        identifiers.add(identifier)
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise SpeechProcessingError(f"segment {identifier} timestamps must be numeric")
        if start < 0 or end <= start or end > source_duration + tolerance:
            raise SpeechProcessingError(f"segment {identifier} has invalid timestamps {start}-{end}")
        if start < previous_start:
            raise SpeechProcessingError("transcript segments are not sorted by start_sec")
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise SpeechProcessingError(f"segment {identifier} has empty text")
        previous_start = start


def update_speech_manifest(
    cache_dir: Path, stage: str, model: str, configuration: dict[str, Any]
) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = read_json(manifest_path)
    speech = manifest.setdefault("speech", {})
    speech[stage] = {"model": model, "configuration": configuration}
    stages = manifest.setdefault("completed_stages", [])
    if stage not in stages:
        stages.append(stage)
    write_json(manifest_path, manifest)


def _transcript_cache_valid(path: Path, source: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = read_json(path)
        validate_transcript(payload, float(source["duration_sec"]))
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        payload.get("source_id") == source.get("source_id")
        and payload.get("source_sha256") == source.get("sha256")
        and payload.get("pipeline_version") == PIPELINE_VERSION
        and payload.get("model") == TRANSCRIPTION_MODEL
        and payload.get("configuration") == TRANSCRIPTION_CONFIG
    )

