"""Full-timeline pyannote diarization with deterministic cache serialization."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.app.cache.manager import PIPELINE_VERSION, read_json, write_json
from backend.app.extraction.transcription import (
    SpeechProcessingError,
    StageResult,
    load_source_and_audio,
    update_speech_manifest,
)


DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZATION_CONFIG = {"device": "cpu"}


def diarize_cached_audio(cache_dir: Path, env_path: Path) -> StageResult:
    source, audio = load_source_and_audio(cache_dir)
    output = cache_dir / "diarization.json"
    if _diarization_cache_valid(output, source):
        return StageResult(read_json(output), "REUSED", 0.0)
    load_dotenv(env_path)
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise SpeechProcessingError(f"HF_TOKEN is missing from {env_path}")

    from pyannote.audio import Pipeline

    started = time.perf_counter()
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        raise SpeechProcessingError(
            "unable to load gated pyannote pipeline; verify repository acceptance and HF_TOKEN"
        )
    diarization = pipeline(str(audio))
    raw_turns = sorted(
        diarization.itertracks(yield_label=True), key=lambda item: (item[0].start, item[0].end, item[2])
    )
    turns: list[dict[str, Any]] = []
    duration = float(source["duration_sec"])
    for segment, _, speaker in raw_turns:
        start = max(0.0, float(segment.start))
        end = min(duration, float(segment.end))
        if end <= start or not speaker:
            continue
        turns.append(
            {
                "turn_id": f"turn_{len(turns) + 1:04d}",
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "speaker": str(speaker),
            }
        )
    runtime = time.perf_counter() - started
    if not turns:
        raise SpeechProcessingError("pyannote returned no speaker-labelled turns")
    payload = {
        "source_id": source["source_id"],
        "source_sha256": source["sha256"],
        "pipeline_version": PIPELINE_VERSION,
        "model": DIARIZATION_MODEL,
        "configuration": DIARIZATION_CONFIG,
        "duration_sec": duration,
        "runtime_sec": round(runtime, 6),
        "turns": turns,
    }
    validate_diarization(payload, duration)
    write_json(output, payload)
    update_speech_manifest(cache_dir, "diarization", DIARIZATION_MODEL, DIARIZATION_CONFIG)
    return StageResult(payload, "CREATED", runtime)


def validate_diarization(payload: dict[str, Any], source_duration: float, tolerance: float = 0.1) -> None:
    if not isinstance(payload.get("turns"), list) or not payload["turns"]:
        raise SpeechProcessingError("diarization must contain a non-empty turns list")
    previous_start = -1.0
    for turn in payload["turns"]:
        start, end, speaker = turn.get("start_sec"), turn.get("end_sec"), turn.get("speaker")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise SpeechProcessingError("diarization timestamps must be numeric")
        if start < 0 or end <= start or end > source_duration + tolerance:
            raise SpeechProcessingError(f"invalid diarization turn timestamps: {start}-{end}")
        if start < previous_start:
            raise SpeechProcessingError("diarization turns are not sorted by start_sec")
        if not isinstance(speaker, str) or not speaker:
            raise SpeechProcessingError("every diarization turn must have a speaker label")
        previous_start = start


def _diarization_cache_valid(path: Path, source: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = read_json(path)
        validate_diarization(payload, float(source["duration_sec"]))
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        payload.get("source_id") == source.get("source_id")
        and payload.get("source_sha256") == source.get("sha256")
        and payload.get("pipeline_version") == PIPELINE_VERSION
        and payload.get("model") == DIARIZATION_MODEL
        and payload.get("configuration") == DIARIZATION_CONFIG
    )

