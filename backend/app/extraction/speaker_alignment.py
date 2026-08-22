"""Deterministic interval-overlap speaker alignment."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any


MIN_MEANINGFUL_OVERLAP_SEC = 0.05
MULTI_SPEAKER_RATIO = 0.25


def interval_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def align_speakers(
    transcript: dict[str, Any], diarization: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    turns = diarization["turns"]
    for segment in transcript["segments"]:
        start = float(segment["start_sec"])
        end = float(segment["end_sec"])
        duration = end - start
        overlaps: dict[str, float] = defaultdict(float)
        for turn in turns:
            if turn["start_sec"] >= end:
                break
            if turn["end_sec"] <= start:
                continue
            overlap = interval_overlap(start, end, turn["start_sec"], turn["end_sec"])
            if overlap > 0:
                overlaps[turn["speaker"]] += overlap
        ranked = sorted(overlaps.items(), key=lambda item: (-item[1], item[0]))
        dominant_speaker = None
        dominant_overlap = 0.0
        if ranked and ranked[0][1] >= MIN_MEANINGFUL_OVERLAP_SEC:
            dominant_speaker, dominant_overlap = ranked[0]
        second_overlap = ranked[1][1] if len(ranked) > 1 else 0.0
        segment["speaker"] = dominant_speaker
        segment["speaker_alignment"] = {
            "overlap_sec": round(dominant_overlap, 6),
            "segment_duration_sec": round(duration, 6),
            "overlap_ratio": round(dominant_overlap / duration, 6) if duration > 0 else 0.0,
            "speaker_overlaps": {
                speaker: round(overlap, 6) for speaker, overlap in ranked
            },
        }
        segment["multi_speaker"] = bool(
            duration > 0 and second_overlap >= duration * MULTI_SPEAKER_RATIO
        )
    runtime = time.perf_counter() - started
    transcript["speaker_alignment_runtime_sec"] = round(runtime, 6)
    transcript["speaker_alignment_method"] = "maximum_total_interval_overlap"
    return transcript, runtime

