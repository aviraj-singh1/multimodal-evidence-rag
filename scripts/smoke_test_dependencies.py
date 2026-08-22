"""Milestone 0 smoke tests for the frozen EvidenceGraph RAG dependencies."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_DIR = ROOT / "data" / "smoke"
CACHE_DIR = ROOT / "data" / "cache" / "models"
DATASET_DIR = ROOT.parent / "demo_dataset"
HF_CACHE = CACHE_DIR / "huggingface"

SMOKE_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


RESULTS: dict[str, tuple[str, str]] = {}


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS[name] = (status, detail)
    suffix = f": {detail}" if detail else ""
    print(f"[{name}] {status}{suffix}", flush=True)


def run_test(name: str, test: Callable[[], str]) -> None:
    try:
        record(name, "PASS", test())
    except Exception as exc:  # each dependency must fail independently
        detail = f"{type(exc).__name__}: {exc}"
        record(name, "FAIL", detail)
        print(traceback.format_exc(limit=4), flush=True)


def command_version(command: str) -> str:
    executable = shutil.which(command)
    if not executable:
        raise RuntimeError(f"{command} is not on PATH")
    result = subprocess.run(
        [executable, "-version"], capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()[0]


def test_python() -> str:
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        raise RuntimeError("interpreter is not running from a virtual environment")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, got {platform.python_version()}")
    return f"{platform.python_version()} ({sys.executable})"


def test_ffmpeg() -> str:
    return command_version("ffmpeg")


def test_ffprobe() -> str:
    detail = command_version("ffprobe")
    videos = sorted(DATASET_DIR.glob("*.mp4")) if DATASET_DIR.exists() else []
    if videos:
        result = subprocess.run(
            [
                shutil.which("ffprobe") or "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "default=noprint_wrappers=1",
                str(videos[0]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        detail += f"; probed {videos[0].name}: {result.stdout.strip().replace(chr(10), ', ')}"
    return detail


def test_pymupdf() -> str:
    import pymupdf

    pdfs = sorted(DATASET_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"no PDF found under {DATASET_DIR}")
    with pymupdf.open(pdfs[0]) as document:
        text = document[0].get_text("text").strip().replace("\n", " ")[:200]
        if not text:
            raise RuntimeError("page 1 returned no text")
        detail = f"{pdfs[0].name}; pages={document.page_count}; page1={text!r}"
    return detail


def test_minilm() -> str:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(HF_CACHE),
        device="cpu",
    )
    embedding = model.encode("Redis reduces repeated database reads.")
    if embedding.ndim != 1 or embedding.shape[0] <= 0:
        raise RuntimeError(f"unexpected embedding shape {embedding.shape}")
    return f"all-MiniLM-L6-v2; dimension={embedding.shape[0]}"


def prepare_audio_sample() -> Path:
    target = SMOKE_DIR / "whisper_sample.wav"
    if target.exists() and target.stat().st_size > 44:
        return target
    audio_candidates = [
        *sorted(DATASET_DIR.glob("*.wav")),
        *sorted(DATASET_DIR.glob("*.mp3")),
        *sorted(DATASET_DIR.glob("*.mp4")),
    ]
    if not audio_candidates:
        raise RuntimeError("no audio/video sample available")
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-ss",
            "0",
            "-t",
            "12",
            "-i",
            str(audio_candidates[0]),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return target


def test_faster_whisper() -> str:
    from faster_whisper import WhisperModel

    sample = prepare_audio_sample()
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(CACHE_DIR / "faster-whisper"),
    )
    segments, _ = model.transcribe(str(sample), beam_size=1, vad_filter=True)
    segment_list = list(segments)
    if not segment_list:
        raise RuntimeError("transcription executed but returned no segments")
    first = segment_list[0]
    if first.start is None or first.end is None:
        raise RuntimeError("timestamp fields are unavailable")
    return (
        f"{model_name} loaded; sample={sample.name}; "
        f"first=[{first.start:.2f}-{first.end:.2f}] {first.text.strip()!r}"
    )


def test_pyannote() -> str:
    import pyannote.audio  # noqa: F401

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        record(
            "pyannote",
            "BLOCKED",
            "HF_TOKEN missing; accept pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0",
        )
        return "__RECORDED__"
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    if pipeline is None:
        raise RuntimeError("model access denied or pipeline did not load")
    sample = prepare_audio_sample()
    diarization = pipeline(str(sample))
    turns = list(diarization.itertracks(yield_label=True))
    if not turns:
        raise RuntimeError("diarization returned no speaker-labelled turns")
    first_segment, _, first_label = turns[0]
    if not first_label or first_segment.start is None or first_segment.end is None:
        raise RuntimeError("speaker label or segment timestamps are unavailable")
    return (
        f"speaker-diarization-3.1 loaded; turns={len(turns)}; "
        f"first=[{first_segment.start:.2f}-{first_segment.end:.2f}] {first_label}"
    )


def run_pyannote() -> None:
    try:
        detail = test_pyannote()
        if detail != "__RECORDED__":
            record("pyannote", "PASS", detail)
    except Exception as exc:
        record("pyannote", "FAIL", f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc(limit=4), flush=True)


def prepare_image_sample() -> Path:
    target = SMOKE_DIR / "ocr_sample.png"
    if target.exists():
        return target
    images = [*sorted(DATASET_DIR.glob("*.png")), *sorted(DATASET_DIR.glob("*.jpg"))]
    if not images:
        raise RuntimeError(f"no image found under {DATASET_DIR}")
    shutil.copy2(images[0], target)
    return target


def test_paddleocr() -> str:
    from paddleocr import PaddleOCR

    sample = prepare_image_sample()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    result = ocr.ocr(str(sample), cls=True)
    if result is None or not isinstance(result, list):
        raise RuntimeError(f"unexpected OCR result type: {type(result).__name__}")
    texts = []
    for page in result:
        if page:
            texts.extend(line[1][0] for line in page if len(line) > 1)
    return f"models loaded; result=list; lines={len(texts)}; sample={texts[:3]!r}"


def test_clip() -> str:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, cache_dir=str(HF_CACHE))
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=str(HF_CACHE))
    model.eval()
    text_inputs = processor(text=["a technical architecture diagram"], return_tensors="pt")
    with torch.inference_mode():
        text_embedding = model.get_text_features(**text_inputs)
    sample = prepare_image_sample()
    with Image.open(sample).convert("RGB") as image:
        image_inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            image_embedding = model.get_image_features(**image_inputs)
    if text_embedding.shape[-1] != image_embedding.shape[-1]:
        raise RuntimeError(
            f"text/image dimensions differ: {text_embedding.shape}, {image_embedding.shape}"
        )
    return (
        f"{model_name}; text_dim={text_embedding.shape[-1]}; "
        f"image_dim={image_embedding.shape[-1]}"
    )


def main() -> int:
    print("DEPENDENCY SMOKE TEST\n", flush=True)
    run_test("Python", test_python)
    run_test("FFmpeg", test_ffmpeg)
    run_test("FFprobe", test_ffprobe)
    run_test("PyMuPDF", test_pymupdf)
    run_test("MiniLM", test_minilm)
    run_test("faster-whisper", test_faster_whisper)
    run_pyannote()
    run_test("PaddleOCR", test_paddleocr)
    run_test("CLIP", test_clip)

    print("\nDEPENDENCY SMOKE TEST SUMMARY\n")
    for name in [
        "Python",
        "FFmpeg",
        "FFprobe",
        "PyMuPDF",
        "MiniLM",
        "faster-whisper",
        "pyannote",
        "PaddleOCR",
        "CLIP",
    ]:
        status, detail = RESULTS.get(name, ("FAIL", "test did not run"))
        shown = f"{status}: {detail}" if detail else status
        print(f"{name:<22}{shown}")

    ready = all(status == "PASS" for status, _ in RESULTS.values()) and len(RESULTS) == 9
    print(f"\nREADY FOR MILESTONE 1: {'YES' if ready else 'NO'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
