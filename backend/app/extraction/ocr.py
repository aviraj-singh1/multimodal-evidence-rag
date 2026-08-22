"""PaddleOCR extraction for cached video frames and standalone images."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from backend.app.cache.manager import (
    PIPELINE_VERSION,
    read_json,
    sha256_file,
    source_id_for_hash,
    write_json,
)


OCR_ENGINE = "PaddleOCR"
OCR_CONFIGURATION = {"language": "en", "use_angle_cls": True}
SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg"}


class ContentExtractionError(RuntimeError):
    """Raised when structured OCR/PDF extraction cannot complete."""


@dataclass(frozen=True)
class ExtractionResult:
    source_id: str
    cache_dir: Path
    payload: dict[str, Any]
    cache_status: str


def create_ocr_engine() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def extract_video_frame_ocr(cache_dir: Path, engine: Any | None = None) -> ExtractionResult:
    source_path = cache_dir / "source.json"
    frames_path = cache_dir / "frames.json"
    if not source_path.is_file() or not frames_path.is_file():
        raise FileNotFoundError(f"video cache lacks source.json or frames.json: {cache_dir}")
    source = read_json(source_path)
    frames_manifest = read_json(frames_path)
    if source.get("source_type") != "video" or source.get("source_id") != cache_dir.name:
        raise ContentExtractionError(f"invalid video source cache: {cache_dir}")
    output = cache_dir / "ocr.json"
    if _ocr_cache_valid(output, source, "video"):
        return ExtractionResult(source["source_id"], cache_dir, read_json(output), "REUSED")
    engine = engine or create_ocr_engine()
    frame_results: list[dict[str, Any]] = []
    for frame in frames_manifest.get("frames", []):
        image_path = cache_dir / frame["path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"frames.json references missing frame: {image_path}")
        regions = run_ocr(engine, image_path, int(frame["width"]), int(frame["height"]), frame["frame_id"])
        frame_results.append(
            {
                "frame_id": frame["frame_id"],
                "frame_path": frame["path"],
                "timestamp_sec": frame["timestamp_sec"],
                "width": frame["width"],
                "height": frame["height"],
                "regions": regions,
            }
        )
    if not frame_results:
        raise ContentExtractionError("frames.json contains no frames")
    payload = {
        "source_id": source["source_id"],
        "source_sha256": source["sha256"],
        "source_type": "video",
        "pipeline_version": PIPELINE_VERSION,
        "engine": OCR_ENGINE,
        "configuration": OCR_CONFIGURATION,
        "frames": frame_results,
    }
    validate_video_ocr(payload, cache_dir, frames_manifest)
    write_json(output, payload)
    update_content_manifest(cache_dir, "ocr", OCR_ENGINE, OCR_CONFIGURATION)
    return ExtractionResult(source["source_id"], cache_dir, payload, "CREATED")


def extract_image_ocr(
    image_path: Path, cache_root: Path, engine: Any | None = None
) -> ExtractionResult:
    image = image_path.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"image does not exist: {image}")
    if image.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"unsupported image extension {image.suffix!r}; expected PNG or JPEG")
    try:
        with Image.open(image) as opened:
            opened.verify()
        with Image.open(image) as opened:
            width, height = opened.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ContentExtractionError(f"invalid or corrupted image {image}: {exc}") from exc
    sha256 = sha256_file(image)
    source_id = source_id_for_hash(sha256)
    cache_dir = cache_root.resolve() / source_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = {
        "source_id": source_id,
        "filename": image.name,
        "source_type": "image",
        "sha256": sha256,
        "width": width,
        "height": height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
    }
    output = cache_dir / "ocr.json"
    if _ocr_cache_valid(output, source, "image"):
        return ExtractionResult(source_id, cache_dir, read_json(output), "REUSED")
    engine = engine or create_ocr_engine()
    regions = run_ocr(engine, image, width, height, "image")
    payload = {
        "source_id": source_id,
        "source_sha256": sha256,
        "source_type": "image",
        "filename": image.name,
        "pipeline_version": PIPELINE_VERSION,
        "engine": OCR_ENGINE,
        "configuration": OCR_CONFIGURATION,
        "width": width,
        "height": height,
        "regions": regions,
    }
    validate_image_ocr(payload)
    write_json(cache_dir / "source.json", source)
    write_json(output, payload)
    write_json(
        cache_dir / "manifest.json",
        {
            "source_id": source_id,
            "sha256": sha256,
            "pipeline_version": PIPELINE_VERSION,
            "completed_stages": ["ocr"],
            "content": {"ocr": {"engine": OCR_ENGINE, "configuration": OCR_CONFIGURATION}},
        },
    )
    return ExtractionResult(source_id, cache_dir, payload, "CREATED")


def run_ocr(engine: Any, path: Path, width: int, height: int, prefix: str) -> list[dict[str, Any]]:
    try:
        result = engine.ocr(str(path), cls=True)
    except Exception as exc:
        raise ContentExtractionError(f"PaddleOCR failed for {path}: {exc}") from exc
    regions: list[dict[str, Any]] = []
    for page in result or []:
        for line in page or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            polygon, recognition = line[0], line[1]
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                continue
            text = str(recognition[0]).strip()
            if not text:
                continue
            confidence = float(recognition[1])
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            bbox = [
                round(max(0.0, min(xs)), 3),
                round(max(0.0, min(ys)), 3),
                round(min(float(width), max(xs)), 3),
                round(min(float(height), max(ys)), 3),
            ]
            regions.append(
                {
                    "region_id": f"ocr_{prefix}_{len(regions) + 1:03d}",
                    "text": text,
                    "confidence": round(confidence, 6),
                    "bbox": bbox,
                }
            )
    return regions


def validate_video_ocr(
    payload: dict[str, Any], cache_dir: Path, frames_manifest: dict[str, Any]
) -> None:
    expected = {frame["frame_id"]: frame for frame in frames_manifest.get("frames", [])}
    if len(payload.get("frames", [])) != len(expected):
        raise ContentExtractionError("OCR frame count differs from frames.json")
    for frame in payload["frames"]:
        manifest_frame = expected.get(frame.get("frame_id"))
        if manifest_frame is None or frame.get("timestamp_sec") != manifest_frame.get("timestamp_sec"):
            raise ContentExtractionError("OCR frame provenance does not match frames.json")
        if not (cache_dir / frame["frame_path"]).is_file():
            raise ContentExtractionError(f"OCR references missing frame {frame['frame_path']}")
        _validate_regions(frame["regions"], int(frame["width"]), int(frame["height"]))


def validate_image_ocr(payload: dict[str, Any]) -> None:
    if payload.get("source_type") != "image" or not payload.get("filename"):
        raise ContentExtractionError("standalone image OCR lacks source provenance")
    _validate_regions(payload.get("regions", []), int(payload["width"]), int(payload["height"]))


def _validate_regions(regions: list[dict[str, Any]], width: int, height: int) -> None:
    for region in regions:
        if not isinstance(region.get("text"), str) or not region["text"].strip():
            raise ContentExtractionError("OCR region text must be non-empty")
        confidence = region.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ContentExtractionError("OCR confidence must be numeric in [0, 1]")
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) for v in bbox):
            raise ContentExtractionError("OCR bbox must contain four numeric coordinates")
        x0, y0, x1, y1 = bbox
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ContentExtractionError(f"OCR bbox outside image bounds: {bbox}")


def _ocr_cache_valid(path: Path, source: dict[str, Any], source_type: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = read_json(path)
        if source_type == "image":
            validate_image_ocr(payload)
    except (RuntimeError, TypeError, ValueError):
        return False
    return (
        payload.get("source_id") == source.get("source_id")
        and payload.get("source_sha256") == source.get("sha256")
        and payload.get("source_type") == source_type
        and payload.get("pipeline_version") == PIPELINE_VERSION
        and payload.get("engine") == OCR_ENGINE
        and payload.get("configuration") == OCR_CONFIGURATION
    )


def update_content_manifest(
    cache_dir: Path, stage: str, engine: str, configuration: dict[str, Any]
) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.setdefault("content", {})[stage] = {
        "engine": engine,
        "configuration": configuration,
    }
    stages = manifest.setdefault("completed_stages", [])
    if stage not in stages:
        stages.append(stage)
    write_json(manifest_path, manifest)

