"""Deterministic filesystem cache management for media preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PIPELINE_VERSION = "0.1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id_for_hash(sha256: str) -> str:
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValueError("sha256 must be a 64-character lowercase hexadecimal digest")
    return f"src_{sha256[:12]}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read valid cache JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"cache JSON must contain an object: {path}")
    return payload


def cache_is_valid(
    cache_dir: Path,
    sha256: str,
    source_type: str,
    settings: dict[str, Any] | None = None,
) -> bool:
    source_path = cache_dir / "source.json"
    manifest_path = cache_dir / "manifest.json"
    audio_path = cache_dir / "audio.wav"
    expected = [source_path, manifest_path, audio_path]
    if source_type == "video":
        expected.append(cache_dir / "frames.json")
    if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return False
    try:
        source = read_json(source_path)
        manifest = read_json(manifest_path)
    except RuntimeError:
        return False
    if source.get("sha256") != sha256 or source.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if manifest.get("sha256") != sha256 or manifest.get("pipeline_version") != PIPELINE_VERSION:
        return False
    if settings is not None and manifest.get("settings") != settings:
        return False
    if source_type == "video":
        try:
            frames = read_json(cache_dir / "frames.json").get("frames", [])
        except RuntimeError:
            return False
        if not frames or not all((cache_dir / frame.get("path", "")).is_file() for frame in frames):
            return False
    return True
