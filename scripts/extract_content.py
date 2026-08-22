"""CLI for Milestone 3 OCR and text-readable PDF extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.extraction.ocr import (  # noqa: E402
    ContentExtractionError,
    create_ocr_engine,
    extract_image_ocr,
    extract_video_frame_ocr,
)
from backend.app.extraction.pdf_blocks import extract_pdf_blocks  # noqa: E402


class LazyOCREngine:
    def __init__(self) -> None:
        self._engine = None

    def ocr(self, *args, **kwargs):
        if self._engine is None:
            self._engine = create_ocr_engine()
        return self._engine.ocr(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frame/image OCR and PDF text blocks.")
    parser.add_argument("--video-source", action="append", default=[], help="Existing src_ cache ID")
    parser.add_argument("--image", action="append", default=[], help="Standalone PNG/JPEG path")
    parser.add_argument("--pdf", action="append", default=[], help="Text-readable PDF path")
    args = parser.parse_args()
    if not (args.video_source or args.image or args.pdf):
        parser.error("provide at least one --video-source, --image, or --pdf")
    cache_root = ROOT / "data" / "cache"
    engine = LazyOCREngine()
    try:
        print("CONTENT EXTRACTION")
        for source_id in args.video_source:
            if not source_id.startswith("src_") or any(part in source_id for part in ("/", "\\", "..")):
                raise ValueError(f"invalid video source ID: {source_id}")
            result = extract_video_frame_ocr(cache_root / source_id, engine)
            frames = result.payload["frames"]
            frames_with_text = sum(bool(frame["regions"]) for frame in frames)
            regions = sum(len(frame["regions"]) for frame in frames)
            print("\nVIDEO FRAME OCR\n")
            print(f"Source:\n{result.source_id}")
            print(f"\nFrames processed:\n{len(frames)}")
            print(f"\nFrames containing OCR text:\n{frames_with_text}")
            print(f"\nOCR regions:\n{regions}")
            print(f"\nStatus:\n{result.cache_status}")
        for image in args.image:
            result = extract_image_ocr(Path(image), cache_root, engine)
            print("\nSTANDALONE IMAGE OCR\n")
            print(f"Source:\n{result.source_id}")
            print(f"File:\n{result.payload['filename']}")
            print(f"Regions:\n{len(result.payload['regions'])}")
            print(f"Status:\n{result.cache_status}")
        for pdf in args.pdf:
            result = extract_pdf_blocks(Path(pdf), cache_root)
            blocks = sum(len(page["blocks"]) for page in result.payload["pages"])
            print("\nPDF EXTRACTION\n")
            print(f"Source:\n{result.source_id}")
            print(f"File:\n{result.payload['filename']}")
            print(f"Pages:\n{result.payload['page_count']}")
            print(f"Blocks:\n{blocks}")
            print(f"Status:\n{result.cache_status}")
    except (FileNotFoundError, ValueError, ContentExtractionError) as exc:
        print(f"\nCONTENT EXTRACTION FAILED\n\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
