"""Pydantic models for canonical source and evidence records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MediaType = Literal["video", "audio", "image", "pdf"]
Modality = Literal["speech", "video_frame", "image", "ocr_text", "pdf_text", "audio_segment"]


class Source(BaseModel):
    """A stable reference to one original media source."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    media_type: MediaType
    original_path: str | None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(min_length=1)
    metadata: dict[str, Any]


class EvidenceNode(BaseModel):
    """One canonical, provenance-preserving extracted observation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    modality: Modality
    content: str | None
    start_sec: float | None
    end_sec: float | None
    speaker: str | None
    page_number: int | None
    bbox: list[float] | None
    frame_path: str | None
    entities: list[str] = Field(default_factory=list)
    confidence: float | None
    provenance: dict[str, Any]
    text_vector_id: str | None
    visual_vector_id: str | None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            if len(value) != 4:
                raise ValueError("bbox must have four coordinates")
            x0, y0, x1, y1 = value
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError("bbox must contain ordered non-negative coordinates")
        return value

    @model_validator(mode="after")
    def validate_locator(self) -> "EvidenceNode":
        if not self.provenance:
            raise ValueError("provenance must not be empty")
        if self.start_sec is not None and self.start_sec < 0:
            raise ValueError("start_sec must be non-negative")
        if self.end_sec is not None and self.end_sec < 0:
            raise ValueError("end_sec must be non-negative")
        if self.start_sec is not None and self.end_sec is not None and self.end_sec < self.start_sec:
            raise ValueError("end_sec must not precede start_sec")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be 1-based")
        return self
