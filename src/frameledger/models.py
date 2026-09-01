from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


CONTENT_KINDS = frozenset({"presentation", "table", "chart", "unknown"})
CONTENT_SELECTOR_BY_KIND: dict[str, str | None] = {
    "presentation": "presentation_states",
    "table": "table_viewport",
    "chart": "boundary_terminal",
    "unknown": None,
}


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    size_bytes: int
    mtime_ns: int
    duration_seconds: float
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "duration_seconds": round(self.duration_seconds, 6),
            "fps": round(self.fps, 6),
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "codec": self.codec,
            "fingerprint": self.fingerprint,
        }


@dataclass
class Sample:
    timestamp: float
    frame_index: int
    gray: np.ndarray = field(repr=False)
    phash: int = 0
    sharpness: float = 0.0
    pixel_delta: float = 0.0
    edge_delta: float = 0.0
    phash_delta: int = 0
    ssim_previous: float = 1.0
    change_score: float = 0.0
    adaptive_ratio: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 6),
            "frame_index": self.frame_index,
            "phash": f"{self.phash:016x}",
            "sharpness": round(self.sharpness, 6),
            "pixel_delta": round(self.pixel_delta, 6),
            "edge_delta": round(self.edge_delta, 6),
            "phash_delta": self.phash_delta,
            "ssim_previous": round(self.ssim_previous, 6),
            "change_score": round(self.change_score, 6),
            "adaptive_ratio": round(self.adaptive_ratio, 6),
        }


@dataclass(frozen=True)
class ContentSegment:
    """One routed candidate span using half-open global sample indexes."""

    segment_id: str
    kind: str
    start_sample_index: int
    stop_sample_index: int
    start_timestamp: float
    candidate_end_timestamp: float
    confidence: float
    selector: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("Content segment id must not be empty")
        if self.kind not in CONTENT_KINDS:
            raise ValueError(f"Unknown content segment kind: {self.kind}")
        expected_selector = CONTENT_SELECTOR_BY_KIND[self.kind]
        if self.selector != expected_selector:
            raise ValueError(
                f"Content segment selector for {self.kind} must be {expected_selector!r}"
            )
        if self.start_sample_index < 0 or self.stop_sample_index <= self.start_sample_index:
            raise ValueError("Content segment indexes must form a non-empty half-open range")
        if self.candidate_end_timestamp < self.start_timestamp:
            raise ValueError("Content segment end must not precede its start")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Content segment confidence must be in [0, 1]")
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "selector": self.selector,
            "start_sample_index": self.start_sample_index,
            "stop_sample_index": self.stop_sample_index,
            "start_timestamp": round(self.start_timestamp, 6),
            "candidate_end_timestamp": round(self.candidate_end_timestamp, 6),
            "confidence": round(self.confidence, 6),
            "reasons": list(self.reasons),
        }


@dataclass
class Candidate:
    sample_index: int
    timestamp: float
    score: float
    reasons: list[str]
    image_path: str | None = None
    segment_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "sample_index": self.sample_index,
            "timestamp": round(self.timestamp, 6),
            "score": round(self.score, 6),
            "reasons": sorted(set(self.reasons)),
            "image_path": self.image_path,
        }
        if self.segment_id is not None:
            payload["segment_id"] = self.segment_id
        return payload


@dataclass
class DroppedCandidate:
    candidate: Candidate
    duplicate_of_timestamp: float
    phash_distance: int
    ssim: float
    block_change: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "duplicate_of_timestamp": round(self.duplicate_of_timestamp, 6),
            "phash_distance": self.phash_distance,
            "ssim": round(self.ssim, 6),
            "block_change": round(self.block_change, 6),
        }


@dataclass
class StrategyResult:
    name: str
    raw_candidates: list[Candidate]
    selected: list[Candidate]
    dropped_duplicates: list[DroppedCandidate]
    capped_candidates: list[Candidate] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    segments: list[ContentSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "strategy": self.name,
            "raw_candidate_count": len(self.raw_candidates),
            "selected_count": len(self.selected),
            "duplicate_dropped_count": len(self.dropped_duplicates),
            "capped_count": len(self.capped_candidates),
            "selected": [candidate.to_dict() for candidate in self.selected],
            "dropped_duplicates": [item.to_dict() for item in self.dropped_duplicates],
            "capped_candidates": [candidate.to_dict() for candidate in self.capped_candidates],
            "metrics": self.metrics,
        }
        if self.segments:
            payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload
