from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .features import RoutingFrameFeatures, routing_frame_features
from .models import Candidate, ContentSegment, CONTENT_SELECTOR_BY_KIND, Sample


SELECTOR_BY_KIND = CONTENT_SELECTOR_BY_KIND


@dataclass(frozen=True)
class VisualRouteDecision:
    """Raw and post-processed routing decision for one eligible sample."""

    sample_index: int
    timestamp: float
    raw_kind: str
    kind: str
    confidence: float
    scores: dict[str, float]
    reasons: tuple[str, ...]
    features: RoutingFrameFeatures

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "timestamp": round(self.timestamp, 6),
            "raw_kind": self.raw_kind,
            "kind": self.kind,
            "confidence": round(self.confidence, 6),
            "scores": {
                name: round(score, 6)
                for name, score in sorted(self.scores.items())
            },
            "reasons": list(self.reasons),
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class VisualRoutePlan:
    """Serializable visual-only routing plan for a bounded sample range."""

    analysis_fps: float
    candidate_end_timestamp: float
    bridge_seconds: float
    minimum_known_seconds: float
    eligible_sample_count: int
    decisions: tuple[VisualRouteDecision, ...]
    segments: tuple[ContentSegment, ...]

    def __post_init__(self) -> None:
        if self.eligible_sample_count <= 0:
            raise ValueError("Visual route plans require at least one eligible sample")
        if len(self.decisions) != self.eligible_sample_count:
            raise ValueError("Route decisions must cover every eligible sample")
        if not self.segments:
            raise ValueError("Visual route plans require at least one content segment")
        if self.segments[0].start_sample_index != 0:
            raise ValueError("The first content segment must start at sample index zero")
        if self.segments[-1].stop_sample_index != self.eligible_sample_count:
            raise ValueError("The final content segment must cover the eligible tail")
        for expected_index, decision in enumerate(self.decisions):
            if decision.sample_index != expected_index:
                raise ValueError("Route decisions must use contiguous global sample indexes")
        for left, right in zip(self.segments, self.segments[1:]):
            if left.stop_sample_index != right.start_sample_index:
                raise ValueError("Content segments must be contiguous and non-overlapping")
        for segment in self.segments:
            if any(
                decision.kind != segment.kind
                for decision in self.decisions[
                    segment.start_sample_index : segment.stop_sample_index
                ]
            ):
                raise ValueError("Content segment kind must match its covered decisions")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_fps": round(self.analysis_fps, 6),
            "candidate_end_timestamp": round(self.candidate_end_timestamp, 6),
            "bridge_seconds": round(self.bridge_seconds, 6),
            "minimum_known_seconds": round(self.minimum_known_seconds, 6),
            "eligible_sample_count": self.eligible_sample_count,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "segments": [segment.to_dict() for segment in self.segments],
        }


def _at_least(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    return float(min(1.0, max(0.0, value / threshold)))


def _at_most(value: float, threshold: float) -> float:
    if value <= threshold:
        return 1.0
    return float(max(0.0, 1.0 - (value - threshold) / max(threshold, 1e-9)))


def classify_routing_features(
    features: RoutingFrameFeatures,
) -> tuple[str, float, dict[str, float], tuple[str, ...]]:
    """Apply conservative, corpus-calibrated visual routing gates."""
    chart_score = (
        0.60 * _at_least(features.dark_ratio, 0.72)
        + 0.40 * _at_most(features.mean_luma, 100.0)
    )
    table_score = (
        0.25 * _at_least(features.mean_luma, 185.0)
        + 0.20 * _at_least(features.bright_ratio, 0.30)
        + 0.25 * _at_least(features.horizontal_line_ratio, 0.008)
        + 0.30 * _at_least(features.edge_density, 0.07)
    )
    ordinary_presentation = features.bright_ratio <= 0.25
    white_embedded_visual = (
        features.horizontal_line_ratio < 0.010
        and features.vertical_line_ratio < 0.008
        and features.edge_density <= 0.12
        and features.mean_luma <= 235.0
    )
    presentation_score = (
        0.24 * _at_least(features.mean_luma, 105.0)
        + 0.20 * _at_most(features.dark_ratio, 0.60)
        + 0.20 * _at_most(features.horizontal_line_ratio, 0.012)
        + 0.16 * _at_most(features.edge_density, 0.25)
        + 0.20
        * max(
            _at_most(features.bright_ratio, 0.25),
            min(
                _at_most(features.horizontal_line_ratio, 0.010),
                _at_most(features.vertical_line_ratio, 0.008),
                _at_most(features.edge_density, 0.12),
                _at_most(features.mean_luma, 235.0),
            ),
        )
    )
    scores = {
        "chart": float(chart_score),
        "table": float(table_score),
        "presentation": float(presentation_score),
    }

    strong_chart = features.dark_ratio >= 0.72 and features.mean_luma <= 100.0
    strong_table = (
        features.mean_luma >= 185.0
        and features.bright_ratio >= 0.30
        and features.horizontal_line_ratio >= 0.008
        and features.edge_density >= 0.07
    )
    presentation_canvas = (
        features.mean_luma >= 105.0
        and features.dark_ratio <= 0.60
        and features.horizontal_line_ratio <= 0.012
        and features.edge_density <= 0.25
        and (ordinary_presentation or white_embedded_visual)
    )

    if strong_chart:
        return (
            "chart",
            chart_score,
            scores,
            (
                "chart:dark_ratio>=0.72",
                "chart:mean_luma<=100",
            ),
        )
    if strong_table:
        return (
            "table",
            table_score,
            scores,
            (
                "table:mean_luma>=185",
                "table:bright_ratio>=0.30",
                "table:horizontal_line_ratio>=0.008",
                "table:edge_density>=0.07",
            ),
        )
    if presentation_canvas:
        presentation_reason = (
            "presentation:white_embedded_visual"
            if not ordinary_presentation and white_embedded_visual
            else "presentation:canvas"
        )
        return (
            "presentation",
            presentation_score,
            scores,
            (
                presentation_reason,
                "presentation:mean_luma>=105",
                "presentation:dark_ratio<=0.60",
                "presentation:horizontal_line_ratio<=0.012",
                "presentation:edge_density<=0.25",
            ),
        )

    confidence = max(0.0, min(1.0, 1.0 - max(scores.values())))
    return (
        "unknown",
        confidence,
        scores,
        ("unknown:no_conservative_layout_rule_matched",),
    )


def classify_routing_frame(
    gray,
    *,
    sample_index: int,
    timestamp: float,
) -> VisualRouteDecision:
    features = routing_frame_features(gray)
    kind, confidence, scores, reasons = classify_routing_features(features)
    return VisualRouteDecision(
        sample_index=sample_index,
        timestamp=timestamp,
        raw_kind=kind,
        kind=kind,
        confidence=confidence,
        scores=scores,
        reasons=reasons,
        features=features,
    )


def _label_runs(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    runs: list[tuple[int, int, str]] = []
    start = 0
    for index in range(1, len(labels)):
        if labels[index] != labels[start]:
            runs.append((start, index, labels[start]))
            start = index
    runs.append((start, len(labels), labels[start]))
    return runs


def _postprocess_route_labels(
    decisions: Sequence[VisualRouteDecision],
    *,
    analysis_fps: float,
    bridge_seconds: float,
    minimum_known_seconds: float,
) -> list[VisualRouteDecision]:
    labels = [decision.raw_kind for decision in decisions]
    annotations: list[list[str]] = [[] for _ in decisions]
    bridge_samples = int(math.floor(bridge_seconds * analysis_fps + 1e-9))

    if bridge_samples > 0:
        changed = True
        while changed:
            changed = False
            runs = _label_runs(labels)
            for run_index in range(1, len(runs) - 1):
                start, stop, middle_kind = runs[run_index]
                left_kind = runs[run_index - 1][2]
                right_kind = runs[run_index + 1][2]
                if stop - start > bridge_samples or left_kind != right_kind:
                    continue
                for index in range(start, stop):
                    labels[index] = left_kind
                    annotations[index].append(
                        f"postprocess:bridge:{middle_kind}->{left_kind}"
                    )
                changed = True
                break

    minimum_known_samples = max(
        1,
        int(math.ceil(minimum_known_seconds * analysis_fps - 1e-9)),
    )
    for start, stop, kind in _label_runs(labels):
        if kind == "unknown" or stop - start >= minimum_known_samples:
            continue
        for index in range(start, stop):
            labels[index] = "unknown"
            annotations[index].append(f"postprocess:short_known:{kind}->unknown")

    return [
        replace(
            decision,
            kind=labels[index],
            confidence=(
                decision.confidence
                if labels[index] == decision.raw_kind
                else 0.0
            ),
            reasons=tuple((*decision.reasons, *annotations[index])),
        )
        for index, decision in enumerate(decisions)
    ]


def detect_content_segments(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float,
    bridge_seconds: float = 1.5,
    minimum_known_seconds: float = 4.0,
) -> VisualRoutePlan:
    """Classify and coalesce eligible samples into contiguous routed spans."""
    if not samples:
        raise ValueError("Content routing requires at least one sample")
    if analysis_fps <= 0 or not math.isfinite(analysis_fps):
        raise ValueError("Routing analysis FPS must be finite and positive")
    if not math.isfinite(candidate_end_timestamp):
        raise ValueError("Routing candidate end timestamp must be finite")
    if candidate_end_timestamp < samples[0].timestamp:
        raise ValueError("Routing candidate end timestamp precedes the sampled range")
    if bridge_seconds < 0 or not math.isfinite(bridge_seconds):
        raise ValueError("Routing bridge duration must be finite and non-negative")
    if minimum_known_seconds <= 0 or not math.isfinite(minimum_known_seconds):
        raise ValueError("Routing minimum known duration must be finite and positive")

    timestamps = [sample.timestamp for sample in samples]
    eligible_count = bisect.bisect_right(timestamps, candidate_end_timestamp + 1e-9)
    if eligible_count == 0:
        raise ValueError("The routed range did not yield any eligible samples")
    raw_decisions = [
        classify_routing_frame(
            samples[index].gray,
            sample_index=index,
            timestamp=samples[index].timestamp,
        )
        for index in range(eligible_count)
    ]
    decisions = _postprocess_route_labels(
        raw_decisions,
        analysis_fps=analysis_fps,
        bridge_seconds=bridge_seconds,
        minimum_known_seconds=minimum_known_seconds,
    )

    segments: list[ContentSegment] = []
    labels = [decision.kind for decision in decisions]
    for ordinal, (start, stop, kind) in enumerate(_label_runs(labels), start=1):
        segment_decisions = decisions[start:stop]
        segment_reasons = sorted(
            {
                reason
                for decision in segment_decisions
                for reason in decision.reasons
            }
        )
        segments.append(
            ContentSegment(
                segment_id=f"route-{ordinal:03d}",
                kind=kind,
                start_sample_index=start,
                stop_sample_index=stop,
                start_timestamp=samples[start].timestamp,
                candidate_end_timestamp=samples[stop - 1].timestamp,
                confidence=float(
                    sum(decision.confidence for decision in segment_decisions)
                    / len(segment_decisions)
                ),
                selector=SELECTOR_BY_KIND[kind],
                reasons=tuple(segment_reasons),
            )
        )

    return VisualRoutePlan(
        analysis_fps=analysis_fps,
        candidate_end_timestamp=candidate_end_timestamp,
        bridge_seconds=bridge_seconds,
        minimum_known_seconds=minimum_known_seconds,
        eligible_sample_count=eligible_count,
        decisions=tuple(decisions),
        segments=tuple(segments),
    )


def _validate_routed_segments(
    samples: Sequence[Sample],
    segments: Sequence[ContentSegment],
) -> None:
    if segments and segments[0].start_sample_index != 0:
        raise ValueError("The first routed segment must start at sample index zero")
    previous_stop: int | None = None
    for segment in segments:
        expected_selector = SELECTOR_BY_KIND[segment.kind]
        if segment.selector != expected_selector:
            raise ValueError(
                f"Segment {segment.segment_id} selector does not match kind {segment.kind}"
            )
        if segment.stop_sample_index > len(samples):
            raise ValueError(f"Segment {segment.segment_id} exceeds the sampled range")
        if previous_stop is not None and segment.start_sample_index != previous_stop:
            raise ValueError("Routed segments must be contiguous and non-overlapping")
        start_sample = samples[segment.start_sample_index]
        end_sample = samples[segment.stop_sample_index - 1]
        if abs(start_sample.timestamp - segment.start_timestamp) > 1e-6:
            raise ValueError(f"Segment {segment.segment_id} start timestamp does not match samples")
        if abs(end_sample.timestamp - segment.candidate_end_timestamp) > 1e-6:
            raise ValueError(f"Segment {segment.segment_id} end timestamp does not match samples")
        previous_stop = segment.stop_sample_index


def select_routed_states(
    samples: list[Sample],
    segments: Sequence[ContentSegment],
    *,
    analysis_fps: float,
    boundary_lookahead_seconds: float = 15.0,
    table_lookahead_seconds: float = 8.0,
    presentation_lookahead_seconds: float = 6.0,
    boundary_transition_threshold: float = 0.025,
    boundary_pixel_delta_threshold: float = 0.004,
    boundary_edge_loss_threshold: float = 0.003,
    boundary_terminal_search_seconds: float = 2.0,
    boundary_terminal_stable_seconds: float = 1.0,
    boundary_terminal_stability_threshold: float = 0.008,
) -> list[Candidate]:
    """Dispatch segment slices and return copied candidates with global indexes."""
    if not samples:
        raise ValueError("Routed selection requires at least one sample")
    if analysis_fps <= 0 or not math.isfinite(analysis_fps):
        raise ValueError("Routing analysis FPS must be finite and positive")
    for name, value in (
        ("Boundary lookahead", boundary_lookahead_seconds),
        ("Table lookahead", table_lookahead_seconds),
        ("Presentation lookahead", presentation_lookahead_seconds),
    ):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    _validate_routed_segments(samples, segments)

    # Local import prevents selection.py from depending on routing models.
    from . import selection as selection_module

    selector_options = {
        "chart": (
            selection_module.select_boundary_terminal_states,
            boundary_lookahead_seconds,
        ),
        "table": (
            selection_module.select_table_viewport_states,
            table_lookahead_seconds,
        ),
        "presentation": (
            selection_module.select_presentation_states,
            presentation_lookahead_seconds,
        ),
    }
    timestamps = [sample.timestamp for sample in samples]
    routed: list[Candidate] = []
    for segment in segments:
        if segment.kind == "unknown":
            continue
        selector, lookahead_seconds = selector_options[segment.kind]
        selector_name = SELECTOR_BY_KIND[segment.kind]
        assert selector_name is not None
        lookahead_stop = bisect.bisect_right(
            timestamps,
            segment.candidate_end_timestamp + lookahead_seconds + 1e-9,
        )
        lookahead_stop = max(segment.stop_sample_index, lookahead_stop)
        local_samples = samples[segment.start_sample_index:lookahead_stop]
        selector_kwargs = {
            "analysis_fps": analysis_fps,
            "candidate_end_timestamp": segment.candidate_end_timestamp,
        }
        if segment.kind == "chart":
            selector_kwargs.update(
                {
                    "transition_threshold": boundary_transition_threshold,
                    "pixel_delta_threshold": boundary_pixel_delta_threshold,
                    "edge_loss_threshold": boundary_edge_loss_threshold,
                    "terminal_search_seconds": boundary_terminal_search_seconds,
                    "terminal_stable_seconds": boundary_terminal_stable_seconds,
                    "terminal_stability_threshold": (
                        boundary_terminal_stability_threshold
                    ),
                }
            )
        local_candidates = selector(local_samples, **selector_kwargs)
        for candidate in local_candidates:
            if not 0 <= candidate.sample_index < len(local_samples):
                raise RuntimeError(
                    f"Selector {selector_name} returned an invalid local sample index"
                )
            global_index = segment.start_sample_index + candidate.sample_index
            global_sample = samples[global_index]
            if global_index >= segment.stop_sample_index:
                raise RuntimeError(
                    f"Selector {selector_name} emitted a lookahead sample"
                )
            if candidate.timestamp > segment.candidate_end_timestamp + 1e-9:
                raise RuntimeError(
                    f"Selector {selector_name} emitted past the segment candidate end"
                )
            if abs(candidate.timestamp - global_sample.timestamp) > 1e-6:
                raise RuntimeError(
                    f"Selector {selector_name} candidate timestamp/index mismatch"
                )
            reasons = list(candidate.reasons)
            reasons.extend(
                [
                    f"route:segment={segment.segment_id}",
                    f"route:kind={segment.kind}",
                    f"route:selector={selector_name}",
                ]
            )
            routed.append(
                Candidate(
                    sample_index=global_index,
                    timestamp=global_sample.timestamp,
                    score=candidate.score,
                    reasons=sorted(set(reasons)),
                    image_path=candidate.image_path,
                    segment_id=segment.segment_id,
                )
            )
    return sorted(routed, key=lambda candidate: (candidate.timestamp, candidate.sample_index))
