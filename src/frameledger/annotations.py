from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import yaml

from .models import Candidate
from .timecode import parse_timecode


class AnnotationError(ValueError):
    pass


def load_annotations(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    if not annotation_path.is_file():
        raise AnnotationError(f"Annotation file does not exist: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AnnotationError("Annotation root must be a mapping")
    for key in ("must_keep", "nice_to_keep", "avoid_duplicates"):
        value = data.get(key, [])
        if not isinstance(value, list):
            raise AnnotationError(f"{key} must be a list")
    review = data.get("review")
    if review is not None and not isinstance(review, dict):
        raise AnnotationError("review must be a mapping")
    return data


def validate_annotations(
    data: dict[str, Any],
    *,
    video_name: str,
    start_seconds: float,
    duration_seconds: float,
    source_sha256: str | None = None,
) -> None:
    try:
        schema_version = int(data.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise AnnotationError("schema_version must be an integer") from error
    if schema_version < 1:
        raise AnnotationError("schema_version must be positive")

    declared_video = data.get("video")
    if isinstance(declared_video, dict):
        declared_video = declared_video.get("basename")
    if declared_video and Path(str(declared_video)).name != video_name:
        raise AnnotationError(
            f"Annotation video {declared_video!r} does not match input {video_name!r}"
        )
    declared_sha256 = data.get("source_sha256")
    if declared_sha256 and source_sha256 and str(declared_sha256) != source_sha256:
        raise AnnotationError("Annotation source_sha256 does not match the benchmark source")
    segment = data.get("segment")
    if segment is not None:
        if not isinstance(segment, dict):
            raise AnnotationError("segment must be a mapping")
        declared_start = parse_timecode(segment.get("start", start_seconds))
        declared_duration = parse_timecode(segment.get("duration", duration_seconds))
        if abs(declared_start - start_seconds) > 0.050 or abs(declared_duration - duration_seconds) > 0.050:
            raise AnnotationError(
                "Annotation segment does not match the benchmark --start/--duration"
            )
    review = data.get("review", {})
    if review:
        coverage = review.get("coverage", "positive_only")
        if coverage not in ("positive_only", "exhaustive", "strategy_candidate_set"):
            raise AnnotationError(
                "review.coverage must be positive_only, exhaustive, or strategy_candidate_set"
            )
        scope = review.get("scope")
        if scope not in (None, "full_segment", "selected_candidates"):
            raise AnnotationError(
                "review.scope must be full_segment or selected_candidates when provided"
            )
        if coverage == "exhaustive":
            if scope != "full_segment" or review.get("unlisted_states") != "reject":
                raise AnnotationError(
                    "exhaustive review requires scope=full_segment and unlisted_states=reject"
                )
            if schema_version >= 4:
                source_review = review.get("source_review")
                if not isinstance(source_review, dict):
                    raise AnnotationError(
                        "schema_version>=4 exhaustive review requires review.source_review"
                    )
                if (
                    source_review.get("basis") != "source_video"
                    or source_review.get("full_segment_watched") is not True
                    or source_review.get("algorithm_outputs_used_as_labels") is not False
                ):
                    raise AnnotationError(
                        "review.source_review must attest basis=source_video, "
                        "full_segment_watched=true, and "
                        "algorithm_outputs_used_as_labels=false"
                    )

                confirmation = review.get("confirmation")
                if not isinstance(confirmation, dict):
                    raise AnnotationError(
                        "schema_version>=4 exhaustive review requires review.confirmation"
                    )
                if (
                    confirmation.get("confirmed_by") != "user"
                    or confirmation.get("labels_complete") is not True
                    or confirmation.get("plateau_windows_confirmed") is not True
                ):
                    raise AnnotationError(
                        "review.confirmation must attest confirmed_by=user, "
                        "labels_complete=true, and plateau_windows_confirmed=true"
                    )
        if coverage == "strategy_candidate_set":
            fingerprint = str(review.get("strategy_file_sha256", ""))
            if (
                scope != "selected_candidates"
                or review.get("unlisted_candidates") != "reject"
                or not review.get("strategy")
                or len(fingerprint) != 64
            ):
                raise AnnotationError(
                    "strategy_candidate_set review requires selected_candidates scope, "
                    "a strategy, a 64-character strategy_file_sha256, and "
                    "unlisted_candidates=reject"
                )
        if coverage in ("exhaustive", "strategy_candidate_set") and data.get(
            "review_status"
        ) != "human_reviewed":
            raise AnnotationError(
                f"review.coverage={coverage} requires review_status=human_reviewed"
            )


def _score_points(points: list[dict[str, Any]], timestamps: list[float]) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    for original_index, item in enumerate(points):
        if not isinstance(item, dict) or "timestamp" not in item:
            raise AnnotationError("Each keep annotation requires timestamp")
        expected = parse_timecode(item["timestamp"])
        has_window = "acceptable_start" in item or "acceptable_end" in item
        if has_window:
            if "acceptable_start" not in item or "acceptable_end" not in item:
                raise AnnotationError(
                    "acceptable_start and acceptable_end must be provided together"
                )
            acceptable_start = parse_timecode(item["acceptable_start"])
            acceptable_end = parse_timecode(item["acceptable_end"])
            if acceptable_start > acceptable_end:
                raise AnnotationError("acceptable_start must not be after acceptable_end")
            if not acceptable_start <= expected <= acceptable_end:
                raise AnnotationError(
                    "timestamp must fall inside acceptable_start/acceptable_end"
                )
            tolerance = None
        else:
            tolerance = float(item.get("tolerance_seconds", 3.0))
            if tolerance < 0:
                raise AnnotationError("tolerance_seconds must be non-negative")
            acceptable_start = expected - tolerance
            acceptable_end = expected + tolerance
        parsed.append(
            {
                "original_index": original_index,
                "timestamp": expected,
                "tolerance_seconds": tolerance,
                "acceptable_start": acceptable_start,
                "acceptable_end": acceptable_end,
                "reason": item.get("reason", ""),
                "kind": item.get("kind"),
            }
        )
    ordered_points = sorted(parsed, key=lambda item: (item["timestamp"], item["original_index"]))
    ordered_candidates = sorted(timestamps)

    # Dynamic programming yields a one-to-one, order-preserving maximum-cardinality
    # match, then minimizes total absolute timestamp error.
    rows = len(ordered_points) + 1
    columns = len(ordered_candidates) + 1
    dp: list[list[tuple[int, float, tuple[tuple[int, int], ...]]]] = [
        [(0, 0.0, ()) for _ in range(columns)] for _ in range(rows)
    ]

    def better(
        left: tuple[int, float, tuple[tuple[int, int], ...]],
        right: tuple[int, float, tuple[tuple[int, int], ...]],
    ) -> tuple[int, float, tuple[tuple[int, int], ...]]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        if abs(left[1] - right[1]) > 1e-12:
            return left if left[1] < right[1] else right
        return left if left[2] <= right[2] else right

    for point_index in range(1, rows):
        for candidate_index in range(1, columns):
            best = better(dp[point_index - 1][candidate_index], dp[point_index][candidate_index - 1])
            point = ordered_points[point_index - 1]
            candidate_time = ordered_candidates[candidate_index - 1]
            error = abs(candidate_time - float(point["timestamp"]))
            if float(point["acceptable_start"]) <= candidate_time <= float(
                point["acceptable_end"]
            ):
                previous = dp[point_index - 1][candidate_index - 1]
                matched = (
                    previous[0] + 1,
                    previous[1] + error,
                    previous[2] + ((point_index - 1, candidate_index - 1),),
                )
                best = better(best, matched)
            dp[point_index][candidate_index] = best

    matched_pairs = {point_index: candidate_index for point_index, candidate_index in dp[-1][-1][2]}
    matches: list[dict[str, Any]] = []
    for point_index, point in enumerate(ordered_points):
        candidate_index = matched_pairs.get(point_index)
        nearest = ordered_candidates[candidate_index] if candidate_index is not None else None
        error = abs(nearest - point["timestamp"]) if nearest is not None else None
        matches.append(
            {
                **{key: value for key, value in point.items() if key != "original_index"},
                "matched": candidate_index is not None,
                "nearest_selected_timestamp": nearest,
                "error_seconds": error,
                "original_index": point["original_index"],
            }
        )
    matches.sort(key=lambda item: item["original_index"])
    for item in matches:
        item.pop("original_index", None)
    matched_errors = [float(item["error_seconds"]) for item in matches if item["matched"]]
    return {
        "total": len(matches),
        "matched": sum(1 for item in matches if item["matched"]),
        "recall": (sum(1 for item in matches if item["matched"]) / len(matches)) if matches else None,
        "median_timestamp_error_seconds": statistics.median(matched_errors) if matched_errors else None,
        "items": matches,
    }


def evaluate_annotations(
    data: dict[str, Any],
    candidates: list[Candidate],
    *,
    strategy_name: str | None = None,
    candidate_set_sha256: str | None = None,
) -> dict[str, Any]:
    timestamps = [candidate.timestamp for candidate in candidates]
    must_keep = _score_points(data.get("must_keep", []), timestamps)
    nice_to_keep = _score_points(data.get("nice_to_keep", []), timestamps)
    duplicate_windows: list[dict[str, Any]] = []
    for item in data.get("avoid_duplicates", []):
        if not isinstance(item, dict):
            raise AnnotationError("Each avoid_duplicates annotation must be a mapping")
        start = parse_timecode(item["start"])
        end = parse_timecode(item["end"])
        maximum = int(item["max_selected_frames"])
        count = sum(1 for timestamp in timestamps if start <= timestamp <= end)
        duplicate_windows.append(
            {
                "start": start,
                "end": end,
                "max_selected_frames": maximum,
                "selected_frames": count,
                "passed": count <= maximum,
                "reason": item.get("reason", ""),
            }
        )
    review = data.get("review", {})
    coverage = review.get("coverage", "positive_only") if isinstance(review, dict) else "positive_only"
    full_segment_review = (
        data.get("review_status") == "human_reviewed"
        and isinstance(review, dict)
        and coverage == "exhaustive"
        and review.get("scope") == "full_segment"
        and review.get("unlisted_states") == "reject"
    )
    reviewed_candidate_set = (
        data.get("review_status") == "human_reviewed"
        and isinstance(review, dict)
        and coverage == "strategy_candidate_set"
        and review.get("scope") == "selected_candidates"
        and review.get("unlisted_candidates") == "reject"
        and strategy_name == review.get("strategy")
    )
    selection_quality = None
    if full_segment_review or reviewed_candidate_set:
        if reviewed_candidate_set:
            expected_fingerprint = str(review.get("strategy_file_sha256"))
            if candidate_set_sha256 != expected_fingerprint:
                raise AnnotationError(
                    "Reviewed strategy candidate set does not match strategy_file_sha256"
                )
            expected_count = review.get("reviewed_candidate_count")
            if expected_count is not None and int(expected_count) != len(timestamps):
                raise AnnotationError(
                    "Reviewed strategy candidate count does not match reviewed_candidate_count"
                )
        accepted_states = _score_points(
            data.get("must_keep", []) + data.get("nice_to_keep", []),
            timestamps,
        )
        accepted_matches = int(accepted_states["matched"])
        accepted_total = int(accepted_states["total"])
        selected_total = len(timestamps)
        precision = accepted_matches / selected_total if selected_total else None
        recall = accepted_matches / accepted_total if full_segment_review and accepted_total else None
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        selection_quality = {
            "review_basis": "full_segment" if full_segment_review else "strategy_candidate_set",
            "review_scope": review.get("scope"),
            "reviewed_strategy": review.get("strategy") if reviewed_candidate_set else None,
            "selection_policy": review.get("selection_policy", "unspecified"),
            "selected_frames": selected_total,
            "accepted_states": accepted_total,
            "accepted_matches": accepted_matches,
            "unlisted_selected_frames": selected_total - accepted_matches,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "source_segment_recall_assessable": full_segment_review,
        }

    if data.get("review_status") != "human_reviewed":
        ground_truth = "annotation_yaml_not_confirmed_as_ground_truth"
    elif coverage == "exhaustive":
        ground_truth = "human_reviewed_full_segment"
    elif coverage == "strategy_candidate_set":
        ground_truth = "human_reviewed_strategy_candidate_set"
    else:
        ground_truth = "human_reviewed_positive_labels_not_exhaustive"

    return {
        "annotation_status": data.get("review_status", "unspecified"),
        "ground_truth": ground_truth,
        "must_keep": must_keep,
        "nice_to_keep": nice_to_keep,
        "selection_quality": selection_quality,
        "avoid_duplicates": {
            "total": len(duplicate_windows),
            "passed": sum(1 for item in duplicate_windows if item["passed"]),
            "items": duplicate_windows,
        },
    }


def candidate_from_dict(data: dict[str, Any]) -> Candidate:
    return Candidate(
        sample_index=int(data["sample_index"]),
        timestamp=float(data["timestamp"]),
        score=float(data.get("score", 0.0)),
        reasons=list(data.get("reasons", [])),
        image_path=data.get("image_path"),
        segment_id=data.get("segment_id"),
    )
