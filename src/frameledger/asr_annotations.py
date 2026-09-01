from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from .report import write_json


ASR_ANCHOR_SCHEMA_VERSION = 1
ASR_ANCHOR_KIND = "asr_anchor_set"
DEFAULT_WINDOW_TOLERANCE_SECONDS = 0.75


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, *, yaml_document: bool) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Evidence file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) if yaml_document else json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not read evidence file {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Evidence document must be a mapping: {path}")
    return value


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _resolve_evidence_path(value: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _segments(document: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    transcript = document.get("transcript")
    raw_segments = transcript.get("segments") if isinstance(transcript, dict) else None
    if document.get("status") != "ok" or not isinstance(raw_segments, list):
        raise ValueError(f"{label} must be one successful ASR document with segments")
    if not all(isinstance(segment, dict) for segment in raw_segments):
        raise ValueError(f"{label} contains a non-mapping segment")
    return raw_segments


def _same_number(left: Any, right: Any) -> bool:
    return abs(_number(left, "timestamp") - _number(right, "timestamp")) <= 1e-6


def _validate_source_and_range(
    document: Mapping[str, Any],
    annotation: Mapping[str, Any],
    *,
    label: str,
) -> None:
    source = document.get("source")
    expected_source = annotation.get("source")
    if not isinstance(source, dict) or not isinstance(expected_source, dict):
        raise ValueError(f"{label} source metadata is missing")
    if source.get("fingerprint") != expected_source.get("video_sha256"):
        raise ValueError(f"{label} source fingerprint does not match the anchor set")
    actual_range = document.get("range")
    expected_range = expected_source.get("range")
    if not isinstance(actual_range, dict) or not isinstance(expected_range, dict):
        raise ValueError(f"{label} range metadata is missing")
    for key in ("start_seconds", "end_seconds", "duration_seconds"):
        if not _same_number(actual_range.get(key), expected_range.get(key)):
            raise ValueError(f"{label} range {key} does not match the anchor set")


def _validate_anchor_list(
    raw: Any,
    *,
    baseline_segments: list[dict[str, Any]],
    label: str,
    related: bool,
) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be a list")
    by_id = {segment.get("id"): segment for segment in baseline_segments}
    seen_ids: set[str] = set()
    anchors: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{label} item {index} must be a mapping")
        anchor_id = item.get("id")
        if not isinstance(anchor_id, str) or not anchor_id or anchor_id in seen_ids:
            raise ValueError(f"{label} item {index} has a missing or duplicate id")
        seen_ids.add(anchor_id)
        segment = by_id.get(item.get("segment_id"))
        if segment is None:
            raise ValueError(f"{anchor_id} does not bind to a baseline segment")
        if segment.get("text") != item.get("baseline_text"):
            raise ValueError(f"{anchor_id} baseline segment text does not match")
        for key in (
            "relative_start_seconds",
            "relative_end_seconds",
            "absolute_start_seconds",
            "absolute_end_seconds",
        ):
            if not _same_number(segment.get(key), item.get(key)):
                raise ValueError(f"{anchor_id} {key} does not match the baseline segment")
        baseline_span = item.get("baseline_span")
        expected_key = "candidate_expected_span" if related else "expected_span"
        expected_span = item.get(expected_key)
        if not isinstance(baseline_span, str) or baseline_span not in str(item["baseline_text"]):
            raise ValueError(f"{anchor_id} baseline_span is not present in baseline_text")
        if not isinstance(expected_span, str) or not expected_span:
            raise ValueError(f"{anchor_id} {expected_key} must be non-empty")
        anchors.append(dict(item))
    return anchors


def _overlaps(segment: Mapping[str, Any], start: float, end: float) -> bool:
    segment_start = _number(segment.get("absolute_start_seconds"), "segment start")
    segment_end = _number(segment.get("absolute_end_seconds"), "segment end")
    return segment_end >= start and segment_start <= end


def _score_anchor(
    anchor: Mapping[str, Any],
    *,
    candidate_segments: list[dict[str, Any]],
    tolerance_seconds: float,
    related: bool,
) -> dict[str, Any]:
    window_start = _number(anchor.get("absolute_start_seconds"), "anchor start")
    window_end = _number(anchor.get("absolute_end_seconds"), "anchor end")
    selected = [
        segment
        for segment in candidate_segments
        if _overlaps(
            segment,
            window_start - tolerance_seconds,
            window_end + tolerance_seconds,
        )
    ]
    window_text = "".join(str(segment.get("text", "")) for segment in selected)
    expected_key = "candidate_expected_span" if related else "expected_span"
    expected_span = str(anchor[expected_key])
    baseline_span = str(anchor["baseline_span"])
    expected_hit = expected_span in window_text
    baseline_persists = baseline_span in window_text
    if expected_hit:
        classification = "expected_exact"
    elif baseline_persists:
        classification = "baseline_error_persists"
    else:
        classification = "other_output"
    off_anchor_segments = [
        segment
        for segment in candidate_segments
        if expected_span in str(segment.get("text", ""))
        and not _overlaps(segment, window_start - tolerance_seconds, window_end + tolerance_seconds)
    ]
    return {
        "id": anchor["id"],
        "baseline_span": baseline_span,
        "expected_span": expected_span,
        "window": {
            "absolute_start_seconds": window_start,
            "absolute_end_seconds": window_end,
            "tolerance_seconds": tolerance_seconds,
        },
        "candidate_segment_ids": [segment.get("id") for segment in selected],
        "candidate_window_text": window_text,
        "expected_exact_hit": expected_hit,
        "baseline_error_persists": baseline_persists,
        "classification": classification,
        "off_anchor_exact_occurrences": [
            {
                "segment_id": segment.get("id"),
                "absolute_start_seconds": segment.get("absolute_start_seconds"),
                "absolute_end_seconds": segment.get("absolute_end_seconds"),
                "text": segment.get("text"),
            }
            for segment in off_anchor_segments
        ],
    }


def _quality_checks(document: Mapping[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    max_compression_ratio: float | None = None
    maximum_identical_run = 0
    current_identical_run = 0
    previous_text: str | None = None
    repeated_ngram_offender_count = 0
    for segment in segments:
        value = segment.get("compression_ratio")
        if value is not None:
            number = _number(value, "compression_ratio")
            max_compression_ratio = (
                number if max_compression_ratio is None else max(max_compression_ratio, number)
            )
        text = str(segment.get("text", "")).strip()
        if text and text == previous_text:
            current_identical_run += 1
        else:
            current_identical_run = 1 if text else 0
        maximum_identical_run = max(maximum_identical_run, current_identical_run)
        previous_text = text
        normalized_text = "".join(text.split())
        seen_ngrams: dict[str, int] = {}
        for position in range(max(0, len(normalized_text) - 32 + 1)):
            ngram = normalized_text[position : position + 32]
            first_position = seen_ngrams.get(ngram)
            if first_position is not None and position - first_position >= 32:
                repeated_ngram_offender_count += 1
                break
            seen_ngrams.setdefault(ngram, position)
    transcript = document.get("transcript")
    full_text = str(transcript.get("text", "")) if isinstance(transcript, dict) else ""
    replacement_character_count = full_text.count("\ufffd")
    pathological_repetition = (
        (max_compression_ratio is not None and max_compression_ratio > 2.4)
        or maximum_identical_run >= 3
        or replacement_character_count > 0
        or repeated_ngram_offender_count > 0
    )
    return {
        "segment_count": len(segments),
        "character_count": len(full_text),
        "replacement_character_count": replacement_character_count,
        "max_segment_compression_ratio": (
            round(max_compression_ratio, 6) if max_compression_ratio is not None else None
        ),
        "max_consecutive_identical_segment_run": maximum_identical_run,
        "repeated_character_ngram_size": 32,
        "repeated_character_ngram_offender_count": repeated_ngram_offender_count,
        "pathological_repetition_detected": pathological_repetition,
    }


def evaluate_asr_anchor_set(
    candidate_asr: str | Path,
    *,
    annotations_path: str | Path,
    output: str | Path | None = None,
    tolerance_seconds: float = DEFAULT_WINDOW_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0 or tolerance_seconds > 5:
        raise ValueError("ASR anchor tolerance must be finite and between 0 and 5 seconds")
    annotation_path = Path(annotations_path).expanduser().resolve()
    project_root = annotation_path.parent.parent
    annotation = _load_mapping(annotation_path, yaml_document=True)
    if annotation.get("schema_version") != ASR_ANCHOR_SCHEMA_VERSION:
        raise ValueError("Unsupported ASR anchor schema_version")
    if annotation.get("kind") != ASR_ANCHOR_KIND:
        raise ValueError(f"ASR annotations kind must be {ASR_ANCHOR_KIND}")
    if annotation.get("review_status") != "human_reviewed":
        raise ValueError("ASR anchors must be human_reviewed")
    if annotation.get("coverage") != "reported_issue_only":
        raise ValueError("ASR anchors must disclose reported_issue_only coverage")
    review = annotation.get("review")
    if not isinstance(review, dict) or review.get("assertions_complete_for_clip") is not False:
        raise ValueError("ASR anchors must explicitly mark the clip review as non-exhaustive")

    baseline = annotation.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("ASR anchor baseline metadata is missing")
    baseline_path = _resolve_evidence_path(
        baseline.get("asr_output"), project_root=project_root, label="baseline.asr_output"
    )
    baseline_sha = _sha256(baseline_path)
    if baseline_sha != baseline.get("asr_sha256"):
        raise ValueError("Baseline ASR SHA-256 does not match the anchor set")
    baseline_document = _load_mapping(baseline_path, yaml_document=False)
    baseline_segments = _segments(baseline_document, "Baseline ASR")
    _validate_source_and_range(baseline_document, annotation, label="Baseline ASR")
    anchors = _validate_anchor_list(
        annotation.get("anchors"),
        baseline_segments=baseline_segments,
        label="anchors",
        related=False,
    )
    related = _validate_anchor_list(
        annotation.get("related_candidates", []),
        baseline_segments=baseline_segments,
        label="related_candidates",
        related=True,
    )

    candidate_path = Path(candidate_asr).expanduser().resolve()
    candidate_document = _load_mapping(candidate_path, yaml_document=False)
    candidate_segments = _segments(candidate_document, "Candidate ASR")
    _validate_source_and_range(candidate_document, annotation, label="Candidate ASR")
    results = [
        _score_anchor(
            anchor,
            candidate_segments=candidate_segments,
            tolerance_seconds=tolerance_seconds,
            related=False,
        )
        for anchor in anchors
    ]
    related_results = [
        _score_anchor(
            anchor,
            candidate_segments=candidate_segments,
            tolerance_seconds=tolerance_seconds,
            related=True,
        )
        for anchor in related
    ]
    exact_hits = sum(bool(item["expected_exact_hit"]) for item in results)
    baseline_persists = sum(bool(item["baseline_error_persists"]) for item in results)
    off_anchor_occurrences = sum(len(item["off_anchor_exact_occurrences"]) for item in results)
    document = {
        "kind": "asr_anchor_evaluation",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "annotations": {
            "path": str(annotation_path),
            "sha256": _sha256(annotation_path),
            "coverage": annotation["coverage"],
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": baseline_sha,
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "parameters": candidate_document.get("parameters"),
            "engine": candidate_document.get("engine"),
        },
        "method": {
            "name": "time_window_exact_substring",
            "tolerance_seconds": tolerance_seconds,
            "claim_limit": "reported anchors only; not transcript accuracy or CER",
        },
        "summary": {
            "confirmed_anchor_count": len(results),
            "expected_exact_hit_count": exact_hits,
            "expected_exact_hit_rate": round(exact_hits / len(results), 6) if results else None,
            "baseline_error_persists_count": baseline_persists,
            "other_output_count": sum(
                item["classification"] == "other_output" for item in results
            ),
            "off_anchor_expected_occurrence_count": off_anchor_occurrences,
            "related_candidate_count": len(related_results),
            "related_candidate_exact_hit_count": sum(
                bool(item["expected_exact_hit"]) for item in related_results
            ),
        },
        "quality_checks": _quality_checks(candidate_document, candidate_segments),
        "anchors": results,
        "related_candidates": related_results,
    }
    if output is not None:
        output_path = Path(output).expanduser().resolve()
        if output_path.exists():
            raise ValueError(f"ASR anchor evaluation output already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, document)
        document["output"] = str(output_path)
    return document
