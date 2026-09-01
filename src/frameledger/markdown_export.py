from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import urllib.parse
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .report import write_json
from .timecode import format_timecode


MARKDOWN_SCHEMA_VERSION = 1
MARKDOWN_POLICY_VERSION = "verified-key-screenshot-markdown-v2"
MARKDOWN_FILENAME = "report.md"
MANIFEST_FILENAME = "manifest.json"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_bytes(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except OSError as error:
        raise ValueError(f"Could not read {label} JSON: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain one object")
    return raw, value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _resolve_path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.path must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    return path.resolve()


def _bound_file(
    base: Path,
    value: Any,
    expected_hash: Any,
    *,
    label: str,
) -> Path:
    path = _resolve_path(base, value, label=label)
    if not path.is_file():
        raise ValueError(f"{label}.path does not exist: {path}")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 no longer matches the semantic ledger")
    return path


def _protect_output(
    output: Path,
    *,
    semantic_path: Path,
    alignment_path: Path,
    phase1_directory: Path,
    upstream_files: Sequence[Path],
) -> None:
    if output.exists():
        raise ValueError(
            f"Markdown output already exists; refusing to overwrite: {output}"
        )
    protected = {
        semantic_path.parent.resolve(),
        alignment_path.parent.resolve(),
        phase1_directory.resolve(),
    }
    protected.update(path.parent.resolve() for path in upstream_files)
    for directory in protected:
        if output == directory or directory in output.parents:
            raise ValueError(
                "Markdown output must be a new directory outside semantic, alignment, "
                "OCR, ASR, Phase 1, and source-video inputs"
            )


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


def _event_id(value: Any, *, prefix: str, label: str) -> str:
    event_id = _text(value, label=label)
    if re.fullmatch(rf"{re.escape(prefix)}-\d+", event_id) is None:
        raise ValueError(f"{label} must use the {prefix}-<number> format")
    return event_id


def _relative_uri(target: Path, *, directory: Path) -> str:
    relative = Path(os.path.relpath(target, start=directory)).as_posix()
    return urllib.parse.quote(relative, safe="/-._~")


def _relative_markdown_link(target: Path, *, directory: Path) -> str:
    return f"<{_relative_uri(target, directory=directory)}>"


def _duplicate_count(values: Sequence[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _require_exact_event_coverage(
    *,
    label: str,
    source_ids: Sequence[str],
    rendered_ids: Sequence[str],
) -> tuple[int, int]:
    source_counter = Counter(source_ids)
    rendered_counter = Counter(rendered_ids)
    if any(count != 1 for count in source_counter.values()):
        raise ValueError(f"Alignment {label} event IDs must be unique")
    missing = sum((source_counter - rendered_counter).values())
    unexpected = sum((rendered_counter - source_counter).values())
    duplicates = _duplicate_count(rendered_ids)
    if missing or unexpected or duplicates or len(rendered_ids) != len(source_ids):
        raise ValueError(
            f"Semantic {label} assignment is not complete and one-to-one "
            f"(missing={missing}, unexpected={unexpected}, duplicates={duplicates})"
        )
    return missing, duplicates


def _validate_source_copy(
    semantic_source: Mapping[str, Any], alignment_source: Mapping[str, Any]
) -> None:
    for key in ("video", "phase1", "ocr", "asr"):
        semantic_value = _mapping(
            semantic_source.get(key), label=f"semantic source {key}"
        )
        alignment_value = _mapping(
            alignment_source.get(key), label=f"alignment source {key}"
        )
        if dict(semantic_value) != dict(alignment_value):
            raise ValueError(
                f"Semantic source {key} no longer matches alignment source"
            )


def _render_markdown(
    *,
    output_path: Path,
    visual_ids: Sequence[str],
    visual_by_id: Mapping[str, Mapping[str, Any]],
    image_by_id: Mapping[str, Path],
) -> str:
    lines: list[str] = []
    for visual_id in visual_ids:
        frame = visual_by_id[visual_id]
        timestamp = _number(frame.get("timestamp"), label=f"{visual_id}.timestamp")
        image_link = _relative_markdown_link(
            image_by_id[visual_id], directory=output_path
        )
        lines.extend(
            [
                f"![{format_timecode(timestamp)}]({image_link})",
                "",
            ]
        )
    return "\n".join(lines)


def run_markdown_export(
    semantic_json: str | Path,
    *,
    output: str | Path,
) -> dict[str, Any]:
    """Combine bound semantic chapters and original evidence frames in Markdown."""

    semantic_path = Path(semantic_json).expanduser().resolve()
    if not semantic_path.is_file():
        raise ValueError(f"Semantic JSON does not exist: {semantic_path}")
    semantic_raw, semantic = _load_json_bytes(semantic_path, label="semantic")
    semantic_sha256 = hashlib.sha256(semantic_raw).hexdigest()
    if semantic.get("kind") != "local_topic_segmentation":
        raise ValueError("Markdown input kind must be local_topic_segmentation")
    if semantic.get("schema_version") != 1:
        raise ValueError("Markdown input semantic schema_version must be 1")
    semantic_coverage = _mapping(semantic.get("coverage"), label="semantic coverage")
    if semantic_coverage.get("complete_event_assignment") is not True:
        raise ValueError("Markdown export requires complete semantic event assignment")

    semantic_source = _mapping(semantic.get("source"), label="semantic source")
    alignment_source = _mapping(
        semantic_source.get("alignment"), label="semantic alignment source"
    )
    alignment_path = _bound_file(
        semantic_path,
        alignment_source.get("path"),
        alignment_source.get("sha256"),
        label="alignment",
    )
    alignment_raw, alignment = _load_json_bytes(alignment_path, label="alignment")
    alignment_sha256 = hashlib.sha256(alignment_raw).hexdigest()
    if alignment.get("kind") != "timestamp_aligned_evidence":
        raise ValueError("Markdown alignment kind must be timestamp_aligned_evidence")
    if alignment.get("schema_version") != 1:
        raise ValueError("Markdown alignment schema_version must be 1")
    alignment_coverage = _mapping(alignment.get("coverage"), label="alignment coverage")
    if alignment_coverage.get("complete_phase1_speech_coverage") is not True:
        raise ValueError("Markdown export requires complete alignment speech coverage")
    alignment_source_data = _mapping(alignment.get("source"), label="alignment source")
    _validate_source_copy(semantic_source, alignment_source_data)

    phase1 = _mapping(semantic_source.get("phase1"), label="semantic Phase 1 source")
    phase1_directory = _resolve_path(
        semantic_path, phase1.get("run_directory"), label="Phase 1 run directory"
    )
    if not phase1_directory.is_dir():
        raise ValueError(f"Phase 1 run directory does not exist: {phase1_directory}")
    video = _mapping(semantic_source.get("video"), label="semantic video source")
    ocr = _mapping(semantic_source.get("ocr"), label="semantic OCR source")
    asr = _mapping(semantic_source.get("asr"), label="semantic ASR source")
    manifest_path = _bound_file(
        semantic_path,
        phase1.get("manifest_path"),
        phase1.get("manifest_sha256"),
        label="Phase 1 manifest",
    )
    strategy_path = _bound_file(
        semantic_path,
        phase1.get("strategy_path"),
        phase1.get("strategy_sha256"),
        label="Phase 1 strategy",
    )
    video_path = _bound_file(
        semantic_path,
        video.get("path"),
        video.get("fingerprint"),
        label="source video",
    )
    ocr_path = _bound_file(
        semantic_path, ocr.get("path"), ocr.get("sha256"), label="OCR"
    )
    asr_path = _bound_file(
        semantic_path, asr.get("path"), asr.get("sha256"), label="ASR"
    )
    audio_path = _bound_file(
        semantic_path,
        asr.get("audio_path"),
        asr.get("audio_sha256"),
        label="ASR audio",
    )

    output_path = Path(output).expanduser().resolve()
    upstream_files = (
        manifest_path,
        strategy_path,
        video_path,
        ocr_path,
        asr_path,
        audio_path,
    )
    _protect_output(
        output_path,
        semantic_path=semantic_path,
        alignment_path=alignment_path,
        phase1_directory=phase1_directory,
        upstream_files=upstream_files,
    )

    visual_items = _list(
        alignment.get("visual_frames"), label="alignment visual_frames"
    )
    speech_items = _list(
        alignment.get("speech_segments"), label="alignment speech_segments"
    )
    chapters_raw = _list(semantic.get("chapters"), label="semantic chapters")
    chapters = [
        _mapping(item, label=f"semantic chapters[{index}]")
        for index, item in enumerate(chapters_raw)
    ]
    if not chapters:
        raise ValueError("Markdown export requires at least one semantic chapter")

    visual_by_id: dict[str, Mapping[str, Any]] = {}
    image_by_id: dict[str, Path] = {}
    image_hashes: dict[str, str] = {}
    previous_visual = -math.inf
    for index, item in enumerate(visual_items):
        frame = _mapping(item, label=f"alignment visual_frames[{index}]")
        event_id = _event_id(
            frame.get("event_id"),
            prefix="visual",
            label=f"visual_frames[{index}].event_id",
        )
        if event_id in visual_by_id:
            raise ValueError(f"Duplicate alignment visual event ID: {event_id}")
        timestamp = _number(frame.get("timestamp"), label=f"{event_id}.timestamp")
        if timestamp < previous_visual - 1e-6:
            raise ValueError("Alignment visual frames are not chronological")
        previous_visual = timestamp
        image = _mapping(frame.get("image"), label=f"{event_id}.image")
        relative_value = image.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            raise ValueError(f"{event_id}.image.path must be a non-empty relative path")
        relative_path = Path(relative_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"{event_id}.image.path must stay inside the Phase 1 run")
        image_path = (phase1_directory / relative_path).resolve()
        if not image_path.is_relative_to(phase1_directory) or not image_path.is_file():
            raise ValueError(
                f"Evidence image does not exist inside Phase 1: {image_path}"
            )
        expected_hash = image.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(image_path) != expected_hash:
            raise ValueError(f"{event_id} image SHA-256 no longer matches alignment")
        visual_by_id[event_id] = frame
        image_by_id[event_id] = image_path
        image_hashes[event_id] = expected_hash

    speech_by_id: dict[str, Mapping[str, Any]] = {}
    previous_speech = -math.inf
    for index, item in enumerate(speech_items):
        speech = _mapping(item, label=f"alignment speech_segments[{index}]")
        event_id = _event_id(
            speech.get("event_id"),
            prefix="speech",
            label=f"speech_segments[{index}].event_id",
        )
        if event_id in speech_by_id:
            raise ValueError(f"Duplicate alignment speech event ID: {event_id}")
        start = _number(
            speech.get("absolute_start_seconds"),
            label=f"{event_id}.absolute_start_seconds",
        )
        if start < previous_speech - 1e-6:
            raise ValueError("Alignment speech segments are not chronological")
        previous_speech = start
        speech_by_id[event_id] = speech

    chapter_ids: list[str] = []
    rendered_visual_ids: list[str] = []
    rendered_speech_ids: list[str] = []
    previous_end: float | None = None
    for index, chapter in enumerate(chapters):
        chapter_id = _event_id(
            chapter.get("chapter_id"),
            prefix="chapter",
            label=f"chapters[{index}].chapter_id",
        )
        if chapter_id in chapter_ids:
            raise ValueError(f"Duplicate semantic chapter ID: {chapter_id}")
        chapter_ids.append(chapter_id)
        start = _number(
            chapter.get("start_seconds"), label=f"{chapter_id}.start_seconds"
        )
        end = _number(chapter.get("end_seconds"), label=f"{chapter_id}.end_seconds")
        if end <= start:
            raise ValueError(f"{chapter_id} range is invalid")
        if previous_end is not None and not math.isclose(
            start, previous_end, abs_tol=1e-6
        ):
            raise ValueError("Semantic chapters are not contiguous")
        previous_end = end
        speech_ids = [
            str(value)
            for value in _list(
                chapter.get("speech_event_ids"), label=f"{chapter_id}.speech_event_ids"
            )
        ]
        visual_ids = [
            str(value)
            for value in _list(
                chapter.get("visual_event_ids"), label=f"{chapter_id}.visual_event_ids"
            )
        ]
        if any(event_id not in speech_by_id for event_id in speech_ids):
            raise ValueError(f"{chapter_id} references an unknown speech event")
        if any(event_id not in visual_by_id for event_id in visual_ids):
            raise ValueError(f"{chapter_id} references an unknown visual event")
        is_last_chapter = index == len(chapters) - 1
        for event_id in speech_ids:
            event_start = _number(
                speech_by_id[event_id].get("absolute_start_seconds"),
                label=f"{event_id}.absolute_start_seconds",
            )
            if not (
                start <= event_start < end
                or (is_last_chapter and math.isclose(event_start, end, abs_tol=1e-6))
            ):
                raise ValueError(f"{chapter_id} contains speech outside its time range")
        for event_id in visual_ids:
            event_time = _number(
                visual_by_id[event_id].get("timestamp"), label=f"{event_id}.timestamp"
            )
            if not (
                start <= event_time < end
                or (is_last_chapter and math.isclose(event_time, end, abs_tol=1e-6))
            ):
                raise ValueError(
                    f"{chapter_id} contains a visual frame outside its time range"
                )
        title_source = chapter.get("title_source_event_id")
        if chapter.get("title_exact_extract") is True:
            if not isinstance(title_source, str) or title_source not in speech_by_id:
                raise ValueError(f"{chapter_id} has an invalid title source event")
            if str(chapter.get("title", "")) != str(
                speech_by_id[title_source].get("text", "")
            ):
                raise ValueError(f"{chapter_id} title is not an exact ASR extract")
        rendered_speech_ids.extend(speech_ids)
        rendered_visual_ids.extend(visual_ids)

    coverage_start = _number(
        semantic_coverage.get("start_seconds"), label="semantic coverage start_seconds"
    )
    coverage_end = _number(
        semantic_coverage.get("end_seconds"), label="semantic coverage end_seconds"
    )
    if not math.isclose(
        _number(chapters[0].get("start_seconds"), label="first chapter start"),
        coverage_start,
        abs_tol=1e-6,
    ) or not math.isclose(
        _number(chapters[-1].get("end_seconds"), label="last chapter end"),
        coverage_end,
        abs_tol=1e-6,
    ):
        raise ValueError("Semantic chapters do not span the declared complete range")

    speech_missing, speech_duplicates = _require_exact_event_coverage(
        label="speech",
        source_ids=list(speech_by_id),
        rendered_ids=rendered_speech_ids,
    )
    visual_missing, visual_duplicates = _require_exact_event_coverage(
        label="visual",
        source_ids=list(visual_by_id),
        rendered_ids=rendered_visual_ids,
    )

    ordered_visual_ids = sorted(
        rendered_visual_ids,
        key=lambda event_id: float(visual_by_id[event_id]["timestamp"]),
    )
    markdown_text = _render_markdown(
        output_path=output_path,
        visual_ids=ordered_visual_ids,
        visual_by_id=visual_by_id,
        image_by_id=image_by_id,
    )
    markdown_bytes = markdown_text.encode("utf-8")
    markdown_sha256 = hashlib.sha256(markdown_bytes).hexdigest()
    coverage = {
        "scope": "visual_frames_only",
        "source_visual_frame_count": len(visual_by_id),
        "rendered_visual_frame_count": len(ordered_visual_ids),
        "unrendered_visual_frame_count": visual_missing,
        "duplicate_visual_render_count": visual_duplicates,
        "complete_report_coverage": True,
    }
    summary = {
        "visual_frame_count": len(ordered_visual_ids),
        "unique_image_sha256_count": len(set(image_hashes.values())),
        "repeated_image_sha256_count": len(image_hashes)
        - len(set(image_hashes.values())),
        "markdown_bytes": len(markdown_bytes),
        "markdown_sha256": markdown_sha256,
    }
    manifest = {
        "kind": "local_evidence_markdown",
        "schema_version": MARKDOWN_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "policy_version": MARKDOWN_POLICY_VERSION,
            "content_scope": "verified_key_screenshots_only",
            "image_order": "absolute_timestamp_ascending",
            "image_policy": "reference_every_bound_visual_event_once",
            "semantic_assignment_validated": True,
            "speech_assignment_validated": speech_missing == 0
            and speech_duplicates == 0,
            "ocr_text_rendered": False,
            "asr_text_rendered": False,
            "semantic_text_rendered": False,
            "source_images_referenced": True,
            "images_copied": False,
            "correction_applied": False,
            "summarization_applied": False,
            "network_ai_used": False,
            "chat_model_used": False,
        },
        "source": {
            "semantic": {"path": str(semantic_path), "sha256": semantic_sha256},
            "alignment": {"path": str(alignment_path), "sha256": alignment_sha256},
            "video": dict(video),
            "phase1": dict(phase1),
            "ocr": dict(ocr),
            "asr": dict(asr),
        },
        "artifact": {
            "path": MARKDOWN_FILENAME,
            "sha256": markdown_sha256,
            "size_bytes": len(markdown_bytes),
        },
        "evidence_images": [
            {
                "event_id": event_id,
                "path": str(image_by_id[event_id]),
                "sha256": image_hashes[event_id],
            }
            for event_id in ordered_visual_ids
        ],
        "coverage": coverage,
        "summary": summary,
    }

    bound_checks = [
        (semantic_path, semantic_sha256, "semantic"),
        (alignment_path, alignment_sha256, "alignment"),
        (manifest_path, str(phase1["manifest_sha256"]), "Phase 1 manifest"),
        (strategy_path, str(phase1["strategy_sha256"]), "Phase 1 strategy"),
        (video_path, str(video["fingerprint"]), "source video"),
        (ocr_path, str(ocr["sha256"]), "OCR"),
        (asr_path, str(asr["sha256"]), "ASR"),
        (audio_path, str(asr["audio_sha256"]), "ASR audio"),
    ]
    bound_checks.extend(
        (image_by_id[event_id], image_hashes[event_id], event_id)
        for event_id in ordered_visual_ids
    )
    for path, expected_hash, label in bound_checks:
        if _sha256(path) != expected_hash:
            raise ValueError(f"{label} changed while Markdown export was running")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = (
        output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_directory.mkdir(exist_ok=False)
    try:
        (temporary_directory / MARKDOWN_FILENAME).write_bytes(markdown_bytes)
        write_json(temporary_directory / MANIFEST_FILENAME, manifest)
        if output_path.exists():
            raise ValueError(
                f"Markdown output appeared during export; refusing to overwrite: {output_path}"
            )
        temporary_directory.replace(output_path)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    markdown_path = output_path / MARKDOWN_FILENAME
    manifest_path_out = output_path / MANIFEST_FILENAME
    return {
        "kind": manifest["kind"],
        "output": str(output_path),
        "markdown_file": str(markdown_path),
        "manifest_json": str(manifest_path_out),
        "source": manifest["source"],
        "summary": summary,
        "coverage": coverage,
    }
