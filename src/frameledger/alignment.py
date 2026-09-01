from __future__ import annotations

import hashlib
import html
import json
import math
import os
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .artifacts import resolve_strategy_artifact
from .report import write_json
from .timecode import format_timecode


ALIGNMENT_SCHEMA_VERSION = 1
ALIGNMENT_POLICY_VERSION = "absolute-time-point-overlap-v1"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain one object")
    return payload


def _number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result != value:
        raise ValueError(f"{label} must be an integer")
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _relative_file(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to its evidence directory")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its evidence directory") from error
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _range(value: Any, *, label: str) -> dict[str, float]:
    payload = _mapping(value, label=label)
    start = _number(payload.get("start_seconds"), label=f"{label}.start_seconds")
    end = _number(payload.get("end_seconds"), label=f"{label}.end_seconds")
    if start < 0 or end <= start:
        raise ValueError(f"{label} must have non-negative start and end after start")
    return {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": round(end - start, 6),
        "start_timecode": format_timecode(start),
        "end_timecode": format_timecode(end),
    }


def _protected_output(output: Path, protected: tuple[Path, ...]) -> None:
    for directory in protected:
        if output == directory or directory in output.parents:
            raise ValueError(
                "Alignment output must be a new directory outside Phase 1, OCR, and ASR inputs"
            )


def _ocr_records(document: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for section in ("frames", "skipped", "failures"):
        for position, raw in enumerate(_list(document.get(section), label=f"OCR {section}")):
            record = dict(_mapping(raw, label=f"OCR {section}[{position}]"))
            sample_index = _integer(
                record.get("sample_index"),
                label=f"OCR {section}[{position}].sample_index",
            )
            if sample_index in records:
                raise ValueError(f"OCR evidence repeats sample_index {sample_index}")
            records[sample_index] = record
    return records


def _microseconds(seconds: float) -> int:
    return int(round(seconds * 1_000_000))


def _relation(segment: Mapping[str, Any], timestamp: float) -> tuple[str, float]:
    start_us = _microseconds(float(segment["absolute_start_seconds"]))
    end_us = _microseconds(float(segment["absolute_end_seconds"]))
    timestamp_us = _microseconds(timestamp)
    if end_us <= timestamp_us:
        return "before_frame", round((timestamp_us - end_us) / 1_000_000, 6)
    if start_us > timestamp_us:
        return "after_frame", round((start_us - timestamp_us) / 1_000_000, 6)
    return "overlaps_frame", 0.0


def _review_html(
    document: Mapping[str, Any],
    *,
    audio_path: Path,
    image_paths: Mapping[str, Path],
    review_directory: Path,
) -> str:
    source = _mapping(document["source"], label="alignment source")
    video = _mapping(source["video"], label="alignment video")
    coverage = _mapping(document["coverage"], label="alignment coverage")
    phase1_range = _mapping(coverage["phase1_range"], label="Phase 1 range")
    asr_range = _mapping(coverage["asr_range"], label="ASR range")
    summary = _mapping(document["summary"], label="alignment summary")
    speech_by_id = {
        str(segment["event_id"]): segment for segment in document["speech_segments"]
    }

    phase_start = float(phase1_range["start_seconds"])
    phase_end = float(phase1_range["end_seconds"])
    phase_span = max(1e-9, phase_end - phase_start)
    speech_left = (float(asr_range["start_seconds"]) - phase_start) / phase_span * 100.0
    speech_width = (
        float(asr_range["end_seconds"]) - float(asr_range["start_seconds"])
    ) / phase_span * 100.0

    markers: list[str] = []
    cards: list[str] = []

    def asset_uri(path: Path) -> str:
        workflow_root = review_directory.parent
        if path.is_relative_to(workflow_root):
            relative = Path(os.path.relpath(path, review_directory)).as_posix()
            return urllib.parse.quote(relative, safe="/._-")
        return path.as_uri()

    for frame in document["visual_frames"]:
        event_id = str(frame["event_id"])
        timestamp = float(frame["timestamp"])
        left = (timestamp - phase_start) / phase_span * 100.0
        alignment = _mapping(frame["temporal_alignment"], label="temporal alignment")
        context = _mapping(frame["nearby_speech"], label="nearby speech")
        alignment_status = str(alignment["status"])
        marker_class = (
            "marker aligned"
            if alignment_status == "segment_at_timestamp"
            else "marker outside"
        )
        seek = (
            f" data-time='{timestamp:.6f}'"
            if float(asr_range["start_seconds"]) <= timestamp < float(asr_range["end_seconds"])
            else ""
        )
        markers.append(
            f"<button class='{marker_class}' style='left:{left:.4f}%' title='"
            f"{html.escape(format_timecode(timestamp), quote=True)}'{seek}></button>"
        )

        image_path = image_paths[event_id]
        ocr = _mapping(frame["ocr"], label="frame OCR")
        observations = ocr.get("observations")
        if not isinstance(observations, list):
            observations = []
        boxes: list[str] = []
        ocr_rows: list[str] = []
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            text = html.escape(str(observation.get("text", "")))
            confidence = observation.get("confidence")
            confidence_text = "n/a" if confidence is None else f"{float(confidence):.1%}"
            bbox = observation.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x, y, width, height = (float(item) * 1000.0 for item in bbox)
                boxes.append(
                    f"<rect x='{x:.3f}' y='{y:.3f}' width='{width:.3f}' "
                    f"height='{height:.3f}'><title>{text} · {confidence_text}</title></rect>"
                )
            ocr_rows.append(
                f"<li><span class='confidence'>{confidence_text}</span>{text}</li>"
            )

        speech_rows: list[str] = []
        for relation in context.get("segments", []):
            if not isinstance(relation, Mapping):
                continue
            segment = speech_by_id.get(str(relation.get("event_id")))
            if segment is None:
                continue
            segment_start = float(segment["absolute_start_seconds"])
            segment_end = float(segment["absolute_end_seconds"])
            relation_name = str(relation.get("relation", "nearby"))
            speech_rows.append(
                "<li class='speech-row'>"
                f"<button class='seek' data-time='{segment_start:.6f}'>"
                f"{format_timecode(segment_start)}–{format_timecode(segment_end)}</button>"
                f"<span class='relation'>{html.escape(relation_name)}</span>"
                f"<span>{html.escape(str(segment.get('text', '')))}</span></li>"
            )
        if alignment_status == "outside_asr_range":
            speech_section = (
                "<p class='coverage-warning'>该视觉帧超出当前 ASR "
                f"{format_timecode(float(asr_range['start_seconds']))}–"
                f"{format_timecode(float(asr_range['end_seconds']))} 覆盖范围；"
                "未建立语音关系，也未自动重跑 ASR。</p>"
            )
        elif speech_rows:
            speech_section = f"<ol class='speech-list'>{''.join(speech_rows)}</ol>"
        else:
            speech_section = "<p class='coverage-warning'>上下文窗口内没有语音段。</p>"

        selection = _mapping(frame["selection"], label="frame selection")
        reasons = ", ".join(str(item) for item in selection.get("reasons", []))
        cards.append(
            f"<article class='frame-card' id='{html.escape(event_id, quote=True)}'>"
            "<div class='visual'>"
            f"<img src='{html.escape(asset_uri(image_path), quote=True)}' alt='Evidence frame'>"
            f"<svg class='boxes' viewBox='0 0 1000 1000' preserveAspectRatio='none'>"
            f"{''.join(boxes)}</svg></div>"
            "<div class='evidence'>"
            f"<h2>{format_timecode(timestamp)}</h2>"
            f"<p><span class='badge'>{html.escape(str(frame['route_kind']))}</span> "
            f"sample {int(frame['sample_index'])} · {html.escape(alignment_status)}</p>"
            f"<p class='reasons'>{html.escape(reasons)}</p>"
            f"<h3>OCR · {html.escape(str(ocr.get('status', 'unknown')))}</h3>"
            f"<ol class='ocr-list'>{''.join(ocr_rows) or '<li>No OCR text.</li>'}</ol>"
            "<h3>Nearby raw ASR</h3>"
            f"{speech_section}</div></article>"
        )

    source_fingerprint = html.escape(str(video.get("fingerprint", "")))
    title = "FrameLedger Phase 2c · timestamp-aligned evidence"
    return f"""<!doctype html>
<html lang="zh-Hans">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ margin:0; color:#18212b; background:#eef2f5; }}
header {{ position:sticky; top:0; z-index:10; padding:18px 24px; background:#fff; border-bottom:1px solid #ccd4dc; }}
h1 {{ margin:0 0 8px; font-size:22px; }} h2 {{ margin:0 0 6px; }} h3 {{ margin:16px 0 6px; font-size:14px; }}
header p {{ margin:5px 0; color:#526170; }} code {{ font-size:11px; }}
audio {{ width:min(720px, 100%); margin-top:8px; }}
.timeline-wrap {{ margin-top:12px; }}
.timeline {{ position:relative; height:32px; border-radius:7px; background:#e5e9ed; overflow:visible; }}
.speech-coverage {{ position:absolute; top:7px; height:18px; border-radius:5px; background:#98c8ff; }}
.marker {{ position:absolute; top:3px; width:10px; height:26px; margin-left:-5px; border:2px solid #fff; border-radius:5px; cursor:pointer; }}
.marker.aligned {{ background:#08775e; }} .marker.outside {{ background:#b36b00; cursor:default; }}
.timeline-labels {{ display:flex; justify-content:space-between; font-size:11px; color:#667380; margin-top:4px; }}
main {{ max-width:1500px; margin:18px auto; padding:0 18px 30px; }}
.frame-card {{ display:grid; grid-template-columns:minmax(420px, 1.15fr) minmax(380px, .85fr); gap:20px; margin:0 0 20px; padding:16px; background:#fff; border:1px solid #d6dde4; border-radius:10px; box-shadow:0 2px 10px #1d2b3a12; }}
.visual {{ position:relative; align-self:start; }} .visual img {{ display:block; width:100%; height:auto; }}
.boxes {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
.boxes rect {{ fill:#00c38918; stroke:#008f6a; stroke-width:1.2; vector-effect:non-scaling-stroke; }}
.badge, .relation {{ display:inline-block; padding:2px 7px; border-radius:999px; background:#e4f2ed; color:#08624e; font-size:11px; }}
.relation {{ margin:0 7px; background:#e8eef6; color:#40566f; }}
.reasons {{ font-size:12px; color:#687684; }}
.ocr-list, .speech-list {{ margin:0; padding-left:24px; }} .ocr-list li, .speech-list li {{ margin:5px 0; }}
.confidence {{ display:inline-block; min-width:54px; margin-right:7px; color:#08775e; font-size:11px; }}
.seek {{ border:0; padding:2px 5px; border-radius:4px; color:#075caa; background:#e8f3ff; cursor:pointer; font-variant-numeric:tabular-nums; }}
.coverage-warning {{ padding:10px; border-left:4px solid #b36b00; background:#fff4dc; color:#714600; }}
@media (max-width:900px) {{ .frame-card {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
<h1>{title}</h1>
<p>{int(summary['visual_frame_count'])} visual frames · {int(summary['speech_segment_count'])} speech segments · {int(summary['frames_with_segment_at_timestamp'])} direct timestamp links · {int(summary['frames_outside_asr_range'])} outside ASR range</p>
<p>Direct links use point-in-half-open-segment time overlap only. Nearby speech is review context; no semantic match, correction, or summarization is claimed.</p>
<p>Video SHA-256 <code>{source_fingerprint}</code></p>
<audio id="audio" controls preload="metadata" src="{html.escape(asset_uri(audio_path), quote=True)}"></audio>
<div class="timeline-wrap"><div class="timeline">
<div class="speech-coverage" style="left:{speech_left:.4f}%;width:{speech_width:.4f}%" title="ASR coverage"></div>
{''.join(markers)}
</div><div class="timeline-labels"><span>{format_timecode(phase_start)}</span><span>{format_timecode(phase_end)}</span></div></div>
</header>
<main>{''.join(cards)}</main>
<script>
const audio = document.getElementById('audio');
const asrStart = {float(asr_range['start_seconds']):.6f};
document.querySelectorAll('[data-time]').forEach((element) => {{
  element.addEventListener('click', () => {{
    const absolute = Number(element.dataset.time);
    audio.currentTime = Math.max(0, absolute - asrStart);
    audio.play().catch(() => {{}});
  }});
}});
</script>
</body>
</html>
"""


def run_evidence_alignment(
    run_directory: str | Path,
    *,
    strategy: str,
    ocr_json: str | Path,
    asr_json: str | Path,
    output: str | Path,
    context_before_seconds: float = 20.0,
    context_after_seconds: float = 20.0,
) -> dict[str, Any]:
    """Bind Phase 1 visual/OCR evidence and bounded ASR on absolute timestamps.

    The context relation is temporal only. It deliberately does not claim that
    nearby speech semantically describes a visual frame.
    """

    before = _number(context_before_seconds, label="context_before_seconds")
    after = _number(context_after_seconds, label="context_after_seconds")
    if before < 0 or after < 0 or before > 120 or after > 120:
        raise ValueError("Alignment context windows must be between 0 and 120 seconds")

    run_root = Path(run_directory).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError(f"Phase 1 run directory does not exist: {run_root}")
    manifest_path = run_root / "manifest.json"
    strategy, strategy_path = resolve_strategy_artifact(run_root, strategy)
    if not manifest_path.is_file():
        raise ValueError(f"Phase 1 manifest does not exist: {manifest_path}")
    if not strategy_path.is_file():
        raise ValueError(f"Phase 1 strategy does not exist: {strategy_path}")

    ocr_path = Path(ocr_json).expanduser().resolve()
    asr_path = Path(asr_json).expanduser().resolve()
    if not ocr_path.is_file():
        raise ValueError(f"OCR JSON does not exist: {ocr_path}")
    if not asr_path.is_file():
        raise ValueError(f"ASR JSON does not exist: {asr_path}")

    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise ValueError(f"Alignment output already exists: {output_path}")
    _protected_output(output_path, (run_root, ocr_path.parent, asr_path.parent))

    manifest = _load_json(manifest_path, label="Phase 1 manifest")
    strategy_document = _load_json(strategy_path, label="Phase 1 strategy")
    ocr_document = _load_json(ocr_path, label="OCR")
    asr_document = _load_json(asr_path, label="ASR")

    if manifest.get("kind") != "candidate_frame_benchmark":
        raise ValueError("Phase 1 manifest kind is not candidate_frame_benchmark")
    if strategy_document.get("strategy") != strategy:
        raise ValueError("Phase 1 strategy JSON does not match the requested strategy")
    if ocr_document.get("kind") != "frame_ocr":
        raise ValueError("OCR JSON kind is not frame_ocr")
    if ocr_document.get("schema_version") != 1:
        raise ValueError("OCR JSON schema_version is not supported")
    if asr_document.get("kind") != "local_speech_transcription":
        raise ValueError("ASR JSON kind is not local_speech_transcription")
    if asr_document.get("schema_version") != 1:
        raise ValueError("ASR JSON schema_version is not supported")
    if asr_document.get("status") != "ok":
        raise ValueError("ASR evidence is not a successful transcription")
    quality = _mapping(
        _mapping(asr_document.get("summary"), label="ASR summary").get("quality_checks"),
        label="ASR quality checks",
    )
    if quality.get("passed") is not True:
        raise ValueError("ASR evidence did not pass its transcript quality gate")

    manifest_source = _mapping(manifest.get("source"), label="Phase 1 source")
    source_video_path = manifest_source.get("path")
    if not isinstance(source_video_path, str) or not source_video_path:
        raise ValueError("Phase 1 source path must be non-empty")
    _protected_output(output_path, (Path(source_video_path).expanduser().resolve().parent,))
    video_sha = manifest_source.get("fingerprint")
    if not isinstance(video_sha, str) or len(video_sha) != 64:
        raise ValueError("Phase 1 source fingerprint must be a SHA-256 string")
    ocr_source = _mapping(ocr_document.get("source"), label="OCR source")
    asr_source = _mapping(asr_document.get("source"), label="ASR source")
    if ocr_source.get("video_sha256") != video_sha:
        raise ValueError("OCR video SHA-256 does not match Phase 1")
    if asr_source.get("fingerprint") != video_sha:
        raise ValueError("ASR video SHA-256 does not match Phase 1")
    try:
        ocr_run_root = Path(str(ocr_source.get("run_directory"))).expanduser().resolve()
    except (TypeError, ValueError) as error:
        raise ValueError("OCR source run_directory is invalid") from error
    if ocr_run_root != run_root:
        raise ValueError("OCR source run_directory does not match Phase 1 input")
    if ocr_source.get("strategy") != strategy:
        raise ValueError("OCR strategy does not match the requested Phase 1 strategy")
    if ocr_source.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("Phase 1 manifest no longer matches the OCR binding")
    if ocr_source.get("strategy_sha256") != _sha256(strategy_path):
        raise ValueError("Phase 1 strategy no longer matches the OCR binding")
    embedded_strategies = _mapping(
        manifest.get("strategies"), label="Phase 1 manifest strategies"
    )
    if embedded_strategies.get(strategy) != strategy_document:
        raise ValueError("Phase 1 manifest embedded strategy differs from strategy JSON")

    phase1_range = _range(manifest.get("range"), label="Phase 1 range")
    asr_range = _range(asr_document.get("range"), label="ASR range")
    tolerance = 1e-6
    if (
        asr_range["start_seconds"] < phase1_range["start_seconds"] - tolerance
        or asr_range["end_seconds"] > phase1_range["end_seconds"] + tolerance
    ):
        raise ValueError("ASR range must be fully contained in the Phase 1 range")

    selected = _list(strategy_document.get("selected"), label="Phase 1 selected frames")
    selected_count = _integer(
        strategy_document.get("selected_count"), label="Phase 1 selected_count"
    )
    if selected_count != len(selected):
        raise ValueError("Phase 1 selected_count does not match selected frames")
    ocr_records = _ocr_records(ocr_document)
    if len(ocr_records) != len(selected):
        raise ValueError("OCR evidence does not cover every selected Phase 1 frame exactly once")

    visual_frames: list[dict[str, Any]] = []
    image_paths: dict[str, Path] = {}
    selected_indexes: set[int] = set()
    for position, raw_candidate in enumerate(selected):
        candidate = _mapping(raw_candidate, label=f"selected frame {position}")
        sample_index = _integer(
            candidate.get("sample_index"), label=f"selected frame {position}.sample_index"
        )
        if sample_index in selected_indexes:
            raise ValueError(f"Phase 1 repeats selected sample_index {sample_index}")
        selected_indexes.add(sample_index)
        timestamp = _number(
            candidate.get("timestamp"), label=f"selected frame {position}.timestamp"
        )
        image_path = _relative_file(
            run_root,
            candidate.get("image_path"),
            label=f"selected frame {position}.image_path",
        )
        ocr_record = ocr_records.get(sample_index)
        if ocr_record is None:
            raise ValueError(f"OCR evidence is missing selected sample_index {sample_index}")
        ocr_timestamp = _number(
            ocr_record.get("timestamp"), label=f"OCR sample {sample_index}.timestamp"
        )
        if abs(timestamp - ocr_timestamp) > tolerance:
            raise ValueError(f"OCR timestamp does not match sample_index {sample_index}")
        if ocr_record.get("image_path") != candidate.get("image_path"):
            raise ValueError(f"OCR image path does not match sample_index {sample_index}")
        image_sha = _sha256(image_path)
        if ocr_record.get("image_sha256") != image_sha:
            raise ValueError(f"OCR image SHA-256 does not match sample_index {sample_index}")

        event_id = f"visual-{position + 1:04d}"
        ocr_payload = dict(ocr_record)
        for key in ("sample_index", "timestamp", "image_path", "image_sha256"):
            ocr_payload.pop(key, None)
        visual_frames.append(
            {
                "event_id": event_id,
                "sample_index": sample_index,
                "timestamp": timestamp,
                "timecode": format_timecode(timestamp),
                "route_kind": str(ocr_record.get("route_kind", "unknown")),
                "segment_id": ocr_record.get("segment_id"),
                "image": {
                    "path": str(candidate.get("image_path")),
                    "sha256": image_sha,
                },
                "selection": {
                    "score": candidate.get("score"),
                    "reasons": list(candidate.get("reasons", [])),
                },
                "ocr": ocr_payload,
                "temporal_alignment": {},
                "nearby_speech": {},
            }
        )
        image_paths[event_id] = image_path
    if selected_indexes != set(ocr_records):
        raise ValueError("OCR evidence includes sample indexes outside the Phase 1 selection")

    transcript = _mapping(asr_document.get("transcript"), label="ASR transcript")
    raw_segments = _list(transcript.get("segments"), label="ASR transcript segments")
    speech_segments: list[dict[str, Any]] = []
    source_segment_ids: set[str] = set()
    previous_start = -math.inf
    for position, raw_segment in enumerate(raw_segments):
        segment = dict(_mapping(raw_segment, label=f"ASR segment {position}"))
        source_id = str(segment.get("id"))
        if source_id in source_segment_ids:
            raise ValueError(f"ASR transcript repeats segment id {source_id}")
        source_segment_ids.add(source_id)
        start = _number(
            segment.get("absolute_start_seconds"),
            label=f"ASR segment {position}.absolute_start_seconds",
        )
        end = _number(
            segment.get("absolute_end_seconds"),
            label=f"ASR segment {position}.absolute_end_seconds",
        )
        if end < start:
            raise ValueError(f"ASR segment {position} ends before it starts")
        if start < previous_start - tolerance:
            raise ValueError("ASR transcript segments are not chronological")
        if (
            start < asr_range["start_seconds"] - tolerance
            or end > asr_range["end_seconds"] + tolerance
        ):
            raise ValueError(f"ASR segment {position} escapes the declared ASR range")
        if not isinstance(segment.get("text"), str):
            raise ValueError(f"ASR segment {position}.text must be a string")
        relative_start = _number(
            segment.get("relative_start_seconds"),
            label=f"ASR segment {position}.relative_start_seconds",
        )
        relative_end = _number(
            segment.get("relative_end_seconds"),
            label=f"ASR segment {position}.relative_end_seconds",
        )
        if (
            abs(start - (asr_range["start_seconds"] + relative_start)) > 1e-3
            or abs(end - (asr_range["start_seconds"] + relative_end)) > 1e-3
        ):
            raise ValueError(
                f"ASR segment {position} absolute and relative timestamps disagree"
            )
        previous_start = start
        speech_segments.append(
            {
                "event_id": f"speech-{position + 1:04d}",
                "source_segment_id": segment.get("id"),
                **segment,
            }
        )
    if transcript.get("segment_count") != len(speech_segments):
        raise ValueError("ASR transcript segment_count does not match its segment list")

    temporal_links: list[dict[str, Any]] = []
    for frame in visual_frames:
        timestamp = float(frame["timestamp"])
        timestamp_us = _microseconds(timestamp)
        if not (
            _microseconds(asr_range["start_seconds"])
            <= timestamp_us
            < _microseconds(asr_range["end_seconds"])
        ):
            frame["temporal_alignment"] = {
                "status": "outside_asr_range",
                "direct_segment_event_ids": [],
                "semantic_match_claimed": False,
            }
            frame["nearby_speech"] = {"window": None, "segments": []}
            continue
        direct_ids: list[str] = []
        for segment in speech_segments:
            if (
                _microseconds(float(segment["absolute_start_seconds"]))
                <= timestamp_us
                < _microseconds(float(segment["absolute_end_seconds"]))
            ):
                direct_ids.append(str(segment["event_id"]))
                temporal_links.append(
                    {
                        "visual_event_id": frame["event_id"],
                        "speech_event_id": segment["event_id"],
                        "relation": "segment_at_visual_timestamp",
                        "clock": "source_video_absolute_microseconds",
                        "semantic_match_claimed": False,
                    }
                )
        frame["temporal_alignment"] = {
            "status": (
                "segment_at_timestamp" if direct_ids else "inside_asr_range_no_segment"
            ),
            "direct_segment_event_ids": direct_ids,
            "semantic_match_claimed": False,
        }
        window_start = max(asr_range["start_seconds"], timestamp - before)
        window_end = min(asr_range["end_seconds"], timestamp + after)
        relations: list[dict[str, Any]] = []
        for segment in speech_segments:
            if (
                _microseconds(float(segment["absolute_end_seconds"]))
                <= _microseconds(window_start)
                or _microseconds(float(segment["absolute_start_seconds"]))
                >= _microseconds(window_end)
            ):
                continue
            relation_name, gap = _relation(segment, timestamp)
            relations.append(
                {
                    "event_id": segment["event_id"],
                    "relation": relation_name,
                    "gap_seconds": gap,
                }
            )
        frame["nearby_speech"] = {
            "window": {
                "start_seconds": round(window_start, 6),
                "end_seconds": round(window_end, 6),
                "start_timecode": format_timecode(window_start),
                "end_timecode": format_timecode(window_end),
            },
            "segments": relations,
            "semantic_match_claimed": False,
        }

    timeline: list[dict[str, Any]] = []
    for frame in visual_frames:
        timeline.append(
            {
                "event_id": frame["event_id"],
                "kind": "visual_frame",
                "timestamp": frame["timestamp"],
            }
        )
    for segment in speech_segments:
        timeline.append(
            {
                "event_id": segment["event_id"],
                "kind": "speech_segment",
                "start_seconds": segment["absolute_start_seconds"],
                "end_seconds": segment["absolute_end_seconds"],
            }
        )
    timeline.sort(
        key=lambda event: (
            float(event.get("timestamp", event.get("start_seconds", 0.0))),
            0 if event["kind"] == "visual_frame" else 1,
            str(event["event_id"]),
        )
    )

    asr_audio = _mapping(asr_document.get("audio"), label="ASR audio")
    audio_path = _relative_file(
        asr_path.parent, asr_audio.get("path"), label="ASR audio.path"
    )
    if asr_audio.get("sha256") != _sha256(audio_path):
        raise ValueError("ASR audio SHA-256 no longer matches its evidence ledger")

    frames_with_direct_segment = sum(
        frame["temporal_alignment"]["status"] == "segment_at_timestamp"
        for frame in visual_frames
    )
    frames_outside = sum(
        frame["temporal_alignment"]["status"] == "outside_asr_range"
        for frame in visual_frames
    )
    document: dict[str, Any] = {
        "schema_version": ALIGNMENT_SCHEMA_VERSION,
        "kind": "timestamp_aligned_evidence",
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "policy_version": ALIGNMENT_POLICY_VERSION,
            "strategy": strategy,
            "context_before_seconds": before,
            "context_after_seconds": after,
            "alignment_basis": "visual_point_in_half_open_speech_segment",
            "clock": "source_video_absolute_microseconds",
            "speech_segment_interval": "half_open",
            "nearby_context_is_alignment": False,
            "semantic_match_claimed": False,
            "correction_applied": False,
            "summarization_applied": False,
        },
        "source": {
            "video": dict(manifest_source),
            "phase1": {
                "run_directory": str(run_root),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "strategy": strategy,
                "strategy_path": str(strategy_path),
                "strategy_sha256": _sha256(strategy_path),
            },
            "ocr": {
                "path": str(ocr_path),
                "sha256": _sha256(ocr_path),
                "policy_version": _mapping(
                    ocr_document.get("parameters"), label="OCR parameters"
                ).get("policy_version"),
            },
            "asr": {
                "path": str(asr_path),
                "sha256": _sha256(asr_path),
                "model_id": _mapping(
                    asr_document.get("parameters"), label="ASR parameters"
                ).get("model_id"),
                "model_revision": _mapping(
                    asr_document.get("parameters"), label="ASR parameters"
                ).get("model_revision"),
                "audio_path": str(audio_path),
                "audio_sha256": asr_audio.get("sha256"),
            },
        },
        "coverage": {
            "phase1_range": phase1_range,
            "asr_range": asr_range,
            "asr_fully_contained_in_phase1": True,
            "complete_phase1_speech_coverage": (
                abs(asr_range["start_seconds"] - phase1_range["start_seconds"])
                <= tolerance
                and abs(asr_range["end_seconds"] - phase1_range["end_seconds"])
                <= tolerance
            ),
        },
        "visual_frames": visual_frames,
        "speech_segments": speech_segments,
        "temporal_links": temporal_links,
        "timeline": timeline,
        "summary": {
            "visual_frame_count": len(visual_frames),
            "ocr_ok_frame_count": sum(
                frame["ocr"].get("status") == "ok" for frame in visual_frames
            ),
            "speech_segment_count": len(speech_segments),
            "frames_with_segment_at_timestamp": frames_with_direct_segment,
            "frames_inside_asr_range_without_segment": sum(
                frame["temporal_alignment"]["status"]
                == "inside_asr_range_no_segment"
                for frame in visual_frames
            ),
            "frames_outside_asr_range": frames_outside,
            "temporal_link_count": len(temporal_links),
            "timeline_event_count": len(timeline),
        },
    }

    review = _review_html(
        document,
        audio_path=audio_path,
        image_paths=image_paths,
        review_directory=output_path,
    )
    output_path.mkdir(parents=True)
    evidence_path = output_path / "evidence.json"
    review_path = output_path / "review.html"
    write_json(evidence_path, document)
    review_path.write_text(review, encoding="utf-8")
    return {
        "kind": document["kind"],
        "output": str(output_path),
        "review_html": str(review_path),
        "evidence_json": str(evidence_path),
        "source": dict(manifest_source),
        "summary": document["summary"],
        "coverage": document["coverage"],
    }
