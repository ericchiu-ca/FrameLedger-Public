from __future__ import annotations

import hashlib
import importlib.metadata
import math
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scenedetect import AdaptiveDetector, ContentDetector, FrameTimecode

from .artifacts import resolve_strategy_artifact
from .annotations import (
    candidate_from_dict,
    evaluate_annotations,
    load_annotations,
    validate_annotations,
)
from .features import resize_gray
from .media import MediaError, iter_video_frames, probe_video, read_frame_at, write_jpeg, write_png
from .models import Candidate, StrategyResult, VideoMetadata
from .report import write_contact_sheet, write_json, write_review_html, write_scan_html
from .routing import detect_content_segments, select_routed_states
from .selection import (
    VALID_STRATEGIES,
    build_samples,
    finalize_strategy,
    scene_times_to_candidates,
    select_adaptive,
    select_boundary_terminal_states,
    select_presentation_states,
    select_settled_end_states,
    select_table_focus_states,
    select_table_viewport_states,
    select_uniform,
)
from .timecode import parse_timecode


MAX_DEFAULT_BENCHMARK_SECONDS = 12 * 60


def _prepare_output(output: str | Path, metadata: VideoMetadata) -> Path:
    target = Path(output).expanduser().resolve()
    if target == metadata.path.parent:
        raise ValueError("Output must not be the source video's directory")
    if target.exists():
        raise ValueError(f"Output directory already exists; refusing to overwrite: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _bounded_range(
    metadata: VideoMetadata,
    start_seconds: float,
    duration_seconds: float | None,
) -> tuple[float, float]:
    if start_seconds < 0 or start_seconds >= metadata.duration_seconds:
        raise ValueError("Start time must be inside the video")
    if duration_seconds is None:
        end_seconds = metadata.duration_seconds
    else:
        if duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        end_seconds = min(metadata.duration_seconds, start_seconds + duration_seconds)
    return start_seconds, end_seconds


def _decode_analysis_and_scenes(
    metadata: VideoMetadata,
    *,
    start_seconds: float,
    end_seconds: float,
    analysis_fps: float,
    analysis_width: int,
    need_content: bool,
    need_adaptive: bool,
    content_threshold: float,
    adaptive_threshold: float,
) -> tuple[list[tuple[float, int, object]], dict[str, list[float]]]:
    detectors: dict[str, object] = {}
    if need_content:
        detectors["content"] = ContentDetector(threshold=content_threshold, min_scene_len=1.0)
    if need_adaptive:
        detectors["adaptive"] = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=1.0,
            min_content_val=max(8.0, content_threshold * 0.55),
        )
    cuts: dict[str, list[float]] = {name: [start_seconds] for name in detectors}
    sampled: list[tuple[float, int, object]] = []
    next_sample_time = start_seconds
    sample_period = 1.0 / analysis_fps
    last_timecode: FrameTimecode | None = None
    detector_width = max(analysis_width, 640)
    for timestamp, frame_index, frame in iter_video_frames(
        metadata,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    ):
        if frame.shape[1] > detector_width:
            detector_height = max(1, int(round(frame.shape[0] * detector_width / frame.shape[1])))
            detector_frame = cv2.resize(
                frame, (detector_width, detector_height), interpolation=cv2.INTER_AREA
            )
        else:
            detector_frame = frame
        timecode = FrameTimecode(frame_index, fps=metadata.fps)
        last_timecode = timecode
        for name, detector in detectors.items():
            for cut in detector.process_frame(timecode, detector_frame):
                cuts[name].append(cut.get_seconds())
        if timestamp + (0.5 / metadata.fps) >= next_sample_time:
            sampled.append((timestamp, frame_index, resize_gray(detector_frame, analysis_width)))
            next_sample_time += sample_period
            while next_sample_time <= timestamp:
                next_sample_time += sample_period
    if last_timecode is not None:
        for name, detector in detectors.items():
            for cut in detector.post_process(last_timecode):
                cuts[name].append(cut.get_seconds())
    return sampled, {name: sorted(set(times)) for name, times in cuts.items()}


def _basic_metrics(result: StrategyResult, segment_seconds: float) -> dict[str, Any]:
    raw_count = len(result.raw_candidates)
    duplicate_count = len(result.dropped_duplicates)
    return {
        "raw_candidates": raw_count,
        "selected_frames": len(result.selected),
        "candidate_frames_per_minute": len(result.selected) / max(segment_seconds / 60.0, 1e-9),
        "duplicate_drop_ratio": duplicate_count / raw_count if raw_count else 0.0,
        "capped_frames": len(result.capped_candidates),
        "annotations": None,
        "ground_truth_status": "not_provided",
    }


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scenedetect_headless": importlib.metadata.version("scenedetect-headless"),
    }


def scan_video(
    video: str | Path,
    *,
    output: str | Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    interval_seconds: float = 60.0,
    maximum_frames: int = 120,
) -> dict[str, Any]:
    if interval_seconds <= 0:
        raise ValueError("Scan interval must be positive")
    metadata = probe_video(video)
    start_seconds, end_seconds = _bounded_range(metadata, start_seconds, duration_seconds)
    output_root = _prepare_output(output, metadata)
    timestamps: list[float] = []
    current = start_seconds
    while current < end_seconds - 1e-6 and len(timestamps) < maximum_frames:
        timestamps.append(current)
        current += interval_seconds
    if not timestamps or timestamps[-1] < end_seconds - min(interval_seconds * 0.25, 2.0):
        timestamps.append(max(start_seconds, end_seconds - 1.0 / metadata.fps))
    if len(timestamps) > maximum_frames:
        raise ValueError(
            f"Scan would produce {len(timestamps)} frames; raise --interval or --max-frames"
        )

    scan_frames: list[tuple[float, str]] = []
    candidates: list[Candidate] = []
    for index, timestamp in enumerate(timestamps):
        frame = read_frame_at(metadata, timestamp)
        relative = Path("frames") / f"scan_{int(round(timestamp * 1000)):012d}.jpg"
        write_jpeg(output_root / relative, frame)
        scan_frames.append((timestamp, relative.as_posix()))
        candidates.append(Candidate(index, timestamp, 0.1, ["coarse_scan"], relative.as_posix()))

    write_contact_sheet(output_root / "contact-sheet.jpg", output_root, candidates)
    write_scan_html(
        output_root / "scan.html",
        metadata,
        scan_frames,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    manifest = {
        "kind": "coarse_scan",
        "source": metadata.to_dict(),
        "range": {"start_seconds": start_seconds, "end_seconds": end_seconds},
        "interval_seconds": interval_seconds,
        "frame_count": len(scan_frames),
        "frames": [{"timestamp": timestamp, "image_path": path} for timestamp, path in scan_frames],
        "environment": _environment(),
    }
    write_json(output_root / "scan.json", manifest)
    source_after = metadata.path.stat()
    if source_after.st_size != metadata.size_bytes or source_after.st_mtime_ns != metadata.mtime_ns:
        raise MediaError("Source video changed during the scan; results are not trustworthy")
    return {"output": str(output_root), "scan_html": str(output_root / "scan.html"), **manifest}


def export_reviewed_frames(
    video: str | Path,
    *,
    annotations_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Export human-selected timestamps as source-resolution evidence frames."""
    metadata = probe_video(video)
    annotations = load_annotations(annotations_path)
    if annotations.get("review_status") != "human_reviewed":
        raise ValueError("Curated export requires review_status=human_reviewed")
    segment = annotations.get("segment")
    if not isinstance(segment, dict) or "start" not in segment or "duration" not in segment:
        raise ValueError("Curated export requires an explicit annotation segment")
    start_seconds = parse_timecode(segment["start"])
    duration_seconds = parse_timecode(segment["duration"])
    start_seconds, end_seconds = _bounded_range(metadata, start_seconds, duration_seconds)
    validate_annotations(
        annotations,
        video_name=metadata.path.name,
        start_seconds=start_seconds,
        duration_seconds=end_seconds - start_seconds,
        source_sha256=metadata.fingerprint,
    )
    points = annotations.get("must_keep", [])
    if not points:
        raise ValueError("Curated export requires at least one must_keep timestamp")
    # Validate point tolerances/windows before creating any output directory.
    evaluate_annotations(annotations, [])

    timestamps: list[float] = []
    for item in points:
        if not isinstance(item, dict) or "timestamp" not in item:
            raise ValueError("Each curated frame requires a timestamp")
        timestamp = parse_timecode(item["timestamp"])
        if timestamp < start_seconds or timestamp > end_seconds + 1e-6:
            raise ValueError("Curated frame timestamp is outside the annotation segment")
        if any(abs(timestamp - existing) < 1e-6 for existing in timestamps):
            raise ValueError("Curated frame timestamps must be unique")
        timestamps.append(timestamp)

    output_root = _prepare_output(output, metadata)
    selected: list[Candidate] = []
    for index, (timestamp, item) in enumerate(zip(timestamps, points, strict=True)):
        frame = read_frame_at(metadata, timestamp)
        if frame.shape[1] != metadata.width or frame.shape[0] != metadata.height:
            raise MediaError(
                f"Evidence frame resolution {frame.shape[1]}x{frame.shape[0]} does not match "
                f"source {metadata.width}x{metadata.height}"
            )
        relative = Path("frames") / f"frame_{int(round(timestamp * 1000)):012d}.png"
        write_png(output_root / relative, frame)
        selected.append(
            Candidate(
                sample_index=index,
                timestamp=timestamp,
                score=1.0,
                reasons=["human_reviewed", str(item.get("kind", "must_keep"))],
                image_path=relative.as_posix(),
            )
        )

    annotation_metrics = evaluate_annotations(annotations, selected)
    result = StrategyResult(
        name="human_curated",
        raw_candidates=list(selected),
        selected=selected,
        dropped_duplicates=[],
        metrics={
            "annotations": annotation_metrics,
            "ground_truth_status": annotation_metrics["ground_truth"],
        },
    )
    write_contact_sheet(output_root / "contact-sheet.jpg", output_root, selected)
    write_review_html(
        output_root / "review.html",
        metadata,
        {"human_curated": result},
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    annotation_path = Path(annotations_path).expanduser().resolve()
    manifest = {
        "kind": "human_curated_frame_export",
        "source": metadata.to_dict(),
        "range": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": end_seconds - start_seconds,
        },
        "annotation_file": str(annotation_path),
        "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        "review_coverage": annotations.get("review", {}).get("coverage", "positive_only"),
        "ground_truth_status": annotation_metrics["ground_truth"],
        "source_segment_recall_assessable": bool(
            (annotation_metrics.get("selection_quality") or {}).get(
                "source_segment_recall_assessable",
                False,
            )
        ),
        "selected": [candidate.to_dict() for candidate in selected],
        "review_html": "review.html",
    }
    write_json(output_root / "manifest.json", manifest)
    source_after = metadata.path.stat()
    if source_after.st_size != metadata.size_bytes or source_after.st_mtime_ns != metadata.mtime_ns:
        raise MediaError("Source video changed during curated export; results are not trustworthy")
    return {**manifest, "output": str(output_root), "review_html": str(output_root / "review.html")}


def run_benchmark(
    video: str | Path,
    *,
    output: str | Path,
    start_seconds: float,
    duration_seconds: float,
    strategies: list[str],
    annotations_path: str | Path | None = None,
    analysis_fps: float = 2.0,
    analysis_width: int = 320,
    uniform_interval_seconds: float = 10.0,
    maximum_frames: int = 80,
    content_threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
    boundary_transition_threshold: float = 0.025,
    boundary_pixel_delta_threshold: float = 0.004,
    boundary_edge_loss_threshold: float = 0.003,
    boundary_terminal_search_seconds: float = 2.0,
    boundary_terminal_stable_seconds: float = 1.0,
    boundary_terminal_stability_threshold: float = 0.008,
    boundary_lookahead_seconds: float = 15.0,
    table_lookahead_seconds: float = 8.0,
    presentation_lookahead_seconds: float = 6.0,
    allow_long_range: bool = False,
) -> dict[str, Any]:
    invalid = sorted(set(strategies) - VALID_STRATEGIES)
    if invalid:
        raise ValueError(f"Unknown strategies: {', '.join(invalid)}")
    if not strategies:
        raise ValueError("At least one strategy is required")
    if duration_seconds > MAX_DEFAULT_BENCHMARK_SECONDS and not allow_long_range:
        raise ValueError(
            f"Phase 1 limits a run to {MAX_DEFAULT_BENCHMARK_SECONDS} seconds; "
            "pass --allow-long-range only after resource costs are understood"
        )
    if not (0 < analysis_fps <= 4):
        raise ValueError("Analysis FPS must be greater than 0 and at most 4")
    if "presentation_states" in strategies and analysis_fps < 1:
        raise ValueError("Presentation analysis FPS must be at least 1")
    if "routed" in strategies and analysis_fps < 1:
        raise ValueError("Routed visual analysis FPS must be at least 1")
    if maximum_frames <= 0:
        raise ValueError("Maximum frame count must be positive")
    if analysis_width < 160 or analysis_width > 1280:
        raise ValueError("Analysis width must be between 160 and 1280")
    if not 0 < boundary_transition_threshold <= 1:
        raise ValueError("Boundary transition threshold must be in (0, 1]")
    if boundary_pixel_delta_threshold < 0 or boundary_edge_loss_threshold < 0:
        raise ValueError("Boundary delta thresholds must be non-negative")
    if boundary_terminal_search_seconds <= 0 or boundary_terminal_stable_seconds <= 0:
        raise ValueError("Boundary terminal search and stable durations must be positive")
    if boundary_terminal_stability_threshold < 0:
        raise ValueError("Boundary terminal stability threshold must be non-negative")
    if not math.isfinite(boundary_lookahead_seconds):
        raise ValueError("Boundary lookahead duration must be finite")
    if boundary_lookahead_seconds < 0:
        raise ValueError("Boundary lookahead duration must be non-negative")
    if boundary_lookahead_seconds > 60 and not allow_long_range:
        raise ValueError(
            "Boundary lookahead is limited to 60 seconds; pass --allow-long-range "
            "only after resource costs are understood"
        )
    if not math.isfinite(table_lookahead_seconds):
        raise ValueError("Table lookahead duration must be finite")
    if table_lookahead_seconds < 0:
        raise ValueError("Table lookahead duration must be non-negative")
    if table_lookahead_seconds > 60 and not allow_long_range:
        raise ValueError(
            "Table lookahead is limited to 60 seconds; pass --allow-long-range "
            "only after resource costs are understood"
        )
    if not math.isfinite(presentation_lookahead_seconds):
        raise ValueError("Presentation lookahead duration must be finite")
    if presentation_lookahead_seconds < 0:
        raise ValueError("Presentation lookahead duration must be non-negative")
    if presentation_lookahead_seconds > 60 and not allow_long_range:
        raise ValueError(
            "Presentation lookahead is limited to 60 seconds; pass --allow-long-range "
            "only after resource costs are understood"
        )

    metadata = probe_video(video)
    start_seconds, end_seconds = _bounded_range(metadata, start_seconds, duration_seconds)
    actual_duration = end_seconds - start_seconds
    annotations = None
    if annotations_path is not None:
        annotations = load_annotations(annotations_path)
        validate_annotations(
            annotations,
            video_name=metadata.path.name,
            start_seconds=start_seconds,
            duration_seconds=actual_duration,
            source_sha256=metadata.fingerprint,
        )

    output_root = _prepare_output(output, metadata)
    stage_times: dict[str, float] = {}
    stage_start = time.perf_counter()
    requested = list(dict.fromkeys(strategies))
    needs_hybrid = "hybrid" in requested
    needs_routing = "routed" in requested
    need_content = "content" in requested or needs_hybrid
    need_adaptive = "adaptive" in requested or needs_hybrid
    analysis_end_seconds = end_seconds
    if "boundary_terminal" in requested:
        analysis_end_seconds = min(
            metadata.duration_seconds,
            end_seconds + boundary_lookahead_seconds,
        )
    if {"table_viewport", "table_focus"}.intersection(requested):
        analysis_end_seconds = max(
            analysis_end_seconds,
            min(metadata.duration_seconds, end_seconds + table_lookahead_seconds),
        )
    if "presentation_states" in requested:
        analysis_end_seconds = max(
            analysis_end_seconds,
            min(metadata.duration_seconds, end_seconds + presentation_lookahead_seconds),
        )
    if needs_routing:
        routing_lookahead_seconds = max(
            boundary_lookahead_seconds,
            table_lookahead_seconds,
            presentation_lookahead_seconds,
        )
        analysis_end_seconds = max(
            analysis_end_seconds,
            min(metadata.duration_seconds, end_seconds + routing_lookahead_seconds),
        )
    sampled_frames, scene_starts = _decode_analysis_and_scenes(
        metadata,
        start_seconds=start_seconds,
        end_seconds=analysis_end_seconds,
        analysis_fps=analysis_fps,
        analysis_width=analysis_width,
        need_content=need_content,
        need_adaptive=need_adaptive,
        content_threshold=content_threshold,
        adaptive_threshold=adaptive_threshold,
    )
    samples = build_samples(sampled_frames)
    selection_samples = [
        sample for sample in samples if sample.timestamp <= end_seconds + 1e-9
    ]
    if not selection_samples:
        raise ValueError("The selected range did not yield any analysis frames")
    stage_times["decode_detect_analyze_seconds"] = time.perf_counter() - stage_start

    routing_plan = None
    if needs_routing:
        stage_start = time.perf_counter()
        routing_plan = detect_content_segments(
            selection_samples,
            analysis_fps=analysis_fps,
            candidate_end_timestamp=end_seconds,
        )
        stage_times["content_route_detect_seconds"] = time.perf_counter() - stage_start

    raw_by_name: dict[str, list[Candidate]] = {}
    raw_by_name["uniform"] = select_uniform(
        selection_samples, interval_seconds=uniform_interval_seconds
    )

    if "content" in requested or needs_hybrid:
        raw_by_name["content"] = scene_times_to_candidates(
            selection_samples,
            [
                timestamp
                for timestamp in scene_starts.get("content", [start_seconds])
                if timestamp <= end_seconds + 1e-9
            ],
            reason="scenedetect_content",
        )

    if "adaptive" in requested or needs_hybrid:
        stage_start = time.perf_counter()
        scene_adaptive = scene_times_to_candidates(
            selection_samples,
            [
                timestamp
                for timestamp in scene_starts.get("adaptive", [start_seconds])
                if timestamp <= end_seconds + 1e-9
            ],
            reason="scenedetect_adaptive",
        )
        sampled_adaptive = select_adaptive(selection_samples, analysis_fps=analysis_fps)
        raw_by_name["adaptive"] = scene_adaptive + sampled_adaptive
        stage_times["adaptive_detect_seconds"] = time.perf_counter() - stage_start

    if "settled" in requested or needs_hybrid:
        stage_start = time.perf_counter()
        raw_by_name["settled"] = select_settled_end_states(
            selection_samples, analysis_fps=analysis_fps
        )
        stage_times["settled_detect_seconds"] = time.perf_counter() - stage_start

    if "boundary_terminal" in requested:
        stage_start = time.perf_counter()
        raw_by_name["boundary_terminal"] = select_boundary_terminal_states(
            samples,
            analysis_fps=analysis_fps,
            transition_threshold=boundary_transition_threshold,
            pixel_delta_threshold=boundary_pixel_delta_threshold,
            edge_loss_threshold=boundary_edge_loss_threshold,
            terminal_search_seconds=boundary_terminal_search_seconds,
            terminal_stable_seconds=boundary_terminal_stable_seconds,
            terminal_stability_threshold=boundary_terminal_stability_threshold,
            candidate_end_timestamp=end_seconds,
        )
        stage_times["boundary_terminal_detect_seconds"] = time.perf_counter() - stage_start

    if "table_viewport" in requested:
        stage_start = time.perf_counter()
        raw_by_name["table_viewport"] = select_table_viewport_states(
            samples,
            analysis_fps=analysis_fps,
            candidate_end_timestamp=end_seconds,
        )
        stage_times["table_viewport_detect_seconds"] = time.perf_counter() - stage_start

    if "table_focus" in requested:
        stage_start = time.perf_counter()
        raw_by_name["table_focus"] = select_table_focus_states(
            samples,
            analysis_fps=analysis_fps,
            candidate_end_timestamp=end_seconds,
        )
        stage_times["table_focus_detect_seconds"] = time.perf_counter() - stage_start

    if "presentation_states" in requested:
        stage_start = time.perf_counter()
        raw_by_name["presentation_states"] = select_presentation_states(
            samples,
            analysis_fps=analysis_fps,
            candidate_end_timestamp=end_seconds,
        )
        stage_times["presentation_states_detect_seconds"] = (
            time.perf_counter() - stage_start
        )

    if needs_routing:
        assert routing_plan is not None
        stage_start = time.perf_counter()
        raw_by_name["routed"] = select_routed_states(
            samples,
            routing_plan.segments,
            analysis_fps=analysis_fps,
            boundary_lookahead_seconds=boundary_lookahead_seconds,
            table_lookahead_seconds=table_lookahead_seconds,
            presentation_lookahead_seconds=presentation_lookahead_seconds,
            boundary_transition_threshold=boundary_transition_threshold,
            boundary_pixel_delta_threshold=boundary_pixel_delta_threshold,
            boundary_edge_loss_threshold=boundary_edge_loss_threshold,
            boundary_terminal_search_seconds=boundary_terminal_search_seconds,
            boundary_terminal_stable_seconds=boundary_terminal_stable_seconds,
            boundary_terminal_stability_threshold=(
                boundary_terminal_stability_threshold
            ),
        )
        stage_times["routed_select_seconds"] = time.perf_counter() - stage_start
        if len(raw_by_name["routed"]) > maximum_frames:
            raise ValueError(
                "Routed selection produced "
                f"{len(raw_by_name['routed'])} frames across "
                f"{sum(segment.kind != 'unknown' for segment in routing_plan.segments)} "
                "known segments; increase --max-frames so routing does not silently "
                "discard segment evidence"
            )

    if needs_hybrid:
        raw_by_name["hybrid"] = (
            raw_by_name["uniform"]
            + raw_by_name.get("content", [])
            + raw_by_name.get("adaptive", [])
            + raw_by_name.get("settled", [])
        )

    results: dict[str, StrategyResult] = {}
    for name in requested:
        result = finalize_strategy(
            name,
            samples,
            raw_by_name.get(name, []),
            maximum_frames=maximum_frames,
            deduplicate_candidates=name
            not in {
                "boundary_terminal",
                "table_viewport",
                "table_focus",
                "presentation_states",
                "routed",
            },
        )
        if name == "routed" and routing_plan is not None:
            result.segments = list(routing_plan.segments)
        result.metrics = _basic_metrics(result, actual_duration)
        if annotations is not None:
            result.metrics["annotations"] = evaluate_annotations(annotations, result.selected)
            result.metrics["ground_truth_status"] = result.metrics["annotations"]["ground_truth"]
        results[name] = result

    stage_start = time.perf_counter()
    raw_indices = sorted(
        {
            candidate.sample_index
            for result in results.values()
            for candidate in (
                result.raw_candidates
                + [item.candidate for item in result.dropped_duplicates]
                + result.capped_candidates
            )
        }
    )
    thumbnail_paths: dict[int, str] = {}
    for sample_index in raw_indices:
        sample = samples[sample_index]
        relative = Path("thumbs") / f"thumb_{int(round(sample.timestamp * 1000)):012d}.jpg"
        write_jpeg(output_root / relative, sample.gray, quality=88)
        thumbnail_paths[sample_index] = relative.as_posix()
    for result in results.values():
        for candidate in result.raw_candidates:
            candidate.image_path = thumbnail_paths[candidate.sample_index]
        for dropped in result.dropped_duplicates:
            dropped.candidate.image_path = thumbnail_paths[dropped.candidate.sample_index]
        for candidate in result.capped_candidates:
            candidate.image_path = thumbnail_paths[candidate.sample_index]

    selected_indices = sorted(
        {candidate.sample_index for result in results.values() for candidate in result.selected}
    )
    frame_paths: dict[int, str] = {}
    for sample_index in selected_indices:
        sample = samples[sample_index]
        frame = read_frame_at(metadata, sample.timestamp)
        if frame.shape[1] != metadata.width or frame.shape[0] != metadata.height:
            raise MediaError(
                f"Evidence frame resolution {frame.shape[1]}x{frame.shape[0]} does not match "
                f"source {metadata.width}x{metadata.height}"
            )
        relative = Path("frames") / f"frame_{int(round(sample.timestamp * 1000)):012d}.png"
        write_png(output_root / relative, frame)
        frame_paths[sample_index] = relative.as_posix()
    for result in results.values():
        for candidate in result.selected:
            candidate.image_path = frame_paths[candidate.sample_index]
    stage_times["evidence_extract_seconds"] = time.perf_counter() - stage_start

    write_json(
        output_root / "analysis-signals.json",
        {
            "analysis_fps": analysis_fps,
            "analysis_width": analysis_width,
            "candidate_end_seconds": end_seconds,
            "analysis_end_seconds": analysis_end_seconds,
            "samples": [sample.to_public_dict() for sample in samples],
        },
    )
    routing_manifest = None
    if routing_plan is not None:
        candidate_counts: dict[str, int] = {
            segment.segment_id: 0 for segment in routing_plan.segments
        }
        for candidate in results["routed"].selected:
            if candidate.segment_id is not None:
                candidate_counts[candidate.segment_id] += 1
        routing_signals = {
            **routing_plan.to_dict(),
            "status": "experimental_visual_only",
            "version": "visual-router-v1",
            "classification_policy": {
                "chart": {"dark_ratio_min": 0.72, "mean_luma_max": 100.0},
                "table": {
                    "mean_luma_min": 185.0,
                    "bright_ratio_min": 0.30,
                    "horizontal_line_ratio_min": 0.008,
                    "edge_density_min": 0.07,
                },
                "presentation": {
                    "mean_luma_min": 105.0,
                    "dark_ratio_max": 0.60,
                    "horizontal_line_ratio_max": 0.012,
                    "edge_density_max": 0.25,
                    "embedded_visual_exception": {
                        "horizontal_line_ratio_max_exclusive": 0.010,
                        "vertical_line_ratio_max_exclusive": 0.008,
                        "edge_density_max": 0.12,
                        "mean_luma_max": 235.0,
                    },
                },
                "postprocess": {
                    "bridge_seconds": routing_plan.bridge_seconds,
                    "minimum_known_seconds": routing_plan.minimum_known_seconds,
                },
            },
            "selector_parameters": {
                "lookahead_seconds": {
                    "chart": boundary_lookahead_seconds,
                    "table": table_lookahead_seconds,
                    "presentation": presentation_lookahead_seconds,
                },
                "boundary_terminal": {
                    "transition_threshold": boundary_transition_threshold,
                    "pixel_delta_threshold": boundary_pixel_delta_threshold,
                    "edge_loss_threshold": boundary_edge_loss_threshold,
                    "terminal_search_seconds": boundary_terminal_search_seconds,
                    "terminal_stable_seconds": boundary_terminal_stable_seconds,
                    "terminal_stability_threshold": (
                        boundary_terminal_stability_threshold
                    ),
                },
            },
            "candidate_counts_by_segment": candidate_counts,
        }
        write_json(output_root / "routing-signals.json", routing_signals)

        segment_duration_by_kind = {
            kind: 0.0 for kind in ("presentation", "table", "chart", "unknown")
        }
        for index, segment in enumerate(routing_plan.segments):
            segment_end = (
                routing_plan.segments[index + 1].start_timestamp
                if index + 1 < len(routing_plan.segments)
                else end_seconds
            )
            visible_start = max(start_seconds, segment.start_timestamp)
            visible_end = min(end_seconds, max(visible_start, segment_end))
            segment_duration_by_kind[segment.kind] += visible_end - visible_start
        unknown_duration = segment_duration_by_kind["unknown"]
        routing_manifest = {
            "status": "experimental_visual_only",
            "version": "visual-router-v1",
            "signals_path": "routing-signals.json",
            "segments": [segment.to_dict() for segment in routing_plan.segments],
            "duration_seconds_by_kind": {
                kind: round(value, 6)
                for kind, value in segment_duration_by_kind.items()
            },
            "unknown_duration_seconds": round(unknown_duration, 6),
            "unknown_ratio": unknown_duration / max(actual_duration, 1e-9),
        }
    for name, result in results.items():
        write_json(output_root / "strategies" / f"{name}.json", result.to_dict())
        write_contact_sheet(
            output_root / "contact-sheets" / f"{name}.jpg",
            output_root,
            result.selected,
        )
    write_review_html(
        output_root / "review.html",
        metadata,
        results,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )

    manifest = {
        "kind": "candidate_frame_benchmark",
        "source": metadata.to_dict(),
        "range": {
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": actual_duration,
            "analysis_end_seconds": analysis_end_seconds,
        },
        "parameters": {
            "strategies": requested,
            "analysis_fps": analysis_fps,
            "analysis_width": analysis_width,
            "uniform_interval_seconds": uniform_interval_seconds,
            "maximum_frames": maximum_frames,
            "content_threshold": content_threshold,
            "adaptive_threshold": adaptive_threshold,
            "boundary_transition_threshold": boundary_transition_threshold,
            "boundary_pixel_delta_threshold": boundary_pixel_delta_threshold,
            "boundary_edge_loss_threshold": boundary_edge_loss_threshold,
            "boundary_terminal_search_seconds": boundary_terminal_search_seconds,
            "boundary_terminal_stable_seconds": boundary_terminal_stable_seconds,
            "boundary_terminal_stability_threshold": boundary_terminal_stability_threshold,
            "boundary_lookahead_seconds": boundary_lookahead_seconds,
            "table_lookahead_seconds": table_lookahead_seconds,
            "presentation_lookahead_seconds": presentation_lookahead_seconds,
            "routing_bridge_seconds": (
                routing_plan.bridge_seconds if routing_plan is not None else None
            ),
            "routing_minimum_known_seconds": (
                routing_plan.minimum_known_seconds if routing_plan is not None else None
            ),
        },
        "environment": _environment(),
        "timestamp_accuracy": "nominal_fps",
        "stage_times_seconds": {key: round(value, 6) for key, value in stage_times.items()},
        "annotation_file": str(Path(annotations_path).expanduser().resolve()) if annotations_path else None,
        "strategies": {name: result.to_dict() for name, result in results.items()},
        "routing": routing_manifest,
        "review_html": "review.html",
    }
    write_json(output_root / "manifest.json", manifest)
    source_after = metadata.path.stat()
    if source_after.st_size != metadata.size_bytes or source_after.st_mtime_ns != metadata.mtime_ns:
        raise MediaError("Source video changed during the run; results are not trustworthy")
    return {**manifest, "output": str(output_root), "review_html": str(output_root / "review.html")}


def evaluate_run(
    run_directory: str | Path,
    *,
    annotations_path: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    run_root = Path(run_directory).expanduser().resolve()
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Run manifest does not exist: {manifest_path}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "candidate_frame_benchmark":
        raise ValueError(f"Not a FrameLedger benchmark run: {run_root}")
    annotations = load_annotations(annotations_path)
    range_data = manifest["range"]
    validate_annotations(
        annotations,
        video_name=Path(manifest["source"]["path"]).name,
        start_seconds=float(range_data["start_seconds"]),
        duration_seconds=float(range_data["duration_seconds"]),
        source_sha256=str(manifest["source"]["fingerprint"]),
    )
    destination = (
        Path(output).expanduser().resolve()
        if output is not None
        else run_root / f"evaluation-{Path(annotations_path).stem}.json"
    )
    if destination.exists():
        raise ValueError(f"Evaluation output already exists; refusing to overwrite: {destination}")
    strategies: dict[str, Any] = {}
    for raw_name in manifest["parameters"]["strategies"]:
        name, strategy_path = resolve_strategy_artifact(run_root, raw_name)
        strategy_bytes = strategy_path.read_bytes()
        strategy_fingerprint = hashlib.sha256(strategy_bytes).hexdigest()
        strategy_data = json.loads(strategy_bytes)
        selected = [candidate_from_dict(item) for item in strategy_data.get("selected", [])]
        strategies[name] = {
            "selected_frames": len(selected),
            "raw_candidates": int(strategy_data.get("raw_candidate_count", 0)),
            "duplicate_dropped_count": int(strategy_data.get("duplicate_dropped_count", 0)),
            "capped_count": int(strategy_data.get("capped_count", 0)),
            "annotations": evaluate_annotations(
                annotations,
                selected,
                strategy_name=name,
                candidate_set_sha256=strategy_fingerprint,
            ),
        }
    evaluation = {
        "kind": "candidate_frame_evaluation",
        "run_directory": str(run_root),
        "source_fingerprint": manifest["source"]["fingerprint"],
        "range": range_data,
        "annotation_file": str(Path(annotations_path).expanduser().resolve()),
        "annotation_status": annotations.get("review_status", "unspecified"),
        "strategies": strategies,
    }
    write_json(destination, evaluation)
    return {**evaluation, "output": str(destination)}
