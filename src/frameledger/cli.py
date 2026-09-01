from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .alignment import run_evidence_alignment
from .asr import DECODING_PROFILE_NAMES, FIXED_ZERO_DECODING_PROFILE, run_local_asr
from .asr_annotations import evaluate_asr_anchor_set
from .media import MediaError, probe_video
from .markdown_export import run_markdown_export
from .ocr import run_frame_ocr
from .pipeline import evaluate_run, export_reviewed_frames, run_benchmark, scan_video
from .semantic import (
    DEFAULT_MAX_CHAPTER_SECONDS,
    DEFAULT_MIN_CHAPTER_SECONDS,
    DEFAULT_TARGET_CHAPTER_SECONDS,
    DEFAULT_TOPIC_WINDOW_SECONDS,
    run_semantic_segmentation,
)
from .timecode import parse_timecode
from .workflow import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    default_asr_helper,
    default_model_path,
    run_complete_workflow,
)


def _timecode(value: str) -> float:
    try:
        return parse_timecode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _strategies(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frameledger",
        description="Evidence-first candidate-frame benchmark for financial videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe", help="Read video metadata without writing output"
    )
    probe.add_argument("video", type=Path)

    scan = subparsers.add_parser(
        "scan",
        help="Generate a sparse visual timeline for choosing an 8-12 minute benchmark range",
    )
    scan.add_argument("video", type=Path)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--start", type=_timecode, default=0.0)
    scan.add_argument("--duration", type=_timecode)
    scan.add_argument("--interval", type=float, default=60.0)
    scan.add_argument("--max-frames", type=int, default=120)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare candidate-frame strategies on one bounded video range",
    )
    benchmark.add_argument("video", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--start", type=_timecode, required=True)
    benchmark.add_argument("--duration", type=_timecode, required=True)
    benchmark.add_argument(
        "--strategies",
        type=_strategies,
        default=_strategies("uniform,content,adaptive,settled,hybrid"),
        help=(
            "Comma-separated selectors. Use 'routed' for the experimental "
            "visual PPT/table/chart router; it is not part of the default hybrid."
        ),
    )
    benchmark.add_argument("--annotations", type=Path)
    benchmark.add_argument("--analysis-fps", type=float, default=2.0)
    benchmark.add_argument(
        "--analysis-width",
        type=int,
        default=320,
        help="Analysis-frame width; use 480 for routed visual benchmarks.",
    )
    benchmark.add_argument("--uniform-interval", type=float, default=10.0)
    benchmark.add_argument("--max-frames", type=int, default=80)
    benchmark.add_argument("--content-threshold", type=float, default=27.0)
    benchmark.add_argument("--adaptive-threshold", type=float, default=3.0)
    benchmark.add_argument("--boundary-transition-threshold", type=float, default=0.025)
    benchmark.add_argument(
        "--boundary-pixel-delta-threshold", type=float, default=0.004
    )
    benchmark.add_argument("--boundary-edge-loss-threshold", type=float, default=0.003)
    benchmark.add_argument(
        "--boundary-terminal-search-seconds", type=float, default=2.0
    )
    benchmark.add_argument(
        "--boundary-terminal-stable-seconds", type=float, default=1.0
    )
    benchmark.add_argument(
        "--boundary-terminal-stability-threshold", type=float, default=0.008
    )
    benchmark.add_argument("--boundary-lookahead-seconds", type=float, default=15.0)
    benchmark.add_argument("--table-lookahead-seconds", type=float, default=8.0)
    benchmark.add_argument("--presentation-lookahead-seconds", type=float, default=6.0)
    benchmark.add_argument("--allow-long-range", action="store_true")

    curate = subparsers.add_parser(
        "curate",
        help="Export human-reviewed timestamps as source-resolution evidence frames",
    )
    curate.add_argument("video", type=Path)
    curate.add_argument("--annotations", type=Path, required=True)
    curate.add_argument("--output", type=Path, required=True)

    ocr = subparsers.add_parser(
        "ocr",
        help="OCR selected PNG evidence from one frozen Phase 1 run",
    )
    ocr.add_argument("run_directory", type=Path)
    ocr.add_argument("--strategy", default="routed")
    ocr.add_argument(
        "--routes",
        type=_strategies,
        default=_strategies("presentation,table"),
    )
    ocr.add_argument("--engine", choices=("apple_vision",), default="apple_vision")
    ocr.add_argument(
        "--apple-vision-helper",
        type=Path,
        help=(
            "Executable implementing frameledger-ocr-helper-v1; otherwise use "
            "FRAMELEDGER_APPLE_VISION_HELPER or PATH discovery."
        ),
    )
    ocr.add_argument(
        "--languages",
        type=_csv,
        default=_csv("zh-Hans,en-US"),
    )
    ocr.add_argument("--output", type=Path, required=True)

    asr = subparsers.add_parser(
        "asr",
        help="Transcribe one bounded source-video range with a local Whisper helper",
    )
    asr.add_argument("video", type=Path)
    asr.add_argument("--start", type=_timecode, required=True)
    asr.add_argument("--duration", type=_timecode, required=True)
    asr.add_argument("--engine", choices=("mlx_whisper",), default="mlx_whisper")
    asr.add_argument("--model", type=Path, required=True)
    asr.add_argument(
        "--model-id",
        default="mlx-community/whisper-large-v3-turbo",
    )
    asr.add_argument("--model-revision", required=True)
    asr.add_argument("--language", default="zh")
    asr.add_argument("--task", choices=("transcribe",), default="transcribe")
    asr.add_argument(
        "--decoding-profile",
        choices=DECODING_PROFILE_NAMES,
        default=FIXED_ZERO_DECODING_PROFILE,
    )
    asr.add_argument(
        "--initial-prompt",
        help="Optional bounded Whisper vocabulary/context prompt recorded in ASR provenance",
    )
    asr.add_argument("--mlx-whisper-helper", type=Path, required=True)
    asr.add_argument("--ffmpeg", type=Path)
    asr.add_argument(
        "--allow-long-range",
        action="store_true",
        help=(
            "Explicitly permit a single-video ASR range longer than 600 seconds; "
            "the exact start and duration remain required."
        ),
    )
    asr.add_argument("--output", type=Path, required=True)

    asr_evaluate = subparsers.add_parser(
        "asr-evaluate",
        help="Score one raw ASR output against human-reviewed reported-issue anchors",
    )
    asr_evaluate.add_argument("candidate_asr", type=Path)
    asr_evaluate.add_argument("--annotations", type=Path, required=True)
    asr_evaluate.add_argument("--tolerance", type=float, default=0.75)
    asr_evaluate.add_argument("--output", type=Path, required=True)

    align = subparsers.add_parser(
        "align",
        help="Align Phase 1 visual/OCR evidence with bounded ASR by absolute time",
    )
    align.add_argument("run_directory", type=Path)
    align.add_argument("--strategy", required=True)
    align.add_argument("--ocr-json", type=Path, required=True)
    align.add_argument("--asr-json", type=Path, required=True)
    align.add_argument("--context-before", type=float, default=20.0)
    align.add_argument("--context-after", type=float, default=20.0)
    align.add_argument("--output", type=Path, required=True)

    semantic = subparsers.add_parser(
        "semantic",
        help="Create deterministic local topic chapters from aligned evidence",
    )
    semantic.add_argument("alignment_json", type=Path)
    semantic.add_argument("--output", type=Path, required=True)
    semantic.add_argument(
        "--min-chapter-seconds", type=float, default=DEFAULT_MIN_CHAPTER_SECONDS
    )
    semantic.add_argument(
        "--target-chapter-seconds", type=float, default=DEFAULT_TARGET_CHAPTER_SECONDS
    )
    semantic.add_argument(
        "--max-chapter-seconds", type=float, default=DEFAULT_MAX_CHAPTER_SECONDS
    )
    semantic.add_argument(
        "--topic-window-seconds", type=float, default=DEFAULT_TOPIC_WINDOW_SECONDS
    )

    markdown = subparsers.add_parser(
        "markdown",
        help="Export verified key screenshots as local Markdown",
    )
    markdown.add_argument("semantic_json", type=Path)
    markdown.add_argument("--output", type=Path, required=True)

    workflow = subparsers.add_parser(
        "workflow",
        help=(
            "Run the complete local visual/OCR/full-range-ASR/alignment/semantic/"
            "Markdown workflow"
        ),
    )
    workflow.add_argument("video", type=Path)
    workflow.add_argument("--output", type=Path, required=True)
    workflow.add_argument("--model", type=Path, default=default_model_path())
    workflow.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    workflow.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    workflow.add_argument(
        "--mlx-whisper-helper", type=Path, default=default_asr_helper()
    )
    workflow.add_argument("--apple-vision-helper", type=Path)
    workflow.add_argument("--ffmpeg", type=Path)
    workflow.add_argument("--analysis-fps", type=float, default=2.0)
    workflow.add_argument("--analysis-width", type=int, default=480)
    workflow.add_argument(
        "--max-frames",
        type=int,
        help="Explicit routed evidence safety cap; default 500 and fails rather than truncates.",
    )

    workflow_ui = subparsers.add_parser(
        "workflow-ui",
        help="Serve the local-only HTML controller for complete workflows",
    )
    workflow_ui.add_argument(
        "--output-root", type=Path, default=Path("outputs/workflows")
    )
    workflow_ui.add_argument("--port", type=int, default=8765)
    workflow_ui.add_argument("--video", type=Path)
    workflow_ui.add_argument("--model", type=Path, default=default_model_path())
    workflow_ui.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    workflow_ui.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    workflow_ui.add_argument(
        "--mlx-whisper-helper", type=Path, default=default_asr_helper()
    )
    workflow_ui.add_argument("--apple-vision-helper", type=Path)
    workflow_ui.add_argument("--ffmpeg", type=Path)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score an existing run against YAML without decoding the video again",
    )
    evaluate.add_argument("run_directory", type=Path)
    evaluate.add_argument("--annotations", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            result = probe_video(args.video).to_dict()
        elif args.command == "scan":
            result = scan_video(
                args.video,
                output=args.output,
                start_seconds=args.start,
                duration_seconds=args.duration,
                interval_seconds=args.interval,
                maximum_frames=args.max_frames,
            )
        elif args.command == "benchmark":
            result = run_benchmark(
                args.video,
                output=args.output,
                start_seconds=args.start,
                duration_seconds=args.duration,
                strategies=args.strategies,
                annotations_path=args.annotations,
                analysis_fps=args.analysis_fps,
                analysis_width=args.analysis_width,
                uniform_interval_seconds=args.uniform_interval,
                maximum_frames=args.max_frames,
                content_threshold=args.content_threshold,
                adaptive_threshold=args.adaptive_threshold,
                boundary_transition_threshold=args.boundary_transition_threshold,
                boundary_pixel_delta_threshold=args.boundary_pixel_delta_threshold,
                boundary_edge_loss_threshold=args.boundary_edge_loss_threshold,
                boundary_terminal_search_seconds=args.boundary_terminal_search_seconds,
                boundary_terminal_stable_seconds=args.boundary_terminal_stable_seconds,
                boundary_terminal_stability_threshold=args.boundary_terminal_stability_threshold,
                boundary_lookahead_seconds=args.boundary_lookahead_seconds,
                table_lookahead_seconds=args.table_lookahead_seconds,
                presentation_lookahead_seconds=args.presentation_lookahead_seconds,
                allow_long_range=args.allow_long_range,
            )
        elif args.command == "curate":
            result = export_reviewed_frames(
                args.video,
                annotations_path=args.annotations,
                output=args.output,
            )
        elif args.command == "ocr":
            result = run_frame_ocr(
                args.run_directory,
                strategy=args.strategy,
                routes=args.routes,
                engine=args.engine,
                languages=args.languages,
                output=args.output,
                helper=args.apple_vision_helper,
            )
        elif args.command == "asr":
            result = run_local_asr(
                args.video,
                start_seconds=args.start,
                duration_seconds=args.duration,
                model_path=args.model,
                model_id=args.model_id,
                model_revision=args.model_revision,
                language=args.language,
                task=args.task,
                decoding_profile=args.decoding_profile,
                initial_prompt=args.initial_prompt,
                output=args.output,
                helper=args.mlx_whisper_helper,
                ffmpeg=args.ffmpeg,
                allow_long_range=args.allow_long_range,
            )
        elif args.command == "asr-evaluate":
            result = evaluate_asr_anchor_set(
                args.candidate_asr,
                annotations_path=args.annotations,
                output=args.output,
                tolerance_seconds=args.tolerance,
            )
        elif args.command == "align":
            result = run_evidence_alignment(
                args.run_directory,
                strategy=args.strategy,
                ocr_json=args.ocr_json,
                asr_json=args.asr_json,
                context_before_seconds=args.context_before,
                context_after_seconds=args.context_after,
                output=args.output,
            )
        elif args.command == "semantic":
            result = run_semantic_segmentation(
                args.alignment_json,
                output=args.output,
                min_chapter_seconds=args.min_chapter_seconds,
                target_chapter_seconds=args.target_chapter_seconds,
                max_chapter_seconds=args.max_chapter_seconds,
                topic_window_seconds=args.topic_window_seconds,
            )
        elif args.command == "markdown":
            result = run_markdown_export(
                args.semantic_json,
                output=args.output,
            )
        elif args.command == "workflow":
            result = run_complete_workflow(
                args.video,
                output=args.output,
                model_path=args.model,
                model_id=args.model_id,
                model_revision=args.model_revision,
                mlx_whisper_helper=args.mlx_whisper_helper,
                apple_vision_helper=args.apple_vision_helper,
                ffmpeg=args.ffmpeg,
                analysis_fps=args.analysis_fps,
                analysis_width=args.analysis_width,
                maximum_frames=args.max_frames,
            )
        elif args.command == "workflow-ui":
            from .local_ui import serve_workflow_ui

            serve_workflow_ui(
                output_root=args.output_root,
                port=args.port,
                default_video=args.video,
                model_path=args.model,
                model_id=args.model_id,
                model_revision=args.model_revision,
                mlx_whisper_helper=args.mlx_whisper_helper,
                apple_vision_helper=args.apple_vision_helper,
                ffmpeg=args.ffmpeg,
            )
            return 0
        else:
            result = evaluate_run(
                args.run_directory,
                annotations_path=args.annotations,
                output=args.output,
            )
    except (MediaError, ValueError, RuntimeError) as error:
        parser.exit(2, f"frameledger: error: {error}\n")
    source_summary = result.get("source")
    if args.command == "probe":
        source_summary = result
    if source_summary is None and result.get("kind") == "candidate_frame_evaluation":
        source_summary = {
            "run_directory": result.get("run_directory"),
            "source_fingerprint": result.get("source_fingerprint"),
            "annotation_status": result.get("annotation_status"),
        }
    summary = {
        "kind": result.get("kind", "probe"),
        "output": result.get("output"),
        "review_html": result.get("review_html") or result.get("scan_html"),
        "ocr_json": result.get("ocr_json"),
        "asr_json": result.get("asr_json"),
        "evidence_json": result.get("evidence_json"),
        "semantic_json": result.get("semantic_json"),
        "markdown_file": result.get("markdown_file"),
        "manifest_json": result.get("manifest_json"),
        "asr_evaluation_json": (
            result.get("output")
            if result.get("kind") == "asr_anchor_evaluation"
            else None
        ),
        "source": source_summary,
    }
    if result.get("kind") == "frame_ocr":
        summary["ocr_summary"] = result.get("summary")
    if result.get("kind") == "local_speech_transcription":
        summary["asr_summary"] = result.get("summary")
    if result.get("kind") == "asr_anchor_evaluation":
        summary["asr_evaluation_summary"] = result.get("summary")
    if result.get("kind") == "timestamp_aligned_evidence":
        summary["alignment_summary"] = result.get("summary")
        summary["coverage"] = result.get("coverage")
    if result.get("kind") == "local_topic_segmentation":
        summary["semantic_summary"] = result.get("summary")
        summary["coverage"] = result.get("coverage")
    if result.get("kind") == "local_evidence_markdown":
        summary["markdown_summary"] = result.get("summary")
        summary["coverage"] = result.get("coverage")
    if result.get("kind") == "complete_local_workflow":
        summary["workflow_json"] = result.get("workflow_json")
        summary["workflow_summary"] = result.get("summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
