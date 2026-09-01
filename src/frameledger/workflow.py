from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .alignment import run_evidence_alignment
from .asr import STANDARD_FALLBACK_DECODING_PROFILE, run_local_asr
from .media import MediaError, probe_video
from .markdown_export import run_markdown_export
from .ocr import run_frame_ocr
from .pipeline import run_benchmark
from .semantic import run_semantic_segmentation
from .timecode import format_timecode


WORKFLOW_SCHEMA_VERSION = 1
WORKFLOW_POLICY_VERSION = "single-video-local-full-range-v3"
DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
DEFAULT_WORKFLOW_MAX_FRAMES = 500
ProgressCallback = Callable[[Mapping[str, Any]], None]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return project_root() / ".cache" / "models" / "whisper-large-v3-turbo"


def default_asr_helper() -> Path:
    return project_root() / "phase2" / "mlx_whisper_asr" / "run"


def default_ocr_helper() -> Path:
    return project_root() / ".cache" / "tools" / "frameledger-apple-vision-ocr"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_existing_file(
    value: str | Path, *, label: str, executable: bool = False
) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{label} is not executable: {path}")
    return path


def _resolve_model(value: str | Path | None) -> Path:
    path = (
        Path(value).expanduser().resolve()
        if value is not None
        else default_model_path()
    )
    if not path.is_dir():
        raise ValueError(
            "Local Whisper model directory does not exist: "
            f"{path}. Download the pinned model before starting the workflow."
        )
    return path


def ensure_apple_vision_helper(value: str | Path | None = None) -> Path:
    if value is not None:
        return _resolve_existing_file(
            value, label="Apple Vision OCR helper", executable=True
        )

    source = project_root() / "phase2" / "apple_vision_ocr" / "main.swift"
    if not source.is_file():
        raise ValueError(f"Apple Vision OCR helper source does not exist: {source}")
    target = default_ocr_helper()
    if (
        target.is_file()
        and os.access(target, os.X_OK)
        and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    module_cache = target.parent / "swift-module-cache"
    module_cache.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    command = [
        "xcrun",
        "swiftc",
        str(source),
        "-o",
        str(temporary),
        "-framework",
        "Vision",
        "-framework",
        "CoreGraphics",
        "-framework",
        "CoreImage",
        "-framework",
        "ImageIO",
    ]
    compile_environment = os.environ.copy()
    xcode_beta = Path("/Applications/Xcode-beta.app/Contents/Developer")
    if xcode_beta.is_dir():
        compile_environment["DEVELOPER_DIR"] = str(xcode_beta)
    compile_environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    compile_environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
        env=compile_environment,
    )
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Could not compile the local Apple Vision helper: {detail}")
    temporary.chmod(0o700)
    temporary.replace(target)
    return target


def _validate_output_root(output: str | Path, *, source: Path) -> Path:
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise ValueError(
            f"Workflow output already exists; refusing to overwrite: {root}"
        )
    source_parent = source.parent.resolve()
    if root == source_parent or source_parent in root.parents:
        raise ValueError(
            "Workflow output cannot be written beside or below the source video"
        )
    frozen = (project_root() / "freezes" / "phase1-v1").resolve()
    if root == frozen or frozen in root.parents:
        raise ValueError("Workflow output cannot modify the frozen Phase 1 baseline")
    return root


def preflight_local_capabilities(
    *, apple_vision_helper: Path, mlx_whisper_helper: Path
) -> dict[str, Any]:
    import numpy as np

    from .media import write_png

    with tempfile.TemporaryDirectory(prefix="frameledger-preflight-") as temporary:
        image = Path(temporary) / "vision.png"
        write_png(image, np.full((160, 640, 3), 255, dtype=np.uint8))
        request = {
            "protocol": "frameledger-ocr-helper-v1",
            "image_path": str(image),
            "languages": ["zh-Hans", "en-US"],
            "route_kind": "presentation",
            "roi_normalized": [0.0, 0.0, 1.0, 1.0],
            "bbox_origin": "top_left_normalized",
        }
        vision = subprocess.run(
            [str(apple_vision_helper)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
    if vision.returncode != 0:
        detail = (vision.stderr or vision.stdout).strip()
        raise RuntimeError(
            "Apple Vision is unavailable in this local session; run outside a sandbox or "
            f"headless session. Helper response: {detail}"
        )
    try:
        vision_response = json.loads(vision.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Apple Vision preflight returned invalid JSON") from error
    if vision_response.get("protocol") != "frameledger-ocr-helper-v1":
        raise RuntimeError("Apple Vision preflight protocol does not match")

    mlx_python = mlx_whisper_helper.parent / ".venv" / "bin" / "python"
    if not mlx_python.is_file():
        raise ValueError(
            "Cannot preflight MLX Metal because the helper runtime Python is missing: "
            f"{mlx_python}"
        )
    metal = subprocess.run(
        [
            str(mlx_python),
            "-c",
            (
                "import mlx.core as mx; "
                "value=mx.array([1.0]); mx.eval(value); "
                "print(str(mx.default_device()))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    if metal.returncode != 0:
        detail = (metal.stderr or metal.stdout).strip()
        raise RuntimeError(
            "MLX Metal is unavailable in this local session; run outside a sandbox or "
            f"headless session. Runtime response: {detail}"
        )
    return {
        "apple_vision": "available",
        "mlx_metal": "available",
        "mlx_device": metal.stdout.strip(),
        "checked_at": _now(),
    }


def _stage_link(stage: Mapping[str, Any], label: str) -> str:
    review = stage.get("review_relative")
    if not isinstance(review, str) or not review:
        return f"<span class='disabled'>{html.escape(label)}</span>"
    download = " download" if review.lower().endswith(".md") else ""
    return (
        f"<a href='{html.escape(review, quote=True)}'{download}>"
        f"{html.escape(label)}</a>"
    )


def _workflow_html(document: Mapping[str, Any]) -> str:
    source = (
        document.get("source") if isinstance(document.get("source"), Mapping) else {}
    )
    stages = (
        document.get("stages") if isinstance(document.get("stages"), Mapping) else {}
    )
    status = str(document.get("status", "unknown"))
    status_class = (
        "ok" if status == "ok" else "error" if status == "error" else "running"
    )
    rows: list[str] = []
    labels = {
        "visual": "视觉路由与关键帧",
        "ocr": "本地 OCR",
        "asr": "整片本地语音识别",
        "alignment": "视觉—语音联合审阅",
        "semantic": "本地语义分段",
        "markdown": "Markdown 关键截图",
    }
    for key, label in labels.items():
        raw = stages.get(key)
        stage = raw if isinstance(raw, Mapping) else {}
        elapsed = stage.get("elapsed_seconds")
        elapsed_text = (
            f"{float(elapsed):.2f}s" if isinstance(elapsed, (int, float)) else "—"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(str(stage.get('status', 'pending')))}</td>"
            f"<td>{elapsed_text}</td>"
            f"<td>{_stage_link(stage, '下载 Markdown' if key == 'markdown' else '打开结果')}</td>"
            "</tr>"
        )
    error = document.get("error")
    error_html = (
        f"<div class='error-box'>{html.escape(str(error))}</div>" if error else ""
    )
    final_link = ""
    final_kind = "markdown"
    final_stage = stages.get(final_kind)
    if not (
        isinstance(final_stage, Mapping)
        and isinstance(final_stage.get("review_relative"), str)
    ):
        final_kind = "semantic"
        final_stage = stages.get(final_kind)
    if not (
        isinstance(final_stage, Mapping)
        and isinstance(final_stage.get("review_relative"), str)
    ):
        final_kind = "alignment"
        final_stage = stages.get(final_kind)
    if isinstance(final_stage, Mapping) and isinstance(
        final_stage.get("review_relative"), str
    ):
        final_label = (
            "下载关键截图 Markdown"
            if final_kind == "markdown"
            else "打开最终语义章节审阅页"
        )
        download = " download" if final_kind == "markdown" else ""
        final_link = (
            f"<a class='primary' href='{html.escape(str(final_stage['review_relative']), quote=True)}'{download}>"
            f"{final_label}</a>"
        )
    return f"""<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FrameLedger 完整本地工作流</title>
<style>
:root {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#17212b; background:#eef2f6; }}
body {{ margin:0; }} main {{ max-width:980px; margin:42px auto; padding:0 20px; }}
.card {{ background:#fff; border:1px solid #d5dde5; border-radius:14px; padding:24px; box-shadow:0 8px 30px #1d2b3a12; }}
h1 {{ margin:0 0 8px; }} p {{ color:#536270; line-height:1.55; }}
.status {{ display:inline-block; padding:5px 10px; border-radius:999px; font-weight:650; }}
.status.ok {{ color:#075d48; background:#dff5ec; }} .status.running {{ color:#775600; background:#fff1c4; }}
.status.error {{ color:#842525; background:#ffe0e0; }}
table {{ width:100%; border-collapse:collapse; margin:22px 0; }} th,td {{ padding:12px; border-bottom:1px solid #e1e6eb; text-align:left; }}
a {{ color:#075eac; }} .primary {{ display:inline-block; padding:11px 16px; border-radius:8px; background:#08775e; color:#fff; text-decoration:none; }}
.disabled {{ color:#8a949e; }} code {{ font-size:11px; overflow-wrap:anywhere; }} .error-box {{ padding:12px; background:#fff0f0; border-left:4px solid #b63b3b; }}
</style></head><body><main><section class="card">
<h1>FrameLedger 完整本地工作流</h1>
<p><span class="status {status_class}">{html.escape(status)}</span>　全流程使用本机视觉算法、Apple Vision、本机 Whisper、确定性语义分段与 Markdown 证据导出，不调用聊天模型或云端转录。</p>
<p><strong>{html.escape(Path(str(source.get("path", ""))).name)}</strong><br>
完整范围 {html.escape(str(document.get("range", {}).get("start_timecode", "")))}–{html.escape(str(document.get("range", {}).get("end_timecode", "")))} ·
<code>{html.escape(str(source.get("fingerprint", "")))}</code></p>
{error_html}
<table><thead><tr><th>阶段</th><th>状态</th><th>耗时</th><th>证据</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
{final_link}
<p><a href="workflow.json">机器可读工作流清单</a></p>
</section></main></body></html>"""


def _write_workflow(root: Path, document: Mapping[str, Any]) -> None:
    json_path = root / "workflow.json"
    html_path = root / "index.html"
    json_temporary = root / f".workflow.{uuid.uuid4().hex}.tmp"
    html_temporary = root / f".index.{uuid.uuid4().hex}.tmp"
    json_temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_temporary.write_text(_workflow_html(document), encoding="utf-8")
    json_temporary.replace(json_path)
    html_temporary.replace(html_path)


def _notify(callback: ProgressCallback | None, document: Mapping[str, Any]) -> None:
    if callback is not None:
        callback(
            {
                "status": document.get("status"),
                "current_stage": document.get("current_stage"),
                "stages": document.get("stages"),
                "output": document.get("output"),
                "error": document.get("error"),
            }
        )


def _validate_stage_result(stage_name: str, result: Mapping[str, Any]) -> None:
    if stage_name == "visual":
        strategies = result.get("strategies")
        routed = strategies.get("routed") if isinstance(strategies, Mapping) else None
        if not isinstance(routed, Mapping) or int(routed.get("selected_count", 0)) <= 0:
            raise RuntimeError(
                "Full workflow visual routing produced no evidence frames"
            )
        if int(routed.get("capped_count", 0)) != 0:
            raise RuntimeError("Full workflow visual routing capped evidence frames")
    elif stage_name == "ocr":
        summary = result.get("summary")
        if not isinstance(summary, Mapping):
            raise RuntimeError("Full workflow OCR result has no summary")
        failures = int(summary.get("failure_frames", 0))
        if failures:
            raise RuntimeError(
                f"Full workflow OCR has {failures} failed evidence frames; see ocr.json"
            )
        covered = int(summary.get("ocr_success_frames", 0)) + int(
            summary.get("skipped_frames", 0)
        )
        if covered != int(summary.get("selected_input_frames", -1)):
            raise RuntimeError(
                "Full workflow OCR did not account for every visual frame"
            )
    elif stage_name == "alignment":
        coverage = result.get("coverage")
        summary = result.get("summary")
        if not isinstance(coverage, Mapping) or not isinstance(summary, Mapping):
            raise RuntimeError("Full workflow alignment lacks coverage evidence")
        if coverage.get("complete_phase1_speech_coverage") is not True:
            raise RuntimeError(
                "Full workflow alignment does not cover the complete visual range"
            )
        if int(summary.get("frames_outside_asr_range", -1)) != 0:
            raise RuntimeError(
                "Full workflow alignment left visual frames outside ASR coverage"
            )
    elif stage_name == "semantic":
        coverage = result.get("coverage")
        summary = result.get("summary")
        if not isinstance(coverage, Mapping) or not isinstance(summary, Mapping):
            raise RuntimeError("Full workflow semantic stage lacks assignment evidence")
        if int(summary.get("chapter_count", 0)) <= 0:
            raise RuntimeError("Full workflow semantic stage produced no chapters")
        if coverage.get("complete_event_assignment") is not True:
            raise RuntimeError(
                "Full workflow semantic stage did not assign every event"
            )
        if any(
            int(coverage.get(key, -1)) != 0
            for key in (
                "unassigned_speech_segment_count",
                "duplicate_speech_assignment_count",
                "unassigned_visual_frame_count",
                "duplicate_visual_assignment_count",
            )
        ):
            raise RuntimeError(
                "Full workflow semantic stage has missing or duplicate events"
            )
    elif stage_name == "markdown":
        coverage = result.get("coverage")
        summary = result.get("summary")
        if not isinstance(coverage, Mapping) or not isinstance(summary, Mapping):
            raise RuntimeError("Full workflow Markdown stage lacks coverage evidence")
        if int(summary.get("visual_frame_count", 0)) <= 0:
            raise RuntimeError("Full workflow Markdown stage produced no screenshots")
        if coverage.get("scope") != "visual_frames_only":
            raise RuntimeError(
                "Full workflow Markdown stage has the wrong content scope"
            )
        if coverage.get("complete_report_coverage") is not True:
            raise RuntimeError(
                "Full workflow Markdown stage did not render every screenshot"
            )
        if any(
            int(coverage.get(key, -1)) != 0
            for key in (
                "unrendered_visual_frame_count",
                "duplicate_visual_render_count",
            )
        ):
            raise RuntimeError(
                "Full workflow Markdown stage has missing or duplicate evidence"
            )
        for key in ("markdown_file", "manifest_json"):
            value = result.get(key)
            if (
                not isinstance(value, str)
                or not Path(value).expanduser().resolve().is_file()
            ):
                raise RuntimeError(f"Full workflow Markdown stage lacks {key}")


def run_complete_workflow(
    video: str | Path,
    *,
    output: str | Path,
    model_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    mlx_whisper_helper: str | Path | None = None,
    apple_vision_helper: str | Path | None = None,
    ffmpeg: str | Path | None = None,
    analysis_fps: float = 2.0,
    analysis_width: int = 480,
    maximum_frames: int | None = None,
    context_before_seconds: float = 20.0,
    context_after_seconds: float = 20.0,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one complete, full-range, local-only FrameLedger workflow."""

    if not math.isfinite(analysis_fps) or analysis_fps < 1 or analysis_fps > 4:
        raise ValueError("Workflow analysis_fps must be between 1 and 4")
    if analysis_width < 160 or analysis_width > 1280:
        raise ValueError("Workflow analysis_width must be between 160 and 1280 pixels")
    if maximum_frames is not None and maximum_frames <= 0:
        raise ValueError("Workflow maximum_frames must be positive")
    if (
        not math.isfinite(context_before_seconds)
        or not 0 <= context_before_seconds <= 120
    ):
        raise ValueError("Workflow context_before_seconds must be between 0 and 120")
    if (
        not math.isfinite(context_after_seconds)
        or not 0 <= context_after_seconds <= 120
    ):
        raise ValueError("Workflow context_after_seconds must be between 0 and 120")
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("Workflow model_id must not be empty")
    model_revision = model_revision.strip().lower()
    if len(model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in model_revision
    ):
        raise ValueError(
            "Workflow model_revision must be one full 40-character hexadecimal commit"
        )

    metadata = probe_video(video)
    source = metadata.path.expanduser().resolve()
    root = _validate_output_root(output, source=source)
    model = _resolve_model(model_path)
    asr_helper = _resolve_existing_file(
        mlx_whisper_helper or default_asr_helper(),
        label="MLX Whisper helper",
        executable=True,
    )
    ocr_helper = ensure_apple_vision_helper(apple_vision_helper)
    ffmpeg_path: Path | None = None
    if ffmpeg is not None:
        ffmpeg_path = _resolve_existing_file(ffmpeg, label="FFmpeg", executable=True)
    else:
        discovered = shutil.which("ffmpeg")
        if discovered is None:
            raise RuntimeError(
                "FFmpeg is unavailable; install it or supply an explicit path"
            )
        ffmpeg_path = Path(discovered).resolve()

    native_preflight = preflight_local_capabilities(
        apple_vision_helper=ocr_helper,
        mlx_whisper_helper=asr_helper,
    )

    root.mkdir(parents=True, exist_ok=False)
    full_duration = float(metadata.duration_seconds)
    resolved_maximum_frames = (
        int(maximum_frames)
        if maximum_frames is not None
        else DEFAULT_WORKFLOW_MAX_FRAMES
    )
    document: dict[str, Any] = {
        "kind": "complete_local_workflow",
        "schema_version": WORKFLOW_SCHEMA_VERSION,
        "policy_version": WORKFLOW_POLICY_VERSION,
        "status": "running",
        "created_at": _now(),
        "completed_at": None,
        "output": str(root),
        "source": metadata.to_dict(),
        "range": {
            "start_seconds": 0.0,
            "end_seconds": round(full_duration, 6),
            "duration_seconds": round(full_duration, 6),
            "start_timecode": format_timecode(0.0),
            "end_timecode": format_timecode(full_duration),
            "coverage_intent": "complete_source_video",
            "range_origin": "probed_full_video",
        },
        "parameters": {
            "visual_strategies": ["routed"],
            "downstream_strategy": "routed",
            "ocr_routes": ["presentation", "table"],
            "analysis_fps": analysis_fps,
            "analysis_width": analysis_width,
            "maximum_frames": resolved_maximum_frames,
            "model_id": model_id,
            "model_revision": model_revision,
            "model_path": str(model),
            "decoding_profile": STANDARD_FALLBACK_DECODING_PROFILE,
            "initial_prompt": None,
            "full_range_asr_explicitly_allowed": True,
            "full_range_visual_explicitly_allowed": True,
            "network_ai_used": False,
            "chat_model_used": False,
        },
        "runtime": {
            "apple_vision_helper": str(ocr_helper),
            "apple_vision_helper_sha256": _sha256(ocr_helper),
            "mlx_whisper_helper": str(asr_helper),
            "mlx_whisper_helper_sha256": _sha256(asr_helper),
            "ffmpeg": str(ffmpeg_path),
            "native_preflight": native_preflight,
        },
        "current_stage": "visual",
        "stages": {
            key: {"status": "pending"}
            for key in ("visual", "ocr", "asr", "alignment", "semantic", "markdown")
        },
        "error": None,
    }
    _write_workflow(root, document)
    _notify(progress_callback, document)

    stage_specs = (
        (
            "visual",
            lambda: run_benchmark(
                source,
                output=root / "visual",
                start_seconds=0.0,
                duration_seconds=full_duration,
                strategies=("routed",),
                analysis_fps=analysis_fps,
                analysis_width=analysis_width,
                maximum_frames=resolved_maximum_frames,
                allow_long_range=True,
            ),
        ),
        (
            "ocr",
            lambda: run_frame_ocr(
                root / "visual",
                strategy="routed",
                routes=("presentation", "table"),
                engine="apple_vision",
                languages=("zh-Hans", "en-US"),
                output=root / "ocr",
                helper=ocr_helper,
            ),
        ),
        (
            "asr",
            lambda: run_local_asr(
                source,
                start_seconds=0.0,
                duration_seconds=full_duration,
                model_path=model,
                model_id=model_id,
                model_revision=model_revision,
                language="zh",
                task="transcribe",
                decoding_profile=STANDARD_FALLBACK_DECODING_PROFILE,
                initial_prompt=None,
                output=root / "asr",
                helper=asr_helper,
                ffmpeg=ffmpeg_path,
                allow_long_range=True,
            ),
        ),
        (
            "alignment",
            lambda: run_evidence_alignment(
                root / "visual",
                strategy="routed",
                ocr_json=root / "ocr" / "ocr.json",
                asr_json=root / "asr" / "asr.json",
                context_before_seconds=context_before_seconds,
                context_after_seconds=context_after_seconds,
                output=root / "alignment",
            ),
        ),
        (
            "semantic",
            lambda: run_semantic_segmentation(
                root / "alignment" / "evidence.json",
                output=root / "semantic",
            ),
        ),
        (
            "markdown",
            lambda: run_markdown_export(
                root / "semantic" / "semantic-segments.json",
                output=root / "markdown",
            ),
        ),
    )

    try:
        for stage_name, runner in stage_specs:
            document["current_stage"] = stage_name
            document["stages"][stage_name] = {"status": "running", "started_at": _now()}
            _write_workflow(root, document)
            _notify(progress_callback, document)
            started = time.perf_counter()
            result = runner()
            _validate_stage_result(stage_name, result)
            artifact_relatives: dict[str, str | None] = {
                "markdown_relative": None,
                "manifest_relative": None,
            }
            if stage_name == "markdown":
                for result_key, relative_key in (
                    ("markdown_file", "markdown_relative"),
                    ("manifest_json", "manifest_relative"),
                ):
                    artifact_path = Path(str(result[result_key])).expanduser().resolve()
                    if not artifact_path.is_relative_to(root):
                        raise RuntimeError(
                            f"Full workflow Markdown {result_key} escaped the run root"
                        )
                    artifact_relatives[relative_key] = artifact_path.relative_to(
                        root
                    ).as_posix()
            elapsed = round(time.perf_counter() - started, 6)
            review_html = result.get("review_html")
            primary_artifact = review_html or result.get("markdown_file")
            review_relative = None
            if isinstance(primary_artifact, str):
                review_path = Path(primary_artifact).expanduser().resolve()
                if review_path.is_relative_to(root):
                    review_relative = review_path.relative_to(root).as_posix()
            document["stages"][stage_name] = {
                "status": "ok",
                "started_at": document["stages"][stage_name]["started_at"],
                "completed_at": _now(),
                "elapsed_seconds": elapsed,
                "output": result.get("output"),
                "review_html": review_html,
                "review_relative": review_relative,
                "markdown_file": result.get("markdown_file"),
                "manifest_json": result.get("manifest_json"),
                **artifact_relatives,
                "summary": result.get("summary"),
                "coverage": result.get("coverage"),
            }
            _write_workflow(root, document)
            _notify(progress_callback, document)

        final_metadata = probe_video(source)
        if (
            final_metadata.fingerprint != metadata.fingerprint
            or final_metadata.size_bytes != metadata.size_bytes
            or final_metadata.mtime_ns != metadata.mtime_ns
        ):
            raise MediaError("Source video changed during the complete workflow")
        document["status"] = "ok"
        document["current_stage"] = None
        document["completed_at"] = _now()
        document["source_integrity_verified_after_workflow"] = True
        _write_workflow(root, document)
        _notify(progress_callback, document)
        return {
            "kind": document["kind"],
            "output": str(root),
            "review_html": str(root / "index.html"),
            "workflow_json": str(root / "workflow.json"),
            "markdown_file": document["stages"]["markdown"].get("markdown_file"),
            "manifest_json": document["stages"]["markdown"].get("manifest_json"),
            "source": metadata.to_dict(),
            "summary": {
                "status": "ok",
                "duration_seconds": full_duration,
                "stage_count": len(stage_specs),
                "alignment": document["stages"]["alignment"].get("summary"),
                "semantic": document["stages"]["semantic"].get("summary"),
                "markdown": document["stages"]["markdown"].get("summary"),
            },
        }
    except Exception as error:
        stage_name = str(document.get("current_stage") or "workflow")
        existing = document["stages"].get(stage_name)
        if isinstance(existing, dict):
            existing["status"] = "error"
            existing["completed_at"] = _now()
            existing["error_type"] = type(error).__name__
            existing["error"] = str(error)
        document["status"] = "error"
        document["completed_at"] = _now()
        document["error"] = f"{stage_name}: {type(error).__name__}: {error}"
        _write_workflow(root, document)
        _notify(progress_callback, document)
        raise
