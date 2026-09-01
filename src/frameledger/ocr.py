from __future__ import annotations

import hashlib
import html
import json
import math
import os
import platform
import shutil
import subprocess
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .artifacts import resolve_strategy_artifact
from .models import CONTENT_KINDS
from .report import write_json
from .timecode import format_timecode


OCR_SCHEMA_VERSION = 1
OCR_POLICY_VERSION = "route-roi-v2"
APPLE_VISION_HELPER_PROTOCOL = "frameledger-ocr-helper-v1"
APPLE_VISION_HELPER_ENV = "FRAMELEDGER_APPLE_VISION_HELPER"

_STRATEGY_KIND = {
    "presentation_states": "presentation",
    "table_viewport": "table",
    "table_focus": "table",
    "boundary_terminal": "chart",
}

# Normalized top-left coordinates: x, y, width, height. These are versioned
# Phase 2 OCR regions and do not alter the frozen Phase 1 selector regions.
_OCR_ROI_BY_KIND: dict[str, tuple[float, float, float, float]] = {
    "presentation": (0.0, 0.0, 1.0, 0.92),
    "table": (0.0, 0.25, 1.0, 0.67),
    "chart": (0.0, 0.0, 1.0, 1.0),
    "unknown": (0.0, 0.0, 1.0, 1.0),
}


class OcrBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrObservation:
    text: str
    confidence: float | None
    bbox: tuple[float, float, float, float]


class OcrBackend(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def recognize(
        self,
        image_path: Path,
        *,
        languages: tuple[str, ...],
        route_kind: str,
        roi_normalized: tuple[float, float, float, float],
    ) -> Sequence[OcrObservation | Mapping[str, Any]]: ...


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_helper(configured: str | Path | None) -> Path | None:
    value = configured or os.environ.get(APPLE_VISION_HELPER_ENV)
    if value:
        text = str(value)
        if "/" in text:
            candidate = Path(text).expanduser().resolve()
            return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
        discovered = shutil.which(text)
        return Path(discovered).resolve() if discovered else None
    discovered = shutil.which("frameledger-apple-vision-ocr")
    return Path(discovered).resolve() if discovered else None


class AppleVisionOcrBackend:
    """Adapter for a separate Apple Vision helper using a small JSON protocol.

    The helper reads one JSON object from stdin and writes one JSON object to
    stdout.  It keeps the Python package free of PyObjC and allows the native
    Vision invocation to be compiled, signed, and tested independently.
    """

    def __init__(
        self,
        *,
        helper: str | Path | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("OCR helper timeout must be finite and positive")
        self.helper_path = _resolve_helper(helper)
        self.helper_sha256 = _sha256(self.helper_path) if self.helper_path else None
        self.timeout_seconds = timeout_seconds
        self._runtime_metadata: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "name": "apple_vision",
            "protocol": APPLE_VISION_HELPER_PROTOCOL,
            "helper_path": str(self.helper_path) if self.helper_path else None,
            "helper_sha256": self.helper_sha256,
            "available": self.helper_path is not None,
            "runtime": self._runtime_metadata or None,
            "platform": platform.platform(),
        }

    def recognize(
        self,
        image_path: Path,
        *,
        languages: tuple[str, ...],
        route_kind: str,
        roi_normalized: tuple[float, float, float, float],
    ) -> Sequence[Mapping[str, Any]]:
        if self.helper_path is None:
            raise OcrBackendError(
                "Apple Vision OCR helper is unavailable; set "
                f"{APPLE_VISION_HELPER_ENV} to an executable implementing "
                f"{APPLE_VISION_HELPER_PROTOCOL}"
            )
        request = {
            "protocol": APPLE_VISION_HELPER_PROTOCOL,
            "image_path": str(image_path),
            "languages": list(languages),
            "route_kind": route_kind,
            "roi_normalized": list(roi_normalized),
            "bbox_origin": "top_left_normalized",
        }
        try:
            completed = subprocess.run(
                [str(self.helper_path)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OcrBackendError(f"Apple Vision OCR helper could not run: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:500]
            raise OcrBackendError(
                f"Apple Vision OCR helper exited with {completed.returncode}: "
                f"{detail or 'no diagnostic output'}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise OcrBackendError("Apple Vision OCR helper returned invalid JSON") from error
        if not isinstance(response, dict):
            raise OcrBackendError("Apple Vision OCR helper response must be an object")
        if response.get("protocol") not in (None, APPLE_VISION_HELPER_PROTOCOL):
            raise OcrBackendError("Apple Vision OCR helper protocol does not match")
        runtime = response.get("engine")
        if isinstance(runtime, dict):
            self._runtime_metadata = {
                key: value
                for key, value in runtime.items()
                if key
                not in {
                    "route_kind",
                    "requested_roi_normalized",
                    "effective_roi_normalized",
                }
            }
        observations = response.get("observations")
        if not isinstance(observations, list):
            raise OcrBackendError("Apple Vision OCR helper response lacks observations")
        return observations


def build_ocr_backend(
    engine: str,
    *,
    helper: str | Path | None = None,
) -> OcrBackend:
    if engine != "apple_vision":
        raise ValueError(f"Unsupported OCR engine: {engine}")
    return AppleVisionOcrBackend(helper=helper)


def _normalize_string_list(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    if not normalized:
        raise ValueError(f"At least one {label} value is required")
    return normalized


def _normalize_observations(
    observations: Sequence[OcrObservation | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, observation in enumerate(observations):
        if isinstance(observation, OcrObservation):
            text = observation.text
            confidence = observation.confidence
            bbox = observation.bbox
        elif isinstance(observation, Mapping):
            text = observation.get("text")
            confidence = observation.get("confidence")
            bbox = observation.get("bbox")
        else:
            raise OcrBackendError("OCR observation must be an object")
        if not isinstance(text, str) or not text.strip():
            raise OcrBackendError("OCR observation text must be non-empty")
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise OcrBackendError("OCR confidence must be in [0, 1]")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise OcrBackendError("OCR bbox must contain x, y, width, height")
        coordinates = tuple(float(value) for value in bbox)
        if any(not math.isfinite(value) for value in coordinates):
            raise OcrBackendError("OCR bbox values must be finite")
        x, y, width, height = coordinates
        if (
            x < 0
            or y < 0
            or width < 0
            or height < 0
            or x > 1
            or y > 1
            or x + width > 1.000001
            or y + height > 1.000001
        ):
            raise OcrBackendError("OCR bbox must use normalized top-left coordinates")
        normalized.append(
            {
                "order": index,
                "text": text.strip(),
                "confidence": round(confidence, 6) if confidence is not None else None,
                "bbox": [round(value, 6) for value in coordinates],
                "bbox_origin": "top_left_normalized",
            }
        )
    return normalized


def _route_segments(
    manifest: Mapping[str, Any],
    strategy_data: Mapping[str, Any],
) -> dict[str, str]:
    segments = strategy_data.get("segments")
    if not isinstance(segments, list):
        routing = manifest.get("routing")
        segments = routing.get("segments") if isinstance(routing, dict) else []
    by_id: dict[str, str] = {}
    if not isinstance(segments, list):
        return by_id
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("OCR input contains an invalid routed segment")
        segment_id = segment.get("segment_id")
        kind = segment.get("kind")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError("OCR input routed segment lacks segment_id")
        if kind not in CONTENT_KINDS:
            raise ValueError(f"OCR input routed segment has unknown kind: {kind}")
        if segment_id in by_id:
            raise ValueError(f"OCR input repeats routed segment id: {segment_id}")
        by_id[segment_id] = str(kind)
    return by_id


def _candidate_route_kind(
    candidate: Mapping[str, Any],
    *,
    strategy: str,
    segment_kinds: Mapping[str, str],
) -> tuple[str | None, str]:
    segment_id = candidate.get("segment_id")
    if segment_id is not None and not isinstance(segment_id, str):
        raise ValueError("OCR candidate segment_id must be a string")
    if strategy == "routed":
        if not segment_id or segment_id not in segment_kinds:
            raise ValueError("Routed OCR candidate is not bound to a known segment")
        return segment_id, segment_kinds[segment_id]
    return segment_id, _STRATEGY_KIND.get(strategy, "unknown")


def _resolve_evidence_image(run_root: Path, image_path: Any) -> tuple[Path, str]:
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("OCR candidate lacks an evidence image path")
    relative = Path(image_path)
    if relative.is_absolute():
        raise ValueError("OCR evidence image path must be relative to the frozen run")
    resolved = (run_root / relative).resolve()
    if not resolved.is_relative_to(run_root):
        raise ValueError("OCR evidence image escapes the frozen run directory")
    if not resolved.is_file():
        raise ValueError(f"OCR evidence image does not exist: {relative.as_posix()}")
    if resolved.suffix.lower() != ".png":
        raise ValueError("OCR v1 accepts only source-resolution PNG evidence frames")
    return resolved, relative.as_posix()


def _validate_output_target(
    output: str | Path,
    *,
    run_root: Path,
    source_path: Any,
) -> Path:
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError(f"OCR output directory already exists; refusing to overwrite: {target}")
    if target == run_root or target.is_relative_to(run_root):
        raise ValueError("OCR output must not modify the frozen Phase 1 run directory")
    if isinstance(source_path, str) and source_path:
        source_parent = Path(source_path).expanduser().resolve().parent
        if target == source_parent or target.is_relative_to(source_parent):
            raise ValueError("OCR output must not be written beside source media")
    return target


def write_ocr_review_html(output_path: Path, document: Mapping[str, Any]) -> None:
    """Write an offline evidence review with image-bound OCR observations."""
    source = document["source"]
    run_root = Path(str(source["run_directory"]))
    review_directory = output_path.parent.resolve()
    workflow_root = review_directory.parent
    frame_cards: list[str] = []
    for frame in document.get("frames", []):
        image_path = (run_root / str(frame["image_path"])).resolve()
        if image_path.is_relative_to(workflow_root):
            relative = Path(os.path.relpath(image_path, review_directory)).as_posix()
            image_uri = urllib.parse.quote(relative, safe="/._-")
        else:
            image_uri = image_path.as_uri()
        boxes: list[str] = []
        rows: list[str] = []
        for observation in frame.get("observations", []):
            x, y, width, height = (float(value) * 1000 for value in observation["bbox"])
            label = html.escape(str(observation["text"]), quote=True)
            confidence = observation.get("confidence")
            confidence_text = "n/a" if confidence is None else f"{float(confidence):.1%}"
            boxes.append(
                f"<rect x='{x:.3f}' y='{y:.3f}' width='{width:.3f}' height='{height:.3f}'>"
                f"<title>{label} · {confidence_text}</title></rect>"
            )
            rows.append(
                "<li>"
                f"<span class='confidence'>{confidence_text}</span>"
                f"<span>{html.escape(str(observation['text']))}</span>"
                "</li>"
            )
        frame_cards.append(
            "<article class='frame-card'>"
            "<div class='visual'>"
            f"<img src='{html.escape(image_uri, quote=True)}' alt='Evidence frame at {format_timecode(float(frame['timestamp']))}'>"
            f"<svg class='boxes' viewBox='0 0 1000 1000' preserveAspectRatio='none'>{''.join(boxes)}</svg>"
            "</div>"
            "<div class='transcript'>"
            f"<h2>{format_timecode(float(frame['timestamp']))}</h2>"
            f"<p>{html.escape(str(frame['route_kind']))} · sample {int(frame['sample_index'])} · "
            f"{len(frame.get('observations', []))} observations</p>"
            f"<ol>{''.join(rows) or '<li>No text observations.</li>'}</ol>"
            "</div></article>"
        )

    exception_cards: list[str] = []
    for category in ("failures", "skipped"):
        for item in document.get(category, []):
            detail = item.get("error") or item.get("reason") or "unspecified"
            exception_cards.append(
                "<li>"
                f"<strong>{html.escape(category[:-1])}</strong> · "
                f"{format_timecode(float(item['timestamp']))} · "
                f"{html.escape(str(item['route_kind']))} · "
                f"{html.escape(str(detail))}"
                "</li>"
            )

    summary = document["summary"]
    engine = document["engine"]
    runtime = engine.get("runtime") or {}
    review = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FrameLedger OCR review</title><style>
:root {{ font-family: ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#17212b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#f2f4f7; }}
header {{ position:sticky; top:0; z-index:4; padding:14px 20px; background:rgba(255,255,255,.97); border-bottom:1px solid #d8dee6; }}
header h1 {{ margin:0 0 6px; font-size:21px; }} header p {{ margin:4px 0; color:#526171; font-size:13px; }}
main {{ max-width:1500px; margin:auto; padding:18px 20px 60px; }}
.frame-card {{ display:grid; grid-template-columns:minmax(0,3fr) minmax(300px,2fr); gap:14px; margin:0 0 20px; padding:12px; background:#fff; border:1px solid #d8dee6; border-radius:10px; }}
.visual {{ position:relative; align-self:start; overflow:hidden; background:#111; }}
.visual img {{ display:block; width:100%; height:auto; }}
.boxes {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }}
.boxes rect {{ fill:rgba(16,185,129,.05); stroke:#10b981; stroke-width:1.5; vector-effect:non-scaling-stroke; }}
.transcript {{ min-width:0; }} .transcript h2 {{ margin:0 0 5px; font-size:18px; font-variant-numeric:tabular-nums; }}
.transcript p {{ color:#607080; font-size:12px; }} ol {{ margin:0; padding-left:26px; }} li {{ margin:0 0 7px; line-height:1.35; }}
.confidence {{ display:inline-block; min-width:54px; margin-right:7px; color:#08775e; font-size:11px; font-variant-numeric:tabular-nums; }}
.exceptions {{ margin-bottom:18px; padding:12px 16px; background:#fff7e8; border:1px solid #e6bd70; border-radius:8px; }}
code {{ overflow-wrap:anywhere; }}
@media(max-width:900px) {{ .frame-card {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
</style></head><body><header><h1>FrameLedger OCR evidence review</h1>
<p>{html.escape(str(source['strategy']))} · {int(summary['ocr_success_frames'])}/{int(summary['selected_input_frames'])} recognized · {int(summary['failure_frames'])} failures · {int(summary['skipped_frames'])} skipped · {int(summary['observation_count'])} observations</p>
<p>Engine: {html.escape(str(engine.get('name')))} · helper SHA-256 <code>{html.escape(str(engine.get('helper_sha256')))}</code> · Vision revision {html.escape(str(runtime.get('request_revision', 'n/a')))}</p>
<p>Boxes and text are raw OCR evidence, not corrected financial facts or reconstructed table cells.</p></header>
<main>{f"<section class='exceptions'><h2>Failures and skips</h2><ul>{''.join(exception_cards)}</ul></section>" if exception_cards else ''}{''.join(frame_cards) or '<p>No frames were recognized.</p>'}</main>
</body></html>"""
    output_path.write_text(review, encoding="utf-8")


def run_frame_ocr(
    run_directory: str | Path,
    *,
    strategy: str,
    routes: Sequence[str],
    engine: str,
    languages: Sequence[str],
    output: str | Path,
    helper: str | Path | None = None,
    backend: OcrBackend | None = None,
) -> dict[str, Any]:
    """OCR selected evidence PNGs from one immutable Phase 1 run."""
    run_root = Path(run_directory).expanduser().resolve()
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Phase 1 run manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("Phase 1 run manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("kind") != "candidate_frame_benchmark":
        raise ValueError(f"Not a FrameLedger Phase 1 benchmark run: {run_root}")

    strategy, strategy_path = resolve_strategy_artifact(run_root, strategy)
    if not strategy_path.is_file():
        raise ValueError(f"OCR strategy result does not exist: {strategy_path}")
    strategy_bytes = strategy_path.read_bytes()
    strategy_sha256 = hashlib.sha256(strategy_bytes).hexdigest()
    try:
        strategy_data = json.loads(strategy_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("OCR strategy result is not valid JSON") from error
    if not isinstance(strategy_data, dict) or strategy_data.get("strategy") != strategy:
        raise ValueError("OCR strategy result does not match the requested strategy")

    requested_routes = _normalize_string_list(
        [route.lower() for route in routes], label="OCR route"
    )
    invalid_routes = sorted(set(requested_routes) - CONTENT_KINDS)
    if invalid_routes:
        raise ValueError(f"Unknown OCR routes: {', '.join(invalid_routes)}")
    requested_languages = _normalize_string_list(languages, label="OCR language")
    engine = engine.strip().lower()
    selected = strategy_data.get("selected")
    if not isinstance(selected, list):
        raise ValueError("OCR strategy result lacks a selected candidate list")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Phase 1 run manifest lacks source metadata")
    video_sha256 = source.get("fingerprint")
    if not isinstance(video_sha256, str) or len(video_sha256) != 64:
        raise ValueError("Phase 1 run manifest lacks a valid source fingerprint")
    output_root = _validate_output_target(
        output,
        run_root=run_root,
        source_path=source.get("path"),
    )
    ocr_backend = (
        backend
        if backend is not None
        else build_ocr_backend(engine, helper=helper)
    )
    segment_kinds = _route_segments(manifest, strategy_data)

    frames: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    image_hashes: dict[Path, str] = {}

    for candidate in selected:
        if not isinstance(candidate, dict):
            raise ValueError("OCR selected candidate must be an object")
        sample_index = int(candidate["sample_index"])
        timestamp = float(candidate["timestamp"])
        if sample_index < 0 or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("OCR candidate index/timestamp is invalid")
        segment_id, route_kind = _candidate_route_kind(
            candidate,
            strategy=strategy,
            segment_kinds=segment_kinds,
        )
        image, relative_image = _resolve_evidence_image(run_root, candidate.get("image_path"))
        image_sha256 = _sha256(image)
        image_hashes[image] = image_sha256
        roi = _OCR_ROI_BY_KIND[route_kind]
        identity = {
            "sample_index": sample_index,
            "timestamp": round(timestamp, 6),
            "segment_id": segment_id,
            "route_kind": route_kind,
            "image_path": relative_image,
            "image_sha256": image_sha256,
            "roi_normalized": [round(value, 6) for value in roi],
            "roi_origin": "top_left_normalized",
        }
        if route_kind not in requested_routes:
            skipped.append(
                {
                    **identity,
                    "status": "skipped",
                    "reason": "skipped_by_route_policy",
                }
            )
            continue
        try:
            observations = _normalize_observations(
                ocr_backend.recognize(
                    image,
                    languages=requested_languages,
                    route_kind=route_kind,
                    roi_normalized=roi,
                )
            )
        except Exception as error:  # Backends are an explicit failure boundary.
            failures.append(
                {
                    **identity,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        frames.append(
            {
                **identity,
                "status": "ok",
                "observations": observations,
                "plain_text": "\n".join(item["text"] for item in observations),
            }
        )

    # Refuse to publish OCR evidence if any frozen input changed while the
    # backend was reading it.
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError("Phase 1 manifest changed during OCR; results are not trustworthy")
    if hashlib.sha256(strategy_path.read_bytes()).hexdigest() != strategy_sha256:
        raise RuntimeError(
            "Phase 1 strategy result changed during OCR; results are not trustworthy"
        )
    for image, expected_hash in image_hashes.items():
        if _sha256(image) != expected_hash:
            raise RuntimeError(f"Phase 1 evidence image changed during OCR: {image.name}")

    document = {
        "kind": "frame_ocr",
        "schema_version": OCR_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "run_directory": str(run_root),
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha256,
            "video_sha256": video_sha256,
            "strategy": strategy,
            "strategy_path": f"strategies/{strategy}.json",
            "strategy_sha256": strategy_sha256,
        },
        "parameters": {
            "policy_version": OCR_POLICY_VERSION,
            "routes": list(requested_routes),
            "languages": list(requested_languages),
            "engine": engine,
        },
        "engine": ocr_backend.describe(),
        "frames": frames,
        "failures": failures,
        "skipped": skipped,
        "summary": {
            "selected_input_frames": len(selected),
            "ocr_success_frames": len(frames),
            "failure_frames": len(failures),
            "skipped_frames": len(skipped),
            "observation_count": sum(len(frame["observations"]) for frame in frames),
        },
    }
    output_root.mkdir(parents=True, exist_ok=False)
    ocr_path = output_root / "ocr.json"
    write_json(ocr_path, document)
    review_path = output_root / "review.html"
    write_ocr_review_html(review_path, document)
    return {
        **document,
        "output": str(output_root),
        "ocr_json": str(ocr_path),
        "review_html": str(review_path),
    }
