from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import Candidate, ContentSegment, StrategyResult, VideoMetadata
from .timecode import format_timecode


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write_contact_sheet(
    output_path: Path,
    root: Path,
    candidates: list[Candidate],
    *,
    columns: int = 4,
    tile_width: int = 360,
    image_height: int = 203,
) -> None:
    if not candidates:
        canvas = np.full((120, 720, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, "No frames selected", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        return
    label_height = 42
    rows = (len(candidates) + columns - 1) // columns
    canvas = np.full((rows * (image_height + label_height), columns * tile_width, 3), 248, dtype=np.uint8)
    for index, candidate in enumerate(candidates):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * (image_height + label_height)
        if candidate.image_path:
            frame = cv2.imread(str(root / candidate.image_path))
        else:
            frame = None
        if frame is None:
            tile = np.full((image_height, tile_width, 3), 180, dtype=np.uint8)
        else:
            tile = _letterbox(frame, tile_width, image_height)
        canvas[y : y + image_height, x : x + tile_width] = tile
        label = f"{index + 1:03d}  {format_timecode(candidate.timestamp)}"
        cv2.putText(
            canvas,
            label,
            (x + 8, y + image_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Could not write contact sheet: {output_path}")


def _metric_summary(metrics: dict[str, Any]) -> str:
    annotations = metrics.get("annotations")
    if not annotations:
        return '<span class="unscored">No human annotations supplied</span>'
    must_keep = annotations.get("must_keep", {})
    recall = must_keep.get("recall")
    recall_text = "n/a" if recall is None else f"{recall:.1%}"
    return (
        f"must_keep recall <strong>{html.escape(recall_text)}</strong>; "
        f"matched {must_keep.get('matched', 0)}/{must_keep.get('total', 0)}"
    )


def _content_segments(strategies: dict[str, StrategyResult]) -> list[ContentSegment]:
    """Return one chronological copy of every routed segment shown in review."""
    segments: list[ContentSegment] = []
    seen: set[tuple[str, str, int, int, str | None]] = set()
    for result in strategies.values():
        for segment in result.segments:
            identity = (
                segment.segment_id,
                segment.kind,
                segment.start_sample_index,
                segment.stop_sample_index,
                segment.selector,
            )
            if identity in seen:
                continue
            seen.add(identity)
            segments.append(segment)
    return sorted(
        segments,
        key=lambda segment: (
            segment.start_timestamp,
            segment.candidate_end_timestamp,
            segment.segment_id,
        ),
    )


def _segment_badge(
    candidate: Candidate,
    segments_by_id: dict[str, ContentSegment],
) -> str:
    if candidate.segment_id is None:
        return ""
    segment = segments_by_id.get(candidate.segment_id)
    kind = segment.kind if segment is not None else "unknown"
    return (
        f"<span class='segment-badge route-kind-{kind}'>"
        f"{html.escape(kind)} · {html.escape(candidate.segment_id)}</span>"
    )


def _routing_overview(
    segments: list[ContentSegment],
    strategies: dict[str, StrategyResult],
    *,
    start_seconds: float,
    end_seconds: float,
) -> str:
    span = max(1e-9, end_seconds - start_seconds)
    timeline_blocks: list[str] = []
    table_rows: list[str] = []
    for position, segment in enumerate(segments, start=1):
        segment_end_timestamp = (
            segments[position].start_timestamp
            if position < len(segments)
            else end_seconds
        )
        visible_start = min(end_seconds, max(start_seconds, segment.start_timestamp))
        visible_end = min(
            end_seconds,
            max(visible_start, segment_end_timestamp),
        )
        left = (visible_start - start_seconds) / span * 100.0
        width = max(0.4, (visible_end - visible_start) / span * 100.0)
        width = min(width, max(0.0, 100.0 - left))
        label = (
            f"{segment.kind} · {segment.segment_id} · "
            f"{format_timecode(segment.start_timestamp)}–"
            f"{format_timecode(segment_end_timestamp)}"
        )
        timeline_blocks.append(
            f"<button class='route-block route-kind-{segment.kind}' "
            f"data-time='{segment.start_timestamp:.6f}' "
            f"style='left:{left:.4f}%;width:{width:.4f}%' "
            f"title='{html.escape(label, quote=True)}' "
            f"aria-label='{html.escape(label, quote=True)}'></button>"
        )
        selector = segment.selector or "none"
        table_rows.append(
            f"<tr id='route-segment-{position:04d}'>"
            f"<td>{format_timecode(segment.start_timestamp)}</td>"
            f"<td>{format_timecode(segment_end_timestamp)}</td>"
            f"<td><span class='segment-badge route-kind-{segment.kind}'>"
            f"{html.escape(segment.kind)}</span></td>"
            f"<td>{segment.confidence:.1%}</td>"
            f"<td>{html.escape(selector)}</td>"
            f"<td><code>{html.escape(segment.segment_id)}</code></td>"
            "</tr>"
        )

    candidate_segment_ids = {
        candidate.segment_id
        for result in strategies.values()
        for candidate in result.selected
        if candidate.segment_id is not None
    }
    empty_known = [
        segment
        for segment in segments
        if segment.kind != "unknown" and segment.segment_id not in candidate_segment_ids
    ]
    warning = ""
    if empty_known:
        items = ", ".join(
            f"{html.escape(segment.segment_id)} ({html.escape(segment.kind)})"
            for segment in empty_known
        )
        warning = (
            "<div class='route-warning' role='alert'><strong>Known routed segments "
            f"with no selected candidates:</strong> {items}</div>"
        )

    return (
        "<section id='content-routing' class='routing-overview'>"
        "<h2>Content routing</h2>"
        "<p>Pure-visual routed spans. Grey means unknown and requires review. "
        "Confidence is a routing heuristic, not a calibrated accuracy estimate. "
        "<a href='routing-signals.json'>Open routing signals JSON</a></p>"
        f"<div class='route-timeline' aria-label='Content segment timeline'>{''.join(timeline_blocks)}</div>"
        f"{warning}"
        "<div class='route-table-wrap'><table class='route-table'>"
        "<thead><tr><th>Start</th><th>End</th><th>Kind</th><th>Heuristic confidence</th>"
        "<th>Selector</th><th>Segment ID</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table></div>"
        "</section>"
    )


def write_review_html(
    output_path: Path,
    metadata: VideoMetadata,
    strategies: dict[str, StrategyResult],
    *,
    start_seconds: float,
    end_seconds: float,
) -> None:
    segments = _content_segments(strategies)
    segments_by_id = {segment.segment_id: segment for segment in segments}
    routing_overview = (
        _routing_overview(
            segments,
            strategies,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        if segments
        else ""
    )
    routing_styles = """
.routing-overview { background: #fff; border: 1px solid #d8dee6; border-radius: 10px; padding: 16px; }
.route-timeline { position: relative; height: 38px; overflow: hidden; margin: 12px 0 16px; background: #e5e7eb; border: 1px solid #cbd3dd; border-radius: 7px; }
.route-block { position: absolute; top: 0; bottom: 0; min-width: 2px; padding: 0; border: 0; border-radius: 0; opacity: .92; }
.route-block:hover, .route-block:focus { opacity: 1; outline: 3px solid rgba(13,91,215,.35); outline-offset: -3px; }
.route-table-wrap { overflow-x: auto; }
.route-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.route-table th, .route-table td { padding: 8px 10px; border-bottom: 1px solid #e1e6ec; text-align: left; white-space: nowrap; }
.route-table th { color: #526171; font-size: 12px; text-transform: uppercase; letter-spacing: .035em; }
.segment-badge { display: inline-block; margin: 7px 0 0 8px; padding: 3px 7px; border-radius: 999px; color: #fff; font-size: 11px; font-weight: 700; vertical-align: middle; }
.route-table .segment-badge { margin: 0; }
.route-kind-presentation { background: #7c3aed; }
.route-kind-table { background: #198754; }
.route-kind-chart { background: #087ea4; }
.route-kind-unknown { background: #6b7280; }
.route-warning { margin: 12px 0; padding: 10px 12px; color: #7a2e0b; background: #fff4df; border: 1px solid #e7bd72; border-radius: 7px; }
""" if segments else ""
    sections: list[str] = []
    for name, result in strategies.items():
        cards: list[str] = []
        for candidate in result.selected:
            image_source = html.escape(candidate.image_path or "")
            reasons = ", ".join(sorted(set(candidate.reasons)))
            segment_badge = _segment_badge(candidate, segments_by_id)
            cards.append(
                "<article class='card'>"
                f"<a href='{image_source}'><img loading='lazy' src='{image_source}' alt='Frame at {format_timecode(candidate.timestamp)}'></a>"
                "<div class='card-body'>"
                f"<button data-time='{candidate.timestamp:.6f}'>{format_timecode(candidate.timestamp)}</button>"
                f"{segment_badge}"
                f"<div class='reason'>{html.escape(reasons)}</div>"
                f"<div class='score'>score {candidate.score:.4f}</div>"
                "</div></article>"
            )
        dropped_cards: list[str] = []
        for dropped in result.dropped_duplicates:
            candidate = dropped.candidate
            image_source = html.escape(candidate.image_path or "")
            segment_badge = _segment_badge(candidate, segments_by_id)
            dropped_cards.append(
                "<article class='card dropped'>"
                f"<a href='{image_source}'><img loading='lazy' src='{image_source}' alt='Dropped frame at {format_timecode(candidate.timestamp)}'></a>"
                "<div class='card-body'>"
                f"<button data-time='{candidate.timestamp:.6f}'>{format_timecode(candidate.timestamp)}</button>"
                f"{segment_badge}"
                f"<div class='reason'>duplicate of {format_timecode(dropped.duplicate_of_timestamp)} · "
                f"pHash {dropped.phash_distance} · SSIM {dropped.ssim:.5f} · local {dropped.block_change:.5f}</div>"
                "</div></article>"
            )
        capped_cards: list[str] = []
        for candidate in result.capped_candidates:
            image_source = html.escape(candidate.image_path or "")
            segment_badge = _segment_badge(candidate, segments_by_id)
            capped_cards.append(
                "<article class='card capped'>"
                f"<a href='{image_source}'><img loading='lazy' src='{image_source}' alt='Capped frame at {format_timecode(candidate.timestamp)}'></a>"
                "<div class='card-body'>"
                f"<button data-time='{candidate.timestamp:.6f}'>{format_timecode(candidate.timestamp)}</button>"
                f"{segment_badge}"
                f"<div class='reason'>frame cap · {html.escape(', '.join(candidate.reasons))}</div>"
                "</div></article>"
            )
        sections.append(
            f"<section id='{html.escape(name)}'>"
            f"<h2>{html.escape(name)}</h2>"
            f"<p>{len(result.selected)} selected / {len(result.raw_candidates)} raw; "
            f"{len(result.dropped_duplicates)} duplicates dropped; {len(result.capped_candidates)} capped. "
            f"{_metric_summary(result.metrics)}</p>"
            f"<p><a href='contact-sheets/{html.escape(name)}.jpg'>Open contact sheet</a> · "
            f"<a href='strategies/{html.escape(name)}.json'>Open JSON</a></p>"
            f"<div class='grid'>{''.join(cards) or '<p>No frames selected.</p>'}</div>"
            f"<details><summary>Inspect {len(dropped_cards)} duplicate drops</summary><div class='grid'>{''.join(dropped_cards) or '<p>None.</p>'}</div></details>"
            f"<details><summary>Inspect {len(capped_cards)} cap drops</summary><div class='grid'>{''.join(capped_cards) or '<p>None.</p>'}</div></details>"
            "</section>"
        )

    source_uri = html.escape(metadata.path.as_uri())
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FrameLedger review</title>
<style>
:root {{ color-scheme: light; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
body {{ margin: 0; background: #f2f4f7; color: #17212b; }}
header {{ position: sticky; top: 0; z-index: 5; padding: 16px 22px; background: rgba(255,255,255,.96); border-bottom: 1px solid #d8dee6; }}
header h1 {{ margin: 0 0 8px; font-size: 22px; }}
header p {{ margin: 4px 0; color: #526171; }}
video {{ width: min(760px, 100%); max-height: 42vh; background: #000; }}
nav {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 10px; }}
nav a {{ color: #0d5bd7; text-decoration: none; font-weight: 600; }}
main {{ max-width: 1480px; margin: auto; padding: 8px 22px 60px; }}
section {{ scroll-margin-top: 260px; margin-top: 28px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
.card {{ overflow: hidden; background: #fff; border: 1px solid #d8dee6; border-radius: 10px; box-shadow: 0 3px 12px rgba(28,39,51,.06); }}
.card img {{ display: block; width: 100%; aspect-ratio: 16/9; object-fit: contain; background: #111; }}
.card-body {{ padding: 10px 12px 12px; }}
button {{ border: 0; border-radius: 6px; padding: 7px 10px; color: #fff; background: #0d5bd7; cursor: pointer; font-variant-numeric: tabular-nums; }}
.reason {{ margin-top: 8px; font-size: 13px; }}
.score, .unscored {{ color: #687888; font-size: 12px; margin-top: 5px; }}
details {{ margin-top: 16px; }} summary {{ cursor: pointer; font-weight: 650; margin-bottom: 10px; }}
.dropped {{ border-color: #dc9c9c; }} .capped {{ border-color: #d4b56c; }}
{routing_styles}</style>
</head>
<body>
<header>
  <h1>FrameLedger candidate-frame review</h1>
  <p>{html.escape(metadata.path.name)} · {format_timecode(start_seconds)}–{format_timecode(end_seconds)}</p>
  <video id="source-video" controls preload="metadata" src="{source_uri}"></video>
  <nav>{''.join(f"<a href='#{html.escape(name)}'>{html.escape(name)}</a>" for name in strategies)}</nav>
</header>
<main>{routing_overview}{''.join(sections)}</main>
<script>
const video = document.getElementById('source-video');
document.querySelectorAll('button[data-time]').forEach((button) => {{
  button.addEventListener('click', () => {{ video.currentTime = Number(button.dataset.time); video.play(); }});
}});
</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def write_scan_html(
    output_path: Path,
    metadata: VideoMetadata,
    frames: list[tuple[float, str]],
    *,
    start_seconds: float,
    end_seconds: float,
) -> None:
    cards = []
    for timestamp, image_path in frames:
        cards.append(
            "<article>"
            f"<img loading='lazy' src='{html.escape(image_path)}' alt='{format_timecode(timestamp)}'>"
            f"<button data-time='{timestamp:.6f}'>{format_timecode(timestamp)}</button>"
            "</article>"
        )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FrameLedger coarse scan</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:20px;background:#f4f6f8;color:#17212b}}
header{{position:sticky;top:0;background:rgba(244,246,248,.96);padding-bottom:12px;z-index:2}}video{{width:min(760px,100%);max-height:42vh;background:#000}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}article{{background:#fff;border:1px solid #d8dee6;border-radius:9px;overflow:hidden;padding-bottom:10px}}
img{{width:100%;aspect-ratio:16/9;object-fit:contain;background:#111;display:block}}button{{margin:9px 10px 0;padding:7px 10px;border:0;border-radius:5px;background:#0d5bd7;color:#fff}}
</style></head><body><header><h1>Coarse visual scan</h1><p>{html.escape(metadata.path.name)} · {format_timecode(start_seconds)}–{format_timecode(end_seconds)}</p>
<video id="source-video" controls preload="metadata" src="{html.escape(metadata.path.as_uri())}"></video></header><main>{''.join(cards)}</main>
<script>const v=document.getElementById('source-video');document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{{v.currentTime=Number(b.dataset.time);v.play();}});</script></body></html>"""
    output_path.write_text(document, encoding="utf-8")
