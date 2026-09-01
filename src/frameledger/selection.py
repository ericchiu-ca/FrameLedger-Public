from __future__ import annotations

import bisect
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .features import (
    PresentationChangeGeometry,
    TableChangeGeometry,
    block_change,
    comparison,
    global_ssim,
    perceptual_hash,
    phash_distance,
    pixel_delta,
    presentation_change_geometry,
    worksheet_change_geometry,
)
from .models import Candidate, DroppedCandidate, Sample, StrategyResult


VALID_STRATEGIES = {
    "uniform",
    "content",
    "adaptive",
    "settled",
    "hybrid",
    "boundary_terminal",
    "table_viewport",
    "table_focus",
    "presentation_states",
    "routed",
}


def build_samples(frame_stream: Iterable[tuple[float, int, object]]) -> list[Sample]:
    from .features import edge_delta, perceptual_hash, pixel_delta, sharpness

    samples: list[Sample] = []
    previous: Sample | None = None
    for timestamp, frame_index, gray in frame_stream:
        current_hash = perceptual_hash(gray)
        current = Sample(
            timestamp=timestamp,
            frame_index=frame_index,
            gray=gray,
            phash=current_hash,
            sharpness=sharpness(gray),
        )
        if previous is not None:
            current.pixel_delta = pixel_delta(previous.gray, gray)
            current.edge_delta = edge_delta(previous.gray, gray)
            current.phash_delta = phash_distance(previous.phash, current_hash)
            current.ssim_previous = global_ssim(previous.gray, gray)
            current.change_score = (
                0.40 * current.pixel_delta
                + 0.20 * current.edge_delta
                + 0.20 * (current.phash_delta / 64.0)
                + 0.20 * max(0.0, 1.0 - current.ssim_previous)
            )
        samples.append(current)
        previous = current
    if not samples:
        raise ValueError("The selected range did not yield any analysis frames")
    return samples


def _candidate(samples: list[Sample], index: int, reason: str, score: float) -> Candidate:
    sample = samples[index]
    return Candidate(
        sample_index=index,
        timestamp=sample.timestamp,
        score=float(score),
        reasons=[reason],
    )


def merge_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    by_index: dict[int, Candidate] = {}
    for item in candidates:
        existing = by_index.get(item.sample_index)
        if existing is None:
            by_index[item.sample_index] = Candidate(
                sample_index=item.sample_index,
                timestamp=item.timestamp,
                score=item.score,
                reasons=list(item.reasons),
                segment_id=item.segment_id,
            )
            continue
        if (
            existing.segment_id is not None
            and item.segment_id is not None
            and existing.segment_id != item.segment_id
        ):
            raise ValueError("Candidates at one sample cannot belong to different route segments")
        if existing.segment_id is None:
            existing.segment_id = item.segment_id
        existing.score = max(existing.score, item.score)
        existing.reasons = sorted(set(existing.reasons + item.reasons))
    return sorted(by_index.values(), key=lambda item: item.timestamp)


def nearest_sample_index(samples: list[Sample], timestamp: float) -> int:
    times = [sample.timestamp for sample in samples]
    position = bisect.bisect_left(times, timestamp)
    if position <= 0:
        return 0
    if position >= len(times):
        return len(times) - 1
    before = position - 1
    return before if abs(times[before] - timestamp) <= abs(times[position] - timestamp) else position


def sample_index_at_or_before(samples: list[Sample], timestamp: float) -> int:
    """Return the last sampled frame that does not exceed the timestamp."""
    times = [sample.timestamp for sample in samples]
    position = bisect.bisect_right(times, timestamp) - 1
    if position < 0:
        raise ValueError("Timestamp precedes the sampled range")
    return min(position, len(samples) - 1)


def select_uniform(samples: list[Sample], interval_seconds: float = 30.0) -> list[Candidate]:
    if interval_seconds <= 0:
        raise ValueError("Uniform interval must be positive")
    start = samples[0].timestamp
    end = samples[-1].timestamp
    timestamps: list[float] = []
    current = start
    while current <= end + 1e-9:
        timestamps.append(current)
        current += interval_seconds
    if not timestamps or timestamps[-1] < end - interval_seconds * 0.25:
        timestamps.append(end)
    return merge_candidates(
        _candidate(samples, nearest_sample_index(samples, timestamp), "density_floor", 0.10)
        for timestamp in timestamps
    )


def scene_times_to_candidates(
    samples: list[Sample],
    scene_starts: Iterable[float],
    *,
    reason: str,
    stable_delay_seconds: float = 0.6,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    first_time = samples[0].timestamp
    for position, timestamp in enumerate(scene_starts):
        target = timestamp if position == 0 or timestamp <= first_time + 1e-6 else timestamp + stable_delay_seconds
        index = nearest_sample_index(samples, target)
        score = max(0.30, samples[index].change_score)
        candidates.append(_candidate(samples, index, reason, score))
    return merge_candidates(candidates)


def _settled_index_after(
    samples: list[Sample],
    start_index: int,
    *,
    stable_samples: int,
    stability_threshold: float,
) -> int:
    stable_run = 0
    for index in range(start_index + 1, len(samples)):
        if samples[index].change_score <= stability_threshold:
            stable_run += 1
            if stable_run >= stable_samples:
                return index
        else:
            stable_run = 0
    return min(start_index, len(samples) - 1)


def select_adaptive(
    samples: list[Sample],
    *,
    analysis_fps: float,
    window_seconds: float = 3.0,
    ratio_threshold: float = 3.0,
    minimum_change: float = 0.018,
    minimum_gap_seconds: float = 2.0,
    stable_seconds: float = 1.0,
    stability_threshold: float = 0.008,
) -> list[Candidate]:
    radius = max(1, int(round(window_seconds * analysis_fps)))
    stable_samples = max(1, int(round(stable_seconds * analysis_fps)))
    candidates: list[Candidate] = []
    last_timestamp = float("-inf")
    for index in range(1, len(samples)):
        left = max(1, index - radius)
        right = min(len(samples), index + radius + 1)
        neighbours = [
            samples[position].change_score
            for position in range(left, right)
            if position != index
        ]
        baseline = statistics.median(neighbours) if neighbours else 0.0
        ratio = samples[index].change_score / max(0.002, baseline)
        samples[index].adaptive_ratio = ratio
        is_local_peak = samples[index].change_score >= max(
            samples[max(1, index - 1)].change_score,
            samples[min(len(samples) - 1, index + 1)].change_score,
        )
        if (
            samples[index].change_score >= minimum_change
            and ratio >= ratio_threshold
            and is_local_peak
            and samples[index].timestamp - last_timestamp >= minimum_gap_seconds
        ):
            settled = _settled_index_after(
                samples,
                index,
                stable_samples=stable_samples,
                stability_threshold=stability_threshold,
            )
            candidates.append(_candidate(samples, settled, "adaptive_change_settled", ratio))
            last_timestamp = samples[settled].timestamp
    return merge_candidates(candidates)


def select_settled_end_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    activity_threshold: float = 0.014,
    cumulative_threshold: float = 0.024,
    stability_threshold: float = 0.006,
    stable_seconds: float = 2.0,
    minimum_gap_seconds: float = 3.0,
) -> list[Candidate]:
    """Select a stable tail after either abrupt activity or accumulated drift."""
    stable_samples = max(1, int(round(stable_seconds * analysis_fps)))
    baseline_index = 0
    active = False
    stable_run = 0
    peak_score = 0.0
    last_selected_time = float("-inf")
    candidates: list[Candidate] = []

    for index in range(1, len(samples)):
        current = samples[index]
        baseline = samples[baseline_index]
        cumulative = comparison(baseline.gray, current.gray, baseline.phash, current.phash)
        cumulative_score = float(cumulative["score"])
        triggered = current.change_score >= activity_threshold or cumulative_score >= cumulative_threshold
        if triggered:
            active = True
            peak_score = max(peak_score, current.change_score, cumulative_score)

        if not active:
            continue
        if current.change_score <= stability_threshold:
            stable_run += 1
        else:
            stable_run = 0

        if stable_run < stable_samples:
            continue

        stable_start = max(baseline_index, index - stable_samples + 1)
        stable_span = comparison(
            samples[stable_start].gray,
            current.gray,
            samples[stable_start].phash,
            current.phash,
        )
        if float(stable_span["score"]) > max(stability_threshold * 1.5, 0.01):
            continue
        if current.timestamp - last_selected_time >= minimum_gap_seconds:
            candidates.append(_candidate(samples, index, "settled_after_change", max(0.4, peak_score)))
            last_selected_time = current.timestamp
        baseline_index = index
        active = False
        stable_run = 0
        peak_score = 0.0

    if active:
        final_index = len(samples) - 1
        final = samples[final_index]
        baseline = samples[baseline_index]
        cumulative = comparison(baseline.gray, final.gray, baseline.phash, final.phash)
        if float(cumulative["score"]) >= cumulative_threshold:
            candidates.append(_candidate(samples, final_index, "segment_end_after_change", max(0.3, peak_score)))
    return merge_candidates(candidates)


def _edge_density(gray: np.ndarray) -> float:
    return float(np.mean(cv2.Canny(gray, 80, 160) > 0))


def _terminal_before_boundary(
    samples: list[Sample],
    *,
    round_start_index: int,
    boundary_index: int,
    analysis_fps: float,
    search_seconds: float,
    stable_seconds: float,
    stability_threshold: float,
    maximum_terminal_index: int | None = None,
) -> int:
    """Return the latest stable real frame before a confirmed boundary.

    Terminal completeness cannot be inferred from a universal percentage of
    pixels: one last arrow may be tiny while still being semantically important.
    The boundary closes the round, so search backwards from it and retain the
    closest quiet pre-transition state.  If drawing continues right up to the
    clear, preserve the last pre-boundary frame instead of dropping the round.
    """
    terminal_index = boundary_index - 1
    if maximum_terminal_index is not None:
        terminal_index = min(terminal_index, maximum_terminal_index)
    terminal_index = max(round_start_index, terminal_index)
    search_samples = max(1, int(round(search_seconds * analysis_fps)))
    stable_samples = max(1, int(round(stable_seconds * analysis_fps)))
    search_start = max(round_start_index, terminal_index - search_samples + 1)
    for run_end in range(terminal_index, search_start - 1, -1):
        run_start = run_end - stable_samples + 1
        if run_start < search_start:
            continue
        if all(
            samples[index].change_score <= stability_threshold
            for index in range(run_start, run_end + 1)
        ):
            reference = samples[run_end]
            plateau_continuity = all(
                phash_distance(reference.phash, samples[position].phash) <= 4
                and global_ssim(reference.gray, samples[position].gray) >= 0.996
                and block_change(reference.gray, samples[position].gray) <= 0.012
                for position in range(run_end, terminal_index + 1)
            )
            if plateau_continuity:
                return run_end
    return terminal_index


def select_boundary_terminal_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    transition_threshold: float = 0.025,
    pre_boundary_seconds: float = 0.5,
    post_boundary_seconds: float = 1.5,
    pixel_delta_threshold: float = 0.004,
    edge_loss_threshold: float = 0.003,
    mask_delta: int = 12,
    minimum_mask_ratio: float = 0.001,
    merge_seconds: float = 3.0,
    reversion_lookback_seconds: float = 2.0,
    terminal_search_seconds: float = 2.0,
    terminal_stable_seconds: float = 1.0,
    terminal_stability_threshold: float = 0.008,
    candidate_end_timestamp: float | None = None,
) -> list[Candidate]:
    """Keep one final canvas state before each persistent clear-like boundary.

    A short quiet period never closes a drawing round.  A boundary must combine a
    strong instantaneous change with a persistent loss of visual edges across a
    multi-second comparison.  The terminal representative is the closest stable
    real frame before the transition, with a last-pre-boundary fallback.
    """
    if analysis_fps <= 0:
        raise ValueError("Analysis FPS must be positive")
    if not 0 < transition_threshold <= 1:
        raise ValueError("Boundary transition threshold must be in (0, 1]")
    if pre_boundary_seconds <= 0 or post_boundary_seconds <= 0:
        raise ValueError("Pre- and post-boundary durations must be positive")
    if reversion_lookback_seconds <= 0:
        raise ValueError("Reversion lookback duration must be positive")
    if terminal_search_seconds <= 0 or terminal_stable_seconds <= 0:
        raise ValueError("Terminal search and stable durations must be positive")
    if terminal_stability_threshold < 0:
        raise ValueError("Terminal stability threshold must be non-negative")
    if candidate_end_timestamp is not None and candidate_end_timestamp < samples[0].timestamp:
        raise ValueError("Candidate end timestamp must not precede the sampled range")

    pre_samples = max(1, int(round(pre_boundary_seconds * analysis_fps)))
    post_samples = max(1, int(round(post_boundary_seconds * analysis_fps)))
    reversion_samples = max(1, int(round(reversion_lookback_seconds * analysis_fps)))
    edge_densities = [_edge_density(sample.gray) for sample in samples]
    boundary_events: list[tuple[int, float, float, float]] = []
    for index in range(pre_samples, len(samples) - post_samples):
        if samples[index].change_score < transition_threshold:
            continue
        before_index = index - pre_samples
        post_indexes = list(range(index + 1, index + post_samples + 1))
        persistent_deltas = [
            pixel_delta(samples[before_index].gray, samples[position].gray)
            for position in post_indexes
        ]
        persistent_edge_losses = [
            edge_densities[before_index] - edge_densities[position]
            for position in post_indexes
        ]
        persistent_delta = float(np.median(persistent_deltas))
        persistent_edge_loss = float(np.median(persistent_edge_losses))
        if persistent_delta < pixel_delta_threshold:
            continue
        if persistent_edge_losses[0] < edge_loss_threshold:
            continue
        if float(np.mean(np.asarray(persistent_edge_losses) >= edge_loss_threshold)) < (2.0 / 3.0):
            continue
        mask_ratios = [
            float(
                np.mean(
                    cv2.absdiff(samples[before_index].gray, samples[position].gray)
                    >= mask_delta
                )
            )
            for position in post_indexes
        ]
        mask_ratio = float(np.median(mask_ratios))
        if mask_ratio < minimum_mask_ratio:
            continue
        history_index = max(0, index - reversion_samples)
        final_post_index = post_indexes[-1]
        history_to_before = pixel_delta(
            samples[history_index].gray, samples[before_index].gray
        )
        history_to_post = pixel_delta(
            samples[history_index].gray, samples[final_post_index].gray
        )
        if (
            history_to_before >= pixel_delta_threshold
            and history_to_post < pixel_delta_threshold * 0.5
            and global_ssim(samples[history_index].gray, samples[final_post_index].gray)
            >= 0.997
        ):
            continue
        boundary_events.append(
            (index, persistent_delta, persistent_edge_loss, mask_ratio)
        )

    clusters: list[list[tuple[int, float, float, float]]] = []
    for event in boundary_events:
        if (
            clusters
            and samples[event[0]].timestamp - samples[clusters[-1][-1][0]].timestamp
            <= merge_seconds
        ):
            clusters[-1].append(event)
        else:
            clusters.append([event])

    candidates: list[Candidate] = []
    round_start_index = 0
    terminal_limit_index = (
        sample_index_at_or_before(samples, candidate_end_timestamp)
        if candidate_end_timestamp is not None
        else None
    )
    for cluster in clusters:
        boundary_index, persistent_delta, edge_loss, mask_ratio = cluster[0]
        if terminal_limit_index is not None and round_start_index > terminal_limit_index:
            break
        actual_terminal_index = _terminal_before_boundary(
            samples,
            round_start_index=round_start_index,
            boundary_index=boundary_index,
            analysis_fps=analysis_fps,
            search_seconds=terminal_search_seconds,
            stable_seconds=terminal_stable_seconds,
            stability_threshold=terminal_stability_threshold,
        )
        terminal_index = actual_terminal_index
        lookahead_confirmed = False
        if (
            terminal_limit_index is not None
            and actual_terminal_index > terminal_limit_index
        ):
            projected_terminal_index = _terminal_before_boundary(
                samples,
                round_start_index=round_start_index,
                boundary_index=boundary_index,
                analysis_fps=analysis_fps,
                search_seconds=terminal_search_seconds,
                stable_seconds=terminal_stable_seconds,
                stability_threshold=terminal_stability_threshold,
                maximum_terminal_index=terminal_limit_index,
            )
            projected = samples[projected_terminal_index]
            plateau_continuity = all(
                phash_distance(projected.phash, samples[position].phash) <= 4
                and global_ssim(projected.gray, samples[position].gray) >= 0.996
                and block_change(projected.gray, samples[position].gray) <= 0.012
                for position in range(projected_terminal_index, actual_terminal_index + 1)
            )
            if not plateau_continuity:
                break
            terminal_index = projected_terminal_index
            lookahead_confirmed = True
        boundary_time = samples[boundary_index].timestamp
        item = _candidate(
            samples,
            terminal_index,
            f"boundary_terminal:boundary={boundary_time:.3f}",
            max(persistent_delta, edge_loss, mask_ratio),
        )
        if lookahead_confirmed:
            item.reasons.extend(
                [
                    "lookahead_confirmed_terminal",
                    f"actual_terminal={samples[actual_terminal_index].timestamp:.3f}",
                ]
            )
        candidates.append(item)
        cluster_end_index = cluster[-1][0]
        round_start_index = min(len(samples) - 1, cluster_end_index + post_samples)
        if terminal_limit_index is not None and boundary_index > terminal_limit_index:
            break
    return merge_candidates(candidates)


def _table_transition_kind(
    geometry: TableChangeGeometry,
    *,
    include_row_focus: bool,
    viewport_changed_ratio: float,
    viewport_row_coverage: float,
    viewport_column_coverage: float,
    focus_changed_ratio: float,
    focus_row_coverage: float,
    focus_column_coverage: float,
) -> str | None:
    """Classify geometry without assigning spreadsheet-specific semantics."""
    if (
        geometry.residual_changed_ratio >= viewport_changed_ratio
        and geometry.residual_row_coverage >= viewport_row_coverage
        and geometry.residual_column_coverage >= viewport_column_coverage
    ):
        return "viewport"
    if (
        include_row_focus
        and geometry.highlight_changed_ratio >= focus_changed_ratio
        and geometry.highlight_row_coverage >= focus_row_coverage
        and geometry.highlight_column_coverage >= focus_column_coverage
    ):
        return "focus"
    return None


def _append_distinct_table_candidate(
    candidates: list[Candidate],
    candidate: Candidate,
    samples: list[Sample],
    *,
    include_row_focus: bool,
    worksheet_top_ratio: float,
    worksheet_bottom_ratio: float,
    mask_delta: int,
    viewport_changed_ratio: float,
    viewport_row_coverage: float,
    viewport_column_coverage: float,
    focus_changed_ratio: float,
    focus_row_coverage: float,
    focus_column_coverage: float,
) -> None:
    """Append unless an earlier semantically equivalent table state exists.

    Viewport-only selection deliberately ignores row-highlight differences.
    Focus selection uses the same geometry classifier to preserve genuinely
    different horizontal selections.  Comparing against every earlier state
    folds a return to an already represented viewport while keeping its first
    occurrence.
    """
    current = samples[candidate.sample_index]
    for previous in candidates:
        earlier = samples[previous.sample_index]
        geometry = worksheet_change_geometry(
            earlier.gray,
            current.gray,
            top_ratio=worksheet_top_ratio,
            bottom_ratio=worksheet_bottom_ratio,
            pixel_threshold=mask_delta,
        )
        if (
            _table_transition_kind(
                geometry,
                include_row_focus=include_row_focus,
                viewport_changed_ratio=viewport_changed_ratio,
                viewport_row_coverage=viewport_row_coverage,
                viewport_column_coverage=viewport_column_coverage,
                focus_changed_ratio=focus_changed_ratio,
                focus_row_coverage=focus_row_coverage,
                focus_column_coverage=focus_column_coverage,
            )
            is None
        ):
            return
    candidates.append(candidate)


def _validate_table_selector_options(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float | None,
    worksheet_top_ratio: float,
    worksheet_bottom_ratio: float,
    mask_delta: int,
    merge_seconds: float,
    stable_seconds: float,
    thresholds: Iterable[tuple[str, float]],
) -> None:
    if not samples:
        raise ValueError("Table selection requires at least one sample")
    if analysis_fps <= 0:
        raise ValueError("Analysis FPS must be positive")
    if candidate_end_timestamp is not None and candidate_end_timestamp < samples[0].timestamp:
        raise ValueError("Candidate end timestamp must not precede the sampled range")
    if not 0 <= worksheet_top_ratio < worksheet_bottom_ratio <= 1:
        raise ValueError("Worksheet ROI ratios must satisfy 0 <= top < bottom <= 1")
    if not 0 <= mask_delta <= 255:
        raise ValueError("Mask delta must be between 0 and 255")
    if merge_seconds < 0:
        raise ValueError("Table transition merge duration must be non-negative")
    if stable_seconds <= 0:
        raise ValueError("Table stable duration must be positive")
    for name, value in thresholds:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")


def _select_table_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    include_row_focus: bool,
    candidate_end_timestamp: float | None,
    worksheet_top_ratio: float,
    worksheet_bottom_ratio: float,
    mask_delta: int,
    merge_seconds: float,
    stable_seconds: float,
    stability_changed_ratio: float,
    viewport_changed_ratio: float,
    viewport_row_coverage: float,
    viewport_column_coverage: float,
    focus_changed_ratio: float,
    focus_row_coverage: float,
    focus_column_coverage: float,
) -> list[Candidate]:
    """Return stable initial and post-transition table states.

    Semantic transitions are clustered until no further viewport/focus event has
    occurred for ``merge_seconds``.  The selected frame is the earliest frame
    that completes the first quiet run after the final event, rather than a
    frame from the scrolling or selection animation.  An unresolved transition
    at the bounded-window tail is deliberately omitted.
    """
    _validate_table_selector_options(
        samples,
        analysis_fps=analysis_fps,
        candidate_end_timestamp=candidate_end_timestamp,
        worksheet_top_ratio=worksheet_top_ratio,
        worksheet_bottom_ratio=worksheet_bottom_ratio,
        mask_delta=mask_delta,
        merge_seconds=merge_seconds,
        stable_seconds=stable_seconds,
        thresholds=(
            ("Stability changed ratio", stability_changed_ratio),
            ("Viewport changed ratio", viewport_changed_ratio),
            ("Viewport row coverage", viewport_row_coverage),
            ("Viewport column coverage", viewport_column_coverage),
            ("Focus changed ratio", focus_changed_ratio),
            ("Focus row coverage", focus_row_coverage),
            ("Focus column coverage", focus_column_coverage),
        ),
    )
    terminal_limit_index = (
        sample_index_at_or_before(samples, candidate_end_timestamp)
        if candidate_end_timestamp is not None
        else len(samples) - 1
    )
    if terminal_limit_index == 0:
        return []

    stable_steps_required = max(1, int(round(stable_seconds * analysis_fps)))
    candidates: list[Candidate] = []
    initial_stable_steps = 0
    initial_resolved = False
    pending_kinds: set[str] = set()
    pending_peak_score = 0.0
    last_event_index: int | None = None
    stable_steps = 0
    stable_candidate_index: int | None = None

    for index in range(1, len(samples)):
        geometry = worksheet_change_geometry(
            samples[index - 1].gray,
            samples[index].gray,
            top_ratio=worksheet_top_ratio,
            bottom_ratio=worksheet_bottom_ratio,
            pixel_threshold=mask_delta,
        )
        event_kind = _table_transition_kind(
            geometry,
            include_row_focus=include_row_focus,
            viewport_changed_ratio=viewport_changed_ratio,
            viewport_row_coverage=viewport_row_coverage,
            viewport_column_coverage=viewport_column_coverage,
            focus_changed_ratio=focus_changed_ratio,
            focus_row_coverage=focus_row_coverage,
            focus_column_coverage=focus_column_coverage,
        )
        quiet = geometry.changed_ratio <= stability_changed_ratio

        if event_kind is not None:
            # Lookahead may confirm a cluster that began inside the eligible
            # range, but it may not begin a new output. A later semantic event
            # before confirmation extends that pending cluster and invalidates
            # its earlier stable frame because the true settled state now lies
            # beyond the candidate boundary.
            if index > terminal_limit_index and not pending_kinds:
                break
            pending_kinds.add(event_kind)
            pending_peak_score = max(
                pending_peak_score,
                geometry.residual_changed_ratio,
                geometry.residual_row_coverage,
                geometry.highlight_changed_ratio,
                geometry.highlight_column_coverage,
            )
            last_event_index = index
            stable_steps = 0
            stable_candidate_index = None
            initial_resolved = True
            continue

        if pending_kinds:
            if quiet:
                stable_steps += 1
                if (
                    stable_candidate_index is None
                    and stable_steps >= stable_steps_required
                    and index <= terminal_limit_index
                ):
                    stable_candidate_index = index
            else:
                stable_steps = 0
                stable_candidate_index = None

            assert last_event_index is not None
            if (
                stable_candidate_index is not None
                and samples[index].timestamp - samples[last_event_index].timestamp
                >= merge_seconds
            ):
                reasons: list[str] = []
                if "viewport" in pending_kinds:
                    reasons.append("table_viewport:post_transition")
                if "focus" in pending_kinds:
                    reasons.append("table_focus:post_row_focus")
                item = _candidate(
                    samples,
                    stable_candidate_index,
                    reasons[0],
                    max(0.20, pending_peak_score),
                )
                item.reasons.extend(reasons[1:])
                _append_distinct_table_candidate(
                    candidates,
                    item,
                    samples,
                    include_row_focus=include_row_focus,
                    worksheet_top_ratio=worksheet_top_ratio,
                    worksheet_bottom_ratio=worksheet_bottom_ratio,
                    mask_delta=mask_delta,
                    viewport_changed_ratio=viewport_changed_ratio,
                    viewport_row_coverage=viewport_row_coverage,
                    viewport_column_coverage=viewport_column_coverage,
                    focus_changed_ratio=focus_changed_ratio,
                    focus_row_coverage=focus_row_coverage,
                    focus_column_coverage=focus_column_coverage,
                )
                pending_kinds.clear()
                pending_peak_score = 0.0
                last_event_index = None
                stable_steps = 0
                stable_candidate_index = None
                if index > terminal_limit_index:
                    break
            continue

        if index > terminal_limit_index:
            break
        if initial_resolved:
            continue
        if quiet:
            initial_stable_steps += 1
        else:
            initial_stable_steps = 0
        if initial_stable_steps >= stable_steps_required:
            reason = (
                "table_focus:initial_stable"
                if include_row_focus
                else "table_viewport:initial_stable"
            )
            _append_distinct_table_candidate(
                candidates,
                _candidate(samples, index, reason, 0.10),
                samples,
                include_row_focus=include_row_focus,
                worksheet_top_ratio=worksheet_top_ratio,
                worksheet_bottom_ratio=worksheet_bottom_ratio,
                mask_delta=mask_delta,
                viewport_changed_ratio=viewport_changed_ratio,
                viewport_row_coverage=viewport_row_coverage,
                viewport_column_coverage=viewport_column_coverage,
                focus_changed_ratio=focus_changed_ratio,
                focus_row_coverage=focus_row_coverage,
                focus_column_coverage=focus_column_coverage,
            )
            initial_resolved = True

    return merge_candidates(candidates)


def select_table_viewport_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float | None = None,
    worksheet_top_ratio: float = 0.25,
    worksheet_bottom_ratio: float = 0.92,
    mask_delta: int = 12,
    merge_seconds: float = 5.0,
    stable_seconds: float = 1.0,
    stability_changed_ratio: float = 0.01,
    viewport_changed_ratio: float = 0.055,
    viewport_row_coverage: float = 0.20,
    viewport_column_coverage: float = 0.50,
) -> list[Candidate]:
    """Keep stable worksheet views after scroll, sort, or sheet transitions."""
    return _select_table_states(
        samples,
        analysis_fps=analysis_fps,
        include_row_focus=False,
        candidate_end_timestamp=candidate_end_timestamp,
        worksheet_top_ratio=worksheet_top_ratio,
        worksheet_bottom_ratio=worksheet_bottom_ratio,
        mask_delta=mask_delta,
        merge_seconds=merge_seconds,
        stable_seconds=stable_seconds,
        stability_changed_ratio=stability_changed_ratio,
        viewport_changed_ratio=viewport_changed_ratio,
        viewport_row_coverage=viewport_row_coverage,
        viewport_column_coverage=viewport_column_coverage,
        focus_changed_ratio=1.0,
        focus_row_coverage=1.0,
        focus_column_coverage=1.0,
    )


def select_table_focus_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float | None = None,
    worksheet_top_ratio: float = 0.25,
    worksheet_bottom_ratio: float = 0.92,
    mask_delta: int = 12,
    merge_seconds: float = 5.0,
    stable_seconds: float = 1.0,
    stability_changed_ratio: float = 0.01,
    viewport_changed_ratio: float = 0.055,
    viewport_row_coverage: float = 0.20,
    viewport_column_coverage: float = 0.50,
    focus_changed_ratio: float = 0.012,
    focus_row_coverage: float = 0.015,
    focus_column_coverage: float = 0.75,
) -> list[Candidate]:
    """Keep table viewport states plus persistent wide row/group focus states."""
    return _select_table_states(
        samples,
        analysis_fps=analysis_fps,
        include_row_focus=True,
        candidate_end_timestamp=candidate_end_timestamp,
        worksheet_top_ratio=worksheet_top_ratio,
        worksheet_bottom_ratio=worksheet_bottom_ratio,
        mask_delta=mask_delta,
        merge_seconds=merge_seconds,
        stable_seconds=stable_seconds,
        stability_changed_ratio=stability_changed_ratio,
        viewport_changed_ratio=viewport_changed_ratio,
        viewport_row_coverage=viewport_row_coverage,
        viewport_column_coverage=viewport_column_coverage,
        focus_changed_ratio=focus_changed_ratio,
        focus_row_coverage=focus_row_coverage,
        focus_column_coverage=focus_column_coverage,
    )


@dataclass
class _PresentationStableState:
    candidate: Candidate
    hard_page_change: bool = False


def _presentation_content_roi(
    gray: np.ndarray,
    *,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> np.ndarray:
    height, width = gray.shape
    x0 = min(width - 1, int(round(width * left_ratio)))
    x1 = max(x0 + 1, min(width, int(round(width * right_ratio))))
    y0 = min(height - 1, int(round(height * top_ratio)))
    y1 = max(y0 + 1, min(height, int(round(height * bottom_ratio))))
    return gray[y0:y1, x0:x1]


def _presentation_is_compact_noise(
    geometry: PresentationChangeGeometry,
    *,
    edge_ratio: float,
) -> bool:
    return (
        geometry.component_count <= 4
        and geometry.largest_component_ratio <= 0.006
        and geometry.largest_component_width_ratio <= 0.08
        and geometry.largest_component_height_ratio <= 0.12
        and geometry.changed_ratio <= 0.012
        and geometry.edge_changed_ratio < edge_ratio
        and geometry.block_spread <= 0.0625
    )


def _presentation_is_meaningful(
    geometry: PresentationChangeGeometry,
    *,
    changed_ratio: float,
    edge_ratio: float,
    block_spread: float,
) -> bool:
    if _presentation_is_compact_noise(geometry, edge_ratio=edge_ratio):
        return False
    return (
        geometry.changed_ratio >= changed_ratio
        or geometry.edge_changed_ratio >= edge_ratio
        or geometry.block_spread >= block_spread
    )


def _presentation_is_hard_page(
    geometry: PresentationChangeGeometry,
    *,
    changed_ratio: float,
    block_spread: float,
) -> bool:
    # A hard page boundary must be both large and spatially broad.  This keeps
    # an embedded chart replacement from being mislabeled as a slide change.
    return (
        geometry.changed_ratio >= changed_ratio
        and geometry.block_spread >= block_spread
    )


def _presentation_duplicate(
    samples: list[Sample],
    left_index: int,
    right_index: int,
    *,
    content_left_ratio: float,
    content_top_ratio: float,
    content_right_ratio: float,
    content_bottom_ratio: float,
    mask_delta: int,
    changed_ratio: float,
    edge_ratio: float,
    ssim_threshold: float,
    phash_threshold: int,
) -> bool:
    left = samples[left_index].gray
    right = samples[right_index].gray
    geometry = presentation_change_geometry(
        left,
        right,
        left_ratio=content_left_ratio,
        top_ratio=content_top_ratio,
        right_ratio=content_right_ratio,
        bottom_ratio=content_bottom_ratio,
        pixel_threshold=mask_delta,
    )
    if geometry.changed_ratio > changed_ratio:
        return False
    if geometry.edge_changed_ratio > edge_ratio:
        return False
    left_roi = _presentation_content_roi(
        left,
        left_ratio=content_left_ratio,
        top_ratio=content_top_ratio,
        right_ratio=content_right_ratio,
        bottom_ratio=content_bottom_ratio,
    )
    right_roi = _presentation_content_roi(
        right,
        left_ratio=content_left_ratio,
        top_ratio=content_top_ratio,
        right_ratio=content_right_ratio,
        bottom_ratio=content_bottom_ratio,
    )
    if global_ssim(left_roi, right_roi) < ssim_threshold:
        return False
    return (
        phash_distance(perceptual_hash(left_roi), perceptual_hash(right_roi))
        <= phash_threshold
    )


def _validate_presentation_selector_options(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float | None,
    content_left_ratio: float,
    content_top_ratio: float,
    content_right_ratio: float,
    content_bottom_ratio: float,
    mask_delta: int,
    stable_seconds: float,
    merge_seconds: float,
    thresholds: Iterable[tuple[str, float]],
    duplicate_phash: int,
) -> None:
    if not samples:
        raise ValueError("Presentation selection requires at least one sample")
    if analysis_fps < 1:
        raise ValueError("Presentation analysis FPS must be at least 1")
    if candidate_end_timestamp is not None and candidate_end_timestamp < samples[0].timestamp:
        raise ValueError("Candidate end timestamp must not precede the sampled range")
    if not 0 <= content_left_ratio < content_right_ratio <= 1:
        raise ValueError("Presentation horizontal ROI must satisfy 0 <= left < right <= 1")
    if not 0 <= content_top_ratio < content_bottom_ratio <= 1:
        raise ValueError("Presentation vertical ROI must satisfy 0 <= top < bottom <= 1")
    if not 0 <= mask_delta <= 255:
        raise ValueError("Mask delta must be between 0 and 255")
    if stable_seconds <= 0:
        raise ValueError("Presentation stable duration must be positive")
    if merge_seconds < 0:
        raise ValueError("Presentation transition merge duration must be non-negative")
    for name, value in thresholds:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not 0 <= duplicate_phash <= 64:
        raise ValueError("Presentation duplicate pHash distance must be in [0, 64]")


def select_presentation_states(
    samples: list[Sample],
    *,
    analysis_fps: float,
    candidate_end_timestamp: float | None = None,
    content_left_ratio: float = 0.04,
    content_top_ratio: float = 0.02,
    content_right_ratio: float = 0.98,
    content_bottom_ratio: float = 0.90,
    mask_delta: int = 12,
    stable_seconds: float = 1.0,
    merge_seconds: float = 2.0,
    stability_changed_ratio: float = 0.002,
    meaningful_changed_ratio: float = 0.003,
    meaningful_edge_ratio: float = 0.005,
    meaningful_block_spread: float = 0.03,
    page_changed_ratio: float = 0.08,
    page_block_spread: float = 0.35,
    duplicate_changed_ratio: float = 0.002,
    duplicate_edge_ratio: float = 0.004,
    duplicate_ssim: float = 0.995,
    duplicate_phash: int = 4,
) -> list[Candidate]:
    """Keep distinct, settled presentation states inside a bounded window.

    The detector first builds a stable-state sequence.  A later state replaces
    the buffered state when it is an additive reveal that retains at least 96%
    of the older edges and loses at most 0.4% of the ROI as old edges.  Page
    changes and destructive replacements preserve the older buffered state.

    Samples after ``candidate_end_timestamp`` are read-only lookahead.  They
    may confirm or invalidate a transition that both began and settled inside
    the eligible range, but can never contribute an output timestamp.
    """
    _validate_presentation_selector_options(
        samples,
        analysis_fps=analysis_fps,
        candidate_end_timestamp=candidate_end_timestamp,
        content_left_ratio=content_left_ratio,
        content_top_ratio=content_top_ratio,
        content_right_ratio=content_right_ratio,
        content_bottom_ratio=content_bottom_ratio,
        mask_delta=mask_delta,
        stable_seconds=stable_seconds,
        merge_seconds=merge_seconds,
        thresholds=(
            ("Stability changed ratio", stability_changed_ratio),
            ("Meaningful changed ratio", meaningful_changed_ratio),
            ("Meaningful edge ratio", meaningful_edge_ratio),
            ("Meaningful block spread", meaningful_block_spread),
            ("Page changed ratio", page_changed_ratio),
            ("Page block spread", page_block_spread),
            ("Duplicate changed ratio", duplicate_changed_ratio),
            ("Duplicate edge ratio", duplicate_edge_ratio),
            ("Duplicate SSIM", duplicate_ssim),
        ),
        duplicate_phash=duplicate_phash,
    )
    terminal_limit_index = (
        sample_index_at_or_before(samples, candidate_end_timestamp)
        if candidate_end_timestamp is not None
        else len(samples) - 1
    )
    if terminal_limit_index == 0:
        return []

    stable_steps_required = max(1, int(round(stable_seconds * analysis_fps)))
    stable_states: list[_PresentationStableState] = []
    initial_stable_steps = 0
    initial_quiet_start_index = 0
    initial_resolved = False
    anchor_index = 0

    event_pending = False
    event_started_index: int | None = None
    last_event_index: int | None = None
    stable_steps = 0
    stable_candidate_index: int | None = None
    quiet_start_index: int | None = None
    pending_hard_page = False
    pending_peak_score = 0.0

    geometry_options = {
        "left_ratio": content_left_ratio,
        "top_ratio": content_top_ratio,
        "right_ratio": content_right_ratio,
        "bottom_ratio": content_bottom_ratio,
        "pixel_threshold": mask_delta,
    }

    def append_pending_state() -> None:
        nonlocal anchor_index
        nonlocal event_pending, event_started_index, last_event_index
        nonlocal stable_steps, stable_candidate_index, quiet_start_index
        nonlocal pending_hard_page, pending_peak_score
        assert stable_candidate_index is not None
        final_geometry = presentation_change_geometry(
            samples[anchor_index].gray,
            samples[stable_candidate_index].gray,
            **geometry_options,
        )
        final_is_additive = (
            final_geometry.retained_edge_ratio >= 0.96
            and final_geometry.edge_loss_ratio <= 0.004
            and final_geometry.edge_gain_ratio >= 0.001
        )
        hard_page = not final_is_additive and (
            pending_hard_page
            or _presentation_is_hard_page(
                final_geometry,
                changed_ratio=page_changed_ratio,
                block_spread=page_block_spread,
            )
        )
        reason = (
            "presentation:hard_page_change"
            if hard_page
            else "presentation:post_settle_meaningful"
        )
        candidate = _candidate(
            samples,
            stable_candidate_index,
            reason,
            max(
                0.20,
                pending_peak_score,
                final_geometry.changed_ratio,
                final_geometry.edge_changed_ratio,
            ),
        )
        if hard_page:
            candidate.reasons.append("presentation:page_change")
        stable_states.append(
            _PresentationStableState(candidate, hard_page_change=hard_page)
        )
        anchor_index = stable_candidate_index
        event_pending = False
        event_started_index = None
        last_event_index = None
        stable_steps = 0
        stable_candidate_index = None
        quiet_start_index = None
        pending_hard_page = False
        pending_peak_score = 0.0

    for index in range(1, len(samples)):
        adjacent_geometry = presentation_change_geometry(
            samples[index - 1].gray,
            samples[index].gray,
            **geometry_options,
        )
        adjacent_quiet = (
            adjacent_geometry.soft_changed_ratio <= stability_changed_ratio
            or _presentation_is_compact_noise(
                adjacent_geometry,
                edge_ratio=meaningful_edge_ratio,
            )
        )

        if not initial_resolved:
            if index > terminal_limit_index:
                break
            if adjacent_quiet:
                quiet_geometry = presentation_change_geometry(
                    samples[initial_quiet_start_index].gray,
                    samples[index].gray,
                    **geometry_options,
                )
                quiet_run_holds = (
                    quiet_geometry.soft_changed_ratio <= stability_changed_ratio
                    or _presentation_is_compact_noise(
                        quiet_geometry,
                        edge_ratio=meaningful_edge_ratio,
                    )
                )
                if quiet_run_holds:
                    initial_stable_steps += 1
                else:
                    initial_stable_steps = 0
                    initial_quiet_start_index = index
            else:
                initial_stable_steps = 0
                initial_quiet_start_index = index
            if initial_stable_steps >= stable_steps_required:
                initial_candidate = _candidate(
                    samples,
                    index,
                    "presentation:initial_stable",
                    0.10,
                )
                stable_states.append(_PresentationStableState(initial_candidate))
                initial_resolved = True
                anchor_index = index
            continue

        adjacent_meaningful = _presentation_is_meaningful(
            adjacent_geometry,
            changed_ratio=meaningful_changed_ratio,
            edge_ratio=meaningful_edge_ratio,
            block_spread=meaningful_block_spread,
        )

        if not event_pending:
            if index > terminal_limit_index:
                break
            cumulative_geometry = presentation_change_geometry(
                samples[anchor_index].gray,
                samples[index].gray,
                **geometry_options,
            )
            if not _presentation_is_meaningful(
                cumulative_geometry,
                changed_ratio=meaningful_changed_ratio,
                edge_ratio=meaningful_edge_ratio,
                block_spread=meaningful_block_spread,
            ):
                continue
            event_pending = True
            event_started_index = index
            last_event_index = index
            stable_steps = 0
            stable_candidate_index = None
            quiet_start_index = None
            pending_hard_page = _presentation_is_hard_page(
                cumulative_geometry,
                changed_ratio=page_changed_ratio,
                block_spread=page_block_spread,
            )
            pending_peak_score = max(
                cumulative_geometry.changed_ratio,
                cumulative_geometry.edge_changed_ratio,
                cumulative_geometry.block_spread,
            )
            continue

        assert event_started_index is not None
        assert last_event_index is not None
        if adjacent_meaningful:
            # A semantic change in lookahead invalidates an earlier candidate;
            # its replacement cannot settle back inside the eligible range.
            last_event_index = index
            stable_steps = 0
            stable_candidate_index = None
            quiet_start_index = None
            pending_hard_page = pending_hard_page or _presentation_is_hard_page(
                adjacent_geometry,
                changed_ratio=page_changed_ratio,
                block_spread=page_block_spread,
            )
            pending_peak_score = max(
                pending_peak_score,
                adjacent_geometry.changed_ratio,
                adjacent_geometry.edge_changed_ratio,
                adjacent_geometry.block_spread,
            )
            continue

        if adjacent_quiet:
            if quiet_start_index is None:
                quiet_start_index = index - 1
            quiet_geometry = presentation_change_geometry(
                samples[quiet_start_index].gray,
                samples[index].gray,
                **geometry_options,
            )
            quiet_run_holds = (
                quiet_geometry.soft_changed_ratio <= stability_changed_ratio
                or _presentation_is_compact_noise(
                    quiet_geometry,
                    edge_ratio=meaningful_edge_ratio,
                )
            )
            if quiet_run_holds:
                stable_steps += 1
                if (
                    stable_candidate_index is None
                    and stable_steps >= stable_steps_required
                    and index <= terminal_limit_index
                ):
                    stable_candidate_index = index
            else:
                stable_steps = 0
                stable_candidate_index = None
                quiet_start_index = index
                last_event_index = index
        else:
            stable_steps = 0
            stable_candidate_index = None
            quiet_start_index = None
            last_event_index = index

        if (
            stable_candidate_index is not None
            and samples[index].timestamp - samples[last_event_index].timestamp
            >= merge_seconds
        ):
            append_pending_state()
            if index > terminal_limit_index:
                break

    # A stable sampled tail is valid evidence even when the bounded input ends
    # before the merge dwell completes.  A tail still in motion is not.
    if event_pending and stable_candidate_index is not None:
        append_pending_state()

    if not stable_states:
        return []

    emitted: list[Candidate] = []
    seen_indexes: list[int] = []
    buffered: _PresentationStableState | None = None
    for state in stable_states:
        current_index = state.candidate.sample_index
        if any(
            _presentation_duplicate(
                samples,
                previous_index,
                current_index,
                content_left_ratio=content_left_ratio,
                content_top_ratio=content_top_ratio,
                content_right_ratio=content_right_ratio,
                content_bottom_ratio=content_bottom_ratio,
                mask_delta=mask_delta,
                changed_ratio=duplicate_changed_ratio,
                edge_ratio=duplicate_edge_ratio,
                ssim_threshold=duplicate_ssim,
                phash_threshold=duplicate_phash,
            )
            for previous_index in seen_indexes
        ):
            continue
        seen_indexes.append(current_index)

        if buffered is None:
            buffered = state
            continue

        relation = presentation_change_geometry(
            samples[buffered.candidate.sample_index].gray,
            samples[current_index].gray,
            **geometry_options,
        )
        additive = (
            not state.hard_page_change
            and relation.retained_edge_ratio >= 0.96
            and relation.edge_loss_ratio <= 0.004
            and relation.edge_gain_ratio >= 0.001
        )
        if additive:
            state.candidate.reasons.append("presentation:additive_build_final")
            buffered = state
        else:
            terminal_reason = (
                "presentation:page_terminal"
                if state.hard_page_change
                else "presentation:pre_replacement"
            )
            buffered.candidate.reasons.append(terminal_reason)
            emitted.append(buffered.candidate)
            buffered = state

    assert buffered is not None
    buffered.candidate.reasons.append("presentation:bounded_tail_stable")
    emitted.append(buffered.candidate)
    return merge_candidates(emitted)


def _priority(candidate: Candidate) -> tuple[int, float, float]:
    reasons = set(candidate.reasons)
    if "settled_after_change" in reasons or "segment_end_after_change" in reasons:
        rank = 4
    elif "scenedetect_content" in reasons or "scenedetect_adaptive" in reasons:
        rank = 3
    elif "adaptive_change_settled" in reasons:
        rank = 2
    else:
        rank = 1
    return rank, candidate.score, candidate.timestamp


def deduplicate(
    samples: list[Sample],
    candidates: Iterable[Candidate],
    *,
    window: int = 6,
    phash_threshold: int = 4,
    ssim_threshold: float = 0.996,
    block_change_threshold: float = 0.012,
) -> tuple[list[Candidate], list[DroppedCandidate]]:
    kept: list[Candidate] = []
    dropped: list[DroppedCandidate] = []
    for item in merge_candidates(candidates):
        duplicate_position: int | None = None
        duplicate_metrics: tuple[int, float, float] | None = None
        current_sample = samples[item.sample_index]
        for position in range(max(0, len(kept) - window), len(kept)):
            previous = kept[position]
            previous_sample = samples[previous.sample_index]
            hamming = phash_distance(previous_sample.phash, current_sample.phash)
            if hamming > phash_threshold:
                continue
            similarity = global_ssim(previous_sample.gray, current_sample.gray)
            if similarity < ssim_threshold:
                continue
            local_change = block_change(previous_sample.gray, current_sample.gray)
            if local_change > block_change_threshold:
                continue
            duplicate_position = position
            duplicate_metrics = (hamming, similarity, local_change)
            break

        if duplicate_position is None or duplicate_metrics is None:
            kept.append(item)
            continue

        previous = kept[duplicate_position]
        hamming, similarity, local_change = duplicate_metrics
        if _priority(item) > _priority(previous):
            item.reasons = sorted(set(item.reasons + previous.reasons))
            kept[duplicate_position] = item
            dropped.append(
                DroppedCandidate(
                    candidate=previous,
                    duplicate_of_timestamp=item.timestamp,
                    phash_distance=hamming,
                    ssim=similarity,
                    block_change=local_change,
                )
            )
        else:
            previous.reasons = sorted(set(previous.reasons + item.reasons))
            dropped.append(
                DroppedCandidate(
                    candidate=item,
                    duplicate_of_timestamp=previous.timestamp,
                    phash_distance=hamming,
                    ssim=similarity,
                    block_change=local_change,
                )
            )
    return sorted(kept, key=lambda item: item.timestamp), dropped


def cap_temporally(candidates: list[Candidate], maximum: int) -> tuple[list[Candidate], list[Candidate]]:
    if maximum <= 0:
        raise ValueError("Maximum frame count must be positive")
    if len(candidates) <= maximum:
        return candidates, []
    start = candidates[0].timestamp
    end = candidates[-1].timestamp
    span = max(1e-9, end - start)
    buckets: list[list[Candidate]] = [[] for _ in range(maximum)]
    for candidate in candidates:
        position = min(maximum - 1, int((candidate.timestamp - start) / span * maximum))
        buckets[position].append(candidate)
    selected = [max(bucket, key=_priority) for bucket in buckets if bucket]
    if len(selected) < maximum:
        selected_ids = {id(item) for item in selected}
        remaining = [item for item in candidates if id(item) not in selected_ids]
        remaining.sort(key=_priority, reverse=True)
        selected.extend(remaining[: maximum - len(selected)])
    selected.sort(key=lambda item: item.timestamp)
    selected_indexes = {item.sample_index for item in selected}
    capped = [item for item in candidates if item.sample_index not in selected_indexes]
    return selected, capped


def finalize_strategy(
    name: str,
    samples: list[Sample],
    raw_candidates: Iterable[Candidate],
    *,
    maximum_frames: int,
    deduplicate_candidates: bool = True,
) -> StrategyResult:
    raw = merge_candidates(raw_candidates)
    if deduplicate_candidates:
        selected, dropped = deduplicate(samples, raw)
    else:
        selected, dropped = list(raw), []
    selected, capped = cap_temporally(selected, maximum_frames)
    return StrategyResult(
        name=name,
        raw_candidates=raw,
        selected=selected,
        dropped_duplicates=dropped,
        capped_candidates=capped,
    )
