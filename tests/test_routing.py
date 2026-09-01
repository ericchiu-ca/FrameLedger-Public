from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from frameledger.features import RoutingFrameFeatures
from frameledger.models import Candidate, ContentSegment, Sample, StrategyResult
from frameledger.routing import (
    classify_routing_features,
    classify_routing_frame,
    detect_content_segments,
    select_routed_states,
)


FRAME_HEIGHT = 270
FRAME_WIDTH = 480


def presentation_frame() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 160, dtype=np.uint8)
    for row in range(6):
        y = 48 + row * 25
        for column in range(12):
            x = 40 + column * 28
            frame[y : y + 6, x : x + 10] = 65
    return frame


def dark_chart_frame() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 28, dtype=np.uint8)
    frame[40:235:35, 28:450] = 52
    frame[35:235, 40:450:55] = 52
    return frame


def table_frame() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 245, dtype=np.uint8)
    frame[22:244:10, 8:472] = 24
    frame[22:244, 8:472:28] = 24
    return frame


def unknown_frame() -> np.ndarray:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH), 245, dtype=np.uint8)


def dark_embedded_presentation_frame() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 190, dtype=np.uint8)
    frame[24:244, 20:250] = 42
    for row in range(7):
        y = 55 + row * 23
        frame[y : y + 4, 290:305] = 80
        frame[y : y + 4, 315:330] = 80
    return frame


def white_embedded_presentation_frame() -> np.ndarray:
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH), 232, dtype=np.uint8)
    frame[55:215, 70:410] = 205
    for row in range(7):
        y = 70 + row * 18
        for column in range(10):
            x = 90 + column * 28
            frame[y : y + 4, x : x + 8] = 120
    return frame


def samples_from_frames(frames: list[np.ndarray], *, fps: float = 2.0) -> list[Sample]:
    return [
        Sample(timestamp=index / fps, frame_index=index, gray=frame.copy())
        for index, frame in enumerate(frames)
    ]


def segment(
    samples: list[Sample],
    *,
    segment_id: str,
    kind: str,
    start: int,
    stop: int,
) -> ContentSegment:
    selector = {
        "presentation": "presentation_states",
        "table": "table_viewport",
        "chart": "boundary_terminal",
        "unknown": None,
    }[kind]
    return ContentSegment(
        segment_id=segment_id,
        kind=kind,
        start_sample_index=start,
        stop_sample_index=stop,
        start_timestamp=samples[start].timestamp,
        candidate_end_timestamp=samples[stop - 1].timestamp,
        confidence=0.9,
        selector=selector,
        reasons=("test",),
    )


class RoutingTests(unittest.TestCase):
    def test_conservative_rules_classify_three_known_layouts_and_unknown(self):
        cases = (
            (presentation_frame(), "presentation"),
            (table_frame(), "table"),
            (dark_chart_frame(), "chart"),
            (unknown_frame(), "unknown"),
        )
        for index, (frame, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                decision = classify_routing_frame(
                    frame,
                    sample_index=index,
                    timestamp=float(index),
                )
                self.assertEqual(decision.kind, expected)
                self.assertEqual(
                    set(decision.scores),
                    {"presentation", "table", "chart"},
                )
                self.assertTrue(decision.reasons)
                self.assertEqual(decision.to_dict()["kind"], expected)

    def test_dark_embedded_chart_stays_presentation(self):
        decision = classify_routing_frame(
            dark_embedded_presentation_frame(),
            sample_index=0,
            timestamp=0.0,
        )

        self.assertEqual(decision.kind, "presentation")
        self.assertGreater(decision.features.dark_ratio, 0.40)
        self.assertLess(decision.features.dark_ratio, 0.72)

    def test_white_embedded_visual_stays_presentation(self):
        decision = classify_routing_frame(
            white_embedded_presentation_frame(),
            sample_index=0,
            timestamp=0.0,
        )

        self.assertEqual(decision.kind, "presentation")
        self.assertGreater(decision.features.bright_ratio, 0.25)
        self.assertLess(decision.features.horizontal_line_ratio, 0.008)
        self.assertLess(decision.features.vertical_line_ratio, 0.008)
        self.assertIn("presentation:white_embedded_visual", decision.reasons)

    def test_realistic_white_embedded_visual_allows_sparse_long_lines(self):
        features = RoutingFrameFeatures(
            mean_luma=232.46,
            dark_ratio=0.0005,
            bright_ratio=0.738,
            edge_density=0.0672,
            horizontal_line_ratio=0.0092,
            vertical_line_ratio=0.0055,
        )

        kind, _confidence, _scores, reasons = classify_routing_features(features)

        self.assertEqual(kind, "presentation")
        self.assertIn("presentation:white_embedded_visual", reasons)

    def test_short_raw_blip_is_bridged_between_matching_kinds(self):
        frames = [presentation_frame() for _ in range(8)]
        frames.extend([table_frame() for _ in range(2)])
        frames.extend([presentation_frame() for _ in range(8)])
        samples = samples_from_frames(frames)

        plan = detect_content_segments(
            samples,
            analysis_fps=2.0,
            candidate_end_timestamp=samples[-1].timestamp,
        )

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].kind, "presentation")
        self.assertTrue(
            any(
                "postprocess:bridge:table->presentation" in decision.reasons
                for decision in plan.decisions
            )
        )

    def test_three_second_high_confidence_table_becomes_unknown(self):
        samples = samples_from_frames([table_frame() for _ in range(6)])

        plan = detect_content_segments(
            samples,
            analysis_fps=2.0,
            candidate_end_timestamp=samples[-1].timestamp,
        )

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].kind, "unknown")
        self.assertIsNone(plan.segments[0].selector)
        self.assertTrue(all(decision.raw_kind == "table" for decision in plan.decisions))
        self.assertTrue(all(decision.confidence == 0.0 for decision in plan.decisions))
        self.assertEqual(plan.segments[0].confidence, 0.0)

    def test_four_second_known_run_is_preserved(self):
        samples = samples_from_frames([table_frame() for _ in range(8)])

        plan = detect_content_segments(
            samples,
            analysis_fps=2.0,
            candidate_end_timestamp=samples[-1].timestamp,
        )

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0].kind, "table")
        self.assertEqual(plan.segments[0].selector, "table_viewport")

    def test_segments_cover_only_eligible_samples_without_gaps(self):
        frames = [dark_chart_frame() for _ in range(8)]
        frames.extend([unknown_frame() for _ in range(4)])
        frames.extend([presentation_frame() for _ in range(8)])
        frames.extend([table_frame() for _ in range(4)])  # read-only lookahead
        samples = samples_from_frames(frames)
        candidate_end = samples[19].timestamp

        plan = detect_content_segments(
            samples,
            analysis_fps=2.0,
            candidate_end_timestamp=candidate_end,
        )

        self.assertEqual(plan.eligible_sample_count, 20)
        self.assertEqual(len(plan.decisions), 20)
        self.assertEqual(plan.segments[0].start_sample_index, 0)
        self.assertEqual(plan.segments[-1].stop_sample_index, 20)
        self.assertEqual(
            [item.segment_id for item in plan.segments],
            [f"route-{index:03d}" for index in range(1, len(plan.segments) + 1)],
        )
        for left, right in zip(plan.segments, plan.segments[1:]):
            self.assertEqual(left.stop_sample_index, right.start_sample_index)
        self.assertTrue(
            all(
                decision.timestamp <= candidate_end
                for decision in plan.decisions
            )
        )
        self.assertEqual(len(plan.to_dict()["segments"]), len(plan.segments))

    def test_unknown_segment_does_not_dispatch_any_selector(self):
        samples = samples_from_frames([unknown_frame() for _ in range(10)])
        routed_segment = segment(
            samples,
            segment_id="route-001",
            kind="unknown",
            start=0,
            stop=8,
        )

        with (
            patch("frameledger.selection.select_boundary_terminal_states") as chart,
            patch("frameledger.selection.select_table_viewport_states") as table,
            patch("frameledger.selection.select_presentation_states") as presentation,
        ):
            candidates = select_routed_states(
                samples,
                [routed_segment],
                analysis_fps=2.0,
            )

        self.assertEqual(candidates, [])
        chart.assert_not_called()
        table.assert_not_called()
        presentation.assert_not_called()

    def test_known_segment_cannot_omit_its_fixed_selector(self):
        samples = samples_from_frames([presentation_frame() for _ in range(8)])

        with self.assertRaisesRegex(ValueError, "must be 'presentation_states'"):
            ContentSegment(
                segment_id="route-001",
                kind="presentation",
                start_sample_index=0,
                stop_sample_index=8,
                start_timestamp=samples[0].timestamp,
                candidate_end_timestamp=samples[7].timestamp,
                confidence=0.9,
                selector=None,
            )

    def test_each_selector_receives_its_bounded_core_plus_fixed_lookahead(self):
        samples = samples_from_frames(
            [presentation_frame() for _ in range(40)],
            fps=1.0,
        )
        cases = (
            (
                "chart",
                "select_boundary_terminal_states",
                15,
            ),
            (
                "table",
                "select_table_viewport_states",
                8,
            ),
            (
                "presentation",
                "select_presentation_states",
                6,
            ),
        )

        for ordinal, (kind, selector_function, lookahead) in enumerate(cases, start=1):
            with self.subTest(kind=kind):
                routed_segment = segment(
                    samples,
                    segment_id=f"route-{ordinal:03d}",
                    kind=kind,
                    start=5,
                    stop=10,
                )
                prefix = segment(
                    samples,
                    segment_id="route-000",
                    kind="unknown",
                    start=0,
                    stop=5,
                )

                def fake_selector(local_samples, **kwargs):
                    self.assertEqual(len(local_samples), 5 + lookahead)
                    self.assertEqual(kwargs["candidate_end_timestamp"], 9.0)
                    return []

                with patch(
                    f"frameledger.selection.{selector_function}",
                    side_effect=fake_selector,
                ) as mocked_selector:
                    candidates = select_routed_states(
                        samples,
                        [prefix, routed_segment],
                        analysis_fps=1.0,
                    )

                self.assertEqual(candidates, [])
                mocked_selector.assert_called_once()

    def test_chart_route_forwards_boundary_terminal_policy(self):
        samples = samples_from_frames(
            [dark_chart_frame() for _ in range(24)],
            fps=1.0,
        )
        routed_segment = segment(
            samples,
            segment_id="route-001",
            kind="chart",
            start=0,
            stop=10,
        )

        with patch(
            "frameledger.selection.select_boundary_terminal_states",
            return_value=[],
        ) as selector:
            select_routed_states(
                samples,
                [routed_segment],
                analysis_fps=1.0,
                boundary_transition_threshold=0.031,
                boundary_pixel_delta_threshold=0.005,
                boundary_edge_loss_threshold=0.004,
                boundary_terminal_search_seconds=2.5,
                boundary_terminal_stable_seconds=1.5,
                boundary_terminal_stability_threshold=0.009,
            )

        kwargs = selector.call_args.kwargs
        self.assertEqual(kwargs["transition_threshold"], 0.031)
        self.assertEqual(kwargs["pixel_delta_threshold"], 0.005)
        self.assertEqual(kwargs["edge_loss_threshold"], 0.004)
        self.assertEqual(kwargs["terminal_search_seconds"], 2.5)
        self.assertEqual(kwargs["terminal_stable_seconds"], 1.5)
        self.assertEqual(kwargs["terminal_stability_threshold"], 0.009)

    def test_local_candidate_index_is_rebased_to_global_sample_index(self):
        samples = samples_from_frames(
            [presentation_frame() for _ in range(22)],
            fps=1.0,
        )
        routed_segment = segment(
            samples,
            segment_id="route-003",
            kind="presentation",
            start=5,
            stop=10,
        )
        prefix = segment(
            samples,
            segment_id="route-002",
            kind="unknown",
            start=0,
            stop=5,
        )

        def fake_selector(local_samples, **kwargs):
            self.assertEqual(len(local_samples), 11)  # 5..15, including 6s lookahead
            self.assertEqual(kwargs["candidate_end_timestamp"], 9.0)
            return [Candidate(2, 7.0, 0.8, ["local"])]

        with patch(
            "frameledger.selection.select_presentation_states",
            side_effect=fake_selector,
        ):
            candidates = select_routed_states(
                samples,
                [prefix, routed_segment],
                analysis_fps=1.0,
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].sample_index, 7)
        self.assertEqual(candidates[0].timestamp, samples[7].timestamp)
        self.assertEqual(candidates[0].segment_id, "route-003")
        self.assertIn("route:kind=presentation", candidates[0].reasons)
        self.assertIn("route:selector=presentation_states", candidates[0].reasons)

    def test_selector_cannot_emit_a_lookahead_candidate(self):
        samples = samples_from_frames(
            [presentation_frame() for _ in range(22)],
            fps=1.0,
        )
        routed_segment = segment(
            samples,
            segment_id="route-001",
            kind="presentation",
            start=5,
            stop=10,
        )
        prefix = segment(
            samples,
            segment_id="route-000",
            kind="unknown",
            start=0,
            stop=5,
        )

        with patch(
            "frameledger.selection.select_presentation_states",
            return_value=[Candidate(5, 10.0, 0.8, ["invalid-lookahead"])],
        ):
            with self.assertRaisesRegex(RuntimeError, "lookahead sample"):
                select_routed_states(
                    samples,
                    [prefix, routed_segment],
                    analysis_fps=1.0,
                )

    def test_optional_segment_fields_preserve_legacy_json_shape(self):
        candidate = Candidate(0, 0.0, 0.1, ["legacy"])
        legacy_candidate_keys = {
            "sample_index",
            "timestamp",
            "score",
            "reasons",
            "image_path",
        }
        self.assertEqual(set(candidate.to_dict()), legacy_candidate_keys)

        result = StrategyResult(
            name="legacy",
            raw_candidates=[candidate],
            selected=[candidate],
            dropped_duplicates=[],
        )
        self.assertNotIn("segments", result.to_dict())

        samples = samples_from_frames([unknown_frame()])
        routed_segment = segment(
            samples,
            segment_id="route-001",
            kind="unknown",
            start=0,
            stop=1,
        )
        candidate.segment_id = "route-001"
        result.segments = [routed_segment]
        self.assertEqual(candidate.to_dict()["segment_id"], "route-001")
        self.assertEqual(
            result.to_dict()["segments"][0]["stop_sample_index"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
