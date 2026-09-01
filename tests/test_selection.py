import unittest

import numpy as np

from frameledger.features import presentation_change_geometry
from frameledger.models import Candidate
from frameledger.selection import (
    VALID_STRATEGIES,
    build_samples,
    deduplicate,
    select_boundary_terminal_states,
    select_presentation_states,
    select_settled_end_states,
    select_table_focus_states,
    select_table_viewport_states,
    select_uniform,
)


def frame_stream(frames, fps=2.0):
    for index, frame in enumerate(frames):
        yield index / fps, index, frame


def table_frame(*, offset=0, focus_row=None, cursor=None, cell=None):
    """Build a deterministic worksheet-like frame for spatial tests."""
    frame = np.full((120, 200), 24, dtype=np.uint8)
    frame[:30, :] = 12
    for visible_row in range(8):
        source_row = visible_row + offset
        y0 = 30 + visible_row * 10
        shade = 42 + (source_row * 17) % 95
        frame[y0 : y0 + 9, :] = shade
        frame[y0 + 2 : y0 + 4, 8 : 55 + (source_row * 11) % 120] = min(
            240, shade + 70
        )
    frame[30:110:10, :] = 8
    frame[30:110, ::25] = 8
    if focus_row is not None:
        y0 = 31 + focus_row * 10
        frame[y0 : y0 + 8, :] = np.clip(
            frame[y0 : y0 + 8, :].astype(np.int16) + 65, 0, 255
        ).astype(np.uint8)
    if cursor is not None:
        y, x = cursor
        frame[y : y + 3, x : x + 3] = 250
    if cell is not None:
        y, x = cell
        frame[y : y + 4, x : x + 8] = 230
    return frame


def presentation_frame(*, page=1, elements=(), cursor=None, cursor_size=(3, 3), cell=False):
    """Build a sparse, deterministic slide with optional reveal elements."""
    frame = np.full((120, 200), 240, dtype=np.uint8)
    frame[108:, :] = 14  # player chrome excluded by the default content ROI
    if page == 1:
        ink = 28
        frame[12:16, 22:122] = ink
        frame[22:24, 22:180] = 110
    else:
        frame[2:108, 8:196] = 38
        ink = 232
        frame[12:17, 42:178] = ink
        frame[25:27, 18:184] = 150

    rectangles = {
        "bullet_1": (38, 42, 30, 154),
        "bullet_2": (53, 57, 30, 168),
        "bullet_3": (68, 72, 30, 144),
        "old_panel": (34, 42, 26, 166),
        "new_panel": (72, 80, 26, 166),
        "annotation": (88, 91, 52, 150),
    }
    for element in elements:
        y0, y1, x0, x1 = rectangles[element]
        frame[y0:y1, x0:x1] = ink
    if cursor is not None:
        y, x = cursor
        cursor_height, cursor_width = cursor_size
        frame[y : y + cursor_height, x : x + cursor_width] = 128
    if cell:
        frame[82:86, 156:164] = 128
    return frame


def dense_presentation_frame(*, annotation=False):
    """Build a text-heavy slide with an isolated annotation in clear space."""
    frame = presentation_frame()
    for row in range(10):
        y = 24 + row * 8
        for column in range(10):
            x = 12 + column * 15
            frame[y : y + 2, x : x + 12] = 28
    if annotation:
        frame[84:102, 170:188] = 112
    return frame


class SelectionTests(unittest.TestCase):
    def test_uniform_includes_tail(self):
        frames = [np.full((72, 128), index, dtype=np.uint8) for index in range(23)]
        samples = build_samples(frame_stream(frames, fps=1.0))
        candidates = select_uniform(samples, interval_seconds=10)
        self.assertEqual([round(item.timestamp) for item in candidates], [0, 10, 20])

    def test_settled_detector_keeps_completed_progression(self):
        base = np.zeros((96, 160), dtype=np.uint8)
        frames = [base.copy() for _ in range(4)]
        for width in (25, 50, 75, 100):
            changed = base.copy()
            changed[44:52, 20 : 20 + width] = 255
            frames.append(changed)
        frames.extend([frames[-1].copy() for _ in range(6)])
        samples = build_samples(frame_stream(frames, fps=2.0))
        candidates = select_settled_end_states(samples, analysis_fps=2.0)
        self.assertTrue(candidates)
        self.assertGreaterEqual(candidates[-1].timestamp, 5.0)
        self.assertIn("settled_after_change", candidates[-1].reasons)

    def test_boundary_terminal_ignores_pause_and_keeps_pre_clear_state(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        base[:, 80:82] = 70

        def annotated(line_count):
            frame = base.copy()
            for index in range(line_count):
                y = 36 + index * 10
                frame[y : y + 4, 20 : 120 + index * 8] = 255
            return frame

        frames = []
        frames.extend([base.copy() for _ in range(4)])
        frames.extend([annotated(1) for _ in range(4)])
        # A long pause after the first stroke must not close the round.
        frames.extend([annotated(1) for _ in range(8)])
        frames.extend([annotated(2) for _ in range(4)])
        frames.extend([annotated(3) for _ in range(8)])
        frames.extend([base.copy() for _ in range(4)])
        frames.extend([annotated(1) for _ in range(4)])
        frames.extend([annotated(2) for _ in range(4)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
            terminal_stable_seconds=1.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].timestamp, 13.5)
        self.assertTrue(any("boundary_terminal" in reason for reason in candidates[0].reasons))

    def test_boundary_terminal_keeps_last_pre_clear_frame_without_quiet_gap(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        frames = [base.copy() for _ in range(4)]
        for width in (30, 60, 90, 120):
            drawing = base.copy()
            drawing[42:44, 20 : 20 + width] = 255
            frames.append(drawing)
        boundary_index = len(frames)
        frames.extend([base.copy() for _ in range(6)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].sample_index, boundary_index - 1)

    def test_boundary_terminal_ignores_transient_flash(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 90
        frames = [base.copy() for _ in range(12)]
        flash = base.copy()
        flash[30:70, 40:120] = 255
        frames.append(flash)
        frames.extend([base.copy() for _ in range(11)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
        )

        self.assertEqual(candidates, [])

    def test_boundary_terminal_rejects_a_temporary_clear_that_recovers(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        annotated = base.copy()
        annotated[40:52, 20:140] = 255
        frames = [annotated.copy() for _ in range(12)]
        # One second of blank canvas is not a persistent round boundary.
        frames.extend([base.copy() for _ in range(2)])
        frames.extend([annotated.copy() for _ in range(12)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
        )

        self.assertEqual(candidates, [])

    def test_boundary_terminal_uses_lookahead_to_confirm_window_end_state(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        annotated = base.copy()
        annotated[42:46, 20:140] = 255
        # The eligible window ends between the 4.0s and 4.5s samples. The same
        # terminal state persists in lookahead until a clear at 5.5s.
        frames = [annotated.copy() for _ in range(11)]
        frames.extend([base.copy() for _ in range(6)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
            candidate_end_timestamp=4.24,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].timestamp, 4.0)
        self.assertIn("lookahead_confirmed_terminal", candidates[0].reasons)
        self.assertTrue(all("segment_end" not in reason for reason in candidates[0].reasons))

    def test_boundary_terminal_does_not_invent_window_end_without_later_clear(self):
        fps = 2.0
        annotated = np.zeros((96, 160), dtype=np.uint8)
        annotated[42:46, 20:140] = 255
        samples = build_samples(
            frame_stream([annotated.copy() for _ in range(20)], fps=fps)
        )

        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(candidates, [])

    def test_boundary_terminal_requires_enough_post_clear_lookahead(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        annotated = base.copy()
        annotated[42:46, 20:140] = 255
        frames = [annotated.copy() for _ in range(11)]
        # The clear is visible, but the sampled range ends before the complete
        # 1.5-second persistence window can be observed.
        frames.extend([base.copy() for _ in range(2)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(candidates, [])

    def test_boundary_terminal_rejects_window_end_if_drawing_continues_in_lookahead(self):
        fps = 2.0
        base = np.zeros((96, 160), dtype=np.uint8)
        base[20:22, :] = 70
        partial = base.copy()
        partial[42:46, 20:100] = 255
        complete = partial.copy()
        for row in (58, 62, 66, 70):
            complete[row : row + 1, 20:140] = 255
        frames = [partial.copy() for _ in range(9)]
        frames.extend([complete.copy() for _ in range(2)])
        frames.extend([base.copy() for _ in range(6)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_boundary_terminal_states(
            samples,
            analysis_fps=fps,
            pre_boundary_seconds=0.5,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(candidates, [])

    def test_table_viewport_keeps_initial_and_stable_post_scroll_state(self):
        fps = 2.0
        initial = table_frame(offset=0)
        frames = [initial.copy() for _ in range(6)]
        # Three steps model a multi-step scroll. They must collapse into one
        # stable viewport, not emit any transitional frame.
        frames.extend([table_frame(offset=1) for _ in range(2)])
        frames.extend([table_frame(offset=2) for _ in range(2)])
        final = table_frame(offset=3)
        frames.extend([final.copy() for _ in range(14)])

        samples = build_samples(frame_stream(frames, fps=fps))
        candidates = select_table_viewport_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)
        self.assertIn("table_viewport:post_transition", candidates[1].reasons)
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, final))

    def test_table_viewport_ignores_wide_row_focus(self):
        fps = 2.0
        initial = table_frame()
        focused = table_frame(focus_row=3)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([focused.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)

    def test_table_focus_keeps_persistent_wide_row_focus(self):
        fps = 2.0
        initial = table_frame()
        focused = table_frame(focus_row=3)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([focused.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_focus_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertIn("table_focus:initial_stable", candidates[0].reasons)
        self.assertIn("table_focus:post_row_focus", candidates[1].reasons)
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, focused))

    def test_table_focus_merges_quick_focus_moves_and_keeps_final_group(self):
        fps = 2.0
        initial = table_frame()
        first_focus = table_frame(focus_row=1)
        final_focus = table_frame(focus_row=5)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([first_focus.copy() for _ in range(6)])
        frames.extend([final_focus.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_focus_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, final_focus))
        self.assertFalse(
            any(np.array_equal(samples[item.sample_index].gray, first_focus) for item in candidates)
        )

    def test_table_focus_preserves_separated_distinct_row_states(self):
        fps = 2.0
        initial = table_frame()
        first_focus = table_frame(focus_row=1)
        second_focus = table_frame(focus_row=5)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([first_focus.copy() for _ in range(14)])
        frames.extend([second_focus.copy() for _ in range(14)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_focus_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, first_focus))
        self.assertTrue(np.array_equal(samples[candidates[2].sample_index].gray, second_focus))

    def test_table_viewport_folds_return_to_prior_view(self):
        fps = 2.0
        initial = table_frame(offset=0)
        scrolled = table_frame(offset=3)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([scrolled.copy() for _ in range(14)])
        frames.extend([initial.copy() for _ in range(14)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, scrolled))

    def test_table_focus_folds_highlight_return_to_baseline(self):
        fps = 2.0
        initial = table_frame()
        focused = table_frame(focus_row=3)
        frames = [initial.copy() for _ in range(6)]
        frames.extend([focused.copy() for _ in range(14)])
        frames.extend([initial.copy() for _ in range(14)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_focus_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, focused))

    def test_table_selectors_ignore_mouse_and_single_cell_changes(self):
        fps = 2.0
        initial = table_frame()
        frames = [initial.copy() for _ in range(6)]
        frames.extend([table_frame(cursor=(54, 72)) for _ in range(4)])
        frames.extend([table_frame(cursor=(76, 130)) for _ in range(4)])
        frames.extend([table_frame(cell=(63, 82)) for _ in range(14)])
        samples = build_samples(frame_stream(frames, fps=fps))

        viewport = select_table_viewport_states(samples, analysis_fps=fps)
        focus = select_table_focus_states(samples, analysis_fps=fps)

        self.assertEqual(len(viewport), 1)
        self.assertEqual(len(focus), 1)

    def test_table_viewport_does_not_emit_unsettled_window_tail(self):
        fps = 2.0
        initial = table_frame()
        frames = [initial.copy() for _ in range(6)]
        frames.extend([table_frame(offset=1) for _ in range(2)])
        frames.extend([table_frame(offset=2) for _ in range(2)])
        # Only two seconds of the final view are visible: less than the five
        # second merge dwell, so the bounded tail is not promoted.
        frames.extend([table_frame(offset=3) for _ in range(4)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)

    def test_table_viewport_lookahead_confirms_pre_end_stable_state(self):
        fps = 2.0
        initial = table_frame(offset=0)
        scrolled = table_frame(offset=3)
        frames = [initial.copy() for _ in range(6)]
        # Transition at 3.0s; the stable candidate is established at 4.0s,
        # inside the 5.0s candidate range. Later read-only samples supply the
        # rest of the five-second merge dwell.
        frames.extend([scrolled.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=5.0,
        )

        self.assertEqual(len(candidates), 2)
        self.assertLessEqual(candidates[1].timestamp, 5.0)
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, scrolled))

    def test_table_viewport_does_not_start_event_in_lookahead(self):
        fps = 2.0
        initial = table_frame(offset=0)
        scrolled = table_frame(offset=3)
        frames = [initial.copy() for _ in range(12)]
        # The transition starts after the 5.0s candidate boundary and has no
        # output eligibility even though it later settles for a long time.
        frames.extend([scrolled.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=5.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)

    def test_table_viewport_does_not_emit_state_that_settles_after_end(self):
        fps = 2.0
        initial = table_frame(offset=0)
        scrolled = table_frame(offset=3)
        frames = [initial.copy() for _ in range(10)]
        # The transition is inside the range at 5.0s, but the one-second stable
        # run completes only in lookahead. Confirmation cannot grant eligibility.
        frames.extend([scrolled.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=5.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)

    def test_table_viewport_lookahead_event_invalidates_pending_state(self):
        fps = 2.0
        initial = table_frame(offset=0)
        first_scroll = table_frame(offset=3)
        later_scroll = table_frame(offset=6)
        frames = [initial.copy() for _ in range(6)]
        # The first view stabilizes inside the range but has not completed its
        # merge dwell by 5.0s. A second scroll in lookahead proves that the first
        # state was not the terminal member of that transition cluster.
        frames.extend([first_scroll.copy() for _ in range(6)])
        frames.extend([later_scroll.copy() for _ in range(16)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_table_viewport_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=5.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertIn("table_viewport:initial_stable", candidates[0].reasons)

    def test_presentation_strategy_is_public(self):
        self.assertIn("presentation_states", VALID_STRATEGIES)

    def test_presentation_rejects_analysis_fps_below_one(self):
        initial = presentation_frame()
        samples = build_samples(
            frame_stream([initial.copy() for _ in range(4)], fps=0.5)
        )

        with self.assertRaisesRegex(ValueError, "at least 1"):
            select_presentation_states(samples, analysis_fps=0.5)

    def test_presentation_one_fps_keeps_stable_final_reveal(self):
        fps = 1.0
        initial = presentation_frame()
        final = presentation_frame(elements=("bullet_1",))
        frames = [initial.copy() for _ in range(3)]
        frames.extend([final.copy() for _ in range(5)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertIn("presentation:bounded_tail_stable", candidates[0].reasons)

    def test_presentation_one_fps_slow_fade_has_no_intermediate_output(self):
        fps = 1.0
        first = presentation_frame(elements=("old_panel",))
        blank = np.full((120, 200), 240, dtype=np.uint8)
        blank[108:, :] = 14
        final = presentation_frame(page=2, elements=("bullet_1",))

        def interpolate(left, right):
            return [
                np.rint(
                    left.astype(np.float32) * (1.0 - step / 22.0)
                    + right.astype(np.float32) * (step / 22.0)
                ).astype(np.uint8)
                for step in range(1, 22)
            ]

        transition = interpolate(first, blank) + [blank] + interpolate(blank, final)
        frames = [first.copy() for _ in range(3)] + transition
        frames.extend([final.copy() for _ in range(5)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, first))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, final))
        self.assertFalse(
            any(
                np.array_equal(samples[item.sample_index].gray, intermediate)
                for item in candidates
                for intermediate in transition
            )
        )

    def test_presentation_static_range_keeps_one_bounded_stable_state(self):
        fps = 2.0
        initial = presentation_frame()
        samples = build_samples(frame_stream([initial.copy() for _ in range(8)], fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertIn("presentation:initial_stable", candidates[0].reasons)
        self.assertIn("presentation:bounded_tail_stable", candidates[0].reasons)

    def test_presentation_additive_builds_collapse_to_final_reveal(self):
        fps = 2.0
        initial = presentation_frame()
        first = presentation_frame(elements=("bullet_1",))
        second = presentation_frame(elements=("bullet_1", "bullet_2"))
        final = presentation_frame(elements=("bullet_1", "bullet_2", "bullet_3"))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([first.copy() for _ in range(12)])
        frames.extend([second.copy() for _ in range(12)])
        frames.extend([final.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertIn("presentation:additive_build_final", candidates[0].reasons)

    def test_presentation_broad_additive_reveal_is_not_a_hard_page(self):
        fps = 2.0
        initial = presentation_frame()
        final = initial.copy()
        # Add content in 24 of the 8x8 ROI blocks.  This deliberately exceeds
        # both hard-page thresholds while retaining every old edge.
        for y in (30, 43, 57, 70, 83, 96):
            for x in (14, 60, 107, 154):
                final[y : y + 9, x : x + 12] = 28
        relation = presentation_change_geometry(initial, final)
        self.assertGreaterEqual(relation.changed_ratio, 0.08)
        self.assertGreaterEqual(relation.block_spread, 0.35)
        self.assertGreaterEqual(relation.retained_edge_ratio, 0.96)
        self.assertLessEqual(relation.edge_loss_ratio, 0.004)
        self.assertGreaterEqual(relation.edge_gain_ratio, 0.001)

        frames = [initial.copy() for _ in range(6)]
        frames.extend([final.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertIn("presentation:additive_build_final", candidates[0].reasons)
        self.assertNotIn("presentation:hard_page_change", candidates[0].reasons)

    def test_presentation_destructive_replacement_preserves_both_states(self):
        fps = 2.0
        before = presentation_frame(elements=("old_panel",))
        after = presentation_frame(elements=("new_panel",))
        frames = [before.copy() for _ in range(6)]
        frames.extend([after.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, before))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, after))

    def test_presentation_hard_page_marks_previous_terminal(self):
        fps = 2.0
        first_page = presentation_frame(elements=("old_panel",))
        second_page = presentation_frame(page=2, elements=("bullet_1",))
        frames = [first_page.copy() for _ in range(6)]
        frames.extend([second_page.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertIn("presentation:page_terminal", candidates[0].reasons)
        self.assertIn("presentation:hard_page_change", candidates[1].reasons)
        self.assertIn("presentation:bounded_tail_stable", candidates[1].reasons)

    def test_presentation_fade_frames_are_never_outputs(self):
        fps = 2.0
        initial = presentation_frame()
        final = presentation_frame(elements=("bullet_1",))
        fades = [
            np.rint(initial.astype(np.float32) * (1.0 - alpha) + final * alpha).astype(
                np.uint8
            )
            for alpha in (0.25, 0.50, 0.75)
        ]
        frames = [initial.copy() for _ in range(6)] + fades
        frames.extend([final.copy() for _ in range(10)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertFalse(
            any(
                np.array_equal(samples[item.sample_index].gray, fade)
                for item in candidates
                for fade in fades
            )
        )

    def test_presentation_slow_fade_drift_never_becomes_a_stable_state(self):
        fps = 2.0
        first = presentation_frame(elements=("old_panel",))
        blank = np.full((120, 200), 240, dtype=np.uint8)
        blank[108:, :] = 14
        final = presentation_frame(page=2, elements=("bullet_1",))

        def interpolate(left, right):
            return [
                np.rint(
                    left.astype(np.float32) * (1.0 - step / 22.0)
                    + right.astype(np.float32) * (step / 22.0)
                ).astype(np.uint8)
                for step in range(1, 22)
            ]

        transition = interpolate(first, blank) + [blank] + interpolate(blank, final)
        # The largest per-sample pixel increment is about ten gray levels,
        # below mask_delta=12, while the cumulative fade is a full page change.
        self.assertLessEqual(
            max(
                int(cvmax)
                for left, right in zip([first] + transition[:-1], transition)
                for cvmax in [np.abs(left.astype(np.int16) - right).max()]
            ),
            10,
        )
        frames = [first.copy() for _ in range(6)] + transition
        frames.extend([final.copy() for _ in range(10)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, first))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, final))
        self.assertFalse(
            any(
                np.array_equal(samples[item.sample_index].gray, intermediate)
                for item in candidates
                for intermediate in transition
            )
        )

    def test_presentation_two_rapid_builds_merge_before_settle(self):
        fps = 2.0
        initial = presentation_frame()
        first = presentation_frame(elements=("bullet_1",))
        final = presentation_frame(elements=("bullet_1", "bullet_2"))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([first.copy() for _ in range(3)])
        frames.extend([final.copy() for _ in range(10)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))

    def test_presentation_separated_non_subsuming_builds_are_preserved(self):
        fps = 2.0
        first = presentation_frame(elements=("old_panel",))
        second = presentation_frame(elements=("new_panel",))
        third = presentation_frame(elements=("bullet_1",))
        frames = [first.copy() for _ in range(6)]
        frames.extend([second.copy() for _ in range(12)])
        frames.extend([third.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, first))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, second))
        self.assertTrue(np.array_equal(samples[candidates[2].sample_index].gray, third))

    def test_presentation_ignores_cursor_and_single_cell_noise(self):
        fps = 2.0
        initial = presentation_frame()
        frames = [initial.copy() for _ in range(6)]
        frames.extend(
            [presentation_frame(cursor=(48, 74), cursor_size=(5, 8)) for _ in range(3)]
        )
        frames.extend(
            [presentation_frame(cursor=(76, 136), cursor_size=(8, 12)) for _ in range(3)]
        )
        frames.extend(
            [presentation_frame(cursor=(54, 92), cursor_size=(12, 8)) for _ in range(3)]
        )
        frames.extend([presentation_frame(cell=True) for _ in range(10)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))

    def test_presentation_annotation_is_meaningful_and_additive(self):
        fps = 2.0
        initial = presentation_frame()
        annotated = presentation_frame(elements=("annotation",))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([annotated.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, annotated))

    def test_presentation_annotation_deletion_preserves_before_and_after(self):
        fps = 2.0
        before = dense_presentation_frame(annotation=True)
        after = dense_presentation_frame(annotation=False)
        relation = presentation_change_geometry(before, after)
        # This intentionally isolates the edge-gain gate: the dense slide keeps
        # at least 96% of its old edges, and the deleted note's absolute edge
        # loss stays under 0.4% of the ROI, but deletion adds no new structure.
        self.assertGreaterEqual(relation.retained_edge_ratio, 0.96)
        self.assertLessEqual(relation.edge_loss_ratio, 0.004)
        self.assertLess(relation.edge_gain_ratio, 0.001)
        self.assertGreaterEqual(relation.changed_ratio, 0.003)

        frames = [before.copy() for _ in range(6)]
        frames.extend([after.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, before))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, after))
        self.assertIn("presentation:pre_replacement", candidates[0].reasons)

    def test_presentation_global_repeat_dedup_folds_return_to_prior_page(self):
        fps = 2.0
        first = presentation_frame(elements=("old_panel",))
        second = presentation_frame(elements=("new_panel",))
        frames = [first.copy() for _ in range(6)]
        frames.extend([second.copy() for _ in range(12)])
        frames.extend([first.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, first))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, second))

    def test_presentation_stable_bounded_tail_does_not_need_merge_dwell(self):
        fps = 2.0
        initial = presentation_frame()
        final = presentation_frame(elements=("bullet_1",))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([final.copy() for _ in range(3)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertIn("presentation:bounded_tail_stable", candidates[0].reasons)

    def test_presentation_unresolved_tail_never_emits_transition_frame(self):
        fps = 2.0
        initial = presentation_frame()
        frames = [initial.copy() for _ in range(6)]
        transitional_frames = []
        for width in (30, 60, 90, 120):
            transition = initial.copy()
            transition[42:46, 30 : 30 + width] = 28
            transitional_frames.append(transition)
        frames.extend(transitional_frames)
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))
        self.assertFalse(
            any(
                np.array_equal(samples[item.sample_index].gray, transitional)
                for item in candidates
                for transitional in transitional_frames
            )
        )

    def test_presentation_lookahead_confirms_pre_end_stable_state(self):
        fps = 2.0
        initial = presentation_frame()
        final = presentation_frame(elements=("bullet_1",))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([final.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].timestamp, 4.0)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, final))
        self.assertTrue(all(item.timestamp <= 4.0 for item in candidates))

    def test_presentation_lookahead_change_invalidates_pending_state(self):
        fps = 2.0
        initial = presentation_frame(elements=("old_panel",))
        first_replacement = presentation_frame(elements=("new_panel",))
        later_replacement = presentation_frame(elements=("bullet_1",))
        frames = [initial.copy() for _ in range(6)]
        frames.extend([first_replacement.copy() for _ in range(4)])
        frames.extend([later_replacement.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))

    def test_presentation_does_not_start_event_after_candidate_end(self):
        fps = 2.0
        initial = presentation_frame(elements=("old_panel",))
        later_page = presentation_frame(page=2, elements=("bullet_1",))
        frames = [initial.copy() for _ in range(10)]
        # The meaningful page event begins at 5.0s, entirely after the 4.0s
        # candidate boundary, despite having ample later samples to settle.
        frames.extend([later_page.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=4.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))
        self.assertTrue(all(item.timestamp <= 4.0 for item in candidates))

    def test_presentation_does_not_promote_state_first_stable_after_end(self):
        fps = 2.0
        initial = presentation_frame(elements=("old_panel",))
        replacement = presentation_frame(elements=("new_panel",))
        frames = [initial.copy() for _ in range(6)]
        # The event starts at 3.0s, but its two-sample stable run completes at
        # 4.0s, after the 3.5s candidate boundary.
        frames.extend([replacement.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(
            samples,
            analysis_fps=fps,
            candidate_end_timestamp=3.5,
        )

        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, initial))
        self.assertTrue(all(item.timestamp <= 3.5 for item in candidates))

    def test_presentation_title_blank_intermediate_is_subsumed_by_final_page(self):
        fps = 2.0
        first_page = presentation_frame(elements=("old_panel",))
        blank_title = presentation_frame(page=2)
        final_page = presentation_frame(page=2, elements=("bullet_1", "bullet_2"))
        frames = [first_page.copy() for _ in range(6)]
        frames.extend([blank_title.copy() for _ in range(6)])
        frames.extend([final_page.copy() for _ in range(12)])
        samples = build_samples(frame_stream(frames, fps=fps))

        candidates = select_presentation_states(samples, analysis_fps=fps)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(np.array_equal(samples[candidates[0].sample_index].gray, first_page))
        self.assertTrue(np.array_equal(samples[candidates[1].sample_index].gray, final_page))
        self.assertFalse(
            any(np.array_equal(samples[item.sample_index].gray, blank_title) for item in candidates)
        )

    def test_dedup_drops_identical_but_preserves_local_edit(self):
        base = np.zeros((96, 160), dtype=np.uint8)
        local_edit = base.copy()
        local_edit[16:40, 32:64] = 255
        samples = build_samples(frame_stream([base, base.copy(), local_edit], fps=1.0))
        candidates = [
            Candidate(0, 0.0, 0.1, ["density_floor"]),
            Candidate(1, 1.0, 0.1, ["density_floor"]),
            Candidate(2, 2.0, 0.5, ["settled_after_change"]),
        ]
        kept, dropped = deduplicate(samples, candidates)
        # Equal-priority duplicates intentionally retain the later settled view.
        self.assertEqual([item.sample_index for item in kept], [1, 2])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].candidate.sample_index, 0)


if __name__ == "__main__":
    unittest.main()
