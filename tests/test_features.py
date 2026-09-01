import unittest

import numpy as np

from frameledger.features import (
    block_change,
    global_ssim,
    perceptual_hash,
    phash_distance,
    pixel_delta,
    presentation_change_geometry,
    worksheet_change_geometry,
)


class FeatureTests(unittest.TestCase):
    def test_identical_frames_are_identical(self):
        frame = np.full((96, 160), 127, dtype=np.uint8)
        self.assertAlmostEqual(global_ssim(frame, frame), 1.0, places=8)
        self.assertEqual(pixel_delta(frame, frame), 0.0)
        self.assertEqual(block_change(frame, frame), 0.0)
        self.assertEqual(phash_distance(perceptual_hash(frame), perceptual_hash(frame)), 0)

    def test_local_change_is_visible_to_block_metric(self):
        before = np.zeros((96, 160), dtype=np.uint8)
        after = before.copy()
        after[20:36, 40:60] = 255
        self.assertGreater(block_change(before, after), pixel_delta(before, after))

    def test_phash_distinguishes_structure(self):
        left = np.zeros((96, 160), dtype=np.uint8)
        left[:, :80] = 255
        right = np.zeros((96, 160), dtype=np.uint8)
        right[:48, :] = 255
        self.assertGreaterEqual(phash_distance(perceptual_hash(left), perceptual_hash(right)), 4)

    def test_worksheet_geometry_strips_narrow_wide_row_highlight(self):
        before = np.full((120, 200), 40, dtype=np.uint8)
        after = before.copy()
        # The default worksheet ROI is y=30..110. A two-row company focus
        # occupies a narrow band but spans essentially every visible column.
        after[60:70, :] = 100

        geometry = worksheet_change_geometry(before, after)

        self.assertGreater(geometry.highlight_changed_ratio, 0.10)
        self.assertGreaterEqual(geometry.highlight_column_coverage, 0.95)
        self.assertLess(geometry.residual_changed_ratio, 0.001)
        self.assertLess(geometry.residual_row_coverage, 0.01)

    def test_worksheet_geometry_keeps_widespread_viewport_change(self):
        before = np.full((120, 200), 40, dtype=np.uint8)
        after = before.copy()
        after[30:110, :] = 110

        geometry = worksheet_change_geometry(before, after)

        self.assertGreater(geometry.residual_changed_ratio, 0.95)
        self.assertGreater(geometry.residual_row_coverage, 0.95)
        self.assertGreater(geometry.residual_column_coverage, 0.95)

    def test_worksheet_geometry_ignores_chrome_above_roi(self):
        before = np.full((120, 200), 40, dtype=np.uint8)
        after = before.copy()
        after[:24, :] = 220

        geometry = worksheet_change_geometry(before, after)

        self.assertEqual(geometry.changed_ratio, 0.0)
        self.assertEqual(geometry.highlight_changed_ratio, 0.0)
        self.assertEqual(geometry.residual_changed_ratio, 0.0)

    def test_presentation_geometry_recognizes_additive_edges(self):
        before = np.full((120, 200), 240, dtype=np.uint8)
        before[12:16, 24:120] = 30
        after = before.copy()
        after[42:46, 30:150] = 30

        geometry = presentation_change_geometry(before, after)

        self.assertGreater(geometry.changed_ratio, 0.003)
        self.assertGreater(geometry.edge_gain_ratio, 0.0)
        self.assertGreaterEqual(geometry.retained_edge_ratio, 0.96)
        self.assertLessEqual(geometry.edge_loss_ratio, 0.004)

    def test_presentation_geometry_exposes_destructive_replacement(self):
        before = np.full((120, 200), 240, dtype=np.uint8)
        before[12:16, 24:120] = 30
        before[40:48, 28:170] = 30
        after = np.full((120, 200), 240, dtype=np.uint8)
        after[12:16, 24:120] = 30
        after[72:80, 28:170] = 30

        geometry = presentation_change_geometry(before, after)

        self.assertLess(geometry.retained_edge_ratio, 0.96)
        self.assertGreater(geometry.edge_loss_ratio, 0.004)
        self.assertGreater(geometry.edge_gain_ratio, 0.004)

    def test_presentation_geometry_ignores_player_chrome_outside_roi(self):
        before = np.full((120, 200), 240, dtype=np.uint8)
        after = before.copy()
        after[110:, :] = 20
        after[:, :6] = 20

        geometry = presentation_change_geometry(before, after)

        self.assertEqual(geometry.changed_ratio, 0.0)
        self.assertEqual(geometry.edge_changed_ratio, 0.0)
        self.assertEqual(geometry.block_spread, 0.0)

    def test_presentation_block_spread_rejects_one_tiny_local_change(self):
        before = np.full((120, 200), 240, dtype=np.uint8)
        after = before.copy()
        after[52:55, 92:95] = 20

        geometry = presentation_change_geometry(before, after)

        self.assertLess(geometry.block_spread, 0.03)


if __name__ == "__main__":
    unittest.main()
