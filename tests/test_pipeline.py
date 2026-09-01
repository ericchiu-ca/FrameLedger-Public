from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from frameledger import pipeline
from frameledger.models import VideoMetadata


class PipelinePresentationTests(unittest.TestCase):
    @staticmethod
    def _metadata(source: Path, *, duration_seconds: float = 100.0) -> VideoMetadata:
        stat = source.stat()
        return VideoMetadata(
            path=source,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            duration_seconds=duration_seconds,
            fps=30.0,
            width=200,
            height=120,
            frame_count=int(duration_seconds * 30),
            codec="test",
            fingerprint="f" * 64,
        )

    @staticmethod
    def _decoded_samples(*_args, **kwargs):
        start = float(kwargs["start_seconds"])
        end = float(kwargs["end_seconds"])
        timestamps = np.arange(start, end + 1e-9, 0.5)
        frame = np.full((120, 200), 240, dtype=np.uint8)
        frames = [
            (float(timestamp), int(round(timestamp * 30)), frame.copy())
            for timestamp in timestamps
        ]
        return frames, {}

    def _run_with_mocked_media(
        self,
        source: Path,
        output: Path,
        **kwargs,
    ):
        metadata = self._metadata(source)
        with (
            patch.object(pipeline, "probe_video", return_value=metadata),
            patch.object(
                pipeline,
                "_decode_analysis_and_scenes",
                side_effect=self._decoded_samples,
            ) as decode,
            patch.object(
                pipeline,
                "select_presentation_states",
                wraps=pipeline.select_presentation_states,
            ) as presentation_selector,
            patch.object(pipeline, "read_frame_at", return_value=np.zeros((120, 200, 3))),
            patch.object(pipeline, "write_jpeg"),
            patch.object(pipeline, "write_png"),
            patch.object(pipeline, "write_contact_sheet"),
            patch.object(pipeline, "write_review_html"),
            patch.object(pipeline, "_environment", return_value={"test": "mocked"}),
        ):
            result = pipeline.run_benchmark(source, output=output, **kwargs)
        return result, decode, presentation_selector

    def test_presentation_uses_six_second_lookahead_without_post_end_candidates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            result, decode, selector = self._run_with_mocked_media(
                source,
                output,
                start_seconds=10.0,
                duration_seconds=10.0,
                strategies=["presentation_states"],
                analysis_fps=2.0,
            )

            self.assertEqual(decode.call_args.kwargs["end_seconds"], 26.0)
            selector_samples = selector.call_args.args[0]
            self.assertEqual(selector.call_args.kwargs["candidate_end_timestamp"], 20.0)
            self.assertEqual(selector_samples[-1].timestamp, 26.0)
            self.assertGreater(selector_samples[-1].timestamp, 20.0)

            selected = result["strategies"]["presentation_states"]["selected"]
            self.assertTrue(selected)
            self.assertTrue(all(candidate["timestamp"] <= 20.0 for candidate in selected))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["range"]["end_seconds"], 20.0)
            self.assertEqual(manifest["range"]["analysis_end_seconds"], 26.0)
            self.assertEqual(
                manifest["parameters"]["presentation_lookahead_seconds"],
                6.0,
            )

    def test_combined_strategies_decode_to_largest_lookahead(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            with (
                patch.object(pipeline, "select_boundary_terminal_states", return_value=[]),
                patch.object(pipeline, "select_table_viewport_states", return_value=[]),
            ):
                result, decode, _ = self._run_with_mocked_media(
                    source,
                    output,
                    start_seconds=10.0,
                    duration_seconds=10.0,
                    strategies=[
                        "presentation_states",
                        "table_viewport",
                        "boundary_terminal",
                    ],
                    analysis_fps=2.0,
                    presentation_lookahead_seconds=6.0,
                    table_lookahead_seconds=8.0,
                    boundary_lookahead_seconds=4.0,
                )

            self.assertEqual(decode.call_args.kwargs["end_seconds"], 28.0)
            self.assertEqual(result["range"]["analysis_end_seconds"], 28.0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["range"]["analysis_end_seconds"], 28.0)

    def test_presentation_rejects_sub_one_fps_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            with patch.object(pipeline, "probe_video") as probe:
                with self.assertRaises(ValueError):
                    pipeline.run_benchmark(
                        source,
                        output=output,
                        start_seconds=10.0,
                        duration_seconds=10.0,
                        strategies=["presentation_states"],
                        analysis_fps=0.5,
                    )

            probe.assert_not_called()
            self.assertFalse(output.exists())

    def test_presentation_rejects_nan_lookahead_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            with patch.object(pipeline, "probe_video") as probe:
                with self.assertRaises(ValueError):
                    pipeline.run_benchmark(
                        source,
                        output=output,
                        start_seconds=10.0,
                        duration_seconds=10.0,
                        strategies=["presentation_states"],
                        analysis_fps=2.0,
                        presentation_lookahead_seconds=float("nan"),
                    )

            probe.assert_not_called()
            self.assertFalse(output.exists())


class CuratedExportAnnotationStatusTests(unittest.TestCase):
    @staticmethod
    def _metadata(source: Path) -> VideoMetadata:
        stat = source.stat()
        return VideoMetadata(
            path=source,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            duration_seconds=20.0,
            fps=20.0,
            width=200,
            height=120,
            frame_count=400,
            codec="test",
            fingerprint="f" * 64,
        )

    def _export(self, *, exhaustive: bool) -> dict:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        source = root / "source.mp4"
        source.write_bytes(b"test video placeholder")
        output = root / "output"
        annotation_path = root / "annotations.yaml"
        review = {
            "coverage": "positive_only",
            "scope": "full_segment",
            "selection_policy": "terminal_states_only",
        }
        schema_version = 3
        if exhaustive:
            schema_version = 4
            review.update(
                {
                    "coverage": "exhaustive",
                    "unlisted_states": "reject",
                    "source_review": {
                        "basis": "source_video",
                        "full_segment_watched": True,
                        "algorithm_outputs_used_as_labels": False,
                    },
                    "confirmation": {
                        "confirmed_by": "user",
                        "labels_complete": True,
                        "plateau_windows_confirmed": True,
                    },
                }
            )
        annotation_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": schema_version,
                    "review_status": "human_reviewed",
                    "review": review,
                    "video": {"basename": source.name},
                    "source_sha256": "f" * 64,
                    "segment": {"start": "00:00:00", "duration": "00:00:10"},
                    "must_keep": [
                        {
                            "timestamp": "00:00:05",
                            "acceptable_start": "00:00:04",
                            "acceptable_end": "00:00:05",
                            "kind": "chart_terminal",
                        }
                    ],
                    "nice_to_keep": [],
                    "avoid_duplicates": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        metadata = self._metadata(source)
        with (
            patch.object(pipeline, "probe_video", return_value=metadata),
            patch.object(
                pipeline,
                "read_frame_at",
                return_value=np.zeros((120, 200, 3), dtype=np.uint8),
            ),
            patch.object(pipeline, "write_png"),
            patch.object(pipeline, "write_contact_sheet"),
            patch.object(pipeline, "write_review_html"),
        ):
            return pipeline.export_reviewed_frames(
                source,
                annotations_path=annotation_path,
                output=output,
            )

    def test_curated_export_reports_exhaustive_ground_truth(self):
        result = self._export(exhaustive=True)

        self.assertEqual(result["ground_truth_status"], "human_reviewed_full_segment")
        self.assertTrue(result["source_segment_recall_assessable"])

    def test_curated_export_preserves_positive_only_status(self):
        result = self._export(exhaustive=False)

        self.assertEqual(
            result["ground_truth_status"],
            "human_reviewed_positive_labels_not_exhaustive",
        )
        self.assertFalse(result["source_segment_recall_assessable"])

if __name__ == "__main__":
    unittest.main()
