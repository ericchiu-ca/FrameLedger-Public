from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from frameledger import pipeline
from frameledger.features import RoutingFrameFeatures
from frameledger.models import Candidate, ContentSegment, VideoMetadata
from frameledger.routing import VisualRouteDecision, VisualRoutePlan


class RoutedPipelineTests(unittest.TestCase):
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
        timestamps = np.arange(start, end + 1e-9, 1.0)
        frame = np.full((120, 200), 180, dtype=np.uint8)
        frames = [
            (float(timestamp), int(round(timestamp * 30)), frame.copy())
            for timestamp in timestamps
        ]
        return frames, {}

    @staticmethod
    def _route_plan(samples, *, kind: str) -> VisualRoutePlan:
        selector = {
            "presentation": "presentation_states",
            "table": "table_viewport",
            "chart": "boundary_terminal",
            "unknown": None,
        }[kind]
        segment = ContentSegment(
            segment_id="route-001",
            kind=kind,
            start_sample_index=0,
            stop_sample_index=len(samples),
            start_timestamp=samples[0].timestamp,
            candidate_end_timestamp=samples[-1].timestamp,
            confidence=0.9,
            selector=selector,
            reasons=("test-route",),
        )
        features = RoutingFrameFeatures(
            mean_luma=180.0,
            dark_ratio=0.0,
            bright_ratio=0.0,
            edge_density=0.01,
            horizontal_line_ratio=0.0,
            vertical_line_ratio=0.0,
        )
        decisions = tuple(
            VisualRouteDecision(
                sample_index=index,
                timestamp=sample.timestamp,
                raw_kind=kind,
                kind=kind,
                confidence=0.9,
                scores={"presentation": 0.0, "table": 0.0, "chart": 0.0},
                reasons=("test-route",),
                features=features,
            )
            for index, sample in enumerate(samples)
        )
        return VisualRoutePlan(
            analysis_fps=1.0,
            candidate_end_timestamp=samples[-1].timestamp,
            bridge_seconds=1.5,
            minimum_known_seconds=4.0,
            eligible_sample_count=len(samples),
            decisions=decisions,
            segments=(segment,),
        )

    def _media_patches(
        self,
        source: Path,
        *,
        detect_side_effect,
        select_side_effect,
    ):
        metadata = self._metadata(source)
        return (
            patch.object(pipeline, "probe_video", return_value=metadata),
            patch.object(
                pipeline,
                "_decode_analysis_and_scenes",
                side_effect=self._decoded_samples,
            ),
            patch.object(
                pipeline,
                "detect_content_segments",
                side_effect=detect_side_effect,
                create=True,
            ),
            patch.object(
                pipeline,
                "select_routed_states",
                side_effect=select_side_effect,
                create=True,
            ),
            patch.object(
                pipeline,
                "read_frame_at",
                return_value=np.zeros((120, 200, 3), dtype=np.uint8),
            ),
            patch.object(pipeline, "write_jpeg"),
            patch.object(pipeline, "write_png"),
            patch.object(pipeline, "write_contact_sheet"),
            patch.object(pipeline, "write_review_html"),
            patch.object(pipeline, "_environment", return_value={"test": "mocked"}),
        )

    def test_routed_rejects_sub_one_fps_before_probe_or_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            with patch.object(pipeline, "probe_video") as probe:
                with self.assertRaisesRegex(ValueError, "(?i)rout.*fps|fps.*rout"):
                    pipeline.run_benchmark(
                        source,
                        output=output,
                        start_seconds=10.0,
                        duration_seconds=10.0,
                        strategies=["routed"],
                        analysis_fps=0.5,
                    )

            probe.assert_not_called()
            self.assertFalse(output.exists())

    def test_routed_uses_maximum_lookahead_and_preserves_global_indexes_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            def detect(eligible_samples, **kwargs):
                self.assertEqual(kwargs["analysis_fps"], 1.0)
                self.assertEqual(kwargs["candidate_end_timestamp"], 20.0)
                self.assertEqual(len(eligible_samples), 11)
                self.assertEqual(eligible_samples[0].timestamp, 10.0)
                self.assertEqual(eligible_samples[-1].timestamp, 20.0)
                return self._route_plan(eligible_samples, kind="presentation")

            def select(all_samples, segments, **kwargs):
                self.assertEqual(len(all_samples), 26)
                self.assertEqual(all_samples[0].timestamp, 10.0)
                self.assertEqual(all_samples[-1].timestamp, 35.0)
                self.assertEqual(segments[0].stop_sample_index, 11)
                self.assertEqual(kwargs["analysis_fps"], 1.0)
                self.assertEqual(kwargs["boundary_lookahead_seconds"], 15.0)
                self.assertEqual(kwargs["table_lookahead_seconds"], 8.0)
                self.assertEqual(kwargs["presentation_lookahead_seconds"], 6.0)
                return [
                    Candidate(
                        sample_index=7,
                        timestamp=all_samples[7].timestamp,
                        score=0.8,
                        reasons=["mock-routed"],
                        segment_id="route-001",
                    )
                ]

            (
                probe,
                decode,
                detector,
                selector,
                read_frame,
                write_jpeg,
                write_png,
                write_contact_sheet,
                write_review_html,
                environment,
            ) = self._media_patches(
                source,
                detect_side_effect=detect,
                select_side_effect=select,
            )
            with (
                probe,
                decode as decode_mock,
                detector as detector_mock,
                selector as selector_mock,
                read_frame as read_frame_mock,
                write_jpeg,
                write_png,
                write_contact_sheet,
                write_review_html,
                environment,
            ):
                result = pipeline.run_benchmark(
                    source,
                    output=output,
                    start_seconds=10.0,
                    duration_seconds=10.0,
                    strategies=["routed"],
                    analysis_fps=1.0,
                    boundary_lookahead_seconds=15.0,
                    table_lookahead_seconds=8.0,
                    presentation_lookahead_seconds=6.0,
                )

            self.assertEqual(decode_mock.call_args.kwargs["end_seconds"], 35.0)
            detector_mock.assert_called_once()
            selector_mock.assert_called_once()
            read_frame_mock.assert_called_once()
            self.assertEqual(read_frame_mock.call_args.args[1], 17.0)

            routed = result["strategies"]["routed"]
            self.assertEqual(routed["selected"][0]["sample_index"], 7)
            self.assertEqual(routed["selected"][0]["timestamp"], 17.0)
            self.assertEqual(routed["selected"][0]["segment_id"], "route-001")
            self.assertEqual(routed["segments"][0]["segment_id"], "route-001")

            routing_signals = json.loads(
                (output / "routing-signals.json").read_text(encoding="utf-8")
            )
            self.assertEqual(routing_signals["eligible_sample_count"], 11)
            self.assertEqual(
                routing_signals["segments"][0]["segment_id"],
                "route-001",
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["range"]["analysis_end_seconds"], 35.0)
            self.assertEqual(manifest["routing"]["signals_path"], "routing-signals.json")
            self.assertEqual(
                manifest["routing"]["segments"][0]["segment_id"],
                "route-001",
            )

    def test_unknown_segment_remains_empty_without_pipeline_fallback_frame(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            plan_holder = {}

            def detect(eligible_samples, **_kwargs):
                plan = self._route_plan(eligible_samples, kind="unknown")
                plan_holder["plan"] = plan
                return plan

            def select(_all_samples, segments, **_kwargs):
                self.assertEqual(tuple(segments), plan_holder["plan"].segments)
                self.assertEqual(segments[0].kind, "unknown")
                return []

            patches = self._media_patches(
                source,
                detect_side_effect=detect,
                select_side_effect=select,
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4] as read_frame,
                patches[5],
                patches[6],
                patches[7],
                patches[8],
                patches[9],
            ):
                result = pipeline.run_benchmark(
                    source,
                    output=output,
                    start_seconds=10.0,
                    duration_seconds=10.0,
                    strategies=["routed"],
                    analysis_fps=1.0,
                )

            self.assertEqual(result["strategies"]["routed"]["selected"], [])
            self.assertEqual(
                result["strategies"]["routed"]["segments"][0]["kind"],
                "unknown",
            )
            read_frame.assert_not_called()

    def test_routed_candidate_overflow_raises_instead_of_silent_temporal_cap(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.mp4"
            source.write_bytes(b"test video placeholder")
            output = root / "output"

            def detect(eligible_samples, **_kwargs):
                return self._route_plan(eligible_samples, kind="presentation")

            def select(all_samples, _segments, **_kwargs):
                return [
                    Candidate(
                        sample_index=index,
                        timestamp=all_samples[index].timestamp,
                        score=0.8,
                        reasons=["mock-routed"],
                        segment_id="route-001",
                    )
                    for index in (1, 4, 7)
                ]

            patches = self._media_patches(
                source,
                detect_side_effect=detect,
                select_side_effect=select,
            )
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                patches[6],
                patches[7],
                patches[8],
                patches[9],
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "(?i)(routed.*(maximum|max)|maximum.*routed|max.*routed)",
                ):
                    pipeline.run_benchmark(
                        source,
                        output=output,
                        start_seconds=10.0,
                        duration_seconds=10.0,
                        strategies=["routed"],
                        analysis_fps=1.0,
                        maximum_frames=2,
                    )

            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
