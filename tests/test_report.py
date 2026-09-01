from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frameledger.models import Candidate, ContentSegment, StrategyResult, VideoMetadata
from frameledger.report import write_review_html


class ReviewReportTests(unittest.TestCase):
    @staticmethod
    def _metadata(path: Path) -> VideoMetadata:
        return VideoMetadata(
            path=path,
            size_bytes=1,
            mtime_ns=1,
            duration_seconds=120.0,
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=3600,
            codec="test",
            fingerprint="f" * 64,
        )

    def test_legacy_review_has_no_routing_surface_without_segments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "review.html"
            candidate = Candidate(
                sample_index=1,
                timestamp=5.0,
                score=0.5,
                reasons=["test"],
                image_path="frames/example.png",
            )
            result = StrategyResult(
                name="uniform",
                raw_candidates=[candidate],
                selected=[candidate],
                dropped_duplicates=[],
            )

            write_review_html(
                output,
                self._metadata(root / "source.mp4"),
                {"uniform": result},
                start_seconds=0.0,
                end_seconds=10.0,
            )

            document = output.read_text(encoding="utf-8")
            self.assertIn("FrameLedger candidate-frame review", document)
            self.assertIn("00:00:05.000", document)
            self.assertNotIn("Content routing", document)
            self.assertNotIn("routing-signals.json", document)
            self.assertNotIn("segment-badge", document)

    def test_routed_review_shows_timeline_table_badges_and_empty_known_warning(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output = root / "review.html"
            segments = [
                ContentSegment(
                    segment_id="seg-p",
                    kind="presentation",
                    start_sample_index=0,
                    stop_sample_index=10,
                    start_timestamp=0.0,
                    candidate_end_timestamp=5.0,
                    confidence=0.91,
                    selector="presentation_states",
                ),
                ContentSegment(
                    segment_id="seg-t",
                    kind="table",
                    start_sample_index=10,
                    stop_sample_index=20,
                    start_timestamp=5.0,
                    candidate_end_timestamp=10.0,
                    confidence=0.82,
                    selector="table_viewport",
                ),
                ContentSegment(
                    segment_id="seg-u",
                    kind="unknown",
                    start_sample_index=20,
                    stop_sample_index=24,
                    start_timestamp=10.0,
                    candidate_end_timestamp=12.0,
                    confidence=0.44,
                    selector=None,
                ),
            ]
            candidate = Candidate(
                sample_index=4,
                timestamp=2.0,
                score=0.8,
                reasons=["presentation:bounded_tail_stable"],
                image_path="frames/presentation.png",
                segment_id="seg-p",
            )
            result = StrategyResult(
                name="routed_visual_states",
                raw_candidates=[candidate],
                selected=[candidate],
                dropped_duplicates=[],
                segments=segments,
            )

            write_review_html(
                output,
                self._metadata(root / "source.mp4"),
                {"routed_visual_states": result},
                start_seconds=0.0,
                end_seconds=12.0,
            )

            document = output.read_text(encoding="utf-8")
            self.assertIn("Content routing", document)
            self.assertIn("href='routing-signals.json'", document)
            self.assertIn("Content segment timeline", document)
            self.assertIn("route-kind-presentation", document)
            self.assertIn("route-kind-table", document)
            self.assertIn("route-kind-unknown", document)
            self.assertIn("background: #6b7280", document)
            self.assertIn("00:00:00.000", document)
            self.assertIn("00:00:05.000", document)
            self.assertIn("91.0%", document)
            self.assertIn("presentation_states", document)
            self.assertIn("table_viewport", document)
            self.assertIn("presentation · seg-p", document)

            warning_start = document.index("Known routed segments with no selected candidates")
            warning_end = document.index("</div>", warning_start)
            warning = document[warning_start:warning_end]
            self.assertIn("seg-t (table)", warning)
            self.assertNotIn("seg-u", warning)


if __name__ == "__main__":
    unittest.main()
