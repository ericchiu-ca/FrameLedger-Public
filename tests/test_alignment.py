from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from frameledger.alignment import run_evidence_alignment
from frameledger.cli import main


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class AlignmentTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        phase1 = root / "phase1"
        frames = phase1 / "frames"
        strategies = phase1 / "strategies"
        frames.mkdir(parents=True)
        strategies.mkdir()
        image1 = frames / "frame_000000020000.png"
        image2 = frames / "frame_000000080000.png"
        image1.write_bytes(b"png-one")
        image2.write_bytes(b"png-two")
        strategy = {
            "strategy": "presentation_states",
            "selected_count": 2,
            "selected": [
                {
                    "sample_index": 1,
                    "timestamp": 20.0,
                    "score": 0.3,
                    "reasons": ["presentation:initial_stable"],
                    "image_path": "frames/frame_000000020000.png",
                },
                {
                    "sample_index": 2,
                    "timestamp": 80.0,
                    "score": 0.4,
                    "reasons": ["presentation:page_terminal"],
                    "image_path": "frames/frame_000000080000.png",
                },
            ],
        }
        strategy_path = strategies / "presentation_states.json"
        _write_json(strategy_path, strategy)
        video_sha = "a" * 64
        manifest = {
            "kind": "candidate_frame_benchmark",
            "source": {
                "path": str(root / "source" / "video.mp4"),
                "fingerprint": video_sha,
            },
            "range": {
                "start_seconds": 10.0,
                "end_seconds": 100.0,
                "duration_seconds": 90.0,
            },
            "strategies": {"presentation_states": strategy},
        }
        manifest_path = phase1 / "manifest.json"
        _write_json(manifest_path, manifest)

        def ocr_frame(
            sample_index: int,
            timestamp: float,
            image_path: str,
            image_sha: str,
            text: str,
        ) -> dict[str, object]:
            return {
                "sample_index": sample_index,
                "timestamp": timestamp,
                "segment_id": None,
                "route_kind": "presentation",
                "image_path": image_path,
                "image_sha256": image_sha,
                "roi_normalized": [0.04, 0.02, 0.94, 0.92],
                "roi_origin": "top_left_normalized",
                "status": "ok",
                "observations": [
                    {
                        "order": 0,
                        "text": text,
                        "confidence": 1.0,
                        "bbox": [0.1, 0.1, 0.5, 0.1],
                        "bbox_origin": "top_left_normalized",
                    }
                ],
                "plain_text": text,
            }

        ocr = {
            "schema_version": 1,
            "kind": "frame_ocr",
            "source": {
                "run_directory": str(phase1.resolve()),
                "manifest_path": "manifest.json",
                "manifest_sha256": _sha256(manifest_path),
                "video_sha256": video_sha,
                "strategy": "presentation_states",
                "strategy_path": "strategies/presentation_states.json",
                "strategy_sha256": _sha256(strategy_path),
            },
            "parameters": {"policy_version": "route-roi-v2"},
            "frames": [
                ocr_frame(
                    1,
                    20.0,
                    "frames/frame_000000020000.png",
                    _sha256(image1),
                    "第一张",
                ),
                ocr_frame(
                    2,
                    80.0,
                    "frames/frame_000000080000.png",
                    _sha256(image2),
                    "第二张",
                ),
            ],
            "skipped": [],
            "failures": [],
            "summary": {
                "selected_input_frames": 2,
                "ocr_success_frames": 2,
                "failure_frames": 0,
                "skipped_frames": 0,
                "observation_count": 2,
            },
        }
        ocr_path = root / "ocr" / "ocr.json"
        _write_json(ocr_path, ocr)

        audio_path = root / "asr" / "audio.wav"
        audio_path.parent.mkdir()
        audio_path.write_bytes(b"bounded-audio")
        segments = [
            {
                "id": 0,
                "relative_start_seconds": 8.0,
                "relative_end_seconds": 12.0,
                "absolute_start_seconds": 18.0,
                "absolute_end_seconds": 22.0,
                "text": "覆盖第一张",
                "words": [],
            },
            {
                "id": 1,
                "relative_start_seconds": 20.0,
                "relative_end_seconds": 25.0,
                "absolute_start_seconds": 30.0,
                "absolute_end_seconds": 35.0,
                "text": "后续语音",
                "words": [],
            },
        ]
        asr = {
            "schema_version": 1,
            "kind": "local_speech_transcription",
            "status": "ok",
            "source": {"fingerprint": video_sha},
            "range": {
                "start_seconds": 10.0,
                "end_seconds": 50.0,
                "duration_seconds": 40.0,
            },
            "parameters": {
                "model_id": "mlx-community/whisper-large-v3-turbo",
                "model_revision": "b" * 40,
            },
            "audio": {"path": "audio.wav", "sha256": _sha256(audio_path)},
            "summary": {"quality_checks": {"passed": True}},
            "transcript": {
                "segment_count": 2,
                "word_count": 0,
                "text": "覆盖第一张后续语音",
                "segments": segments,
            },
        }
        asr_path = root / "asr" / "asr.json"
        _write_json(asr_path, asr)
        return {
            "phase1": phase1,
            "manifest": manifest_path,
            "strategy": strategy_path,
            "image1": image1,
            "image2": image2,
            "ocr": ocr_path,
            "asr": asr_path,
            "audio": audio_path,
        }

    def _run(self, paths: dict[str, Path], output: Path) -> dict[str, object]:
        return run_evidence_alignment(
            paths["phase1"],
            strategy="presentation_states",
            ocr_json=paths["ocr"],
            asr_json=paths["asr"],
            context_before_seconds=5.0,
            context_after_seconds=5.0,
            output=output,
        )

    def test_alignment_writes_bound_timeline_and_offline_review(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            result = self._run(paths, root / "aligned")
            document = json.loads(Path(result["evidence_json"]).read_text(encoding="utf-8"))

            self.assertEqual(document["kind"], "timestamp_aligned_evidence")
            self.assertEqual(document["summary"]["visual_frame_count"], 2)
            self.assertEqual(document["summary"]["speech_segment_count"], 2)
            self.assertEqual(document["summary"]["temporal_link_count"], 1)
            self.assertEqual(
                document["visual_frames"][0]["temporal_alignment"]["status"],
                "segment_at_timestamp",
            )
            self.assertEqual(
                document["visual_frames"][1]["temporal_alignment"]["status"],
                "outside_asr_range",
            )
            self.assertEqual(document["temporal_links"][0]["speech_event_id"], "speech-0001")
            self.assertFalse(document["parameters"]["semantic_match_claimed"])
            review = Path(result["review_html"]).read_text(encoding="utf-8")
            self.assertIn("第一张", review)
            self.assertIn("覆盖第一张", review)
            self.assertIn("outside_asr_range", review)
            self.assertIn("data-time='18.000000'", review)
            self.assertNotIn("https://", review)

    def test_skipped_ocr_frame_is_preserved_as_visual_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            ocr = json.loads(paths["ocr"].read_text(encoding="utf-8"))
            skipped = ocr["frames"].pop()
            skipped["status"] = "skipped"
            skipped["reason"] = "skipped_by_route_policy"
            skipped.pop("observations")
            skipped.pop("plain_text")
            ocr["skipped"].append(skipped)
            _write_json(paths["ocr"], ocr)

            result = self._run(paths, root / "aligned")
            document = json.loads(Path(result["evidence_json"]).read_text(encoding="utf-8"))
            self.assertEqual(document["visual_frames"][1]["ocr"]["status"], "skipped")
            self.assertEqual(document["summary"]["ocr_ok_frame_count"], 1)

    def test_tampered_phase1_image_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            paths["image1"].write_bytes(b"tampered")
            output = root / "aligned"
            with self.assertRaisesRegex(ValueError, "image SHA-256"):
                self._run(paths, output)
            self.assertFalse(output.exists())

    def test_source_and_quality_mismatches_fail_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            asr = json.loads(paths["asr"].read_text(encoding="utf-8"))
            asr["source"]["fingerprint"] = "c" * 64
            _write_json(paths["asr"], asr)
            with self.assertRaisesRegex(ValueError, "ASR video SHA-256"):
                self._run(paths, root / "wrong-source")

            paths = self._fixture(root / "quality")
            asr = json.loads(paths["asr"].read_text(encoding="utf-8"))
            asr["summary"]["quality_checks"]["passed"] = False
            _write_json(paths["asr"], asr)
            with self.assertRaisesRegex(ValueError, "quality gate"):
                self._run(paths, root / "bad-quality")

    def test_cli_summary_and_output_path_guards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "align",
                        str(paths["phase1"]),
                        "--strategy",
                        "presentation_states",
                        "--ocr-json",
                        str(paths["ocr"]),
                        "--asr-json",
                        str(paths["asr"]),
                        "--output",
                        str(root / "aligned"),
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["kind"], "timestamp_aligned_evidence")
            self.assertEqual(summary["alignment_summary"]["temporal_link_count"], 1)
            self.assertFalse(summary["coverage"]["complete_phase1_speech_coverage"])

            with self.assertRaisesRegex(ValueError, "outside Phase 1"):
                self._run(paths, paths["phase1"] / "nested-output")


if __name__ == "__main__":
    unittest.main()
