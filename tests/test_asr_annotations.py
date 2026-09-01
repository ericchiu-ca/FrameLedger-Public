from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from frameledger.asr_annotations import evaluate_asr_anchor_set
from frameledger.cli import build_parser


SOURCE_SHA = "a" * 64


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _asr(segments):
    return {
        "status": "ok",
        "source": {"fingerprint": SOURCE_SHA},
        "range": {
            "start_seconds": 120.0,
            "end_seconds": 130.0,
            "duration_seconds": 10.0,
        },
        "parameters": {"initial_prompt": "头寸平仓"},
        "engine": {"name": "test"},
        "transcript": {
            "text": "".join(segment["text"] for segment in segments),
            "segments": segments,
        },
    }


def _segment(segment_id, start, end, text, compression=1.2):
    return {
        "id": segment_id,
        "relative_start_seconds": start - 120.0,
        "relative_end_seconds": end - 120.0,
        "absolute_start_seconds": start,
        "absolute_end_seconds": end,
        "text": text,
        "compression_ratio": compression,
    }


class AsrAnnotationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        baseline_path = root / "outputs" / "baseline" / "asr.json"
        baseline_segment = _segment(4, 122.0, 124.0, "把你的投顺平仓掉")
        related_segment = _segment(5, 126.0, 128.0, "没有看到介入股票")
        _write_json(baseline_path, _asr([baseline_segment, related_segment]))
        baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        annotation_path = root / "annotations" / "anchor.yaml"
        annotation_path.parent.mkdir(parents=True)
        annotation = {
            "schema_version": 1,
            "kind": "asr_anchor_set",
            "review_status": "human_reviewed",
            "coverage": "reported_issue_only",
            "source": {
                "video_sha256": SOURCE_SHA,
                "range": {
                    "start_seconds": 120.0,
                    "end_seconds": 130.0,
                    "duration_seconds": 10.0,
                },
            },
            "baseline": {
                "asr_output": "outputs/baseline/asr.json",
                "asr_sha256": baseline_sha,
            },
            "review": {"assertions_complete_for_clip": False},
            "anchors": [
                {
                    "id": "head-position",
                    "segment_id": 4,
                    "relative_start_seconds": 2.0,
                    "relative_end_seconds": 4.0,
                    "absolute_start_seconds": 122.0,
                    "absolute_end_seconds": 124.0,
                    "baseline_text": "把你的投顺平仓掉",
                    "baseline_span": "投顺平仓",
                    "expected_span": "头寸平仓",
                }
            ],
            "related_candidates": [
                {
                    "id": "borrow-stock-related",
                    "segment_id": 5,
                    "relative_start_seconds": 6.0,
                    "relative_end_seconds": 8.0,
                    "absolute_start_seconds": 126.0,
                    "absolute_end_seconds": 128.0,
                    "baseline_text": "没有看到介入股票",
                    "baseline_span": "介入股票",
                    "candidate_expected_span": "借入股票",
                }
            ],
        }
        annotation_path.write_text(
            yaml.safe_dump(annotation, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return baseline_path, annotation_path

    def test_scores_confirmed_and_related_windows_without_claiming_cer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, annotation_path = self._fixture(root)
            candidate_path = root / "candidate.json"
            _write_json(
                candidate_path,
                _asr(
                    [
                        _segment(10, 121.9, 124.1, "把你的头寸平仓掉", 1.4),
                        _segment(11, 125.9, 128.1, "没有看到借入股票", 1.3),
                    ]
                ),
            )
            output = root / "evaluation.json"
            result = evaluate_asr_anchor_set(
                candidate_path,
                annotations_path=annotation_path,
                output=output,
            )
            self.assertEqual(result["summary"]["expected_exact_hit_count"], 1)
            self.assertEqual(result["summary"]["related_candidate_exact_hit_count"], 1)
            self.assertEqual(result["summary"]["off_anchor_expected_occurrence_count"], 0)
            self.assertEqual(result["method"]["claim_limit"], "reported anchors only; not transcript accuracy or CER")
            self.assertEqual(result["quality_checks"]["max_segment_compression_ratio"], 1.4)
            self.assertFalse(result["quality_checks"]["pathological_repetition_detected"])
            self.assertTrue(output.is_file())

    def test_rejects_a_tampered_baseline_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path, annotation_path = self._fixture(root)
            baseline_path.write_text("{}", encoding="utf-8")
            candidate_path = root / "candidate.json"
            _write_json(candidate_path, _asr([]))
            with self.assertRaisesRegex(ValueError, "Baseline ASR SHA-256"):
                evaluate_asr_anchor_set(
                    candidate_path,
                    annotations_path=annotation_path,
                )

    def test_rejects_a_mismatched_anchor_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, annotation_path = self._fixture(root)
            annotation = yaml.safe_load(annotation_path.read_text(encoding="utf-8"))
            annotation["anchors"][0]["baseline_text"] = "wrong"
            annotation_path.write_text(
                yaml.safe_dump(annotation, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            candidate_path = root / "candidate.json"
            _write_json(candidate_path, _asr([]))
            with self.assertRaisesRegex(ValueError, "baseline segment text"):
                evaluate_asr_anchor_set(
                    candidate_path,
                    annotations_path=annotation_path,
                )

    def test_cli_exposes_reported_issue_asr_evaluation(self):
        args = build_parser().parse_args(
            [
                "asr-evaluate",
                "/tmp/candidate.json",
                "--annotations",
                "/tmp/anchors.yaml",
                "--output",
                "/tmp/evaluation.json",
            ]
        )
        self.assertEqual(args.command, "asr-evaluate")
        self.assertEqual(args.tolerance, 0.75)
