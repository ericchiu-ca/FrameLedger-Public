from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from frameledger.semantic import run_semantic_segmentation


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SemanticSegmentationTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        duration_seconds: float = 600.0,
        speech_step_seconds: float = 40.0,
        repeated_ocr_title: bool = False,
    ) -> dict[str, Any]:
        source_directory = root / "source"
        source_directory.mkdir(parents=True)
        video_path = source_directory / "episode.mp4"
        video_path.write_bytes(b"immutable-source-video")
        video_sha256 = _sha256(video_path)

        visual_directory = root / "visual"
        frame_directory = visual_directory / "frames"
        strategy_directory = visual_directory / "strategies"
        frame_directory.mkdir(parents=True)
        strategy_directory.mkdir()

        starts: list[float] = []
        cursor = 0.0
        while cursor < duration_seconds:
            starts.append(round(cursor, 6))
            cursor += speech_step_seconds

        source_segments: list[dict[str, Any]] = []
        aligned_segments: list[dict[str, Any]] = []
        for index, start in enumerate(starts):
            end = min(duration_seconds, start + min(30.0, speech_step_seconds * 0.75))
            if index == 2:
                text = '</script><script>alert("semantic-xss")</script>'
            else:
                topic = (index * 3) // max(1, len(starts))
                topic_name = ("锤子线", "看涨抱线", "启明星")[min(topic, 2)]
                text = f"{topic_name} 原始语音片段 {index:02d}，保持原文。"
            source_segment = {
                "id": index,
                "relative_start_seconds": start,
                "relative_end_seconds": round(end, 6),
                "absolute_start_seconds": start,
                "absolute_end_seconds": round(end, 6),
                "text": text,
                "words": [],
            }
            source_segments.append(source_segment)
            aligned_segments.append(
                {
                    "event_id": f"speech-{index + 1:04d}",
                    "source_segment_id": index,
                    **source_segment,
                }
            )

        audio_directory = root / "asr"
        audio_directory.mkdir()
        audio_path = audio_directory / "audio.wav"
        audio_path.write_bytes(b"local-bounded-audio")
        asr_document = {
            "schema_version": 1,
            "kind": "local_speech_transcription",
            "status": "ok",
            "source": {
                "path": str(video_path),
                "fingerprint": video_sha256,
            },
            "range": {
                "start_seconds": 0.0,
                "end_seconds": duration_seconds,
                "duration_seconds": duration_seconds,
            },
            "parameters": {
                "model_id": "local-test-whisper",
                "model_revision": "b" * 40,
            },
            "audio": {
                "path": "audio.wav",
                "sha256": _sha256(audio_path),
                "duration_seconds": duration_seconds,
            },
            "summary": {
                "segment_count": len(source_segments),
                "quality_checks": {"passed": True},
            },
            "transcript": {
                "segment_count": len(source_segments),
                "text": "".join(segment["text"] for segment in source_segments),
                "segments": source_segments,
            },
        }
        asr_path = audio_directory / "asr.json"
        _write_json(asr_path, asr_document)

        visual_starts = starts[1::2] or starts[:1]
        speech_event_by_start = {
            float(segment["absolute_start_seconds"]): segment["event_id"]
            for segment in aligned_segments
        }
        selected: list[dict[str, Any]] = []
        ocr_frames: list[dict[str, Any]] = []
        visual_frames: list[dict[str, Any]] = []
        for index, timestamp in enumerate(visual_starts):
            relative_image = f"frames/frame_{index + 1:04d}.png"
            image_path = visual_directory / relative_image
            image_path.write_bytes(f"png-{index}".encode("utf-8"))
            if repeated_ocr_title:
                title = "完全相同的重复页面标题"
            elif index == 1:
                title = "<img src=x onerror=alert('ocr-xss')>"
            else:
                title = ("锤子线", "看涨抱线", "启明星")[
                    min((index * 3) // max(1, len(visual_starts)), 2)
                ]
            candidate = {
                "sample_index": index + 1,
                "timestamp": timestamp,
                "score": 0.5,
                "reasons": ["presentation:page_terminal"],
                "image_path": relative_image,
            }
            selected.append(candidate)
            ocr_record = {
                "sample_index": index + 1,
                "timestamp": timestamp,
                "segment_id": f"route-{index + 1:03d}",
                "route_kind": "presentation",
                "image_path": relative_image,
                "image_sha256": _sha256(image_path),
                "status": "ok",
                "observations": [
                    {
                        "order": 0,
                        "text": title,
                        "confidence": 1.0,
                        "bbox": [0.1, 0.1, 0.7, 0.1],
                        "bbox_origin": "top_left_normalized",
                    }
                ],
                "plain_text": title,
            }
            ocr_frames.append(ocr_record)
            visual_frames.append(
                {
                    "event_id": f"visual-{index + 1:04d}",
                    "sample_index": index + 1,
                    "timestamp": timestamp,
                    "timecode": self._timecode(timestamp),
                    "route_kind": "presentation",
                    "segment_id": f"route-{index + 1:03d}",
                    "image": {
                        "path": relative_image,
                        "sha256": _sha256(image_path),
                    },
                    "selection": {
                        "score": 0.5,
                        "reasons": ["presentation:page_terminal"],
                    },
                    "ocr": {
                        "status": "ok",
                        "observations": ocr_record["observations"],
                        "plain_text": title,
                    },
                    "temporal_alignment": {
                        "status": "segment_at_timestamp",
                        "direct_segment_event_ids": [speech_event_by_start[timestamp]],
                        "semantic_match_claimed": False,
                    },
                    "nearby_speech": {
                        "window": None,
                        "segments": [],
                        "semantic_match_claimed": False,
                    },
                }
            )

        temporal_links = [
            {
                "visual_event_id": frame["event_id"],
                "speech_event_id": frame["temporal_alignment"][
                    "direct_segment_event_ids"
                ][0],
                "relation": "segment_at_visual_timestamp",
                "clock": "source_video_absolute_microseconds",
                "semantic_match_claimed": False,
            }
            for frame in visual_frames
        ]
        timeline = [
            {
                "event_id": frame["event_id"],
                "kind": "visual_frame",
                "timestamp": frame["timestamp"],
            }
            for frame in visual_frames
        ] + [
            {
                "event_id": segment["event_id"],
                "kind": "speech_segment",
                "start_seconds": segment["absolute_start_seconds"],
                "end_seconds": segment["absolute_end_seconds"],
            }
            for segment in aligned_segments
        ]
        timeline.sort(
            key=lambda event: (
                float(event.get("timestamp", event.get("start_seconds", 0.0))),
                0 if event["kind"] == "visual_frame" else 1,
                str(event["event_id"]),
            )
        )

        strategy_document = {
            "strategy": "routed",
            "selected_count": len(selected),
            "selected": selected,
        }
        strategy_path = strategy_directory / "routed.json"
        _write_json(strategy_path, strategy_document)
        manifest_document = {
            "schema_version": 1,
            "kind": "candidate_frame_benchmark",
            "source": {
                "path": str(video_path),
                "fingerprint": video_sha256,
            },
            "range": {
                "start_seconds": 0.0,
                "end_seconds": duration_seconds,
                "duration_seconds": duration_seconds,
            },
            "strategies": {"routed": strategy_document},
        }
        manifest_path = visual_directory / "manifest.json"
        _write_json(manifest_path, manifest_document)

        ocr_document = {
            "schema_version": 1,
            "kind": "frame_ocr",
            "source": {
                "run_directory": str(visual_directory),
                "manifest_path": "manifest.json",
                "manifest_sha256": _sha256(manifest_path),
                "video_sha256": video_sha256,
                "strategy": "routed",
                "strategy_path": "strategies/routed.json",
                "strategy_sha256": _sha256(strategy_path),
            },
            "parameters": {"policy_version": "route-roi-v2"},
            "frames": ocr_frames,
            "skipped": [],
            "failures": [],
            "summary": {
                "selected_input_frames": len(ocr_frames),
                "ocr_success_frames": len(ocr_frames),
                "failure_frames": 0,
                "skipped_frames": 0,
                "observation_count": len(ocr_frames),
            },
        }
        ocr_directory = root / "ocr"
        ocr_path = ocr_directory / "ocr.json"
        _write_json(ocr_path, ocr_document)

        range_document = {
            "start_seconds": 0.0,
            "end_seconds": duration_seconds,
            "duration_seconds": duration_seconds,
            "start_timecode": self._timecode(0.0),
            "end_timecode": self._timecode(duration_seconds),
        }
        alignment_document = {
            "schema_version": 1,
            "kind": "timestamp_aligned_evidence",
            "parameters": {
                "policy_version": "absolute-time-point-overlap-v1",
                "semantic_match_claimed": False,
                "correction_applied": False,
                "summarization_applied": False,
            },
            "source": {
                "video": {
                    "path": str(video_path),
                    "size_bytes": video_path.stat().st_size,
                    "mtime_ns": video_path.stat().st_mtime_ns,
                    "duration_seconds": duration_seconds,
                    "fingerprint": video_sha256,
                },
                "phase1": {
                    "run_directory": str(visual_directory),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "strategy": "routed",
                    "strategy_path": str(strategy_path),
                    "strategy_sha256": _sha256(strategy_path),
                },
                "ocr": {
                    "path": str(ocr_path),
                    "sha256": _sha256(ocr_path),
                    "policy_version": "route-roi-v2",
                },
                "asr": {
                    "path": str(asr_path),
                    "sha256": _sha256(asr_path),
                    "model_id": "local-test-whisper",
                    "model_revision": "b" * 40,
                    "audio_path": str(audio_path),
                    "audio_sha256": _sha256(audio_path),
                },
            },
            "coverage": {
                "phase1_range": range_document,
                "asr_range": dict(range_document),
                "asr_fully_contained_in_phase1": True,
                "complete_phase1_speech_coverage": True,
            },
            "visual_frames": visual_frames,
            "speech_segments": aligned_segments,
            "temporal_links": temporal_links,
            "timeline": timeline,
            "summary": {
                "visual_frame_count": len(visual_frames),
                "ocr_ok_frame_count": len(visual_frames),
                "speech_segment_count": len(aligned_segments),
                "frames_with_segment_at_timestamp": len(visual_frames),
                "frames_inside_asr_range_without_segment": 0,
                "frames_outside_asr_range": 0,
                "temporal_link_count": len(temporal_links),
                "timeline_event_count": len(timeline),
            },
        }
        alignment_directory = root / "alignment"
        alignment_path = alignment_directory / "evidence.json"
        _write_json(alignment_path, alignment_document)
        return {
            "source": video_path,
            "visual": visual_directory,
            "manifest": manifest_path,
            "strategy": strategy_path,
            "ocr_directory": ocr_directory,
            "ocr": ocr_path,
            "asr_directory": audio_directory,
            "asr": asr_path,
            "audio": audio_path,
            "alignment_directory": alignment_directory,
            "alignment": alignment_path,
            "alignment_document": alignment_document,
        }

    @staticmethod
    def _timecode(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remainder = seconds - hours * 3600 - minutes * 60
        return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"

    @staticmethod
    def _run(alignment: Path, output: Path, **overrides: Any) -> dict[str, Any]:
        parameters = {
            "min_chapter_seconds": 30.0,
            "target_chapter_seconds": 180.0,
            "max_chapter_seconds": 480.0,
            "topic_window_seconds": 45.0,
        }
        parameters.update(overrides)
        return run_semantic_segmentation(
            alignment,
            output=output,
            **parameters,
        )

    def test_deterministic_complete_assignment_preserves_raw_text_and_is_offline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            result_one = self._run(paths["alignment"], root / "results" / "semantic-one")
            result_two = self._run(paths["alignment"], root / "results" / "semantic-two")
            document_one = json.loads(
                Path(result_one["semantic_json"]).read_text(encoding="utf-8")
            )
            document_two = json.loads(
                Path(result_two["semantic_json"]).read_text(encoding="utf-8")
            )

            self.assertEqual(document_one["kind"], "local_topic_segmentation")
            self.assertEqual(document_one["chapters"], document_two["chapters"])
            chapters = document_one["chapters"]
            self.assertGreater(len(chapters), 0)

            source_speech = paths["alignment_document"]["speech_segments"]
            source_visual = paths["alignment_document"]["visual_frames"]
            expected_speech_ids = [segment["event_id"] for segment in source_speech]
            expected_visual_ids = [frame["event_id"] for frame in source_visual]
            assigned_speech_ids = [
                event_id for chapter in chapters for event_id in chapter["speech_event_ids"]
            ]
            assigned_visual_ids = [
                event_id for chapter in chapters for event_id in chapter["visual_event_ids"]
            ]
            self.assertCountEqual(assigned_speech_ids, expected_speech_ids)
            self.assertEqual(len(assigned_speech_ids), len(set(assigned_speech_ids)))
            self.assertCountEqual(assigned_visual_ids, expected_visual_ids)
            self.assertEqual(len(assigned_visual_ids), len(set(assigned_visual_ids)))

            speech_starts = {
                float(segment["absolute_start_seconds"]) for segment in source_speech
            }
            phase_range = paths["alignment_document"]["coverage"]["phase1_range"]
            for index, chapter in enumerate(chapters):
                self.assertIn(float(chapter["start_seconds"]), speech_starts)
                self.assertLess(float(chapter["start_seconds"]), float(chapter["end_seconds"]))
                if index:
                    self.assertEqual(
                        float(chapters[index - 1]["end_seconds"]),
                        float(chapter["start_seconds"]),
                    )
            self.assertEqual(float(chapters[0]["start_seconds"]), phase_range["start_seconds"])
            self.assertEqual(float(chapters[-1]["end_seconds"]), phase_range["end_seconds"])

            speech_by_id = {segment["event_id"]: segment["text"] for segment in source_speech}
            visual_by_id = {
                frame["event_id"]: frame["ocr"]["plain_text"] for frame in source_visual
            }
            for chapter in chapters:
                for event_id in chapter["speech_event_ids"]:
                    self.assertIn(speech_by_id[event_id], chapter["raw_text"])
                self.assertTrue(chapter["title_exact_extract"])
                title_source_id = chapter["title_source_event_id"]
                source_text = speech_by_id.get(title_source_id, visual_by_id.get(title_source_id))
                self.assertIsNotNone(source_text)
                self.assertIn(chapter["title"], source_text)

            parameters = document_one["parameters"]
            self.assertFalse(parameters["network_ai_used"])
            self.assertFalse(parameters["chat_model_used"])
            self.assertFalse(parameters["correction_applied"])
            self.assertFalse(parameters["summarization_applied"])
            self.assertEqual(parameters["visual_title_lookahead_seconds"], 5.0)
            self.assertEqual(parameters["visual_title_asr_window_seconds"], 14.0)
            self.assertGreaterEqual(parameters["visual_title_min_ocr_confidence"], 0.25)
            coverage = document_one["coverage"]
            self.assertTrue(coverage["complete_event_assignment"])
            self.assertEqual(coverage["source_speech_segment_count"], len(source_speech))
            self.assertEqual(coverage["assigned_speech_segment_count"], len(source_speech))
            self.assertEqual(coverage["unassigned_speech_segment_count"], 0)
            self.assertEqual(coverage["duplicate_speech_assignment_count"], 0)
            self.assertEqual(coverage["source_visual_frame_count"], len(source_visual))
            self.assertEqual(coverage["assigned_visual_frame_count"], len(source_visual))
            self.assertEqual(coverage["unassigned_visual_frame_count"], 0)
            self.assertEqual(coverage["duplicate_visual_assignment_count"], 0)
            summary = document_one["summary"]
            self.assertEqual(summary["chapter_count"], len(chapters))
            self.assertEqual(summary["speech_segment_count"], len(source_speech))
            self.assertEqual(summary["visual_frame_count"], len(source_visual))

            review = Path(result_one["review_html"]).read_text(encoding="utf-8")
            self.assertNotIn("http://", review)
            self.assertNotIn("https://", review)
            self.assertNotIn('</script><script>alert("semantic-xss")</script>', review)
            self.assertNotIn("<img src=x onerror=alert('ocr-xss')>", review)
            self.assertIn("semantic-xss", review)
            self.assertIn("Content-Security-Policy", review)

    def test_repeated_ocr_title_does_not_create_a_boundary_at_every_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root, repeated_ocr_title=True)
            result = self._run(paths["alignment"], root / "results" / "semantic")
            document = json.loads(Path(result["semantic_json"]).read_text(encoding="utf-8"))
            chapter_starts = {
                float(chapter["start_seconds"]) for chapter in document["chapters"][1:]
            }
            repeated_title_times = {
                float(frame["timestamp"])
                for frame in paths["alignment_document"]["visual_frames"]
            }
            self.assertLess(
                len(chapter_starts & repeated_title_times),
                len(repeated_title_times),
            )

    def test_max_duration_forces_only_asr_anchored_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(
                root,
                duration_seconds=1000.0,
                speech_step_seconds=40.0,
                repeated_ocr_title=True,
            )
            result = self._run(
                paths["alignment"],
                root / "results" / "semantic",
                target_chapter_seconds=120.0,
                max_chapter_seconds=160.0,
            )
            document = json.loads(Path(result["semantic_json"]).read_text(encoding="utf-8"))
            speech_starts = {
                float(segment["absolute_start_seconds"])
                for segment in paths["alignment_document"]["speech_segments"]
            }
            for chapter in document["chapters"]:
                self.assertLessEqual(
                    float(chapter["end_seconds"]) - float(chapter["start_seconds"]),
                    160.0 + 1e-6,
                )
                self.assertIn(float(chapter["start_seconds"]), speech_starts)
            self.assertGreater(document["summary"]["forced_boundary_count"], 0)

    def test_tampered_bound_inputs_are_rejected_before_output(self) -> None:
        cases = (
            ("source", lambda path: path.write_bytes(path.read_bytes() + b"tampered")),
            ("manifest", lambda path: path.write_text(path.read_text() + "\n", encoding="utf-8")),
            ("strategy", lambda path: path.write_text(path.read_text() + "\n", encoding="utf-8")),
            ("asr", lambda path: path.write_text(path.read_text() + "\n", encoding="utf-8")),
            ("ocr", lambda path: path.write_text(path.read_text() + "\n", encoding="utf-8")),
            ("audio", lambda path: path.write_bytes(path.read_bytes() + b"tampered")),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case_name, tamper in cases:
                with self.subTest(case=case_name):
                    case_root = root / case_name
                    paths = self._fixture(case_root)
                    tamper(paths[case_name])
                    output = case_root / "results" / "semantic"
                    with self.assertRaisesRegex(ValueError, "(?i)(sha-?256|hash)"):
                        self._run(paths["alignment"], output)
                    self.assertFalse(output.exists())

    def test_low_confidence_titles_and_non_chronological_tracks_are_rejected_or_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            low_confidence = self._fixture(root / "low-confidence")
            document = low_confidence["alignment_document"]
            for frame in document["visual_frames"]:
                for observation in frame["ocr"]["observations"]:
                    observation["confidence"] = 0.1
            _write_json(low_confidence["alignment"], document)
            result = self._run(
                low_confidence["alignment"], root / "low-confidence-result"
            )
            semantic = json.loads(
                Path(result["semantic_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(semantic["summary"]["visual_title_change_count"], 0)
            self.assertEqual(semantic["summary"]["visual_title_change_mapped_count"], 0)

            overlapping = self._fixture(root / "overlap")
            overlap_document = overlapping["alignment_document"]
            overlap_document["speech_segments"][1]["absolute_start_seconds"] = 20.0
            _write_json(overlapping["alignment"], overlap_document)
            with self.assertRaisesRegex(ValueError, "(?i)overlap"):
                self._run(overlapping["alignment"], root / "overlap-result")

            unordered_visual = self._fixture(root / "unordered-visual")
            visual_document = unordered_visual["alignment_document"]
            visual_document["visual_frames"][0]["timestamp"] = 160.0
            visual_document["visual_frames"][1]["timestamp"] = 80.0
            _write_json(unordered_visual["alignment"], visual_document)
            with self.assertRaisesRegex(ValueError, "(?i)(visual|chronological)"):
                self._run(
                    unordered_visual["alignment"], root / "unordered-visual-result"
                )

    def test_bad_kind_and_incomplete_coverage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad_kind = self._fixture(root / "bad-kind")
            document = bad_kind["alignment_document"]
            document["kind"] = "not_alignment_evidence"
            _write_json(bad_kind["alignment"], document)
            bad_kind_output = root / "bad-kind-result"
            with self.assertRaisesRegex(ValueError, "(?i)kind"):
                self._run(bad_kind["alignment"], bad_kind_output)
            self.assertFalse(bad_kind_output.exists())

            incomplete = self._fixture(root / "incomplete")
            document = incomplete["alignment_document"]
            document["coverage"]["complete_phase1_speech_coverage"] = False
            document["coverage"]["asr_range"]["end_seconds"] = 500.0
            document["coverage"]["asr_range"]["duration_seconds"] = 500.0
            _write_json(incomplete["alignment"], document)
            incomplete_output = root / "incomplete-result"
            with self.assertRaisesRegex(ValueError, "(?i)(complete|coverage)"):
                self._run(incomplete["alignment"], incomplete_output)
            self.assertFalse(incomplete_output.exists())

    def test_output_must_be_outside_every_upstream_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            upstream_directories = (
                paths["alignment_directory"],
                paths["visual"],
                paths["ocr_directory"],
                paths["asr_directory"],
                paths["source"].parent,
            )
            for index, upstream in enumerate(upstream_directories):
                output = upstream / f"semantic-output-{index}"
                with self.subTest(upstream=upstream):
                    with self.assertRaisesRegex(ValueError, "(?i)(output|outside|upstream)"):
                        self._run(paths["alignment"], output)
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
