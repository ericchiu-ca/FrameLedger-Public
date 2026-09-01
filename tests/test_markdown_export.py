from __future__ import annotations

import hashlib
import json
import re
import struct
import tempfile
import unittest
import urllib.parse
import zlib
from pathlib import Path
from typing import Any, Callable

from frameledger.markdown_export import run_markdown_export


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _png_bytes(red: int, green: int, blue: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    pixels = zlib.compress(bytes((0, red, green, blue)))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", pixels)
        + chunk(b"IEND", b"")
    )


class MarkdownExportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Any]:
        source_directory = root / "source"
        source_directory.mkdir(parents=True)
        video_path = source_directory / "episode <iframe onload=video-xss>.mp4"
        video_path.write_bytes(b"immutable-source-video")

        phase1_directory = root / "phase1"
        frame_directory = phase1_directory / "frames"
        strategy_directory = phase1_directory / "strategies"
        frame_directory.mkdir(parents=True)
        strategy_directory.mkdir()
        image_paths = (
            frame_directory / "截图 one #(1).png",
            frame_directory / "第二张 (2).png",
        )
        image_paths[0].write_bytes(_png_bytes(220, 20, 60))
        image_paths[1].write_bytes(_png_bytes(20, 80, 220))

        relative_images = tuple(
            image.relative_to(phase1_directory).as_posix() for image in image_paths
        )
        manifest_path = phase1_directory / "manifest.json"
        strategy_path = strategy_directory / "routed.json"
        _write_json(
            strategy_path,
            {
                "strategy": "routed",
                "selected_count": 2,
                "selected": [
                    {
                        "sample_index": index + 1,
                        "timestamp": timestamp,
                        "image_path": relative_images[index],
                    }
                    for index, timestamp in enumerate((2.0, 7.0))
                ],
            },
        )
        _write_json(
            manifest_path,
            {
                "kind": "candidate_frame_benchmark",
                "schema_version": 1,
                "range": {"start_seconds": 0.0, "end_seconds": 20.0},
                "strategies": {"routed": {"selected_count": 2}},
            },
        )

        audio_path = root / "asr" / "audio.wav"
        audio_path.parent.mkdir()
        audio_path.write_bytes(b"local-pcm-audio")
        unsafe_title = (
            '章节 </script><script>alert("title-xss")</script> '
            "[click](javascript:alert(1)) ```"
        )
        closing_text = '收尾 <svg onload=alert("speech-xss")> ````'
        speech_segments = [
            {
                "event_id": "speech-0001",
                "id": 0,
                "absolute_start_seconds": 0.0,
                "absolute_end_seconds": 9.0,
                "text": unsafe_title,
            },
            {
                "event_id": "speech-0002",
                "id": 1,
                "absolute_start_seconds": 10.0,
                "absolute_end_seconds": 20.0,
                "text": closing_text,
            },
        ]
        asr_path = audio_path.parent / "asr.json"
        _write_json(
            asr_path,
            {
                "kind": "local_speech_transcription",
                "schema_version": 1,
                "status": "ok",
                "audio": {"path": "audio.wav", "sha256": _sha256(audio_path)},
                "transcript": {"segments": speech_segments},
            },
        )

        ocr_texts = (
            'OCR </script><script>alert("ocr-xss")</script> `````',
            '<img src=x onerror=alert("image-xss")> [bad](data:text/html,x)',
        )
        ocr_path = root / "ocr" / "ocr.json"
        _write_json(
            ocr_path,
            {
                "kind": "frame_ocr",
                "schema_version": 1,
                "frames": [
                    {
                        "sample_index": index + 1,
                        "image_path": relative_images[index],
                        "image_sha256": _sha256(image_paths[index]),
                        "plain_text": ocr_texts[index],
                    }
                    for index in range(2)
                ],
            },
        )

        video_source = {
            "path": str(video_path),
            "size_bytes": video_path.stat().st_size,
            "mtime_ns": video_path.stat().st_mtime_ns,
            "duration_seconds": 20.0,
            "fingerprint": _sha256(video_path),
        }
        phase1_source = {
            "run_directory": str(phase1_directory),
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "strategy": "routed",
            "strategy_path": str(strategy_path),
            "strategy_sha256": _sha256(strategy_path),
        }
        ocr_source = {
            "path": str(ocr_path),
            "sha256": _sha256(ocr_path),
            "policy_version": "test-route-roi-v1",
        }
        asr_source = {
            "path": str(asr_path),
            "sha256": _sha256(asr_path),
            "audio_path": str(audio_path),
            "audio_sha256": _sha256(audio_path),
            "model_id": "local-test-model",
            "model_revision": "a" * 40,
        }
        source = {
            "video": video_source,
            "phase1": phase1_source,
            "ocr": ocr_source,
            "asr": asr_source,
        }
        visual_frames = [
            {
                "event_id": f"visual-{index + 1:04d}",
                "sample_index": index + 1,
                "timestamp": timestamp,
                "timecode": timecode,
                "route_kind": "presentation",
                "segment_id": "route-001",
                "image": {
                    "path": relative_images[index],
                    "sha256": _sha256(image_paths[index]),
                },
                "ocr": {"status": "ok", "plain_text": ocr_texts[index]},
            }
            for index, (timestamp, timecode) in enumerate(
                ((2.0, "00:00:02.000"), (7.0, "00:00:07.000"))
            )
        ]
        range_payload = {
            "start_seconds": 0.0,
            "end_seconds": 20.0,
            "duration_seconds": 20.0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:20.000",
        }
        alignment_document = {
            "kind": "timestamp_aligned_evidence",
            "schema_version": 1,
            "source": source,
            "coverage": {
                "phase1_range": range_payload,
                "asr_range": dict(range_payload),
                "asr_fully_contained_in_phase1": True,
                "complete_phase1_speech_coverage": True,
            },
            "visual_frames": visual_frames,
            "speech_segments": speech_segments,
            "summary": {
                "visual_frame_count": 2,
                "speech_segment_count": 2,
                "frames_outside_asr_range": 0,
            },
        }
        alignment_path = root / "alignment" / "evidence.json"
        _write_json(alignment_path, alignment_document)

        semantic_source = {
            "alignment": {
                "path": str(alignment_path),
                "sha256": _sha256(alignment_path),
            },
            **source,
        }
        semantic_document = {
            "kind": "local_topic_segmentation",
            "schema_version": 1,
            "source": semantic_source,
            "coverage": {
                **range_payload,
                "source_speech_segment_count": 2,
                "assigned_speech_segment_count": 2,
                "unassigned_speech_segment_count": 0,
                "duplicate_speech_assignment_count": 0,
                "source_visual_frame_count": 2,
                "assigned_visual_frame_count": 2,
                "unassigned_visual_frame_count": 0,
                "duplicate_visual_assignment_count": 0,
                "complete_event_assignment": True,
            },
            "chapters": [
                {
                    "chapter_id": "chapter-001",
                    "start_seconds": 0.0,
                    "end_seconds": 10.0,
                    "duration_seconds": 10.0,
                    "start_timecode": "00:00:00.000",
                    "end_timecode": "00:00:10.000",
                    "title": unsafe_title,
                    "title_source_event_id": "speech-0001",
                    "title_exact_extract": True,
                    "keywords": [
                        "安全测试",
                        "x](javascript:alert(2))",
                        '<style onload=alert("keyword-xss")>',
                    ],
                    "raw_text": unsafe_title,
                    "speech_event_ids": ["speech-0001"],
                    "visual_event_ids": ["visual-0001", "visual-0002"],
                },
                {
                    "chapter_id": "chapter-002",
                    "start_seconds": 10.0,
                    "end_seconds": 20.0,
                    "duration_seconds": 10.0,
                    "start_timecode": "00:00:10.000",
                    "end_timecode": "00:00:20.000",
                    "title": closing_text,
                    "title_source_event_id": "speech-0002",
                    "title_exact_extract": True,
                    "keywords": [],
                    "raw_text": closing_text,
                    "speech_event_ids": ["speech-0002"],
                    "visual_event_ids": [],
                },
            ],
            "summary": {
                "chapter_count": 2,
                "speech_segment_count": 2,
                "visual_frame_count": 2,
            },
        }
        semantic_path = root / "semantic" / "semantic-segments.json"
        _write_json(semantic_path, semantic_document)
        return {
            "root": root,
            "semantic": semantic_path,
            "semantic_document": semantic_document,
            "alignment": alignment_path,
            "images": image_paths,
        }

    def test_exports_complete_bound_markdown_with_relative_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            output = paths["root"] / "export"

            result = run_markdown_export(paths["semantic"], output=output)

            report_path = output / "report.md"
            manifest_path = output / "manifest.json"
            self.assertEqual(result["kind"], "local_evidence_markdown")
            self.assertEqual(
                Path(result["markdown_file"]).resolve(), report_path.resolve()
            )
            self.assertNotIn("review_html", result)
            self.assertEqual(
                Path(result["manifest_json"]).resolve(), manifest_path.resolve()
            )
            self.assertEqual(
                {item.name for item in output.iterdir()},
                {"report.md", "manifest.json"},
            )

            report = report_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "local_evidence_markdown")
            self.assertEqual(manifest["artifact"]["path"], "report.md")
            self.assertEqual(manifest["artifact"]["sha256"], _sha256(report_path))
            self.assertEqual(manifest["summary"]["visual_frame_count"], 2)
            self.assertEqual(manifest["coverage"]["scope"], "visual_frames_only")
            self.assertTrue(manifest["coverage"]["complete_report_coverage"])
            self.assertEqual(manifest["coverage"]["unrendered_visual_frame_count"], 0)
            self.assertEqual(manifest["coverage"]["duplicate_visual_render_count"], 0)

            image_links = re.findall(r"!\[[^\n]*?\]\(<([^>\n]+)>\)", report)
            self.assertEqual(len(image_links), 2)
            evidence_by_path = {
                Path(item["path"]).resolve(): item
                for item in manifest["evidence_images"]
            }
            resolved_links: list[Path] = []
            for encoded_link in image_links:
                self.assertEqual(urllib.parse.urlsplit(encoded_link).scheme, "")
                decoded_link = urllib.parse.unquote(encoded_link)
                target = (report_path.parent / decoded_link).resolve()
                resolved_links.append(target)
                self.assertTrue(target.is_file())
                self.assertIn(target, evidence_by_path)
                self.assertEqual(_sha256(target), evidence_by_path[target]["sha256"])
            self.assertCountEqual(
                resolved_links, [path.resolve() for path in paths["images"]]
            )
            self.assertTrue(any("%20" in link for link in image_links))
            self.assertTrue(any("%23" in link for link in image_links))
            self.assertFalse(any(output.rglob("*.png")))

    def test_markdown_contains_only_screenshot_embeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            output = paths["root"] / "export"
            report_path = Path(
                run_markdown_export(paths["semantic"], output=output)["markdown_file"]
            )
            report = report_path.read_text(encoding="utf-8")
            nonempty_lines = [line for line in report.splitlines() if line]
            self.assertEqual(len(nonempty_lines), 2)
            self.assertTrue(
                all(
                    re.fullmatch(r"!\[\d\d:\d\d:\d\d\.\d{3}\]\(<[^>]+>\)", line)
                    for line in nonempty_lines
                )
            )
            for forbidden in (
                "章节",
                "OCR",
                "ASR",
                "SHA-256",
                "script",
                "javascript:",
                "data:",
                "http://",
                "https://",
            ):
                self.assertNotIn(forbidden, report)

    def test_tampered_image_fails_closed_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            image = paths["images"][0]
            image.write_bytes(image.read_bytes() + b"tampered")
            output = paths["root"] / "export"

            with self.assertRaisesRegex(ValueError, "(?i)(image|sha-256)"):
                run_markdown_export(paths["semantic"], output=output)

            self.assertFalse(output.exists())

    def test_duplicate_or_missing_visual_assignment_fails_closed(self) -> None:
        mutations: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
            (
                "duplicate",
                lambda document: document["chapters"][0]["visual_event_ids"].append(
                    "visual-0001"
                ),
            ),
            (
                "missing",
                lambda document: document["chapters"][0].__setitem__(
                    "visual_event_ids", ["visual-0001"]
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate in mutations:
                with self.subTest(case=name):
                    paths = self._fixture(root / name)
                    document = paths["semantic_document"]
                    mutate(document)
                    _write_json(paths["semantic"], document)
                    output = paths["root"] / "export"

                    with self.assertRaisesRegex(
                        ValueError, "(?i)(assignment|one-to-one|missing|duplicate)"
                    ):
                        run_markdown_export(paths["semantic"], output=output)

                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
