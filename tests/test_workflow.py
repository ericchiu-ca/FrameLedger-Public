from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frameledger.cli import build_parser
from frameledger.local_ui import RUN_NAME_PATTERN, _frontend_html, _frontend_js
from frameledger.models import VideoMetadata
from frameledger.workflow import (
    DEFAULT_MODEL_REVISION,
    DEFAULT_WORKFLOW_MAX_FRAMES,
    run_complete_workflow,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


class CompleteWorkflowTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source_dir = root / "source"
        source_dir.mkdir()
        source = source_dir / "episode-124.mp4"
        source.write_bytes(b"immutable-complete-video")
        metadata = VideoMetadata(
            path=source,
            size_bytes=source.stat().st_size,
            mtime_ns=source.stat().st_mtime_ns,
            duration_seconds=1633.05,
            fps=20.0,
            width=1920,
            height=1080,
            frame_count=32661,
            codec="h264",
            fingerprint=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        return (
            source,
            metadata,
            model,
            _executable(root / "asr-helper"),
            _executable(root / "ocr-helper"),
            _executable(root / "ffmpeg"),
        )

    @staticmethod
    def _stage(stage: str, calls: list[str]):
        def run(*args, **kwargs):
            calls.append(stage)
            output = Path(kwargs["output"])
            output.mkdir(parents=True)
            result = {
                "kind": stage,
                "output": str(output),
                "summary": {"stage": stage},
            }
            if stage != "markdown":
                review = output / "review.html"
                review.write_text(f"<!doctype html><p>{stage}</p>", encoding="utf-8")
                result["review_html"] = str(review)
            if stage == "visual":
                result["strategies"] = {
                    "routed": {"selected_count": 2, "capped_count": 0}
                }
            if stage == "ocr":
                result["summary"] = {
                    "selected_input_frames": 2,
                    "ocr_success_frames": 2,
                    "failure_frames": 0,
                    "skipped_frames": 0,
                }
            if stage == "alignment":
                result["coverage"] = {"complete_phase1_speech_coverage": True}
                result["summary"] = {"frames_outside_asr_range": 0}
            if stage == "semantic":
                semantic_json = output / "semantic-segments.json"
                semantic_json.write_text("{}\n", encoding="utf-8")
                result["kind"] = "local_topic_segmentation"
                result["semantic_json"] = str(semantic_json)
                result["coverage"] = {
                    "complete_event_assignment": True,
                    "unassigned_speech_segment_count": 0,
                    "duplicate_speech_assignment_count": 0,
                    "unassigned_visual_frame_count": 0,
                    "duplicate_visual_assignment_count": 0,
                }
                result["summary"] = {
                    "chapter_count": 3,
                    "speech_segment_count": 8,
                    "visual_frame_count": 2,
                    "forced_boundary_count": 0,
                }
            if stage == "markdown":
                markdown_file = output / "report.md"
                manifest_json = output / "manifest.json"
                markdown_file.write_text("# Evidence report\n", encoding="utf-8")
                manifest_json.write_text("{}\n", encoding="utf-8")
                result["kind"] = "local_evidence_markdown"
                result["markdown_file"] = str(markdown_file)
                result["manifest_json"] = str(manifest_json)
                result["coverage"] = {
                    "scope": "visual_frames_only",
                    "source_visual_frame_count": 2,
                    "rendered_visual_frame_count": 2,
                    "unrendered_visual_frame_count": 0,
                    "duplicate_visual_render_count": 0,
                    "complete_report_coverage": True,
                }
                result["summary"] = {
                    "visual_frame_count": 2,
                    "markdown_bytes": markdown_file.stat().st_size,
                }
            return result

        return run

    def test_complete_workflow_uses_probed_full_duration_and_writes_frontend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, model, asr_helper, ocr_helper, ffmpeg = self._fixture(
                root
            )
            output = root / "outputs" / "episode-124"
            calls: list[str] = []
            original = source.read_bytes()
            with (
                patch(
                    "frameledger.workflow.probe_video", return_value=metadata
                ) as probe,
                patch(
                    "frameledger.workflow.run_benchmark",
                    side_effect=self._stage("visual", calls),
                ) as visual,
                patch(
                    "frameledger.workflow.run_frame_ocr",
                    side_effect=self._stage("ocr", calls),
                ) as ocr,
                patch(
                    "frameledger.workflow.run_local_asr",
                    side_effect=self._stage("asr", calls),
                ) as asr,
                patch(
                    "frameledger.workflow.run_evidence_alignment",
                    side_effect=self._stage("alignment", calls),
                ) as alignment,
                patch(
                    "frameledger.workflow.run_semantic_segmentation",
                    side_effect=self._stage("semantic", calls),
                ) as semantic,
                patch(
                    "frameledger.workflow.run_markdown_export",
                    side_effect=self._stage("markdown", calls),
                ) as markdown,
                patch(
                    "frameledger.workflow.preflight_local_capabilities",
                    return_value={
                        "apple_vision": "available",
                        "mlx_metal": "available",
                    },
                ),
            ):
                result = run_complete_workflow(
                    source,
                    output=output,
                    model_path=model,
                    mlx_whisper_helper=asr_helper,
                    apple_vision_helper=ocr_helper,
                    ffmpeg=ffmpeg,
                )

            self.assertEqual(
                calls,
                ["visual", "ocr", "asr", "alignment", "semantic", "markdown"],
            )
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(visual.call_args.kwargs["start_seconds"], 0.0)
            self.assertEqual(visual.call_args.kwargs["duration_seconds"], 1633.05)
            self.assertTrue(visual.call_args.kwargs["allow_long_range"])
            self.assertEqual(
                visual.call_args.kwargs["maximum_frames"],
                DEFAULT_WORKFLOW_MAX_FRAMES,
            )
            self.assertEqual(asr.call_args.kwargs["start_seconds"], 0.0)
            self.assertEqual(asr.call_args.kwargs["duration_seconds"], 1633.05)
            self.assertTrue(asr.call_args.kwargs["allow_long_range"])
            self.assertEqual(alignment.call_args.kwargs["strategy"], "routed")
            self.assertEqual(ocr.call_args.kwargs["routes"], ("presentation", "table"))
            self.assertEqual(
                Path(semantic.call_args.args[0]).resolve(),
                (output / "alignment" / "evidence.json").resolve(),
            )
            self.assertEqual(
                Path(semantic.call_args.kwargs["output"]).resolve(),
                (output / "semantic").resolve(),
            )
            self.assertEqual(
                Path(markdown.call_args.args[0]).resolve(),
                (output / "semantic" / "semantic-segments.json").resolve(),
            )
            self.assertEqual(
                Path(markdown.call_args.kwargs["output"]).resolve(),
                (output / "markdown").resolve(),
            )
            document = json.loads(
                (output / "workflow.json").read_text(encoding="utf-8")
            )
            frontend = (output / "index.html").read_text(encoding="utf-8")
            self.assertEqual(document["status"], "ok")
            self.assertEqual(document["range"]["range_origin"], "probed_full_video")
            self.assertEqual(document["range"]["end_seconds"], 1633.05)
            self.assertTrue(document["parameters"]["full_range_asr_explicitly_allowed"])
            self.assertFalse(document["parameters"]["chat_model_used"])
            self.assertTrue(document["source_integrity_verified_after_workflow"])
            self.assertIn("markdown/report.md", frontend)
            self.assertIn("download", frontend)
            self.assertEqual(document["stages"]["semantic"]["status"], "ok")
            self.assertEqual(
                document["stages"]["semantic"]["summary"]["chapter_count"], 3
            )
            self.assertEqual(document["stages"]["markdown"]["status"], "ok")
            self.assertEqual(
                document["stages"]["markdown"]["summary"]["visual_frame_count"], 2
            )
            self.assertEqual(
                Path(document["stages"]["markdown"]["markdown_file"]).resolve(),
                (output / "markdown" / "report.md").resolve(),
            )
            self.assertEqual(
                Path(document["stages"]["markdown"]["manifest_json"]).resolve(),
                (output / "markdown" / "manifest.json").resolve(),
            )
            self.assertEqual(result["summary"]["stage_count"], 6)
            self.assertEqual(
                Path(result["markdown_file"]).resolve(),
                (output / "markdown" / "report.md").resolve(),
            )
            self.assertEqual(
                Path(result["manifest_json"]).resolve(),
                (output / "markdown" / "manifest.json").resolve(),
            )
            self.assertNotIn("https://", frontend)
            self.assertNotIn("http://", frontend)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(
                result["review_html"], str((output / "index.html").resolve())
            )

    def test_failure_short_circuits_and_keeps_machine_readable_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, model, asr_helper, ocr_helper, ffmpeg = self._fixture(
                root
            )
            output = root / "outputs" / "failed"
            calls: list[str] = []
            with (
                patch("frameledger.workflow.probe_video", return_value=metadata),
                patch(
                    "frameledger.workflow.run_benchmark",
                    side_effect=self._stage("visual", calls),
                ),
                patch(
                    "frameledger.workflow.run_frame_ocr",
                    side_effect=RuntimeError("OCR stopped"),
                ),
                patch("frameledger.workflow.run_local_asr") as asr,
                patch("frameledger.workflow.run_evidence_alignment") as alignment,
                patch("frameledger.workflow.run_semantic_segmentation") as semantic,
                patch("frameledger.workflow.run_markdown_export") as markdown,
                patch(
                    "frameledger.workflow.preflight_local_capabilities",
                    return_value={
                        "apple_vision": "available",
                        "mlx_metal": "available",
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "OCR stopped"):
                    run_complete_workflow(
                        source,
                        output=output,
                        model_path=model,
                        mlx_whisper_helper=asr_helper,
                        apple_vision_helper=ocr_helper,
                        ffmpeg=ffmpeg,
                    )
            asr.assert_not_called()
            alignment.assert_not_called()
            semantic.assert_not_called()
            markdown.assert_not_called()
            document = json.loads(
                (output / "workflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["status"], "error")
            self.assertEqual(document["stages"]["visual"]["status"], "ok")
            self.assertEqual(document["stages"]["ocr"]["status"], "error")
            self.assertIn("OCR stopped", document["error"])
            self.assertIn("visual/review.html", (output / "index.html").read_text())

    def test_ocr_ledger_failures_are_a_hard_workflow_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, model, asr_helper, ocr_helper, ffmpeg = self._fixture(
                root
            )
            output = root / "outputs" / "ocr-ledger-failure"
            calls: list[str] = []

            def failed_ocr(*args, **kwargs):
                result = self._stage("ocr", calls)(*args, **kwargs)
                result["summary"].update({"ocr_success_frames": 0, "failure_frames": 2})
                return result

            with (
                patch("frameledger.workflow.probe_video", return_value=metadata),
                patch(
                    "frameledger.workflow.run_benchmark",
                    side_effect=self._stage("visual", calls),
                ),
                patch("frameledger.workflow.run_frame_ocr", side_effect=failed_ocr),
                patch("frameledger.workflow.run_local_asr") as asr,
                patch("frameledger.workflow.run_evidence_alignment") as alignment,
                patch("frameledger.workflow.run_semantic_segmentation") as semantic,
                patch("frameledger.workflow.run_markdown_export") as markdown,
                patch(
                    "frameledger.workflow.preflight_local_capabilities",
                    return_value={
                        "apple_vision": "available",
                        "mlx_metal": "available",
                    },
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "2 failed evidence frames"):
                    run_complete_workflow(
                        source,
                        output=output,
                        model_path=model,
                        mlx_whisper_helper=asr_helper,
                        apple_vision_helper=ocr_helper,
                        ffmpeg=ffmpeg,
                    )
            asr.assert_not_called()
            alignment.assert_not_called()
            semantic.assert_not_called()
            markdown.assert_not_called()
            document = json.loads((output / "workflow.json").read_text())
            self.assertEqual(document["stages"]["ocr"]["status"], "error")

    def test_markdown_incomplete_coverage_is_a_hard_workflow_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, model, asr_helper, ocr_helper, ffmpeg = self._fixture(
                root
            )
            output = root / "outputs" / "markdown-coverage-failure"
            calls: list[str] = []

            def incomplete_markdown(*args, **kwargs):
                result = self._stage("markdown", calls)(*args, **kwargs)
                result["coverage"]["complete_report_coverage"] = False
                result["coverage"]["unrendered_visual_frame_count"] = 1
                return result

            with (
                patch("frameledger.workflow.probe_video", return_value=metadata),
                patch(
                    "frameledger.workflow.run_benchmark",
                    side_effect=self._stage("visual", calls),
                ),
                patch(
                    "frameledger.workflow.run_frame_ocr",
                    side_effect=self._stage("ocr", calls),
                ),
                patch(
                    "frameledger.workflow.run_local_asr",
                    side_effect=self._stage("asr", calls),
                ),
                patch(
                    "frameledger.workflow.run_evidence_alignment",
                    side_effect=self._stage("alignment", calls),
                ),
                patch(
                    "frameledger.workflow.run_semantic_segmentation",
                    side_effect=self._stage("semantic", calls),
                ),
                patch(
                    "frameledger.workflow.run_markdown_export",
                    side_effect=incomplete_markdown,
                ),
                patch(
                    "frameledger.workflow.preflight_local_capabilities",
                    return_value={
                        "apple_vision": "available",
                        "mlx_metal": "available",
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "did not render every screenshot"
                ):
                    run_complete_workflow(
                        source,
                        output=output,
                        model_path=model,
                        mlx_whisper_helper=asr_helper,
                        apple_vision_helper=ocr_helper,
                        ffmpeg=ffmpeg,
                    )

            self.assertEqual(
                calls,
                ["visual", "ocr", "asr", "alignment", "semantic", "markdown"],
            )
            document = json.loads(
                (output / "workflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["status"], "error")
            self.assertEqual(document["stages"]["semantic"]["status"], "ok")
            self.assertEqual(document["stages"]["markdown"]["status"], "error")
            self.assertIn("markdown: RuntimeError", document["error"])
            self.assertIn("semantic/review.html", (output / "index.html").read_text())

    def test_output_cannot_be_beside_source_or_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, metadata, model, asr_helper, ocr_helper, ffmpeg = self._fixture(
                root
            )
            with patch("frameledger.workflow.probe_video", return_value=metadata):
                with self.assertRaisesRegex(ValueError, "beside or below"):
                    run_complete_workflow(
                        source,
                        output=source.parent / "result",
                        model_path=model,
                        mlx_whisper_helper=asr_helper,
                        apple_vision_helper=ocr_helper,
                        ffmpeg=ffmpeg,
                    )
                existing = root / "existing"
                existing.mkdir()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    run_complete_workflow(
                        source,
                        output=existing,
                        model_path=model,
                        mlx_whisper_helper=asr_helper,
                        apple_vision_helper=ocr_helper,
                        ffmpeg=ffmpeg,
                    )

    def test_cli_and_local_frontend_are_standalone_and_local_only(self):
        workflow = build_parser().parse_args(
            ["workflow", "/tmp/video.mp4", "--output", "/tmp/run"]
        )
        self.assertEqual(workflow.command, "workflow")
        self.assertEqual(workflow.model_revision, DEFAULT_MODEL_REVISION)
        self.assertIsNone(workflow.max_frames)
        ui = build_parser().parse_args(["workflow-ui", "--port", "0"])
        self.assertEqual(ui.command, "workflow-ui")
        semantic = build_parser().parse_args(
            [
                "semantic",
                "/tmp/alignment/evidence.json",
                "--output",
                "/tmp/semantic",
            ]
        )
        self.assertEqual(semantic.command, "semantic")
        self.assertEqual(semantic.min_chapter_seconds, 30.0)
        self.assertEqual(semantic.target_chapter_seconds, 180.0)
        self.assertEqual(semantic.max_chapter_seconds, 480.0)
        self.assertEqual(semantic.topic_window_seconds, 45.0)
        markdown = build_parser().parse_args(
            [
                "markdown",
                "/tmp/semantic/semantic-segments.json",
                "--output",
                "/tmp/markdown",
            ]
        )
        self.assertEqual(markdown.command, "markdown")
        self.assertEqual(
            markdown.semantic_json,
            Path("/tmp/semantic/semantic-segments.json"),
        )
        self.assertEqual(markdown.output, Path("/tmp/markdown"))
        frontend = _frontend_html(token="test-token", default_video="/tmp/video.mp4")
        javascript = _frontend_js()
        self.assertIn("无需聊天模型", frontend)
        self.assertIn("Markdown 证据导出", frontend)
        self.assertIn("把一个或多个视频拖到这里", frontend)
        self.assertIn('type="file" multiple', frontend)
        self.assertIn("选择输出文件夹", frontend)
        self.assertIn("批量生成", frontend)
        self.assertIn('src="/app.js"', frontend)
        self.assertIn("/api/choose-videos", javascript)
        self.assertIn("/api/choose-output", javascript)
        self.assertIn("/api/import", javascript)
        self.assertIn("/api/run-batch", javascript)
        self.assertIn("dragover", javascript)
        self.assertIn("preventDefault", javascript)
        self.assertIn("textContent", javascript)
        self.assertIn("semantic:", javascript)
        self.assertIn("语义", javascript)
        self.assertIn("markdown:", javascript)
        self.assertIn("Markdown 关键截图", javascript)
        self.assertIn("下载关键截图 Markdown", javascript)
        self.assertIn("report_url", javascript)
        self.assertNotIn("https://", frontend)
        self.assertNotIn('src="http', frontend)
        escaped = _frontend_html(
            token="test-token",
            default_video='</script><script>alert("x")</script>',
        )
        self.assertNotIn('</script><script>alert("x")</script>', escaped)
        self.assertIsNotNone(RUN_NAME_PATTERN.fullmatch("episode-124-v1"))
        self.assertIsNotNone(RUN_NAME_PATTERN.fullmatch("会员第124期"))
        self.assertIsNone(RUN_NAME_PATTERN.fullmatch("../escape"))


if __name__ == "__main__":
    unittest.main()
