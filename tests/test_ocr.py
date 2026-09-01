from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frameledger.cli import build_parser, main
from frameledger.ocr import AppleVisionOcrBackend, OcrObservation, run_frame_ocr


class _FakeBackend:
    def __init__(self, *, fail_route: str | None = None) -> None:
        self.fail_route = fail_route
        self.calls: list[dict[str, object]] = []

    def describe(self):
        return {"name": "fake_ocr", "version": "test-v1"}

    def recognize(
        self,
        image_path,
        *,
        languages,
        route_kind,
        roi_normalized,
    ):
        self.calls.append(
            {
                "image_path": image_path,
                "languages": languages,
                "route_kind": route_kind,
                "roi_normalized": roi_normalized,
            }
        )
        if route_kind == self.fail_route or self.fail_route == "all":
            raise RuntimeError("synthetic OCR failure")
        if route_kind == "presentation":
            return [
                OcrObservation(
                    text="做空的几种方式",
                    confidence=0.98,
                    bbox=(0.1, 0.1, 0.4, 0.1),
                )
            ]
        return [
            {
                "text": "FB 13.98",
                "confidence": 0.91,
                "bbox": [0.1, 0.2, 0.3, 0.05],
            }
        ]


class FrameOcrTests(unittest.TestCase):
    @staticmethod
    def _make_run(root: Path, *, escaping_image: bool = False) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        media_root = root / "media"
        media_root.mkdir()
        source = media_root / "source.mp4"
        source.write_bytes(b"source placeholder")

        run = root / "phase1-run"
        frames = run / "frames"
        strategies = run / "strategies"
        frames.mkdir(parents=True)
        strategies.mkdir()
        for name, payload in (("p.png", b"ppt"), ("t.png", b"table"), ("c.png", b"chart")):
            (frames / name).write_bytes(payload)

        segments = [
            {"segment_id": "seg-p", "kind": "presentation"},
            {"segment_id": "seg-t", "kind": "table"},
            {"segment_id": "seg-c", "kind": "chart"},
        ]
        manifest = {
            "kind": "candidate_frame_benchmark",
            "source": {
                "path": str(source),
                "fingerprint": "a" * 64,
            },
            "routing": {"segments": segments},
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        strategy = {
            "strategy": "routed",
            "segments": segments,
            "selected": [
                {
                    "sample_index": 1,
                    "timestamp": 5.0,
                    "segment_id": "seg-p",
                    "image_path": "../outside.png" if escaping_image else "frames/p.png",
                },
                {
                    "sample_index": 2,
                    "timestamp": 10.0,
                    "segment_id": "seg-t",
                    "image_path": "frames/t.png",
                },
                {
                    "sample_index": 3,
                    "timestamp": 15.0,
                    "segment_id": "seg-c",
                    "image_path": "frames/c.png",
                },
            ],
        }
        (strategies / "routed.json").write_text(json.dumps(strategy), encoding="utf-8")
        if escaping_image:
            (root / "outside.png").write_bytes(b"outside")
        return run, source

    def test_ocr_reads_frozen_run_and_emits_bound_success_and_skip_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, _source = self._make_run(root)
            output = root / "ocr-output"
            backend = _FakeBackend()

            result = run_frame_ocr(
                run,
                strategy="routed",
                routes=["presentation", "table"],
                engine="apple_vision",
                languages=["zh-Hans", "en-US"],
                output=output,
                backend=backend,
            )

            document = json.loads((output / "ocr.json").read_text(encoding="utf-8"))
            review = (output / "review.html").read_text(encoding="utf-8")
            self.assertEqual(result["kind"], "frame_ocr")
            self.assertEqual(result["review_html"], str((output / "review.html").resolve()))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["parameters"]["policy_version"], "route-roi-v2")
            self.assertEqual(document["source"]["video_sha256"], "a" * 64)
            self.assertEqual(
                document["source"]["manifest_sha256"],
                hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                document["source"]["strategy_sha256"],
                hashlib.sha256((run / "strategies/routed.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(len(document["frames"]), 2)
            self.assertEqual(len(document["failures"]), 0)
            self.assertEqual(len(document["skipped"]), 1)
            self.assertEqual(document["skipped"][0]["route_kind"], "chart")
            self.assertEqual(document["skipped"][0]["reason"], "skipped_by_route_policy")
            self.assertEqual(document["frames"][0]["segment_id"], "seg-p")
            self.assertEqual(document["frames"][0]["plain_text"], "做空的几种方式")
            self.assertEqual(
                document["frames"][0]["observations"][0]["bbox_origin"],
                "top_left_normalized",
            )
            self.assertEqual(
                document["frames"][0]["image_sha256"],
                hashlib.sha256((run / "frames/p.png").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                [call["route_kind"] for call in backend.calls],
                ["presentation", "table"],
            )
            self.assertEqual(backend.calls[0]["languages"], ("zh-Hans", "en-US"))
            self.assertEqual(backend.calls[0]["roi_normalized"], (0.0, 0.0, 1.0, 0.92))
            self.assertEqual(backend.calls[1]["roi_normalized"], (0.0, 0.25, 1.0, 0.67))
            self.assertIn("做空的几种方式", review)
            self.assertIn("../phase1-run/frames/p.png", review)
            self.assertNotIn("https://", review)

    def test_backend_failure_is_recorded_without_losing_successful_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, _source = self._make_run(root)
            output = root / "ocr-output"

            run_frame_ocr(
                run,
                strategy="routed",
                routes=["presentation", "table"],
                engine="apple_vision",
                languages=["zh-Hans"],
                output=output,
                backend=_FakeBackend(fail_route="table"),
            )

            document = json.loads((output / "ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(document["summary"]["ocr_success_frames"], 1)
            self.assertEqual(document["summary"]["failure_frames"], 1)
            self.assertEqual(document["failures"][0]["route_kind"], "table")
            self.assertEqual(document["failures"][0]["error_type"], "RuntimeError")
            self.assertIn("synthetic OCR failure", document["failures"][0]["error"])
            self.assertEqual(
                document["failures"][0]["image_sha256"],
                hashlib.sha256((run / "frames/t.png").read_bytes()).hexdigest(),
            )

    def test_all_backend_failures_still_write_zero_success_ledger(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, _source = self._make_run(root)
            output = root / "ocr-output"

            result = run_frame_ocr(
                run,
                strategy="routed",
                routes=["presentation", "table"],
                engine="apple_vision",
                languages=["zh-Hans"],
                output=output,
                backend=_FakeBackend(fail_route="all"),
            )

            self.assertEqual(result["summary"]["ocr_success_frames"], 0)
            self.assertEqual(result["summary"]["failure_frames"], 2)
            self.assertTrue((output / "ocr.json").is_file())
            document = json.loads((output / "ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(document["frames"], [])
            self.assertEqual(len(document["failures"]), 2)

    def test_output_cannot_modify_phase1_run_and_images_cannot_escape_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run, _source = self._make_run(root)
            backend = _FakeBackend()
            with self.assertRaisesRegex(ValueError, "frozen Phase 1"):
                run_frame_ocr(
                    run,
                    strategy="routed",
                    routes=["presentation"],
                    engine="apple_vision",
                    languages=["zh-Hans"],
                    output=run / "ocr",
                    backend=backend,
                )
            self.assertEqual(backend.calls, [])

            escaping_run, _source = self._make_run(root / "escape-case", escaping_image=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                run_frame_ocr(
                    escaping_run,
                    strategy="routed",
                    routes=["presentation"],
                    engine="apple_vision",
                    languages=["zh-Hans"],
                    output=root / "escape-output",
                    backend=_FakeBackend(),
                )

    def test_apple_vision_adapter_uses_json_helper_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            helper = root / "vision-helper"
            helper.write_text("placeholder", encoding="utf-8")
            helper.chmod(0o700)
            image = root / "frame.png"
            image.write_bytes(b"png")
            response = {
                "protocol": "frameledger-ocr-helper-v1",
                "engine": {"revision": 3, "os_version": "test"},
                "observations": [
                    {
                        "text": "测试",
                        "confidence": 0.9,
                        "bbox": [0.1, 0.2, 0.3, 0.1],
                    }
                ],
            }
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(response),
                stderr="",
            )
            backend = AppleVisionOcrBackend(helper=helper)

            with patch("frameledger.ocr.subprocess.run", return_value=completed) as run:
                observations = backend.recognize(
                    image,
                    languages=("zh-Hans", "en-US"),
                    route_kind="presentation",
                    roi_normalized=(0.04, 0.02, 0.94, 0.88),
                )

            request = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(run.call_args.args[0], [str(helper.resolve())])
            self.assertEqual(
                backend.describe()["helper_sha256"],
                hashlib.sha256(helper.read_bytes()).hexdigest(),
            )
            self.assertEqual(request["protocol"], "frameledger-ocr-helper-v1")
            self.assertEqual(request["image_path"], str(image))
            self.assertEqual(request["languages"], ["zh-Hans", "en-US"])
            self.assertEqual(request["bbox_origin"], "top_left_normalized")
            self.assertEqual(observations[0]["text"], "测试")
            self.assertEqual(backend.describe()["runtime"]["revision"], 3)

    def test_external_fake_helper_executable_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            helper = root / "fake-vision-helper"
            helper.write_text(
                "\n".join(
                    [
                        f"#!{sys.executable}",
                        "import json, sys",
                        "request = json.load(sys.stdin)",
                        "json.dump({",
                        "    'protocol': request['protocol'],",
                        "    'engine': {'name': 'fake-native-helper'},",
                        "    'observations': [{",
                        "        'text': request['route_kind'],",
                        "        'confidence': 0.75,",
                        "        'bbox': [0.1, 0.2, 0.3, 0.1],",
                        "    }],",
                        "}, sys.stdout)",
                    ]
                ),
                encoding="utf-8",
            )
            helper.chmod(0o700)
            image = root / "frame.png"
            image.write_bytes(b"png")
            backend = AppleVisionOcrBackend(helper=helper)

            observations = backend.recognize(
                image,
                languages=("zh-Hans",),
                route_kind="presentation",
                roi_normalized=(0.04, 0.02, 0.94, 0.88),
            )

            self.assertEqual(observations[0]["text"], "presentation")
            self.assertEqual(backend.describe()["runtime"]["name"], "fake-native-helper")

    def test_cli_exposes_frozen_run_ocr_defaults(self):
        args = build_parser().parse_args(
            ["ocr", "/tmp/run", "--output", "/tmp/ocr-output"]
        )
        self.assertEqual(args.command, "ocr")
        self.assertEqual(args.strategy, "routed")
        self.assertEqual(args.routes, ["presentation", "table"])
        self.assertEqual(args.engine, "apple_vision")
        self.assertEqual(args.languages, ["zh-Hans", "en-US"])
        self.assertIsNone(args.apple_vision_helper)

        unavailable = AppleVisionOcrBackend(helper="/definitely/missing/helper")
        self.assertIsNone(unavailable.describe()["helper_sha256"])

        explicit = build_parser().parse_args(
            [
                "ocr",
                "/tmp/run",
                "--apple-vision-helper",
                "/tmp/vision-helper",
                "--output",
                "/tmp/ocr-output",
            ]
        )
        self.assertEqual(explicit.apple_vision_helper, Path("/tmp/vision-helper"))

    def test_cli_summary_exposes_zero_ocr_success(self):
        result = {
            "kind": "frame_ocr",
            "output": "/tmp/ocr-output",
            "ocr_json": "/tmp/ocr-output/ocr.json",
            "source": {"video_sha256": "a" * 64},
            "summary": {
                "selected_input_frames": 2,
                "ocr_success_frames": 0,
                "failure_frames": 2,
                "skipped_frames": 0,
                "observation_count": 0,
            },
        }
        stdout = io.StringIO()
        with (
            patch("frameledger.cli.run_frame_ocr", return_value=result),
            redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "ocr",
                    "/tmp/run",
                    "--output",
                    "/tmp/ocr-output",
                ]
            )

        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["ocr_summary"]["ocr_success_frames"], 0)
        self.assertEqual(summary["ocr_summary"]["failure_frames"], 2)


if __name__ == "__main__":
    unittest.main()
