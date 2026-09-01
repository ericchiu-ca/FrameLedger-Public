from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frameledger.asr import (
    AsrBackendError,
    FIXED_ZERO_DECODING_PROFILE,
    MlxWhisperAsrBackend,
    STANDARD_FALLBACK_DECODING_PROFILE,
    _normalize_transcript,
    run_local_asr,
)
from frameledger.cli import build_parser
from frameledger.models import VideoMetadata


MODEL_REVISION = "a" * 40


class _FakeBackend:
    def __init__(self, *, fail: bool = False, result=None) -> None:
        self.fail = fail
        self.result = result
        self.calls: list[dict[str, object]] = []

    def describe(self):
        return {
            "name": "fake_mlx_whisper",
            "runtime": {
                "elapsed_seconds": 0.5,
                "model": {"tree_sha256": "b" * 64},
            },
        }

    def transcribe(
        self,
        audio_path,
        *,
        language,
        task,
        word_timestamps,
        initial_prompt=None,
        decoding_profile=None,
    ):
        self.calls.append(
            {
                "audio_path": str(audio_path),
                "language": language,
                "task": task,
                "word_timestamps": word_timestamps,
                "initial_prompt": initial_prompt,
                "decoding_profile": decoding_profile,
            }
        )
        if self.fail:
            raise RuntimeError("synthetic ASR failure")
        if self.result is not None:
            return self.result
        return {
            "language": "zh",
            "text": " 做空真的能赚钱吗？",
            "segments": [
                {
                    "id": 0,
                    "start": 0.5,
                    "end": 1.5,
                    "text": " 做空真的能赚钱吗？",
                    "tokens": [1, 2, 3],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.1,
                    "no_speech_prob": 0.01,
                    "words": [
                        {
                            "word": " 做空",
                            "start": 0.5,
                            "end": 0.9,
                            "probability": 0.95,
                        },
                        {
                            "word": "真的能赚钱吗？",
                            "start": 0.9,
                            "end": 1.5,
                            "probability": 0.9,
                        },
                    ],
                }
            ],
        }


class LocalAsrTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source_dir = root / "source"
        source_dir.mkdir()
        source = source_dir / "episode.mp4"
        source.write_bytes(b"read-only-source-video")
        fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata = VideoMetadata(
            path=source,
            size_bytes=source.stat().st_size,
            mtime_ns=source.stat().st_mtime_ns,
            duration_seconds=30.0,
            fps=20.0,
            width=1920,
            height=1080,
            frame_count=600,
            codec="h264",
            fingerprint=fingerprint,
        )
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        ffmpeg = root / "fake-ffmpeg"
        ffmpeg.write_text(
            "\n".join(
                [
                    f"#!{sys.executable}",
                    "import sys, wave",
                    "from pathlib import Path",
                    "if '-version' in sys.argv:",
                    "    print('ffmpeg version test-v1')",
                    "    raise SystemExit(0)",
                    "destination = Path(sys.argv[-1])",
                    "with wave.open(str(destination), 'wb') as audio:",
                    "    audio.setnchannels(1)",
                    "    audio.setsampwidth(2)",
                    "    audio.setframerate(16000)",
                    "    audio.writeframes(b'\\0\\0' * 32000)",
                ]
            ),
            encoding="utf-8",
        )
        ffmpeg.chmod(0o700)
        return source, metadata, model, ffmpeg

    def test_bounded_asr_writes_bound_audio_transcript_and_offline_review(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, metadata, model, ffmpeg = self._fixture(root)
            output = root / "output"
            backend = _FakeBackend()
            original = source.read_bytes()

            with patch("frameledger.asr.probe_video", return_value=metadata):
                result = run_local_asr(
                    source,
                    start_seconds=10.0,
                    duration_seconds=2.0,
                    model_path=model,
                    model_id="mlx-community/test-model",
                    model_revision=MODEL_REVISION,
                    language="zh",
                    task="transcribe",
                    output=output,
                    ffmpeg=ffmpeg,
                    backend=backend,
                )

            document = json.loads((output / "asr.json").read_text(encoding="utf-8"))
            review = (output / "review.html").read_text(encoding="utf-8")
            self.assertEqual(result["kind"], "local_speech_transcription")
            self.assertEqual(document["status"], "ok")
            self.assertEqual(document["source"]["fingerprint"], metadata.fingerprint)
            self.assertEqual(document["parameters"]["policy_version"], "bounded-local-asr-v1")
            self.assertEqual(document["parameters"]["model_revision"], MODEL_REVISION)
            self.assertEqual(document["parameters"]["prompt_mode"], "unprompted")
            self.assertIsNone(document["parameters"]["initial_prompt"])
            self.assertIsNone(document["parameters"]["initial_prompt_sha256"])
            self.assertEqual(document["parameters"]["initial_prompt_character_count"], 0)
            self.assertFalse(document["parameters"]["long_range_explicitly_allowed"])
            self.assertEqual(
                document["parameters"]["decoding_profile"],
                FIXED_ZERO_DECODING_PROFILE,
            )
            self.assertEqual(document["parameters"]["temperature"], 0.0)
            self.assertEqual(document["parameters"]["compression_ratio_threshold"], 2.4)
            self.assertEqual(document["parameters"]["logprob_threshold"], -1.0)
            self.assertEqual(document["parameters"]["no_speech_threshold"], 0.6)
            self.assertTrue(document["parameters"]["condition_on_previous_text"])
            self.assertEqual(document["audio"]["sample_rate_hz"], 16000)
            self.assertEqual(document["audio"]["duration_seconds"], 2.0)
            self.assertEqual(
                document["transcript"]["segments"][0]["absolute_start_seconds"],
                10.5,
            )
            self.assertEqual(
                document["transcript"]["segments"][0]["words"][0]["absolute_start_seconds"],
                10.5,
            )
            self.assertEqual(document["summary"]["segment_count"], 1)
            self.assertEqual(document["summary"]["word_count"], 2)
            self.assertEqual(document["summary"]["real_time_factor"], 0.25)
            self.assertTrue(document["summary"]["quality_checks"]["passed"])
            self.assertEqual(
                document["summary"]["quality_checks"]["max_segment_compression_ratio"],
                1.1,
            )
            self.assertEqual(source.read_bytes(), original)
            self.assertIn("<audio controls", review)
            self.assertIn("做空真的能赚钱吗", review)
            self.assertNotIn("https://", review)
            self.assertIn("Inference mode: unprompted", review)
            self.assertIn("prompt SHA-256 none · 0 characters", review)
            self.assertIn("no post-processing correction", review)
            self.assertEqual(backend.calls[0]["language"], "zh")
            self.assertIsNone(backend.calls[0]["initial_prompt"])
            self.assertEqual(
                backend.calls[0]["decoding_profile"],
                FIXED_ZERO_DECODING_PROFILE,
            )

    def test_tail_timestamp_quantization_is_clamped_and_disclosed(self):
        transcript = _normalize_transcript(
            {
                "language": "zh",
                "text": "尾声",
                "segments": [
                    {
                        "id": 0,
                        "start": 9.5,
                        "end": 10.02,
                        "text": "尾声",
                        "tokens": [],
                        "words": [
                            {
                                "word": "尾声",
                                "start": 9.5,
                                "end": 10.02,
                                "probability": 0.9,
                            }
                        ],
                    }
                ],
            },
            absolute_offset=100.0,
            audio_duration=10.0,
        )
        segment = transcript["segments"][0]
        self.assertEqual(segment["relative_end_seconds"], 10.0)
        self.assertEqual(segment["absolute_end_seconds"], 110.0)
        self.assertEqual(segment["source_relative_end_seconds"], 10.02)
        self.assertTrue(segment["end_clamped_to_audio_duration"])
        self.assertEqual(transcript["timestamp_clamp_count"], 2)

    def test_prompted_asr_records_exact_trimmed_prompt_and_review_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, metadata, model, ffmpeg = self._fixture(root)
            output = root / "prompted-output"
            backend = _FakeBackend()
            prompt = "头寸平仓。借入股票卖空。"

            with patch("frameledger.asr.probe_video", return_value=metadata):
                run_local_asr(
                    source,
                    start_seconds=10.0,
                    duration_seconds=2.0,
                    model_path=model,
                    model_id="mlx-community/test-model",
                    model_revision=MODEL_REVISION,
                    language="zh",
                    task="transcribe",
                    output=output,
                    ffmpeg=ffmpeg,
                    backend=backend,
                    initial_prompt=f"  {prompt}  ",
                    decoding_profile=STANDARD_FALLBACK_DECODING_PROFILE,
                )

            document = json.loads((output / "asr.json").read_text(encoding="utf-8"))
            review = (output / "review.html").read_text(encoding="utf-8")
            parameters = document["parameters"]
            self.assertEqual(parameters["prompt_mode"], "prompted")
            self.assertEqual(parameters["initial_prompt"], prompt)
            self.assertEqual(
                parameters["initial_prompt_sha256"],
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(parameters["initial_prompt_character_count"], len(prompt))
            self.assertEqual(
                parameters["decoding_profile"],
                STANDARD_FALLBACK_DECODING_PROFILE,
            )
            self.assertEqual(parameters["temperature"], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            self.assertEqual(backend.calls[0]["initial_prompt"], prompt)
            self.assertEqual(
                backend.calls[0]["decoding_profile"],
                STANDARD_FALLBACK_DECODING_PROFILE,
            )
            self.assertIn("Inference mode: prompted", review)
            self.assertIn(parameters["initial_prompt_sha256"], review)
            self.assertIn(f"{len(prompt)} characters", review)

    def test_transcription_failure_is_preserved_in_machine_ledger(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, metadata, model, ffmpeg = self._fixture(root)
            output = root / "output"
            with patch("frameledger.asr.probe_video", return_value=metadata):
                with self.assertRaisesRegex(RuntimeError, "during transcription"):
                    run_local_asr(
                        source,
                        start_seconds=10.0,
                        duration_seconds=2.0,
                        model_path=model,
                        model_id="mlx-community/test-model",
                        model_revision=MODEL_REVISION,
                        language="zh",
                        task="transcribe",
                        output=output,
                        ffmpeg=ffmpeg,
                        backend=_FakeBackend(fail=True),
                        initial_prompt="头寸平仓",
                    )
            document = json.loads((output / "asr.json").read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "error")
            self.assertEqual(document["failure"]["stage"], "transcription")
            self.assertIn("synthetic ASR failure", document["failure"]["error"])
            self.assertEqual(document["parameters"]["prompt_mode"], "prompted")
            self.assertEqual(document["parameters"]["initial_prompt"], "头寸平仓")
            self.assertTrue((output / "audio.wav").is_file())

    def test_transcript_quality_gate_preserves_raw_failure_without_review(self):
        high_compression_result = {
            "language": "zh",
            "text": "就是",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "就是",
                    "tokens": [1],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 2.400001,
                    "no_speech_prob": 0.0,
                    "words": [],
                }
            ],
        }
        repeated_segments_result = {
            "language": "zh",
            "text": "就是就是就是",
            "segments": [
                {
                    "id": index,
                    "start": index * 0.5,
                    "end": (index + 1) * 0.5,
                    "text": " 就是 ",
                    "tokens": [index + 1],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                    "words": [],
                }
                for index in range(3)
            ],
        }
        replacement_character_result = {
            "language": "zh",
            "text": "转写包含�替换字符",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "转写包含�替换字符",
                    "tokens": [1],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                    "words": [],
                }
            ],
        }
        repeated_fragment = "这是一个足够长的重复转写片段用于检测模型在同一段内发生病态循环行为"
        repeated_ngram_result = {
            "language": "zh",
            "text": repeated_fragment + "中间过渡" + repeated_fragment,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": repeated_fragment + "中间过渡" + repeated_fragment,
                    "tokens": [1],
                    "temperature": 0.0,
                    "avg_logprob": -0.1,
                    "compression_ratio": 1.0,
                    "no_speech_prob": 0.0,
                    "words": [],
                }
            ],
        }
        cases = (
            (
                "compression",
                high_compression_result,
                "compression_ratio_failed",
                "max_segment_compression_ratio",
                2.400001,
            ),
            (
                "repetition",
                repeated_segments_result,
                "consecutive_identical_nonempty_segments_failed",
                "max_consecutive_identical_nonempty_segments",
                3,
            ),
            (
                "replacement-character",
                replacement_character_result,
                "replacement_character_failed",
                "replacement_character_count",
                1,
            ),
            (
                "repeated-ngram",
                repeated_ngram_result,
                "repeated_character_ngram_failed",
                "repeated_character_ngram_offender_count",
                1,
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for label, result, failure_key, metric_key, expected_metric in cases:
                with self.subTest(label=label):
                    case_root = root / label
                    case_root.mkdir()
                    source, metadata, model, ffmpeg = self._fixture(case_root)
                    output = case_root / "output"
                    with patch("frameledger.asr.probe_video", return_value=metadata):
                        with self.assertRaisesRegex(RuntimeError, "quality gate"):
                            run_local_asr(
                                source,
                                start_seconds=10.0,
                                duration_seconds=2.0,
                                model_path=model,
                                model_id="mlx-community/test-model",
                                model_revision=MODEL_REVISION,
                                language="zh",
                                task="transcribe",
                                output=output,
                                ffmpeg=ffmpeg,
                                backend=_FakeBackend(result=result),
                            )
                    document = json.loads(
                        (output / "asr.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(document["status"], "error")
                    self.assertEqual(document["failure"]["stage"], "transcript_quality")
                    self.assertFalse(document["quality_checks"]["passed"])
                    self.assertTrue(document["quality_checks"][failure_key])
                    self.assertEqual(
                        document["quality_checks"][metric_key],
                        expected_metric,
                    )
                    self.assertEqual(document["transcript"]["text"], result["text"])
                    self.assertTrue((output / "audio.wav").is_file())
                    self.assertFalse((output / "review.html").exists())

    def test_range_revision_and_sidecar_guards_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, metadata, model, ffmpeg = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "limited to 600"):
                run_local_asr(
                    source,
                    start_seconds=0,
                    duration_seconds=601,
                    model_path=model,
                    model_id="model",
                    model_revision=MODEL_REVISION,
                    language="zh",
                    task="transcribe",
                    output=root / "too-long",
                    ffmpeg=ffmpeg,
                    backend=_FakeBackend(),
                )
            metadata = replace(metadata, duration_seconds=700.0, frame_count=14000)
            allowed_output = root / "explicit-long-range"
            with (
                patch("frameledger.asr.probe_video", return_value=metadata),
                patch(
                    "frameledger.asr._extract_audio",
                    side_effect=RuntimeError("synthetic extraction stop"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "audio extraction"):
                    run_local_asr(
                        source,
                        start_seconds=0,
                        duration_seconds=601,
                        model_path=model,
                        model_id="model",
                        model_revision=MODEL_REVISION,
                        language="zh",
                        task="transcribe",
                        output=allowed_output,
                        ffmpeg=ffmpeg,
                        backend=_FakeBackend(),
                        allow_long_range=True,
                    )
            allowed_document = json.loads(
                (allowed_output / "asr.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                allowed_document["parameters"]["long_range_explicitly_allowed"]
            )
            with self.assertRaisesRegex(ValueError, "40-character"):
                run_local_asr(
                    source,
                    start_seconds=0,
                    duration_seconds=2,
                    model_path=model,
                    model_id="model",
                    model_revision="main",
                    language="zh",
                    task="transcribe",
                    output=root / "bad-revision",
                    ffmpeg=ffmpeg,
                    backend=_FakeBackend(),
                )
            invalid_profile_output = root / "bad-decoding-profile"
            with self.assertRaisesRegex(ValueError, "decoding profile"):
                run_local_asr(
                    source,
                    start_seconds=0,
                    duration_seconds=2,
                    model_path=model,
                    model_id="model",
                    model_revision=MODEL_REVISION,
                    language="zh",
                    task="transcribe",
                    output=invalid_profile_output,
                    ffmpeg=ffmpeg,
                    backend=_FakeBackend(),
                    decoding_profile="unknown",
                )
            self.assertFalse(invalid_profile_output.exists())
            with patch("frameledger.asr.probe_video", return_value=metadata):
                with self.assertRaisesRegex(ValueError, "beside or below"):
                    run_local_asr(
                        source,
                        start_seconds=0,
                        duration_seconds=2,
                        model_path=model,
                        model_id="model",
                        model_revision=MODEL_REVISION,
                        language="zh",
                        task="transcribe",
                        output=source.parent / "sidecar-output",
                        ffmpeg=ffmpeg,
                        backend=_FakeBackend(),
                    )

    def test_initial_prompt_guards_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source, _metadata, model, ffmpeg = self._fixture(root)
            invalid_prompts = (
                ("empty", "   ", "must not be empty"),
                ("control", "头寸\n平仓", "control characters"),
                ("too-long", "术" * 1001, "limited to 1000"),
            )
            for label, prompt, message in invalid_prompts:
                with self.subTest(label=label):
                    output = root / f"bad-prompt-{label}"
                    with self.assertRaisesRegex(ValueError, message):
                        run_local_asr(
                            source,
                            start_seconds=0,
                            duration_seconds=2,
                            model_path=model,
                            model_id="model",
                            model_revision=MODEL_REVISION,
                            language="zh",
                            task="transcribe",
                            output=output,
                            ffmpeg=ffmpeg,
                            backend=_FakeBackend(),
                            initial_prompt=prompt,
                        )
                    self.assertFalse(output.exists())

    def test_mlx_adapter_uses_strict_helper_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            helper = root / "helper"
            helper.write_bytes(b"helper")
            helper.chmod(0o700)
            model = root / "model"
            model.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"wav")
            response = {
                "protocol": "frameledger-asr-helper-v1",
                "engine": {"elapsed_seconds": 1.0},
                "result": {"language": "zh", "text": "测试", "segments": []},
            }
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(response),
                stderr="",
            )
            backend = MlxWhisperAsrBackend(model_path=model, helper=helper)
            with patch("frameledger.asr.subprocess.run", return_value=completed) as run:
                result = backend.transcribe(
                    audio,
                    language="zh",
                    task="transcribe",
                    word_timestamps=True,
                )
            request = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(request["protocol"], "frameledger-asr-helper-v1")
            self.assertEqual(request["model_path"], str(model.resolve()))
            self.assertEqual(request["word_timestamps"], True)
            self.assertNotIn("initial_prompt", request)
            self.assertEqual(result["text"], "测试")
            self.assertEqual(backend.describe()["runtime"]["elapsed_seconds"], 1.0)

    def test_mlx_adapter_requires_matching_prompt_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            helper = root / "helper"
            helper.write_bytes(b"helper")
            helper.chmod(0o700)
            model = root / "model"
            model.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"wav")
            prompt = "头寸平仓"
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            response = {
                "protocol": "frameledger-asr-helper-v1",
                "engine": {
                    "elapsed_seconds": 1.0,
                    "initial_prompt_sha256": prompt_sha256,
                    "initial_prompt_character_count": len(prompt),
                },
                "result": {"language": "zh", "text": "头寸平仓", "segments": []},
            }
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(response),
                stderr="",
            )
            backend = MlxWhisperAsrBackend(model_path=model, helper=helper)
            with patch("frameledger.asr.subprocess.run", return_value=completed) as run:
                backend.transcribe(
                    audio,
                    language="zh",
                    task="transcribe",
                    word_timestamps=True,
                    initial_prompt=prompt,
                )
            request = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(request["initial_prompt"], prompt)

            mismatched_response = {
                **response,
                "engine": {
                    **response["engine"],
                    "initial_prompt_sha256": "0" * 64,
                },
            }
            mismatched = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(mismatched_response),
                stderr="",
            )
            backend = MlxWhisperAsrBackend(model_path=model, helper=helper)
            with patch("frameledger.asr.subprocess.run", return_value=mismatched):
                with self.assertRaisesRegex(AsrBackendError, "confirm the requested"):
                    backend.transcribe(
                        audio,
                        language="zh",
                        task="transcribe",
                        word_timestamps=True,
                        initial_prompt=prompt,
                    )
            self.assertEqual(
                backend.describe()["runtime"]["initial_prompt_sha256"],
                "0" * 64,
            )

    def test_mlx_adapter_requires_matching_explicit_decoding_profile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            helper = root / "helper"
            helper.write_bytes(b"helper")
            helper.chmod(0o700)
            model = root / "model"
            model.mkdir()
            audio = root / "audio.wav"
            audio.write_bytes(b"wav")
            engine = {
                "decoding_profile": STANDARD_FALLBACK_DECODING_PROFILE,
                "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
            }
            response = {
                "protocol": "frameledger-asr-helper-v1",
                "engine": engine,
                "result": {"language": "zh", "text": "测试", "segments": []},
            }
            completed = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(response),
                stderr="",
            )
            backend = MlxWhisperAsrBackend(model_path=model, helper=helper)
            with patch("frameledger.asr.subprocess.run", return_value=completed) as run:
                backend.transcribe(
                    audio,
                    language="zh",
                    task="transcribe",
                    word_timestamps=True,
                    decoding_profile=STANDARD_FALLBACK_DECODING_PROFILE,
                )
            request = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(
                request["decoding_profile"],
                STANDARD_FALLBACK_DECODING_PROFILE,
            )

            mismatched = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        **response,
                        "engine": {**engine, "compression_ratio_threshold": 99.0},
                    }
                ),
                stderr="",
            )
            backend = MlxWhisperAsrBackend(model_path=model, helper=helper)
            with patch("frameledger.asr.subprocess.run", return_value=mismatched):
                with self.assertRaisesRegex(AsrBackendError, "decoding profile"):
                    backend.transcribe(
                        audio,
                        language="zh",
                        task="transcribe",
                        word_timestamps=True,
                        decoding_profile=STANDARD_FALLBACK_DECODING_PROFILE,
                    )
            self.assertEqual(
                backend.describe()["runtime"]["compression_ratio_threshold"],
                99.0,
            )

    def test_cli_requires_bounded_local_model_inputs(self):
        args = build_parser().parse_args(
            [
                "asr",
                "/tmp/video.mp4",
                "--start",
                "00:02:00",
                "--duration",
                "00:03:00",
                "--model",
                "/tmp/model",
                "--model-revision",
                MODEL_REVISION,
                "--initial-prompt",
                "头寸平仓",
                "--mlx-whisper-helper",
                "/tmp/helper",
                "--output",
                "/tmp/asr",
            ]
        )
        self.assertEqual(args.command, "asr")
        self.assertEqual(args.start, 120.0)
        self.assertEqual(args.duration, 180.0)
        self.assertEqual(args.language, "zh")
        self.assertEqual(args.task, "transcribe")
        self.assertEqual(args.model_revision, MODEL_REVISION)
        self.assertEqual(args.initial_prompt, "头寸平仓")
        self.assertEqual(args.decoding_profile, FIXED_ZERO_DECODING_PROFILE)
        self.assertFalse(args.allow_long_range)

        explicit_long = build_parser().parse_args(
            [
                "asr",
                "/tmp/video.mp4",
                "--start",
                "0",
                "--duration",
                "00:27:00",
                "--model",
                "/tmp/model",
                "--model-revision",
                MODEL_REVISION,
                "--mlx-whisper-helper",
                "/tmp/helper",
                "--allow-long-range",
                "--output",
                "/tmp/asr-full",
            ]
        )
        self.assertTrue(explicit_long.allow_long_range)
