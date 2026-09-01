from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "phase2/mlx_whisper_asr/main.py"


class MlxWhisperAsrHelperTests(unittest.TestCase):
    def _valid_request_fixture(self, root: Path) -> tuple[dict[str, object], dict[str, str]]:
        audio = root / "audio.wav"
        audio.write_bytes(b"wav")
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
        (root / "mlx_whisper.py").write_text(
            "def transcribe(audio, **kwargs):\n"
            "    return {\"language\": kwargs[\"language\"], "
            "\"text\": kwargs.get(\"initial_prompt\", \"__ABSENT__\"), "
            "\"segments\": [], \"decode_options\": kwargs}\n",
            encoding="utf-8",
        )
        request: dict[str, object] = {
            "protocol": "frameledger-asr-helper-v1",
            "audio_path": str(audio),
            "model_path": str(model),
            "language": "zh",
            "task": "transcribe",
            "word_timestamps": True,
        }
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root)
        return request, environment

    def test_usage_error_keeps_stdout_empty_and_writes_json_stderr(self):
        completed = subprocess.run(
            [sys.executable, str(HELPER), "unexpected"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        error = json.loads(completed.stderr)
        self.assertEqual(error["protocol"], "frameledger-asr-helper-v1")
        self.assertIn("no arguments", error["error"])

    def test_invalid_request_fails_before_importing_mlx(self):
        completed = subprocess.run(
            [sys.executable, str(HELPER)],
            input="{}",
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        error = json.loads(completed.stderr)
        self.assertEqual(error["error_type"], "HelperError")
        self.assertIn("protocol", error["error"])

    def test_prompt_is_trimmed_passed_and_echoed_while_absence_remains_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request, environment = self._valid_request_fixture(root)
            prompt = "头寸平仓。借入股票卖空。"
            prompted_request = {**request, "initial_prompt": f"  {prompt}  "}
            completed = subprocess.run(
                [sys.executable, str(HELPER)],
                input=json.dumps(prompted_request, ensure_ascii=False),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = json.loads(completed.stdout)
            self.assertEqual(response["result"]["text"], prompt)
            self.assertEqual(
                response["engine"]["initial_prompt_sha256"],
                hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(
                response["engine"]["initial_prompt_character_count"],
                len(prompt),
            )
            self.assertEqual(response["engine"]["temperature"], 0.0)
            self.assertEqual(response["engine"]["decoding_profile"], "fixed_zero_v1")
            self.assertEqual(response["engine"]["compression_ratio_threshold"], 2.4)
            self.assertEqual(response["engine"]["logprob_threshold"], -1.0)
            self.assertEqual(response["engine"]["no_speech_threshold"], 0.6)
            self.assertTrue(response["engine"]["condition_on_previous_text"])
            self.assertEqual(response["result"]["decode_options"]["temperature"], 0.0)
            self.assertEqual(
                response["engine"]["implementation_sha256"],
                hashlib.sha256(HELPER.read_bytes()).hexdigest(),
            )

            completed = subprocess.run(
                [sys.executable, str(HELPER)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = json.loads(completed.stdout)
            self.assertEqual(response["result"]["text"], "__ABSENT__")
            self.assertIsNone(response["engine"]["initial_prompt_sha256"])
            self.assertEqual(response["engine"]["initial_prompt_character_count"], 0)

    def test_standard_fallback_profile_is_passed_and_echoed_exactly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request, environment = self._valid_request_fixture(root)
            request["decoding_profile"] = "standard_fallback_v1"
            completed = subprocess.run(
                [sys.executable, str(HELPER)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            response = json.loads(completed.stdout)
            expected_temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            self.assertEqual(
                response["engine"]["decoding_profile"],
                "standard_fallback_v1",
            )
            self.assertEqual(response["engine"]["temperature"], expected_temperature)
            self.assertEqual(response["engine"]["compression_ratio_threshold"], 2.4)
            self.assertEqual(response["engine"]["logprob_threshold"], -1.0)
            self.assertEqual(response["engine"]["no_speech_threshold"], 0.6)
            self.assertTrue(response["engine"]["condition_on_previous_text"])
            decode_options = response["result"]["decode_options"]
            self.assertEqual(decode_options["temperature"], expected_temperature)
            self.assertEqual(decode_options["compression_ratio_threshold"], 2.4)
            self.assertEqual(decode_options["logprob_threshold"], -1.0)
            self.assertEqual(decode_options["no_speech_threshold"], 0.6)
            self.assertTrue(decode_options["condition_on_previous_text"])

    def test_invalid_decoding_profile_fails_before_importing_mlx(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request, environment = self._valid_request_fixture(root)
            request["decoding_profile"] = "unknown"
            completed = subprocess.run(
                [sys.executable, str(HELPER)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
                env=environment,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")
            error = json.loads(completed.stderr)
            self.assertEqual(error["error_type"], "HelperError")
            self.assertIn("decoding_profile", error["error"])

    def test_invalid_initial_prompts_fail_with_json_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request, environment = self._valid_request_fixture(root)
            invalid_prompts = (
                ("empty", "   ", "must not be empty"),
                ("control", "头寸\n平仓", "control characters"),
                ("too-long", "术" * 1001, "limited to 1000"),
            )
            for label, prompt, message in invalid_prompts:
                with self.subTest(label=label):
                    completed = subprocess.run(
                        [sys.executable, str(HELPER)],
                        input=json.dumps(
                            {**request, "initial_prompt": prompt},
                            ensure_ascii=False,
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=15,
                        env=environment,
                    )
                    self.assertEqual(completed.returncode, 2)
                    self.assertEqual(completed.stdout, "")
                    error = json.loads(completed.stderr)
                    self.assertEqual(error["error_type"], "HelperError")
                    self.assertIn(message, error["error"])
