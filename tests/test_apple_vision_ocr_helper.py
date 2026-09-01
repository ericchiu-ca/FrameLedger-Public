from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "phase2" / "apple_vision_ocr" / "main.swift"
KNOWN_SUCCESS_STDERR_LINES = {
    "IOServiceMatchingfailed for: AppleM2ScalerParavirtDriver",
}


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def _write_fixture_png(path: Path) -> None:
    """Write a dependency-free, high-contrast PNG suitable for a Vision smoke."""
    width, height = 640, 160
    pixels = bytearray([255] * (width * height * 3))
    glyphs = {
        "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
        "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    }
    scale = 12
    cursor_x = 42
    top = 35
    for character in "HELLO 123":
        if character == " ":
            cursor_x += 4 * scale
            continue
        for row, bits in enumerate(glyphs[character]):
            for column, bit in enumerate(bits):
                if bit != "1":
                    continue
                for dy in range(scale):
                    for dx in range(scale):
                        x = cursor_x + column * scale + dx
                        y = top + row * scale + dy
                        offset = (y * width + x) * 3
                        pixels[offset : offset + 3] = b"\x00\x00\x00"
        cursor_x += 7 * scale

    scanlines = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


class AppleVisionOCRHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if shutil.which("xcrun") is None:
            raise unittest.SkipTest("Apple xcrun is unavailable")
        cls._temporary = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls._temporary.name)
        cls.binary = cls.temp_root / "frameledger-vision-ocr"
        compile_environment = os.environ.copy()
        xcode_beta = Path("/Applications/Xcode-beta.app/Contents/Developer")
        if xcode_beta.is_dir():
            compile_environment["DEVELOPER_DIR"] = str(xcode_beta)
        module_cache = cls.temp_root / "module-cache"
        module_cache.mkdir()
        compile_environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
        compile_environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
        compile_result = subprocess.run(
            [
                "xcrun",
                "swiftc",
                str(SOURCE),
                "-o",
                str(cls.binary),
                "-framework",
                "Vision",
                "-framework",
                "CoreGraphics",
                "-framework",
                "CoreImage",
                "-framework",
                "ImageIO",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=compile_environment,
        )
        if compile_result.returncode != 0:
            raise AssertionError(
                "Swift helper failed to compile:\n"
                f"stdout:\n{compile_result.stdout}\n"
                f"stderr:\n{compile_result.stderr}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def _run(
        self,
        *arguments: str,
        stdin_payload: dict[str, object] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), *arguments],
            input=(json.dumps(stdin_payload) if stdin_payload is not None else None),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_usage_error_keeps_stdout_empty_and_writes_json_stderr(self) -> None:
        result = self._run()
        self.assertEqual(result.returncode, 64)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["schema_version"], 1)
        self.assertEqual(error["error"]["code"], "usage")

    def test_non_png_is_rejected_before_vision(self) -> None:
        fake_png = self.temp_root / "not-an-image.png"
        fake_png.write_text("not a PNG", encoding="utf-8")
        result = self._run("--image", str(fake_png), "--languages", "en-US")
        self.assertEqual(result.returncode, 65)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["error"]["code"], "invalid_input")

    def test_real_vision_smoke_or_explicit_native_code_8_skip(self) -> None:
        fixture = self.temp_root / "hello-123.png"
        _write_fixture_png(fixture)
        requested_roi = [0.05, 0.10, 0.90, 0.80]
        result = self._run(
            stdin_payload={
                "protocol": "frameledger-ocr-helper-v1",
                "image_path": str(fixture),
                "languages": ["en-US"],
                "route_kind": "presentation",
                "roi_normalized": requested_roi,
                "bbox_origin": "top_left_normalized",
            }
        )
        if result.returncode != 0:
            self.assertEqual(result.stdout, "")
            error = json.loads(result.stderr)["error"]
            if (
                error.get("code") == "vision_request_failed"
                and (
                    error.get("native_code") == 8
                    or (
                        error.get("native_domain") == "Foundation._GenericObjCError"
                        and error.get("native_code") == 0
                    )
                )
            ):
                self.skipTest(
                    "Apple Vision compiled but is unavailable in this sandbox "
                    f"({error.get('native_domain')} code {error.get('native_code')})"
                )
            self.fail(f"Unexpected Vision failure: {error}")

        payload = json.loads(result.stdout)
        unexpected_stderr = [
            line
            for line in result.stderr.splitlines()
            if line and line not in KNOWN_SUCCESS_STDERR_LINES
        ]
        self.assertEqual(unexpected_stderr, [])
        self.assertEqual(
            set(payload), {"protocol", "engine", "observations"}
        )
        self.assertEqual(payload["protocol"], "frameledger-ocr-helper-v1")
        self.assertEqual(payload["engine"]["bbox_origin"], "top_left_normalized")
        self.assertEqual(
            payload["engine"]["reading_order"],
            "top_to_bottom_then_left_to_right",
        )
        self.assertEqual(payload["engine"]["order_base"], 0)
        self.assertEqual(payload["engine"]["route_kind"], "presentation")
        self.assertEqual(payload["engine"]["requested_roi_normalized"], requested_roi)
        self.assertEqual(payload["engine"]["recognition_languages"], ["en-US"])
        self.assertGreater(payload["engine"]["request_revision"], 0)
        self.assertGreater(payload["engine"]["operating_system"]["major"], 0)
        self.assertIsInstance(payload["observations"], list)
        for index, observation in enumerate(payload["observations"]):
            self.assertEqual(observation["order"], index)
            self.assertTrue(observation["text"])
            self.assertGreaterEqual(observation["confidence"], 0.0)
            self.assertLessEqual(observation["confidence"], 1.0)
            bbox = observation["bbox"]
            self.assertEqual(len(bbox), 4)
            for value in bbox:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            x, y, width, height = bbox
            self.assertGreaterEqual(x, requested_roi[0] - 0.000001)
            self.assertGreaterEqual(y, requested_roi[1] - 0.000001)
            self.assertLessEqual(
                x + width,
                requested_roi[0] + requested_roi[2] + 0.000001,
            )
            self.assertLessEqual(
                y + height,
                requested_roi[1] + requested_roi[3] + 0.000001,
            )


if __name__ == "__main__":
    unittest.main()
