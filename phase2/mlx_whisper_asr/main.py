#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import sys
import time
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROTOCOL = "frameledger-asr-helper-v1"
FIXED_ZERO_DECODING_PROFILE = "fixed_zero_v1"
STANDARD_FALLBACK_DECODING_PROFILE = "standard_fallback_v1"
DECODING_PROFILE_NAMES = (
    FIXED_ZERO_DECODING_PROFILE,
    STANDARD_FALLBACK_DECODING_PROFILE,
)


class HelperError(RuntimeError):
    pass


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _model_manifest(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    tree_digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _sha256(path)
        files.append({"path": relative, "size_bytes": size, "sha256": digest})
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(size).encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\n")
    if not files:
        raise HelperError(f"Local MLX Whisper model has no files: {root}")
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "tree_sha256": tree_digest.hexdigest(),
        "files": files,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    raise HelperError(f"ASR result contains unsupported value {type(value).__name__}")


def _required_text(request: Mapping[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HelperError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_initial_prompt(request: Mapping[str, Any]) -> str | None:
    value = request.get("initial_prompt")
    if value is None:
        return None
    if not isinstance(value, str):
        raise HelperError("initial_prompt must be a string")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise HelperError("initial_prompt must not contain control characters")
    prompt = value.strip()
    if not prompt:
        raise HelperError("initial_prompt must not be empty")
    if len(prompt) > 1000:
        raise HelperError("initial_prompt is limited to 1000 characters")
    return prompt


def _decoding_parameters(profile: str) -> dict[str, Any]:
    if profile not in DECODING_PROFILE_NAMES:
        choices = ", ".join(DECODING_PROFILE_NAMES)
        raise HelperError(f"decoding_profile must be one of: {choices}")
    temperature: float | tuple[float, ...]
    if profile == FIXED_ZERO_DECODING_PROFILE:
        temperature = 0.0
    else:
        temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    return {
        "decoding_profile": profile,
        "temperature": temperature,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }


def _requested_decoding_parameters(request: Mapping[str, Any]) -> dict[str, Any]:
    value = request.get("decoding_profile", FIXED_ZERO_DECODING_PROFILE)
    if not isinstance(value, str):
        raise HelperError("decoding_profile must be a string")
    return _decoding_parameters(value.strip())


def _run(request: Mapping[str, Any]) -> dict[str, Any]:
    if request.get("protocol") != PROTOCOL:
        raise HelperError(f"protocol must be {PROTOCOL}")
    audio_path = Path(_required_text(request, "audio_path")).expanduser().resolve()
    model_path = Path(_required_text(request, "model_path")).expanduser().resolve()
    if not audio_path.is_file() or audio_path.suffix.lower() != ".wav":
        raise HelperError("audio_path must name one existing WAV file")
    if not model_path.is_dir():
        raise HelperError("model_path must name one existing local model directory")
    language = _required_text(request, "language")
    task = _required_text(request, "task")
    if task != "transcribe":
        raise HelperError("Phase 2b supports task=transcribe only")
    if request.get("word_timestamps") is not True:
        raise HelperError("Phase 2b requires word_timestamps=true")
    initial_prompt = _optional_initial_prompt(request)
    decoding = _requested_decoding_parameters(request)

    # A local directory is mandatory. These flags make an accidental network
    # fallback fail rather than silently fetching different model files.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    model = _model_manifest(model_path)
    started = time.perf_counter()
    try:
        import mlx_whisper
    except Exception as error:  # pragma: no cover - exercised in isolated runtime
        raise HelperError(f"mlx_whisper import failed: {error}") from error

    transcribe_options: dict[str, Any] = {
        "path_or_hf_repo": str(model_path),
        "language": language,
        "task": task,
        "word_timestamps": True,
        "verbose": None,
        "temperature": decoding["temperature"],
        "compression_ratio_threshold": decoding["compression_ratio_threshold"],
        "logprob_threshold": decoding["logprob_threshold"],
        "no_speech_threshold": decoding["no_speech_threshold"],
        "condition_on_previous_text": decoding["condition_on_previous_text"],
        "fp16": True,
    }
    if initial_prompt is not None:
        transcribe_options["initial_prompt"] = initial_prompt
    result = mlx_whisper.transcribe(str(audio_path), **transcribe_options)
    elapsed = time.perf_counter() - started
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    implementation_path = Path(__file__).resolve()
    return {
        "protocol": PROTOCOL,
        "engine": {
            "name": "mlx_whisper",
            "packages": {
                "mlx-whisper": _package_version("mlx-whisper"),
                "mlx": _package_version("mlx"),
                "huggingface-hub": _package_version("huggingface-hub"),
                "numpy": _package_version("numpy"),
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "implementation_path": str(implementation_path),
            "implementation_sha256": _sha256(implementation_path),
            "language": language,
            "task": task,
            "word_timestamps": True,
            "precision": "float16",
            **decoding,
            "initial_prompt_sha256": (
                hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest()
                if initial_prompt is not None
                else None
            ),
            "initial_prompt_character_count": (
                len(initial_prompt) if initial_prompt is not None else 0
            ),
            "model": model,
            "elapsed_seconds": round(elapsed, 6),
            "peak_rss_raw": peak_rss,
            "peak_rss_unit": "bytes_on_macos",
        },
        "result": _json_value(result),
    }


def main() -> int:
    if len(sys.argv) != 1:
        print(
            json.dumps({"protocol": PROTOCOL, "error": "helper accepts no arguments"}),
            file=sys.stderr,
        )
        return 2
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise HelperError("request must be a JSON object")
        response = _run(request)
    except Exception as error:
        print(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
