from __future__ import annotations

import hashlib
import html
import json
import math
import os
import platform
import shutil
import subprocess
import unicodedata
import wave
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .media import probe_video
from .report import write_json
from .timecode import format_timecode


ASR_SCHEMA_VERSION = 1
ASR_POLICY_VERSION = "bounded-local-asr-v1"
MLX_WHISPER_HELPER_PROTOCOL = "frameledger-asr-helper-v1"
MLX_WHISPER_HELPER_ENV = "FRAMELEDGER_MLX_WHISPER_HELPER"
MAX_ASR_SMOKE_SECONDS = 600.0
ASR_END_TIMESTAMP_TOLERANCE_SECONDS = 1.0
FIXED_ZERO_DECODING_PROFILE = "fixed_zero_v1"
STANDARD_FALLBACK_DECODING_PROFILE = "standard_fallback_v1"
DECODING_PROFILE_NAMES = (
    FIXED_ZERO_DECODING_PROFILE,
    STANDARD_FALLBACK_DECODING_PROFILE,
)
TRANSCRIPT_COMPRESSION_RATIO_LIMIT = 2.4
CONSECUTIVE_IDENTICAL_SEGMENT_LIMIT = 3
REPEATED_CHARACTER_NGRAM_SIZE = 32
TRANSCRIPT_QUALITY_POLICY_VERSION = "asr-transcript-quality-v2"


class AsrBackendError(RuntimeError):
    pass


class AsrBackend(Protocol):
    def describe(self) -> dict[str, Any]: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        task: str,
        word_timestamps: bool,
        initial_prompt: str | None = None,
        decoding_profile: str | None = None,
    ) -> Mapping[str, Any]: ...


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_initial_prompt(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("ASR initial prompt must be a string")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("ASR initial prompt must not contain control characters")
    prompt = value.strip()
    if not prompt:
        raise ValueError("ASR initial prompt must not be empty")
    if len(prompt) > 1000:
        raise ValueError("ASR initial prompt is limited to 1000 characters")
    return prompt


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decoding_parameters(profile: str) -> dict[str, Any]:
    if profile not in DECODING_PROFILE_NAMES:
        choices = ", ".join(DECODING_PROFILE_NAMES)
        raise ValueError(f"ASR decoding profile must be one of: {choices}")
    temperature: float | list[float]
    if profile == FIXED_ZERO_DECODING_PROFILE:
        temperature = 0.0
    else:
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    return {
        "decoding_profile": profile,
        "temperature": temperature,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }


def _resolve_executable(configured: str | Path | None, fallback: str) -> Path | None:
    if configured:
        text = str(configured)
        if "/" in text:
            candidate = Path(text).expanduser().resolve()
            return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None
        discovered = shutil.which(text)
        return Path(discovered).resolve() if discovered else None
    discovered = shutil.which(fallback)
    return Path(discovered).resolve() if discovered else None


class MlxWhisperAsrBackend:
    """Invoke an isolated MLX Whisper helper through strict JSON stdin/stdout."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        helper: str | Path | None = None,
        timeout_seconds: float = 1800.0,
    ) -> None:
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("ASR helper timeout must be finite and positive")
        self.model_path = Path(model_path).expanduser().resolve()
        self.helper_path = _resolve_executable(
            helper or os.environ.get(MLX_WHISPER_HELPER_ENV),
            "frameledger-mlx-whisper-asr",
        )
        self.helper_sha256 = _sha256(self.helper_path) if self.helper_path else None
        self.timeout_seconds = timeout_seconds
        self._runtime_metadata: dict[str, Any] = {}

    def describe(self) -> dict[str, Any]:
        return {
            "name": "mlx_whisper",
            "protocol": MLX_WHISPER_HELPER_PROTOCOL,
            "helper_path": str(self.helper_path) if self.helper_path else None,
            "helper_sha256": self.helper_sha256,
            "model_path": str(self.model_path),
            "available": self.helper_path is not None and self.model_path.is_dir(),
            "runtime": self._runtime_metadata or None,
            "orchestrator_platform": platform.platform(),
        }

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str,
        task: str,
        word_timestamps: bool,
        initial_prompt: str | None = None,
        decoding_profile: str | None = None,
    ) -> Mapping[str, Any]:
        if self.helper_path is None:
            raise AsrBackendError(
                "MLX Whisper ASR helper is unavailable; pass --mlx-whisper-helper or set "
                f"{MLX_WHISPER_HELPER_ENV}"
            )
        if not self.model_path.is_dir():
            raise AsrBackendError(f"Local MLX Whisper model is unavailable: {self.model_path}")
        request = {
            "protocol": MLX_WHISPER_HELPER_PROTOCOL,
            "audio_path": str(audio_path.resolve()),
            "model_path": str(self.model_path),
            "language": language,
            "task": task,
            "word_timestamps": word_timestamps,
        }
        if initial_prompt is not None:
            request["initial_prompt"] = initial_prompt
        expected_decoding = None
        if decoding_profile is not None:
            expected_decoding = _decoding_parameters(decoding_profile)
            request["decoding_profile"] = decoding_profile
        try:
            completed = subprocess.run(
                [str(self.helper_path)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AsrBackendError(f"MLX Whisper ASR helper could not run: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise AsrBackendError(
                f"MLX Whisper ASR helper exited with {completed.returncode}: "
                f"{detail or 'no diagnostic output'}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AsrBackendError("MLX Whisper ASR helper returned invalid JSON") from error
        if not isinstance(response, dict) or response.get("protocol") != MLX_WHISPER_HELPER_PROTOCOL:
            raise AsrBackendError("MLX Whisper ASR helper protocol does not match")
        runtime = response.get("engine")
        result = response.get("result")
        if not isinstance(runtime, dict) or not isinstance(result, dict):
            raise AsrBackendError("MLX Whisper ASR helper response lacks engine/result objects")
        self._runtime_metadata = runtime
        if initial_prompt is not None:
            expected_sha256 = _text_sha256(initial_prompt)
            actual_sha256 = runtime.get("initial_prompt_sha256")
            actual_character_count = runtime.get("initial_prompt_character_count")
            if (
                actual_sha256 != expected_sha256
                or not isinstance(actual_character_count, int)
                or isinstance(actual_character_count, bool)
                or actual_character_count != len(initial_prompt)
            ):
                raise AsrBackendError(
                    "MLX Whisper ASR helper did not confirm the requested initial prompt"
                )
        if expected_decoding is not None and any(
            runtime.get(key) != expected for key, expected in expected_decoding.items()
        ):
            raise AsrBackendError(
                "MLX Whisper ASR helper did not confirm the requested decoding profile"
            )
        return result


def _validate_output_target(output: str | Path, *, source: Path) -> Path:
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise ValueError(f"ASR output already exists; refusing to overwrite: {root}")
    if root.parent == source.parent or source.parent in root.parents:
        raise ValueError("ASR output cannot be written beside or below the source video")
    return root


def _ffmpeg_metadata(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(path), "-version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    first_line = (completed.stdout or completed.stderr).splitlines()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "version_line": first_line[0] if first_line else None,
    }


def _extract_audio(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> dict[str, Any]:
    command = [
        str(ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=max(120.0, min(900.0, duration_seconds * 3.0)),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:1000]
        raise RuntimeError(
            f"FFmpeg audio extraction failed with {completed.returncode}: "
            f"{detail or 'no diagnostic output'}"
        )
    if not destination.is_file():
        raise RuntimeError("FFmpeg reported success but did not create the WAV artifact")
    with wave.open(str(destination), "rb") as audio:
        channels = audio.getnchannels()
        sample_rate = audio.getframerate()
        sample_width = audio.getsampwidth()
        frames = audio.getnframes()
    if channels != 1 or sample_rate != 16000 or sample_width != 2 or frames <= 0:
        raise RuntimeError("Extracted WAV is not mono 16 kHz signed 16-bit PCM")
    actual_duration = frames / sample_rate
    if abs(actual_duration - duration_seconds) > 1.0:
        raise RuntimeError(
            "Extracted WAV duration differs from the requested duration by more than one second"
        )
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "frame_count": frames,
        "duration_seconds": round(actual_duration, 6),
    }


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"ASR result {label} is not numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"ASR result {label} is not finite")
    return number


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, "metric")
    return round(number, 6)


def _normalize_transcript(
    result: Mapping[str, Any],
    *,
    absolute_offset: float,
    audio_duration: float,
) -> dict[str, Any]:
    raw_segments = result.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError("ASR result lacks a segment list")
    segments: list[dict[str, Any]] = []
    word_count = 0
    timestamp_clamp_count = 0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise RuntimeError("ASR segment must be an object")
        start = _finite_number(raw.get("start"), f"segment {index} start")
        end = _finite_number(raw.get("end"), f"segment {index} end")
        if (
            start < -1e-6
            or start > audio_duration + 1e-6
            or end < start
            or end > audio_duration + ASR_END_TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise RuntimeError(f"ASR segment {index} timestamps fall outside the audio window")
        raw_end = end
        segment_end_clamped = end > audio_duration
        if segment_end_clamped:
            end = audio_duration
            timestamp_clamp_count += 1
        text = str(raw.get("text", ""))
        words: list[dict[str, Any]] = []
        raw_words = raw.get("words", [])
        if not isinstance(raw_words, list):
            raise RuntimeError(f"ASR segment {index} words must be a list")
        for word_index, raw_word in enumerate(raw_words):
            if not isinstance(raw_word, dict):
                raise RuntimeError("ASR word must be an object")
            word_start = _finite_number(
                raw_word.get("start"), f"segment {index} word {word_index} start"
            )
            word_end = _finite_number(
                raw_word.get("end"), f"segment {index} word {word_index} end"
            )
            if (
                word_start < -1e-6
                or word_start > audio_duration + 1e-6
                or word_end < word_start
                or word_end > audio_duration + ASR_END_TIMESTAMP_TOLERANCE_SECONDS
            ):
                raise RuntimeError("ASR word timestamps fall outside the audio window")
            raw_word_end = word_end
            word_end_clamped = word_end > audio_duration
            if word_end_clamped:
                word_end = audio_duration
                timestamp_clamp_count += 1
            word_payload = {
                    "word": str(raw_word.get("word", "")),
                    "relative_start_seconds": round(word_start, 6),
                    "relative_end_seconds": round(word_end, 6),
                    "absolute_start_seconds": round(absolute_offset + word_start, 6),
                    "absolute_end_seconds": round(absolute_offset + word_end, 6),
                    "probability": _optional_number(raw_word.get("probability")),
                }
            if word_end_clamped:
                word_payload["source_relative_end_seconds"] = round(raw_word_end, 6)
                word_payload["end_clamped_to_audio_duration"] = True
            words.append(word_payload)
        word_count += len(words)
        tokens = raw.get("tokens", [])
        if not isinstance(tokens, list):
            raise RuntimeError(f"ASR segment {index} tokens must be a list")
        segment_payload = {
                "id": int(raw.get("id", index)),
                "relative_start_seconds": round(start, 6),
                "relative_end_seconds": round(end, 6),
                "absolute_start_seconds": round(absolute_offset + start, 6),
                "absolute_end_seconds": round(absolute_offset + end, 6),
                "text": text,
                "tokens": [int(token) for token in tokens],
                "temperature": _optional_number(raw.get("temperature")),
                "avg_logprob": _optional_number(raw.get("avg_logprob")),
                "compression_ratio": _optional_number(raw.get("compression_ratio")),
                "no_speech_prob": _optional_number(raw.get("no_speech_prob")),
                "words": words,
            }
        if segment_end_clamped:
            segment_payload["source_relative_end_seconds"] = round(raw_end, 6)
            segment_payload["end_clamped_to_audio_duration"] = True
        segments.append(segment_payload)
    plain_text = str(result.get("text", "")).strip()
    if not plain_text:
        plain_text = "".join(segment["text"] for segment in segments).strip()
    language = result.get("language")
    return {
        "language": str(language) if language is not None else None,
        "text": plain_text,
        "segments": segments,
        "segment_count": len(segments),
        "word_count": word_count,
        "timestamp_policy": "clamp_end_within_audio_tail_tolerance_v1",
        "timestamp_clamp_count": timestamp_clamp_count,
    }


def _transcript_quality_checks(transcript: Mapping[str, Any]) -> dict[str, Any]:
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError("Normalized ASR transcript lacks a segment list")

    compression_ratios: list[float] = []
    compression_ratio_offenders: list[dict[str, Any]] = []
    repeated_ngram_offenders: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise RuntimeError("Normalized ASR segment must be an object")
        ratio = segment.get("compression_ratio")
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
            numeric_ratio = float(ratio)
            compression_ratios.append(numeric_ratio)
            if numeric_ratio > TRANSCRIPT_COMPRESSION_RATIO_LIMIT:
                compression_ratio_offenders.append(
                    {
                        "segment_index": index,
                        "segment_id": segment.get("id"),
                        "compression_ratio": round(numeric_ratio, 6),
                    }
                )
        normalized_text = "".join(str(segment.get("text", "")).split())
        seen_ngrams: dict[str, int] = {}
        for position in range(
            max(0, len(normalized_text) - REPEATED_CHARACTER_NGRAM_SIZE + 1)
        ):
            ngram = normalized_text[position : position + REPEATED_CHARACTER_NGRAM_SIZE]
            first_position = seen_ngrams.get(ngram)
            if (
                first_position is not None
                and position - first_position >= REPEATED_CHARACTER_NGRAM_SIZE
            ):
                repeated_ngram_offenders.append(
                    {
                        "segment_index": index,
                        "segment_id": segment.get("id"),
                        "ngram_size": REPEATED_CHARACTER_NGRAM_SIZE,
                        "first_position": first_position,
                        "second_position": position,
                        "ngram_sha256": _text_sha256(ngram),
                        "ngram_preview": ngram[:80],
                    }
                )
                break
            seen_ngrams.setdefault(ngram, position)

    repetition_runs: list[dict[str, Any]] = []
    maximum_run = 0
    previous_text: str | None = None
    run_start = 0
    run_count = 0

    def record_run(end_index: int) -> None:
        if previous_text is None or run_count < CONSECUTIVE_IDENTICAL_SEGMENT_LIMIT:
            return
        repetition_runs.append(
            {
                "start_segment_index": run_start,
                "end_segment_index": end_index,
                "count": run_count,
                "text_sha256": _text_sha256(previous_text),
                "text_preview": previous_text[:200],
            }
        )

    for index, segment in enumerate(segments):
        normalized_text = " ".join(str(segment.get("text", "")).split())
        if not normalized_text:
            record_run(index - 1)
            previous_text = None
            run_count = 0
            continue
        if normalized_text == previous_text:
            run_count += 1
        else:
            record_run(index - 1)
            previous_text = normalized_text
            run_start = index
            run_count = 1
        maximum_run = max(maximum_run, run_count)
    record_run(len(segments) - 1)

    maximum_compression_ratio = max(compression_ratios) if compression_ratios else None
    full_text = str(transcript.get("text", ""))
    replacement_character_count = full_text.count("\ufffd")
    failures: list[str] = []
    if compression_ratio_offenders:
        failures.append("max_segment_compression_ratio")
    if repetition_runs:
        failures.append("consecutive_identical_nonempty_segments")
    if replacement_character_count:
        failures.append("replacement_character")
    if repeated_ngram_offenders:
        failures.append("repeated_character_ngram_within_segment")
    return {
        "policy_version": TRANSCRIPT_QUALITY_POLICY_VERSION,
        "passed": not failures,
        "failures": failures,
        "compression_ratio_threshold": TRANSCRIPT_COMPRESSION_RATIO_LIMIT,
        "max_segment_compression_ratio": (
            round(maximum_compression_ratio, 6)
            if maximum_compression_ratio is not None
            else None
        ),
        "compression_ratio_failed": bool(compression_ratio_offenders),
        "compression_ratio_offenders": compression_ratio_offenders,
        "consecutive_identical_nonempty_segment_threshold": (
            CONSECUTIVE_IDENTICAL_SEGMENT_LIMIT
        ),
        "max_consecutive_identical_nonempty_segments": maximum_run,
        "consecutive_identical_nonempty_segments_failed": bool(repetition_runs),
        "repetition_runs": repetition_runs,
        "replacement_character_count": replacement_character_count,
        "replacement_character_failed": bool(replacement_character_count),
        "repeated_character_ngram_size": REPEATED_CHARACTER_NGRAM_SIZE,
        "repeated_character_ngram_offender_count": len(repeated_ngram_offenders),
        "repeated_character_ngram_failed": bool(repeated_ngram_offenders),
        "repeated_character_ngram_offenders": repeated_ngram_offenders,
    }


def write_asr_review_html(path: Path, document: Mapping[str, Any]) -> None:
    transcript = document.get("transcript", {})
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    source = document["source"]
    range_data = document["range"]
    parameters = document["parameters"]
    runtime = document["engine"].get("runtime") or {}
    rows: list[str] = []
    for segment in segments:
        absolute_start = format_timecode(float(segment["absolute_start_seconds"]))
        absolute_end = format_timecode(float(segment["absolute_end_seconds"]))
        relative_start = format_timecode(float(segment["relative_start_seconds"]))
        relative_end = format_timecode(float(segment["relative_end_seconds"]))
        words = segment.get("words", [])
        word_html = " ".join(
            (
                "<span class='word' title='"
                + html.escape(
                    f"{format_timecode(float(word['absolute_start_seconds']))}–"
                    f"{format_timecode(float(word['absolute_end_seconds']))} · "
                    f"p={word.get('probability')}"
                )
                + "'>"
                + html.escape(str(word.get("word", "")))
                + "</span>"
            )
            for word in words
        )
        rows.append(
            "<article><div class='time'>"
            f"<strong>{html.escape(absolute_start)}–{html.escape(absolute_end)}</strong>"
            f"<small>clip {html.escape(relative_start)}–{html.escape(relative_end)}</small>"
            "</div><div class='text'>"
            f"<p>{html.escape(str(segment.get('text', '')))}</p>"
            f"<details><summary>{len(words)} aligned units</summary><div class='words'>{word_html}</div></details>"
            "</div></article>"
        )
    elapsed = runtime.get("elapsed_seconds")
    prompt_mode = str(parameters.get("prompt_mode", "unprompted"))
    prompt_sha256 = parameters.get("initial_prompt_sha256")
    prompt_character_count = int(parameters.get("initial_prompt_character_count", 0))
    prompt_detail = (
        f"prompt SHA-256 {prompt_sha256} · {prompt_character_count} characters"
        if prompt_sha256
        else f"prompt SHA-256 none · {prompt_character_count} characters"
    )
    html_text = f"""<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FrameLedger local ASR review</title><style>
:root {{ font-family:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17212b; }}
* {{ box-sizing:border-box; }} body {{ margin:0;background:#f3f5f7; }}
header {{ position:sticky;top:0;z-index:2;padding:14px 20px;background:rgba(255,255,255,.97);border-bottom:1px solid #d8dee6; }}
h1 {{ margin:0 0 6px;font-size:21px; }} header p {{ margin:4px 0;color:#526171;font-size:13px; }}
audio {{ width:min(760px,100%);margin-top:8px; }} main {{ max-width:1200px;margin:auto;padding:18px 20px 60px; }}
article {{ display:grid;grid-template-columns:280px 1fr;gap:16px;margin:0 0 12px;padding:14px;background:#fff;border:1px solid #d8dee6;border-radius:9px; }}
.time {{ font-variant-numeric:tabular-nums; }} .time small {{ display:block;margin-top:5px;color:#607080; }}
.text p {{ margin:0 0 8px;font-size:18px;line-height:1.55; }} summary {{ cursor:pointer;color:#526171;font-size:12px; }}
.words {{ margin-top:8px;line-height:1.8; }} .word {{ border-bottom:1px dotted #10b981; }} code {{ overflow-wrap:anywhere; }}
@media(max-width:760px) {{ article {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
</style></head><body><header><h1>FrameLedger local ASR evidence review</h1>
<p>{html.escape(Path(str(source['path'])).name)} · {format_timecode(float(range_data['start_seconds']))}–{format_timecode(float(range_data['end_seconds']))}</p>
<p>MLX Whisper · {html.escape(str(document['parameters']['model_id']))} · {len(segments)} segments · engine {html.escape(str(elapsed))}s</p>
<p>Inference mode: {html.escape(prompt_mode)} · {html.escape(prompt_detail)}</p>
<p>Raw local model output; no post-processing correction, spelling normalization, financial normalization, or speaker attribution.</p>
<audio controls preload="metadata" src="audio.wav"></audio></header><main>{''.join(rows)}</main></body></html>"""
    path.write_text(html_text, encoding="utf-8")


def _write_failure(
    path: Path,
    *,
    source: Mapping[str, Any],
    range_data: Mapping[str, Any],
    parameters: Mapping[str, Any],
    stage: str,
    error: Exception,
    extractor: Mapping[str, Any] | None = None,
    audio: Mapping[str, Any] | None = None,
    engine: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | None = None,
    quality_checks: Mapping[str, Any] | None = None,
) -> None:
    document = {
        "kind": "local_speech_transcription",
        "schema_version": ASR_SCHEMA_VERSION,
        "status": "error",
        "created_at": datetime.now(UTC).isoformat(),
        "source": dict(source),
        "range": dict(range_data),
        "parameters": dict(parameters),
        "extractor": dict(extractor) if extractor else None,
        "audio": dict(audio) if audio else None,
        "engine": dict(engine) if engine else None,
        "failure": {
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    }
    if transcript is not None:
        document["transcript"] = dict(transcript)
    if quality_checks is not None:
        document["quality_checks"] = dict(quality_checks)
    write_json(path, document)


def run_local_asr(
    video: str | Path,
    *,
    start_seconds: float,
    duration_seconds: float,
    model_path: str | Path,
    model_id: str,
    model_revision: str,
    language: str,
    task: str,
    output: str | Path,
    helper: str | Path | None = None,
    ffmpeg: str | Path | None = None,
    backend: AsrBackend | None = None,
    initial_prompt: str | None = None,
    decoding_profile: str = FIXED_ZERO_DECODING_PROFILE,
    allow_long_range: bool = False,
) -> dict[str, Any]:
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("ASR --start must be finite and non-negative")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("ASR --duration must be finite and positive")
    if duration_seconds > MAX_ASR_SMOKE_SECONDS and not allow_long_range:
        raise ValueError(
            f"Phase 2b ASR ranges are limited to {MAX_ASR_SMOKE_SECONDS:.0f} seconds "
            "unless --allow-long-range is explicitly enabled"
        )
    language = language.strip().lower()
    if not language:
        raise ValueError("ASR language must not be empty")
    if task != "transcribe":
        raise ValueError("Phase 2b supports task=transcribe only")
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("ASR model ID must not be empty")
    model_revision = model_revision.strip().lower()
    if len(model_revision) != 40 or any(character not in "0123456789abcdef" for character in model_revision):
        raise ValueError("ASR model revision must be one full 40-character hexadecimal commit")
    initial_prompt = _normalize_initial_prompt(initial_prompt)
    if not isinstance(decoding_profile, str):
        raise ValueError("ASR decoding profile must be a string")
    decoding_profile = decoding_profile.strip()
    decoding_parameters = _decoding_parameters(decoding_profile)

    metadata = probe_video(video)
    source_path = metadata.path.resolve()
    end_seconds = start_seconds + duration_seconds
    if start_seconds >= metadata.duration_seconds or end_seconds > metadata.duration_seconds + 1e-6:
        raise ValueError("ASR range falls outside the source video duration")
    output_root = _validate_output_target(output, source=source_path)
    resolved_ffmpeg = _resolve_executable(ffmpeg, "ffmpeg")
    if resolved_ffmpeg is None:
        raise RuntimeError("FFmpeg is unavailable; install it or pass --ffmpeg")
    model_root = Path(model_path).expanduser().resolve()
    if not model_root.is_dir():
        raise ValueError(f"Local ASR model directory does not exist: {model_root}")

    source = metadata.to_dict()
    source["path"] = str(source_path)
    range_data = {
        "start_seconds": round(start_seconds, 6),
        "end_seconds": round(end_seconds, 6),
        "duration_seconds": round(duration_seconds, 6),
        "start_timecode": format_timecode(start_seconds),
        "end_timecode": format_timecode(end_seconds),
    }
    parameters = {
        "policy_version": ASR_POLICY_VERSION,
        "engine": "mlx_whisper",
        "model_id": model_id,
        "model_revision": model_revision,
        "model_path": str(model_root),
        "language": language,
        "task": task,
        "word_timestamps": True,
        "precision": "float16",
        "quality_policy_version": TRANSCRIPT_QUALITY_POLICY_VERSION,
        "long_range_explicitly_allowed": bool(allow_long_range),
        **decoding_parameters,
        "prompt_mode": "prompted" if initial_prompt is not None else "unprompted",
        "initial_prompt": initial_prompt,
        "initial_prompt_sha256": (
            _text_sha256(initial_prompt) if initial_prompt is not None else None
        ),
        "initial_prompt_character_count": len(initial_prompt) if initial_prompt is not None else 0,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    asr_path = output_root / "asr.json"
    audio_path = output_root / "audio.wav"
    extractor = _ffmpeg_metadata(resolved_ffmpeg)
    try:
        audio = _extract_audio(
            resolved_ffmpeg,
            source_path,
            audio_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
    except Exception as error:
        _write_failure(
            asr_path,
            source=source,
            range_data=range_data,
            parameters=parameters,
            stage="audio_extraction",
            error=error,
            extractor=extractor,
        )
        raise RuntimeError(f"Local ASR failed during audio extraction; see {asr_path}") from error

    asr_backend = backend or MlxWhisperAsrBackend(
        model_path=model_root,
        helper=helper,
    )
    try:
        raw_result = asr_backend.transcribe(
            audio_path,
            language=language,
            task=task,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            decoding_profile=decoding_profile,
        )
        transcript = _normalize_transcript(
            raw_result,
            absolute_offset=start_seconds,
            audio_duration=float(audio["duration_seconds"]),
        )
    except Exception as error:
        _write_failure(
            asr_path,
            source=source,
            range_data=range_data,
            parameters=parameters,
            stage="transcription",
            error=error,
            extractor=extractor,
            audio=audio,
            engine=asr_backend.describe(),
        )
        raise RuntimeError(f"Local ASR failed during transcription; see {asr_path}") from error
    quality_checks = _transcript_quality_checks(transcript)

    if _sha256(source_path) != metadata.fingerprint:
        error = RuntimeError("Source video changed during ASR; results are not trustworthy")
        _write_failure(
            asr_path,
            source=source,
            range_data=range_data,
            parameters=parameters,
            stage="source_integrity",
            error=error,
            extractor=extractor,
            audio=audio,
            engine=asr_backend.describe(),
        )
        raise error
    if _sha256(audio_path) != audio["sha256"]:
        error = RuntimeError("Extracted audio changed during ASR; results are not trustworthy")
        _write_failure(
            asr_path,
            source=source,
            range_data=range_data,
            parameters=parameters,
            stage="audio_integrity",
            error=error,
            extractor=extractor,
            audio=audio,
            engine=asr_backend.describe(),
        )
        raise error

    if not quality_checks["passed"]:
        error = RuntimeError(
            "ASR transcript failed quality checks: "
            + ", ".join(str(item) for item in quality_checks["failures"])
        )
        _write_failure(
            asr_path,
            source=source,
            range_data=range_data,
            parameters=parameters,
            stage="transcript_quality",
            error=error,
            extractor=extractor,
            audio=audio,
            engine=asr_backend.describe(),
            transcript=transcript,
            quality_checks=quality_checks,
        )
        raise RuntimeError(
            f"Local ASR failed transcript quality gate; see {asr_path}"
        ) from error

    engine = asr_backend.describe()
    runtime = engine.get("runtime") or {}
    elapsed = runtime.get("elapsed_seconds")
    real_time_factor = None
    if isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)):
        real_time_factor = round(float(elapsed) / duration_seconds, 6)
    document = {
        "kind": "local_speech_transcription",
        "schema_version": ASR_SCHEMA_VERSION,
        "status": "ok",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "range": range_data,
        "parameters": parameters,
        "extractor": extractor,
        "audio": audio,
        "engine": engine,
        "transcript": transcript,
        "summary": {
            "segment_count": transcript["segment_count"],
            "word_count": transcript["word_count"],
            "character_count": len(transcript["text"]),
            "engine_elapsed_seconds": elapsed,
            "real_time_factor": real_time_factor,
            "quality_checks": quality_checks,
        },
    }
    write_json(asr_path, document)
    review_path = output_root / "review.html"
    write_asr_review_html(review_path, document)
    return {
        **document,
        "output": str(output_root),
        "asr_json": str(asr_path),
        "review_html": str(review_path),
    }
