from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import sys
import urllib.parse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .report import write_json
from .timecode import format_timecode


SEMANTIC_SCHEMA_VERSION = 1
SEMANTIC_POLICY_VERSION = "deterministic-local-topic-segmentation-v1"
DEFAULT_TOPIC_WINDOW_SECONDS = 45.0
DEFAULT_MIN_CHAPTER_SECONDS = 30.0
DEFAULT_TARGET_CHAPTER_SECONDS = 180.0
DEFAULT_MAX_CHAPTER_SECONDS = 480.0
VISUAL_TITLE_LOOKBACK_SECONDS = 210.0
VISUAL_TITLE_LOOKAHEAD_SECONDS = 5.0
VISUAL_TITLE_ASR_WINDOW_SECONDS = 14.0
VISUAL_TITLE_MIN_OCR_CONFIDENCE = 0.25
VISUAL_TITLE_MIN_MAPPING_SIMILARITY = 0.12
VISUAL_TITLE_MIN_COMPONENT = 0.55

WEIGHTS = {
    "lexical_novelty": 0.44,
    "discourse_transition": 0.25,
    "pause": 0.11,
    "visual_title_change": 0.20,
}

_FINANCE_TERMS = (
    "K线",
    "k线",
    "形态",
    "抄底",
    "逃顶",
    "见底",
    "见顶",
    "止跌",
    "锤子线",
    "上吊线",
    "流星线",
    "启明星",
    "起明星",
    "黄昏之星",
    "倒锤子线",
    "看涨",
    "看跌",
    "抱线",
    "报线",
    "吞没",
    "孕线",
    "运线",
    "实体",
    "影线",
    "引线",
    "趋势",
    "成交量",
    "放量",
    "缩量",
    "支撑",
    "压力",
    "均线",
    "财报",
    "营收",
    "利润",
    "现金流",
    "估值",
    "公司",
    "行业",
    "市场",
    "排序",
)

_TITLE_EXCLUSIONS = (
    "youtube会员专享",
    "视野环球财经",
    "globalfinance",
    "disclaimer",
    "不构成任何投资建议",
    "投资有风险",
    "入市需谨慎",
    "人市需谨慎",
    "分享个人投资理念",
)

_TITLE_PENALTIES = (
    "大家好",
    "欢迎收看",
    "会员专享",
    "我们来看",
    "大家来看",
    "对不对",
    "好吧",
    "好吗",
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain one object")
    return value


def _load_bound_json(
    path: Path, expected_hash: str, *, label: str
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"{label} SHA-256 no longer matches the alignment ledger")
        value = json.loads(raw)
    except OSError as error:
        raise ValueError(f"Could not read {label} JSON: {path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse {label} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON must contain one object")
    return value


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _number(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _source_file(
    alignment_path: Path, value: Any, expected_hash: Any, *, label: str
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.path must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = alignment_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label}.path does not exist: {path}")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ValueError(f"{label} SHA-256 no longer matches the alignment ledger")
    return path


def _validate_parameters(
    minimum: float, target: float, maximum: float, window: float
) -> tuple[float, float, float, float]:
    values = {
        "min_chapter_seconds": minimum,
        "target_chapter_seconds": target,
        "max_chapter_seconds": maximum,
        "topic_window_seconds": window,
    }
    converted: dict[str, float] = {}
    for label, value in values.items():
        converted[label] = _number(value, label=label)
        if converted[label] <= 0:
            raise ValueError(f"{label} must be positive")
    minimum = converted["min_chapter_seconds"]
    target = converted["target_chapter_seconds"]
    maximum = converted["max_chapter_seconds"]
    window = converted["topic_window_seconds"]
    if minimum > target or target > maximum:
        raise ValueError(
            "Semantic chapter constraints must satisfy min <= target <= max"
        )
    if maximum > 3600 or window > 300:
        raise ValueError("Semantic chapter constraints are outside the supported range")
    return minimum, target, maximum, window


def _protect_output(
    output: Path,
    *,
    alignment_path: Path,
    source: Mapping[str, Any],
    upstream_files: Sequence[Path],
) -> None:
    if output.exists():
        raise ValueError(f"Semantic output already exists; refusing to overwrite: {output}")
    protected: set[Path] = {alignment_path.parent.resolve()}
    protected.update(path.parent.resolve() for path in upstream_files)
    phase1 = source.get("phase1")
    if isinstance(phase1, Mapping) and isinstance(phase1.get("run_directory"), str):
        protected.add(Path(str(phase1["run_directory"])).expanduser().resolve())
    video = source.get("video")
    if isinstance(video, Mapping) and isinstance(video.get("path"), str):
        protected.add(Path(str(video["path"])).expanduser().resolve().parent)
    for directory in protected:
        if output == directory or directory in output.parents:
            raise ValueError(
                "Semantic output must be a new directory outside alignment, ASR, OCR, "
                "Phase 1, and source-video inputs"
            )


def _normalise_text(text: str) -> str:
    return "".join(
        character.lower()
        for character in text
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )


def _normalise_match_text(text: str) -> str:
    """Normalise recurring OCR/ASR variants for scoring, never for displayed text."""

    value = _normalise_text(text)
    for source, target in (
        ("抱线", "包线"),
        ("报线", "包线"),
        ("爆线", "包线"),
        ("运线", "孕线"),
        ("起明星", "启明星"),
        ("引线", "影线"),
        ("指跌", "止跌"),
    ):
        value = value.replace(source, target)
    return value


def _features(text: str) -> Counter[str]:
    normalised = _normalise_text(text)
    features: Counter[str] = Counter()
    for size, weight in ((2, 1.0), (3, 1.35)):
        for index in range(max(0, len(normalised) - size + 1)):
            features[f"{size}:{normalised[index:index + size]}"] += weight
    for token in re.findall(r"[a-z0-9]{2,}", normalised):
        features[f"word:{token}"] += 1.5
    return features


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _text_similarity(left: str, right: str) -> float:
    return _cosine(_features(left), _features(right))


def _match_similarity(left: str, right: str) -> float:
    return _cosine(
        _features(_normalise_match_text(left)),
        _features(_normalise_match_text(right)),
    )


def _window_text(
    speech: Sequence[Mapping[str, Any]], start: float, end: float
) -> str:
    return " ".join(
        str(segment["text"])
        for segment in speech
        if float(segment["absolute_end_seconds"]) > start
        and float(segment["absolute_start_seconds"]) < end
    )


def _discourse_evidence(text: str) -> tuple[float, list[str]]:
    compact = _normalise_text(text)
    patterns: tuple[tuple[str, float, str], ...] = (
        (r"直接进入(主题|正题)", 1.0, "direct_topic_entry"),
        (r"进入正题", 1.0, "direct_topic_entry"),
        (r"(接着|继续)往下看", 0.98, "continue_to_next"),
        (r"(那|那么)?(我们|咱们)?再看一个", 0.98, "another_topic"),
        (r"^排在第[一二三四五六七八九十]", 0.96, "ranked_topic"),
        (r"^(今天|本期).{0,16}(主题|讲解|来讲)", 0.90, "episode_topic"),
        (r"我先把.{0,22}(排序|大方向)", 0.92, "overview_transition"),
        (r"^(首先|其次|接下来|最后).{0,18}(讲|看|介绍|分析|解释|把)", 0.86, "ordered_transition"),
        (r"^(下面|接下来).{0,12}(看|讲|说|分析)", 0.84, "next_topic"),
        (r"^(总结|小结|结论|所以这就是|以上就是)", 0.94, "summary_transition"),
        (r"^这是第[一二三四五六七八九十]种", 0.80, "numbered_topic"),
    )
    evidence: list[str] = []
    score = 0.0
    for pattern, weight, name in patterns:
        if re.search(pattern, compact):
            score = max(score, weight)
            evidence.append(name)
    return score, evidence


def _visual_title(frame: Mapping[str, Any]) -> dict[str, Any] | None:
    ocr = frame.get("ocr")
    if not isinstance(ocr, Mapping) or ocr.get("status") != "ok":
        return None
    observations = ocr.get("observations")
    if not isinstance(observations, list):
        return None
    candidates: list[tuple[float, float, float, int, str, float]] = []
    for raw in observations:
        if not isinstance(raw, Mapping):
            continue
        text = str(raw.get("text", "")).strip()
        normalised = _normalise_text(text)
        if len(normalised) < 4 or len(normalised) > 80:
            continue
        if any(term in normalised for term in _TITLE_EXCLUSIONS):
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            _, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (y, width, height)):
            continue
        if y > 0.36 or width < 0.10:
            continue
        confidence_value = raw.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float))
            else 0.0
        )
        if not math.isfinite(confidence):
            continue
        if confidence < VISUAL_TITLE_MIN_OCR_CONFIDENCE:
            continue
        title_rank = (
            confidence * 0.20
            + max(0.0, 1.0 - y) * 0.55
            + min(1.0, width) * 0.25
        )
        candidates.append((-title_rank, y, -width, -len(normalised), text, confidence))
    if not candidates:
        return None
    candidates.sort()
    selected = candidates[0]
    return {
        "raw_title": selected[4],
        "ocr_confidence": round(selected[5], 6),
    }


def _changed_visual_titles(
    visual: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    previous_title: str | None = None
    for frame in visual:
        title_record = _visual_title(frame)
        if title_record is None:
            continue
        title = str(title_record["raw_title"])
        if previous_title is not None and _text_similarity(previous_title, title) >= 0.72:
            continue
        changes.append(
            {
                "visual_event_id": str(frame["event_id"]),
                "timestamp": float(frame["timestamp"]),
                "raw_title": title,
                "ocr_confidence": title_record["ocr_confidence"],
            }
        )
        previous_title = title
    return changes


def _candidate_records(
    speech: Sequence[Mapping[str, Any]], *, window_seconds: float
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(1, len(speech)):
        current = speech[index]
        timestamp = float(current["absolute_start_seconds"])
        left = _window_text(speech, timestamp - window_seconds, timestamp)
        right = _window_text(speech, timestamp, timestamp + window_seconds)
        novelty = 1.0 - _text_similarity(left, right)
        previous_end = float(speech[index - 1]["absolute_end_seconds"])
        pause_seconds = max(0.0, timestamp - previous_end)
        pause = min(1.0, pause_seconds / 1.5)
        # One short look-ahead catches a transition split across two Whisper
        # segments without moving the boundary several phrases too early.
        discourse_text = " ".join(
            str(item["text"]) for item in speech[index : index + 2]
        )
        discourse, cues = _discourse_evidence(discourse_text)
        records.append(
            {
                "speech_index": index,
                "speech_event_id": str(current["event_id"]),
                "timestamp": timestamp,
                "lexical_novelty": round(novelty, 6),
                "discourse_transition": round(discourse, 6),
                "pause": round(pause, 6),
                "pause_seconds": round(pause_seconds, 6),
                "visual_title_change": 0.0,
                "discourse_cues": cues,
                "visual_evidence": [],
            }
        )
    return records


def _attach_visual_evidence(
    candidates: list[dict[str, Any]],
    speech: Sequence[Mapping[str, Any]],
    title_changes: Sequence[Mapping[str, Any]],
    *,
    range_start: float,
    range_end: float,
    minimum: float,
) -> None:
    previous_mapped_time = range_start
    for title in title_changes:
        visual_time = float(title["timestamp"])
        raw_title = str(title["raw_title"])
        match_title = re.split(r"[：:]", raw_title, maxsplit=1)[-1]
        if visual_time < range_start + minimum:
            continue
        eligible = [
            candidate
            for candidate in candidates
            if max(
                range_start + minimum,
                previous_mapped_time + minimum,
                visual_time - VISUAL_TITLE_LOOKBACK_SECONDS,
            )
            <= float(candidate["timestamp"])
            <= min(range_end - minimum, visual_time + VISUAL_TITLE_LOOKAHEAD_SECONDS)
        ]
        best: dict[str, Any] | None = None
        best_rank = -1.0
        best_similarity = 0.0
        for candidate in eligible:
            timestamp = float(candidate["timestamp"])
            right = _window_text(
                speech,
                timestamp,
                min(range_end, timestamp + VISUAL_TITLE_ASR_WINDOW_SECONDS),
            )
            similarity = _match_similarity(match_title, right)
            title_terms = {
                term
                for term in _FINANCE_TERMS
                if term in match_title and _normalise_match_text(term) in _normalise_match_text(right)
            }
            term_bonus = min(0.16, len(title_terms) * 0.08)
            timing = max(
                0.0,
                1.0 - abs(visual_time - timestamp) / VISUAL_TITLE_LOOKBACK_SECONDS,
            )
            rank = (
                similarity * 0.72
                + term_bonus
                + float(candidate["discourse_transition"]) * 0.22
                + timing * 0.06
            )
            tie_break = (
                rank,
                similarity,
                float(candidate["discourse_transition"]),
                -timestamp,
            )
            current_tie = (
                best_rank,
                best_similarity,
                float(best["discourse_transition"]) if best is not None else -1.0,
                -float(best["timestamp"]) if best is not None else -math.inf,
            )
            if tie_break > current_tie:
                best = candidate
                best_rank = rank
                best_similarity = similarity
        if best is None or best_similarity < 0.045 or best_rank < 0.28:
            continue
        ocr_confidence = float(title["ocr_confidence"])
        mapping_component = min(
            1.0,
            max(
                0.0,
                0.55 * min(1.0, best_rank / 0.60) + 0.45 * ocr_confidence,
            ),
        )
        best["visual_title_change"] = round(
            max(float(best["visual_title_change"]), mapping_component), 6
        )
        best["visual_evidence"].append(
            {
                "visual_event_id": title["visual_event_id"],
                "visual_timestamp": round(visual_time, 6),
                "raw_ocr_title": raw_title,
                "ocr_confidence": round(ocr_confidence, 6),
                "title_to_asr_similarity": round(best_similarity, 6),
                "mapping_score": round(best_rank, 6),
                "visual_component": round(mapping_component, 6),
                "lookback_seconds": round(max(0.0, visual_time - float(best["timestamp"])), 6),
                "visual_timestamp_used_as_boundary": False,
            }
        )
        previous_mapped_time = float(best["timestamp"])


def _score_candidates(candidates: list[dict[str, Any]]) -> None:
    for candidate in candidates:
        components = {
            key: float(candidate[key])
            for key in (
                "lexical_novelty",
                "discourse_transition",
                "pause",
                "visual_title_change",
            )
        }
        score = sum(components[key] * WEIGHTS[key] for key in WEIGHTS)
        candidate["score"] = round(score, 6)
        candidate["score_components"] = components


def _boundary_priority(candidate: Mapping[str, Any]) -> float:
    return float(candidate["score"])


def _select_boundaries(
    candidates: Sequence[dict[str, Any]],
    *,
    range_start: float,
    range_end: float,
    minimum: float,
    target: float,
    maximum: float,
) -> list[dict[str, Any]]:
    natural = [
        candidate
        for candidate in candidates
        if (
            float(candidate["discourse_transition"]) >= 0.90
            or (
                float(candidate["visual_title_change"]) >= VISUAL_TITLE_MIN_COMPONENT
                and max(
                    (
                        float(item.get("title_to_asr_similarity", 0.0))
                        for item in candidate.get("visual_evidence", [])
                        if isinstance(item, Mapping)
                    ),
                    default=0.0,
                )
                >= VISUAL_TITLE_MIN_MAPPING_SIMILARITY
            )
            or (
                float(candidate["lexical_novelty"]) >= 0.97
                and float(candidate["pause"]) >= 0.45
            )
        )
        and range_start + minimum <= float(candidate["timestamp"]) <= range_end - minimum
    ]

    # Collapse several consecutive phrases that describe the same transition.
    collapsed: list[dict[str, Any]] = []
    for candidate in natural:
        if not collapsed or float(candidate["timestamp"]) - float(collapsed[-1]["timestamp"]) >= 18.0:
            collapsed.append(candidate)
            continue
        if _boundary_priority(candidate) > _boundary_priority(collapsed[-1]):
            collapsed[-1] = candidate

    # Keep strong natural boundaries when they respect the minimum chapter size.
    selected: list[dict[str, Any]] = []
    for candidate in collapsed:
        timestamp = float(candidate["timestamp"])
        if selected and timestamp - float(selected[-1]["timestamp"]) < minimum:
            if _boundary_priority(candidate) > _boundary_priority(selected[-1]):
                selected[-1] = candidate
            continue
        if timestamp - (float(selected[-1]["timestamp"]) if selected else range_start) < minimum:
            continue
        selected.append(candidate)

    while selected and range_end - float(selected[-1]["timestamp"]) < minimum:
        selected.pop()

    # A topic with no strong cue must still be reviewable. Split only gaps that violate
    # the declared maximum, choosing the strongest ASR boundary near the target length.
    changed = True
    while changed:
        changed = False
        points = [range_start] + [float(item["timestamp"]) for item in selected] + [range_end]
        for left, right in zip(points, points[1:]):
            if right - left <= maximum + 1e-6:
                continue
            ideal = min(left + target, right - minimum)
            eligible = [
                candidate
                for candidate in candidates
                if left + minimum <= float(candidate["timestamp"]) <= right - minimum
            ]
            if not eligible:
                raise ValueError("No ASR segment boundary can satisfy max_chapter_seconds")
            chosen = max(
                eligible,
                key=lambda item: (
                    float(item["score"]) - abs(float(item["timestamp"]) - ideal) / maximum,
                    -abs(float(item["timestamp"]) - ideal),
                    -float(item["timestamp"]),
                ),
            )
            chosen = dict(chosen)
            chosen["forced_by_max_duration"] = True
            selected.append(chosen)
            selected.sort(key=lambda item: float(item["timestamp"]))
            changed = True
            break
    return selected


def _chapter_title(
    segments: Sequence[Mapping[str, Any]], *, chapter_start: float
) -> tuple[str, str]:
    nearby = [
        segment
        for segment in segments
        if float(segment["absolute_start_seconds"]) < chapter_start + 32.0
    ]
    if not nearby:
        nearby = list(segments[:1])
    best: Mapping[str, Any] | None = None
    best_score = -math.inf
    for position, segment in enumerate(nearby):
        text = str(segment["text"]).strip()
        compact = _normalise_text(text)
        if not compact:
            continue
        term_count = sum(term in text for term in _FINANCE_TERMS)
        discourse, _ = _discourse_evidence(text)
        length_score = 1.0 - min(1.0, abs(len(compact) - 13) / 28.0)
        penalty = sum(term.lower() in compact for term in _TITLE_PENALTIES) * 0.45
        score = term_count * 0.8 + discourse * 0.35 + length_score * 0.3 - penalty - position * 0.008
        if score > best_score:
            best = segment
            best_score = score
    if best is None:
        return "（无语音文本）", ""
    return str(best["text"]), str(best["event_id"])


def _keywords(raw_text: str, *, limit: int = 8) -> list[str]:
    values: list[tuple[int, int, str]] = []
    for position, term in enumerate(_FINANCE_TERMS):
        count = raw_text.count(term)
        if count:
            values.append((count, -position, term))
    values.sort(reverse=True)
    selected = [value[2] for value in values[:limit]]
    if len(selected) < limit:
        latin = Counter(re.findall(r"(?i)\b[a-z][a-z0-9.-]{1,15}\b", raw_text))
        for token, _ in latin.most_common(limit - len(selected)):
            if token not in selected:
                selected.append(token)
    return selected


def _assign_chapters(
    speech: Sequence[Mapping[str, Any]],
    visual: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    *,
    range_start: float,
    range_end: float,
) -> list[dict[str, Any]]:
    starts = [range_start] + [float(item["timestamp"]) for item in boundaries]
    ends = starts[1:] + [range_end]
    chapters: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(starts, ends)):
        chapter_speech = [
            item
            for item in speech
            if start <= float(item["absolute_start_seconds"]) < end
            or (
                index == len(starts) - 1
                and math.isclose(float(item["absolute_start_seconds"]), end, abs_tol=1e-6)
            )
        ]
        chapter_visual = [
            item
            for item in visual
            if start <= float(item["timestamp"]) < end
            or (
                index == len(starts) - 1
                and math.isclose(float(item["timestamp"]), end, abs_tol=1e-6)
            )
        ]
        title, title_source = _chapter_title(chapter_speech, chapter_start=start)
        raw_text = " ".join(str(item["text"]) for item in chapter_speech)
        if index == 0:
            boundary = {
                "kind": "range_start",
                "timestamp": round(start, 6),
                "timecode": format_timecode(start),
                "forced": False,
            }
        else:
            candidate = boundaries[index - 1]
            boundary = {
                "kind": (
                    "max_duration_split"
                    if candidate.get("forced_by_max_duration")
                    else "evidence_boundary"
                ),
                "timestamp": round(start, 6),
                "timecode": format_timecode(start),
                "speech_event_id": candidate["speech_event_id"],
                "score": candidate["score"],
                "score_components": candidate["score_components"],
                "discourse_cues": list(candidate["discourse_cues"]),
                "pause_seconds": candidate["pause_seconds"],
                "visual_evidence": list(candidate["visual_evidence"]),
                "forced": bool(candidate.get("forced_by_max_duration", False)),
                "boundary_basis": "existing_asr_segment_start",
            }
        chapters.append(
            {
                "chapter_id": f"chapter-{index + 1:03d}",
                "start_seconds": round(start, 6),
                "end_seconds": round(end, 6),
                "duration_seconds": round(end - start, 6),
                "start_timecode": format_timecode(start),
                "end_timecode": format_timecode(end),
                "title": title,
                "title_source_event_id": title_source,
                "title_exact_extract": bool(title_source),
                "keywords": _keywords(raw_text),
                "raw_text": raw_text,
                "speech_event_ids": [str(item["event_id"]) for item in chapter_speech],
                "visual_event_ids": [str(item["event_id"]) for item in chapter_visual],
                "boundary": boundary,
            }
        )
    return chapters


def _duplicates(values: Iterable[str]) -> int:
    counter = Counter(values)
    return sum(count - 1 for count in counter.values() if count > 1)


def _asset_uri(path: Path, *, review_directory: Path) -> str:
    relative = Path(os.path.relpath(path, review_directory)).as_posix()
    return urllib.parse.quote(relative, safe="/._-")


def _review_html(document: Mapping[str, Any], *, audio_path: Path, review_directory: Path) -> str:
    coverage = _mapping(document["coverage"], label="semantic coverage")
    chapters = _list(document["chapters"], label="semantic chapters")
    range_start = float(coverage["start_seconds"])
    range_end = float(coverage["end_seconds"])
    span = max(1e-9, range_end - range_start)
    blocks: list[str] = []
    cards: list[str] = []
    palette = ("#08775e", "#245f9e", "#8c5a17", "#74458e", "#2d6f7a")
    for index, raw_chapter in enumerate(chapters):
        chapter = _mapping(raw_chapter, label=f"chapter {index}")
        start = float(chapter["start_seconds"])
        end = float(chapter["end_seconds"])
        left = (start - range_start) / span * 100.0
        width = (end - start) / span * 100.0
        colour = palette[index % len(palette)]
        title = html.escape(str(chapter["title"]))
        blocks.append(
            f"<button class='chapter-block' style='left:{left:.5f}%;width:{width:.5f}%;"
            f"background:{colour}' data-time='{start:.6f}' title='{title}'></button>"
        )
        boundary = _mapping(chapter["boundary"], label=f"chapter {index} boundary")
        components = boundary.get("score_components")
        component_html = ""
        if isinstance(components, Mapping):
            component_html = (
                "<dl class='signals'>"
                f"<div><dt>语义变化</dt><dd>{float(components.get('lexical_novelty', 0)):.2f}</dd></div>"
                f"<div><dt>转场措辞</dt><dd>{float(components.get('discourse_transition', 0)):.2f}</dd></div>"
                f"<div><dt>停顿</dt><dd>{float(components.get('pause', 0)):.2f}</dd></div>"
                f"<div><dt>PPT 标题</dt><dd>{float(components.get('visual_title_change', 0)):.2f}</dd></div>"
                "</dl>"
            )
        visual_evidence = boundary.get("visual_evidence")
        visual_rows = ""
        if isinstance(visual_evidence, list) and visual_evidence:
            visual_rows = "<ul class='visual-evidence'>" + "".join(
                "<li>画面 "
                f"{format_timecode(float(item.get('visual_timestamp', 0)))}："
                f"{html.escape(str(item.get('raw_ocr_title', '')))}"
                f"（向前绑定 {float(item.get('lookback_seconds', 0)):.1f}s）</li>"
                for item in visual_evidence
                if isinstance(item, Mapping)
            ) + "</ul>"
        keywords = chapter.get("keywords")
        keyword_html = "".join(
            f"<span>{html.escape(str(item))}</span>"
            for item in keywords
        ) if isinstance(keywords, list) else ""
        title_claim = (
            "标题为 ASR 原句摘录"
            if chapter.get("title_exact_extract") is True
            else "本章没有可用的 ASR 标题摘录"
        )
        cards.append(
            f"<article class='chapter' id='{html.escape(str(chapter['chapter_id']), quote=True)}'>"
            "<header><div>"
            f"<p class='eyebrow'>第 {index + 1} 章 · {format_timecode(start)}–{format_timecode(end)}</p>"
            f"<h2>{title}</h2></div>"
            f"<button class='seek' data-time='{start:.6f}'>从这里播放</button></header>"
            f"<div class='keywords'>{keyword_html}</div>"
            f"{component_html}{visual_rows}"
            f"<p class='counts'>{len(chapter['speech_event_ids'])} 个原始语音段 · "
            f"{len(chapter['visual_event_ids'])} 个视觉帧 · {title_claim}</p>"
            "<details><summary>查看未经改写的章节 ASR</summary>"
            f"<p class='transcript'>{html.escape(str(chapter['raw_text']))}</p></details>"
            "</article>"
        )
    audio_uri = html.escape(_asset_uri(audio_path, review_directory=review_directory), quote=True)
    source = _mapping(document["source"], label="semantic source")
    alignment = _mapping(source["alignment"], label="semantic alignment source")
    parameters = _mapping(document["parameters"], label="semantic parameters")
    return f"""<!doctype html>
<html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; object-src 'none'; base-uri 'none'">
<title>FrameLedger · 本地语义分段</title>
<style>
:root {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#17212b; background:#eef2f6; }}
body {{ margin:0; }} main {{ max-width:1120px; margin:32px auto; padding:0 20px 80px; }}
.hero,.chapter {{ background:#fff; border:1px solid #d4dce4; border-radius:14px; box-shadow:0 8px 28px #14283b10; }}
.hero {{ padding:24px; margin-bottom:18px; }} h1 {{ margin:0 0 8px; }} .subtle,.counts {{ color:#5d6a76; }}
audio {{ width:100%; margin:14px 0; }} .timeline {{ position:relative; height:34px; background:#dce3e9; border-radius:8px; overflow:hidden; }}
.chapter-block {{ position:absolute; top:0; bottom:0; border:0; border-right:2px solid #fff; opacity:.9; cursor:pointer; }}
.chapter-list {{ display:grid; gap:14px; }} .chapter {{ padding:21px 23px; }} .chapter header {{ display:flex; justify-content:space-between; gap:18px; align-items:flex-start; }}
.chapter h2 {{ margin:3px 0 10px; font-size:1.28rem; }} .eyebrow {{ margin:0; color:#6b7781; font-size:.83rem; }}
button.seek {{ border:0; background:#08775e; color:#fff; padding:9px 12px; border-radius:7px; cursor:pointer; white-space:nowrap; }}
.keywords {{ display:flex; gap:6px; flex-wrap:wrap; margin:5px 0 14px; }} .keywords span {{ background:#edf4f1; color:#075d48; border-radius:999px; padding:4px 8px; font-size:.82rem; }}
.signals {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin:10px 0; }} .signals div {{ background:#f3f6f8; padding:9px; border-radius:7px; }}
.signals dt {{ color:#65727d; font-size:.76rem; }} .signals dd {{ margin:3px 0 0; font-weight:700; }}
.visual-evidence {{ background:#f5f7fa; padding:10px 10px 10px 30px; border-radius:7px; color:#3c4a56; }}
details {{ border-top:1px solid #e2e7eb; padding-top:12px; }} summary {{ cursor:pointer; color:#075eac; }} .transcript {{ line-height:1.7; white-space:pre-wrap; }}
code {{ font-size:11px; overflow-wrap:anywhere; }} a {{ color:#075eac; }}
@media(max-width:700px) {{ .signals {{ grid-template-columns:repeat(2,1fr); }} .chapter header {{ display:block; }} button.seek {{ margin-top:8px; }} }}
</style></head><body><main>
<section class="hero"><h1>本地语义分段</h1>
<p class="subtle">{len(chapters)} 个章节 · {format_timecode(range_start)}–{format_timecode(range_end)}。边界来自已有 ASR 起点；PPT 标题只作软证据。未调用网络 AI、聊天模型，也没有纠正或总结原始识别文本。</p>
<audio id="audio" controls preload="metadata" src="{audio_uri}"></audio>
<div class="timeline" aria-label="章节时间轴">{''.join(blocks)}</div>
<p><a href="semantic-segments.json">机器可读语义账本</a> · 上游对齐 SHA-256：<code>{html.escape(str(alignment['sha256']))}</code></p>
<p class="subtle">算法 {html.escape(str(parameters['policy_version']))}；这是确定性本地分段，不声称生成了新的事实。</p></section>
<section class="chapter-list">{''.join(cards)}</section>
</main><script src="review.js"></script></body></html>"""


_REVIEW_JS = """(() => {
  'use strict';
  const audio = document.getElementById('audio');
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-time]');
    if (!target || !audio) return;
    const value = Number(target.dataset.time);
    if (!Number.isFinite(value)) return;
    audio.currentTime = value;
    audio.play().catch(() => {});
  });
})();
"""


def run_semantic_segmentation(
    alignment_json: str | Path,
    *,
    output: str | Path,
    min_chapter_seconds: float = DEFAULT_MIN_CHAPTER_SECONDS,
    target_chapter_seconds: float = DEFAULT_TARGET_CHAPTER_SECONDS,
    max_chapter_seconds: float = DEFAULT_MAX_CHAPTER_SECONDS,
    topic_window_seconds: float = DEFAULT_TOPIC_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Segment one complete aligned evidence timeline using local deterministic evidence."""

    minimum, target, maximum, window = _validate_parameters(
        min_chapter_seconds,
        target_chapter_seconds,
        max_chapter_seconds,
        topic_window_seconds,
    )
    alignment_path = Path(alignment_json).expanduser().resolve()
    if not alignment_path.is_file():
        raise ValueError(f"Alignment JSON does not exist: {alignment_path}")
    alignment_sha256 = _sha256(alignment_path)
    alignment = _load_bound_json(
        alignment_path, alignment_sha256, label="alignment"
    )
    if alignment.get("kind") != "timestamp_aligned_evidence":
        raise ValueError("Semantic input kind must be timestamp_aligned_evidence")
    if alignment.get("schema_version") != 1:
        raise ValueError("Semantic input alignment schema_version must be 1")
    coverage_source = _mapping(alignment.get("coverage"), label="alignment coverage")
    if coverage_source.get("complete_phase1_speech_coverage") is not True:
        raise ValueError("Semantic segmentation requires complete Phase 1 speech coverage")
    phase_range = _mapping(coverage_source.get("phase1_range"), label="Phase 1 range")
    range_start = _number(phase_range.get("start_seconds"), label="Phase 1 start")
    range_end = _number(phase_range.get("end_seconds"), label="Phase 1 end")
    if range_start < 0 or range_end <= range_start:
        raise ValueError("Semantic input range is invalid")

    source = _mapping(alignment.get("source"), label="alignment source")
    video_source = _mapping(source.get("video"), label="alignment video source")
    phase1_source = _mapping(source.get("phase1"), label="alignment Phase 1 source")
    asr_source = _mapping(source.get("asr"), label="alignment ASR source")
    ocr_source = _mapping(source.get("ocr"), label="alignment OCR source")
    video_path = _source_file(
        alignment_path,
        video_source.get("path"),
        video_source.get("fingerprint"),
        label="source video",
    )
    manifest_path = _source_file(
        alignment_path,
        phase1_source.get("manifest_path"),
        phase1_source.get("manifest_sha256"),
        label="Phase 1 manifest",
    )
    strategy_path = _source_file(
        alignment_path,
        phase1_source.get("strategy_path"),
        phase1_source.get("strategy_sha256"),
        label="Phase 1 strategy",
    )
    asr_path = _source_file(
        alignment_path, asr_source.get("path"), asr_source.get("sha256"), label="ASR"
    )
    ocr_path = _source_file(
        alignment_path, ocr_source.get("path"), ocr_source.get("sha256"), label="OCR"
    )
    audio_path = _source_file(
        alignment_path,
        asr_source.get("audio_path"),
        asr_source.get("audio_sha256"),
        label="ASR audio",
    )
    output_path = Path(output).expanduser().resolve()
    _protect_output(
        output_path,
        alignment_path=alignment_path,
        source=source,
        upstream_files=(
            video_path,
            manifest_path,
            strategy_path,
            asr_path,
            ocr_path,
            audio_path,
        ),
    )

    raw_speech = _list(alignment.get("speech_segments"), label="alignment speech_segments")
    if not raw_speech:
        raise ValueError("Semantic input contains no speech segments")
    speech: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    previous_start = -math.inf
    previous_end = -math.inf
    for index, raw in enumerate(raw_speech):
        segment = dict(_mapping(raw, label=f"speech segment {index}"))
        event_id = segment.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"speech segment {index}.event_id must be non-empty")
        if event_id in event_ids:
            raise ValueError(f"Semantic input repeats speech event_id {event_id}")
        event_ids.add(event_id)
        start = _number(segment.get("absolute_start_seconds"), label=f"speech {event_id} start")
        end = _number(segment.get("absolute_end_seconds"), label=f"speech {event_id} end")
        if start < previous_start - 1e-6 or end < start:
            raise ValueError("Semantic input speech segments are not chronological")
        if start < previous_end - 1e-6:
            raise ValueError("Semantic input speech segments overlap in source time")
        if start < range_start - 1e-3 or end > range_end + 1e-3:
            raise ValueError(f"Speech segment {event_id} escapes the complete range")
        if not isinstance(segment.get("text"), str):
            raise ValueError(f"Speech segment {event_id}.text must be a string")
        previous_start = start
        previous_end = end
        speech.append(segment)

    # Prove that the aligned raw transcript is still the bound ASR transcript.
    asr_document = _load_bound_json(
        asr_path, str(asr_source["sha256"]), label="ASR"
    )
    _load_bound_json(ocr_path, str(ocr_source["sha256"]), label="OCR")
    transcript = _mapping(asr_document.get("transcript"), label="ASR transcript")
    original_segments = _list(transcript.get("segments"), label="ASR transcript segments")
    if len(original_segments) != len(speech):
        raise ValueError("Aligned speech count no longer matches the bound ASR transcript")
    for index, (aligned, raw) in enumerate(zip(speech, original_segments)):
        original = _mapping(raw, label=f"ASR segment {index}")
        for key in ("id", "absolute_start_seconds", "absolute_end_seconds", "text"):
            if aligned.get(key) != original.get(key):
                raise ValueError(f"Aligned speech segment {index}.{key} differs from bound ASR")

    raw_visual = _list(alignment.get("visual_frames"), label="alignment visual_frames")
    visual: list[dict[str, Any]] = []
    visual_ids: set[str] = set()
    previous_visual_timestamp = -math.inf
    for index, raw in enumerate(raw_visual):
        frame = dict(_mapping(raw, label=f"visual frame {index}"))
        event_id = frame.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"visual frame {index}.event_id must be non-empty")
        if event_id in visual_ids:
            raise ValueError(f"Semantic input repeats visual event_id {event_id}")
        visual_ids.add(event_id)
        timestamp = _number(frame.get("timestamp"), label=f"visual {event_id} timestamp")
        if not range_start <= timestamp <= range_end:
            raise ValueError(f"Visual frame {event_id} escapes the complete range")
        if timestamp < previous_visual_timestamp - 1e-6:
            raise ValueError("Semantic input visual frames are not chronological")
        previous_visual_timestamp = timestamp
        visual.append(frame)

    candidates = _candidate_records(speech, window_seconds=window)
    title_changes = _changed_visual_titles(visual)
    _attach_visual_evidence(
        candidates,
        speech,
        title_changes,
        range_start=range_start,
        range_end=range_end,
        minimum=minimum,
    )
    _score_candidates(candidates)
    boundaries = _select_boundaries(
        candidates,
        range_start=range_start,
        range_end=range_end,
        minimum=minimum,
        target=target,
        maximum=maximum,
    )
    chapters = _assign_chapters(
        speech,
        visual,
        boundaries,
        range_start=range_start,
        range_end=range_end,
    )

    assigned_speech = [item for chapter in chapters for item in chapter["speech_event_ids"]]
    assigned_visual = [item for chapter in chapters for item in chapter["visual_event_ids"]]
    unassigned_speech = len(event_ids - set(assigned_speech))
    unassigned_visual = len(visual_ids - set(assigned_visual))
    duplicate_speech = _duplicates(assigned_speech)
    duplicate_visual = _duplicates(assigned_visual)
    complete_assignment = (
        len(assigned_speech) == len(speech)
        and len(assigned_visual) == len(visual)
        and unassigned_speech == 0
        and unassigned_visual == 0
        and duplicate_speech == 0
        and duplicate_visual == 0
    )
    if not complete_assignment:
        raise RuntimeError("Semantic chapter assignment is not complete and one-to-one")

    coverage = {
        "start_seconds": round(range_start, 6),
        "end_seconds": round(range_end, 6),
        "duration_seconds": round(range_end - range_start, 6),
        "start_timecode": format_timecode(range_start),
        "end_timecode": format_timecode(range_end),
        "source_speech_segment_count": len(speech),
        "assigned_speech_segment_count": len(assigned_speech),
        "unassigned_speech_segment_count": unassigned_speech,
        "duplicate_speech_assignment_count": duplicate_speech,
        "source_visual_frame_count": len(visual),
        "assigned_visual_frame_count": len(assigned_visual),
        "unassigned_visual_frame_count": unassigned_visual,
        "duplicate_visual_assignment_count": duplicate_visual,
        "overlapping_source_speech_segment_count": 0,
        "complete_event_assignment": complete_assignment,
    }
    summary = {
        "chapter_count": len(chapters),
        "natural_boundary_count": sum(
            not bool(item.get("forced_by_max_duration")) for item in boundaries
        ),
        "forced_boundary_count": sum(
            bool(item.get("forced_by_max_duration")) for item in boundaries
        ),
        "speech_segment_count": len(speech),
        "visual_frame_count": len(visual),
        "visual_title_change_count": len(title_changes),
        "visual_title_change_mapped_count": sum(
            bool(item.get("visual_evidence")) for item in candidates
        ),
    }
    document: dict[str, Any] = {
        "kind": "local_topic_segmentation",
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "policy_version": SEMANTIC_POLICY_VERSION,
            "algorithm": "char_ngram_novelty_plus_discourse_pause_and_lagged_visual_title",
            "candidate_boundary_basis": "existing_asr_segment_starts_only",
            "topic_window_seconds": window,
            "min_chapter_seconds": minimum,
            "target_chapter_seconds": target,
            "max_chapter_seconds": maximum,
            "weights": dict(WEIGHTS),
            "visual_title_lookback_seconds": VISUAL_TITLE_LOOKBACK_SECONDS,
            "visual_title_lookahead_seconds": VISUAL_TITLE_LOOKAHEAD_SECONDS,
            "visual_title_asr_window_seconds": VISUAL_TITLE_ASR_WINDOW_SECONDS,
            "visual_title_min_ocr_confidence": VISUAL_TITLE_MIN_OCR_CONFIDENCE,
            "visual_title_min_mapping_similarity": VISUAL_TITLE_MIN_MAPPING_SIMILARITY,
            "visual_title_min_component": VISUAL_TITLE_MIN_COMPONENT,
            "visual_title_repeat_similarity": 0.72,
            "visual_timestamp_is_hard_boundary": False,
            "title_policy": "exact_raw_asr_segment_extract",
            "determinism_scope": "chapters_and_scores_for_identical_input_bytes_and_parameters",
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
                "python_version": sys.version.split()[0],
            },
            "semantic_embeddings_used": False,
            "correction_applied": False,
            "summarization_applied": False,
            "network_ai_used": False,
            "chat_model_used": False,
        },
        "source": {
            "alignment": {
                "path": str(alignment_path),
                "sha256": alignment_sha256,
            },
            "video": dict(video_source),
            "phase1": dict(phase1_source),
            "ocr": dict(ocr_source),
            "asr": dict(asr_source),
        },
        "coverage": coverage,
        "chapters": chapters,
        "summary": summary,
    }
    for path, expected, label in (
        (alignment_path, alignment_sha256, "alignment"),
        (video_path, str(video_source["fingerprint"]), "source video"),
        (manifest_path, str(phase1_source["manifest_sha256"]), "Phase 1 manifest"),
        (strategy_path, str(phase1_source["strategy_sha256"]), "Phase 1 strategy"),
        (ocr_path, str(ocr_source["sha256"]), "OCR"),
        (asr_path, str(asr_source["sha256"]), "ASR"),
        (audio_path, str(asr_source["audio_sha256"]), "ASR audio"),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"{label} changed while semantic segmentation was running")
    output_path.mkdir(parents=True, exist_ok=False)
    semantic_path = output_path / "semantic-segments.json"
    review_path = output_path / "review.html"
    script_path = output_path / "review.js"
    write_json(semantic_path, document)
    review_path.write_text(
        _review_html(document, audio_path=audio_path, review_directory=output_path),
        encoding="utf-8",
    )
    script_path.write_text(_REVIEW_JS, encoding="utf-8")
    return {
        "kind": document["kind"],
        "output": str(output_path),
        "review_html": str(review_path),
        "semantic_json": str(semantic_path),
        "source": document["source"],
        "summary": summary,
        "coverage": coverage,
    }
