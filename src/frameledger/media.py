from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import cv2

from .features import resize_gray
from .models import VideoMetadata


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm"}


class MediaError(RuntimeError):
    pass


def validate_video_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".part" or resolved.name.lower().endswith(".part"):
        raise MediaError(f"Incomplete download is not a valid input: {resolved}")
    if not resolved.exists():
        raise MediaError(f"Video does not exist: {resolved}")
    if not resolved.is_file():
        raise MediaError(f"Expected one video file, not a directory: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise MediaError(f"Unsupported video extension {resolved.suffix!r}; expected one of {allowed}")
    return resolved


def _fingerprint(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash the full source so every run is bound to one immutable artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_fourcc(value: int) -> str:
    if value <= 0:
        return "unknown"
    text = "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
    return text.strip("\x00 ") or "unknown"


def probe_video(path: str | Path) -> VideoMetadata:
    video = validate_video_path(path)
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise MediaError(f"OpenCV could not open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        codec = _decode_fourcc(int(capture.get(cv2.CAP_PROP_FOURCC)))
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise MediaError(f"Video metadata is incomplete or invalid: {video}")
    finally:
        capture.release()
    stat = video.stat()
    return VideoMetadata(
        path=video,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        duration_seconds=frame_count / fps,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        codec=codec,
        fingerprint=_fingerprint(video),
    )


def iter_analysis_frames(
    metadata: VideoMetadata,
    *,
    start_seconds: float,
    end_seconds: float,
    analysis_fps: float,
    analysis_width: int,
) -> Iterator[tuple[float, int, object]]:
    if analysis_fps <= 0:
        raise ValueError("Analysis FPS must be positive")
    capture = cv2.VideoCapture(str(metadata.path))
    if not capture.isOpened():
        raise MediaError(f"OpenCV could not open video: {metadata.path}")
    start_frame = max(0, int(round(start_seconds * metadata.fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    next_sample_time = start_seconds
    sample_period = 1.0 / analysis_fps
    frame_index = start_frame
    try:
        while frame_index < metadata.frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / metadata.fps
            if timestamp > end_seconds + 1e-9:
                break
            if timestamp + (0.5 / metadata.fps) >= next_sample_time:
                yield timestamp, frame_index, resize_gray(frame, analysis_width)
                next_sample_time += sample_period
                while next_sample_time <= timestamp:
                    next_sample_time += sample_period
            frame_index += 1
    finally:
        capture.release()


def iter_video_frames(
    metadata: VideoMetadata,
    *,
    start_seconds: float,
    end_seconds: float,
):
    """Sequentially decode a bounded range and yield absolute nominal timestamps."""
    capture = cv2.VideoCapture(str(metadata.path))
    if not capture.isOpened():
        raise MediaError(f"OpenCV could not open video: {metadata.path}")
    start_frame = max(0, int(round(start_seconds * metadata.fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    try:
        while frame_index < metadata.frame_count:
            ok, frame = capture.read()
            if not ok:
                raise MediaError(
                    f"Decode failed at frame {frame_index} ({frame_index / metadata.fps:.3f}s)"
                )
            timestamp = frame_index / metadata.fps
            if timestamp > end_seconds + 1e-9:
                break
            yield timestamp, frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def read_frame_at(metadata: VideoMetadata, timestamp: float):
    target = min(max(timestamp, 0.0), max(0.0, metadata.duration_seconds - 1.0 / metadata.fps))
    capture = cv2.VideoCapture(str(metadata.path))
    try:
        if not capture.isOpened():
            raise MediaError(f"OpenCV could not open video: {metadata.path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(target * metadata.fps)))
        ok, frame = capture.read()
        if not ok:
            raise MediaError(f"Could not decode frame at {target:.3f}s from {metadata.path}")
        return frame
    finally:
        capture.release()


def write_jpeg(path: Path, frame, *, quality: int = 94) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise MediaError(f"Could not write JPEG: {path}")


def write_png(path: Path, frame, *, compression: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_PNG_COMPRESSION, compression]):
        raise MediaError(f"Could not write PNG: {path}")
