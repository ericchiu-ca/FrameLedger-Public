from __future__ import annotations

import math


def parse_timecode(value: str | int | float) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into non-negative seconds."""
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid timecodes")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("Timecode cannot be empty")
        parts = text.split(":")
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, seconds_part = parts
            seconds = int(minutes) * 60 + float(seconds_part)
        elif len(parts) == 3:
            hours, minutes, seconds_part = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_part)
        else:
            raise ValueError(f"Invalid timecode: {value!r}")
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Timecode must be a finite non-negative value: {value!r}")
    return seconds


def format_timecode(seconds: float, *, milliseconds: bool = True) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Seconds must be finite and non-negative")
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
