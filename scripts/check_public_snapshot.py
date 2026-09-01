#!/usr/bin/env python3
"""Verify that the public snapshot contains only approved repository material."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    ".python-version",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "docs",
    "phase2",
    "pyproject.toml",
    "scripts",
    "src",
    "tests",
    "uv.lock",
}
LOCAL_ONLY_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "outputs",
}
FORBIDDEN_DIRECTORY_NAMES = {
    ".frameledger-batches",
    ".frameledger-imports",
    "annotations",
    "freezes",
    "samples",
}
FORBIDDEN_SUFFIXES = {
    ".gguf",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npz",
    ".pem",
    ".pfx",
    ".png",
    ".safetensors",
    ".wav",
    ".webm",
}
ABSOLUTE_PRIVATE_PATH_MARKERS = (
    "/" + "Users" + "/",
    "/" + "home" + "/",
    ":\\" + "Users" + "\\",
)
PRIVATE_KEY_MARKER = re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY")
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024


def is_local_only(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(part in LOCAL_ONLY_DIRECTORY_NAMES for part in relative.parts)


def main() -> int:
    errors: list[str] = []
    checked_files = 0

    for entry in ROOT.iterdir():
        if entry.name not in ALLOWED_TOP_LEVEL | LOCAL_ONLY_DIRECTORY_NAMES:
            errors.append(f"unexpected top-level path: {entry.name}")

    for path in ROOT.rglob("*"):
        if is_local_only(path):
            continue
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts):
            errors.append(f"forbidden directory content: {relative}")
            continue
        if path.is_symlink():
            errors.append(f"symlink requires explicit review: {relative}")
            continue
        if not path.is_file():
            continue
        checked_files += 1
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            errors.append(f"file exceeds 5 MiB public limit: {relative}")
        lower_name = path.name.lower()
        if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden media/model/key file: {relative}")
        if lower_name.endswith(".tar.gz"):
            errors.append(f"archive requires explicit review: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            errors.append(f"non-UTF-8 or unreadable file requires review: {relative}")
            continue
        if any(marker in text for marker in ABSOLUTE_PRIVATE_PATH_MARKERS):
            errors.append(f"private absolute path marker: {relative}")
        if PRIVATE_KEY_MARKER.search(text):
            errors.append(f"private key marker: {relative}")

    if errors:
        print("Public snapshot check failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Public snapshot allowlist check passed for {checked_files} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
