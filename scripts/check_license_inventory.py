#!/usr/bin/env python3
"""Verify that the documented runtime license inventory matches both lockfiles."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_LOCK = ROOT / "uv.lock"
ASR_LOCK = ROOT / "phase2" / "mlx_whisper_asr" / "requirements.lock.txt"
INVENTORY = ROOT / "docs" / "THIRD_PARTY_LICENSES.md"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_main_lock() -> dict[str, str]:
    data = tomllib.loads(MAIN_LOCK.read_text(encoding="utf-8"))
    packages: dict[str, str] = {}
    for package in data["package"]:
        name = normalize(package["name"])
        if name == "frameledger":
            continue
        packages[name] = package["version"]
    return packages


def read_asr_lock() -> dict[str, str]:
    packages: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)")
    for line in ASR_LOCK.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line.strip())
        if match:
            packages[normalize(match.group(1))] = match.group(2)
    return packages


def read_inventory() -> dict[str, tuple[str, str, str]]:
    packages: dict[str, tuple[str, str, str]] = {}
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            raise ValueError(f"invalid inventory row: {line}")
        name = normalize(cells[0].strip("`"))
        packages[name] = (cells[1].strip("`"), cells[2], cells[3].strip("`"))
    return packages


def expected_runtime(name: str, main: dict[str, str], asr: dict[str, str]) -> str:
    if name in main and name in asr:
        return "main + ASR"
    if name in main:
        return "main"
    return "ASR"


def main() -> int:
    main_packages = read_main_lock()
    asr_packages = read_asr_lock()
    documented = read_inventory()
    expected_names = set(main_packages) | set(asr_packages)
    errors: list[str] = []

    for name in sorted(expected_names):
        expected_version = main_packages.get(name, asr_packages.get(name))
        if name in main_packages and name in asr_packages:
            if main_packages[name] != asr_packages[name]:
                errors.append(
                    f"{name}: main {main_packages[name]} != ASR {asr_packages[name]}"
                )
                continue
            expected_version = main_packages[name]
        row = documented.get(name)
        if row is None:
            errors.append(f"{name}: missing from {INVENTORY.relative_to(ROOT)}")
            continue
        version, runtime, license_expression = row
        if version != expected_version:
            errors.append(f"{name}: documented {version} != locked {expected_version}")
        expected = expected_runtime(name, main_packages, asr_packages)
        if runtime != expected:
            errors.append(f"{name}: documented runtime {runtime!r} != {expected!r}")
        if license_expression.lower() in {"", "unknown", "tbd", "待确认"}:
            errors.append(f"{name}: license metadata is not recorded")

    for name in sorted(set(documented) - expected_names):
        errors.append(f"{name}: documented but absent from both lockfiles")

    if errors:
        print("License inventory check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "License inventory matches "
        f"{len(main_packages)} main and {len(asr_packages)} ASR locked dependencies "
        f"({len(expected_names)} unique packages)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
