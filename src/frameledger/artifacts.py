from __future__ import annotations

import re
from pathlib import Path


STRATEGY_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def resolve_strategy_artifact(run_root: Path, strategy: object) -> tuple[str, Path]:
    """Resolve one strategy JSON without allowing the identifier to become a path."""
    if not isinstance(strategy, str):
        raise ValueError("Strategy name must be text")
    name = strategy.strip().lower()
    if not STRATEGY_NAME_PATTERN.fullmatch(name):
        raise ValueError("Strategy name must be a lowercase identifier")
    strategies_root = (run_root / "strategies").resolve()
    path = (strategies_root / f"{name}.json").resolve()
    if not path.is_relative_to(strategies_root):
        raise ValueError("Strategy artifact escapes the run strategies directory")
    return name, path
