from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frameledger.artifacts import resolve_strategy_artifact


class StrategyArtifactTests(unittest.TestCase):
    def test_strategy_identifier_is_confined_to_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            strategies = run_root / "strategies"
            strategies.mkdir(parents=True)
            expected = strategies / "presentation_states.json"
            expected.write_text("{}", encoding="utf-8")

            name, path = resolve_strategy_artifact(
                run_root, " presentation_states "
            )
            self.assertEqual(name, "presentation_states")
            self.assertEqual(path, expected.resolve())

            for value in (
                "../outside",
                "/tmp/outside",
                "presentation/states",
                "presentation\\states",
                "",
                123,
            ):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(ValueError, "Strategy"):
                        resolve_strategy_artifact(run_root, value)

    def test_strategy_symlink_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strategies = root / "run" / "strategies"
            strategies.mkdir(parents=True)
            outside = root / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            (strategies / "hybrid.json").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_strategy_artifact(root / "run", "hybrid")


if __name__ == "__main__":
    unittest.main()
