"""Train a minimal PPO baseline for F2a/F2b with the provided XPT target reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.semifinal_training_common import add_common_args, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="F2 PPO example: shape transition -> XPT target")
    add_common_args(
        parser,
        default_save_dir=ROOT / "results" / "f2_ppo",
        default_reference=ROOT / "configs" / "xpt_reference_targets.json",
        default_start_shot="13844_500",
        default_max_steps=500,
    )
    args = parser.parse_args()
    return run_training(args, task="F2")


if __name__ == "__main__":
    raise SystemExit(main())
