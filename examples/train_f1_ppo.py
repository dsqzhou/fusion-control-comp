"""Train a minimal PPO baseline for F1 with the provided limiter target reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.semifinal_training_common import add_common_args, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="F1 PPO example: divertor-like start -> limiter target")
    add_common_args(
        parser,
        default_save_dir=ROOT / "results" / "f1_ppo",
        default_reference=ROOT / "configs" / "f1_reference_targets.json",
        default_start_shot="13844_500",
        default_max_steps=300,
    )
    args = parser.parse_args()
    return run_training(args, task="F1")


if __name__ == "__main__":
    raise SystemExit(main())
