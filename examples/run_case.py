"""Load a 21311/21316 training case and optionally step the HFM env."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.case_references import CASE_SHOT_IDS, build_case_reference  # noqa: E402
from environment.shot_registry import get_fge_init_config_for_shot, get_shot_psm_config_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a 21311/21316 training case")
    parser.add_argument("--shot", choices=CASE_SHOT_IDS, default="21316_150")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config or (root / "configs" / f"case_{args.shot}.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    max_steps = int(args.max_steps or cfg.get("max_steps", 100))
    init_cfg = get_fge_init_config_for_shot(args.shot)
    reference = build_case_reference(args.shot, max_steps)

    print(f"shot_id={args.shot}")
    print(f"psm={get_shot_psm_config_path(args.shot)}")
    print("fge_init:")
    for key, value in init_cfg.items():
        print(f"  {key}: {value}")
    print(
        "reference[0/last]: "
        f"Ip={reference['Ip'][0]:.0f}/{reference['Ip'][-1]:.0f} "
        f"Rmax={reference['Rmax'][0]:.3f}/{reference['Rmax'][-1]:.3f} "
        f"kappa={reference['kappa'][0]:.3f}/{reference['kappa'][-1]:.3f}"
    )
    if args.dry_run:
        return 0

    from environment.hfm_simulator import HFMSimulator

    env = HFMSimulator(cfg)
    try:
        obs, info = env.reset(seed=0)
        print(f"reset ok, shot_id={info['shot_id']}, Ip={float(np.asarray(obs['Ip']).reshape(-1)[0]):.1f}")
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        for step in range(min(3, max_steps)):
            obs, reward, terminated, truncated, _ = env.step(action)
            print(
                f"step {step}: reward={float(reward):.4f}, "
                f"terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
