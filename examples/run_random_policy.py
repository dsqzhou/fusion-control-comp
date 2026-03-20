"""Run a random 12D action policy for a few steps."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from environment import HFMSimulator


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "env_default.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = HFMSimulator(config)
    obs, info = env.reset(seed=42)
    print(obs)
    print("reset ok, obs keys:", list(obs.keys()))
    print("shot_id:", info.get("shot_id"))

    for step in range(5):
        action = np.random.uniform(env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"step {step} reward={reward} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            break

    env.close()
    print("done.")


if __name__ == "__main__":
    main()
