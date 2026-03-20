"""Use DictObsFlattenWrapper for flat observation training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from environment import DictObsFlattenWrapper, HFMSimulator


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "env_default.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = HFMSimulator(config)
    env = DictObsFlattenWrapper(env)

    obs, _ = env.reset(seed=42)
    print("flatten obs shape:", obs.shape, "dtype:", obs.dtype)

    action = np.random.uniform(env.action_space.low, env.action_space.high)
    obs, _, _, _, _ = env.step(action)
    print("step ok, obs shape:", obs.shape)

    env.close()


if __name__ == "__main__":
    main()
