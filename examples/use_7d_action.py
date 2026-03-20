"""
Use Action7DTo12DWrapper to train with 7D action (example).
"""
# ruff: noqa: E402, I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from environment import Action7DTo12DWrapper, DictObsFlattenWrapper, HFMSimulator


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "env_default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = HFMSimulator(config)
    env = Action7DTo12DWrapper(env)
    env = DictObsFlattenWrapper(
        env,
        keys=[
            "Ip", "R", "Z",
            "reference_Ip", "reference_R", "reference_Z",
            "lcfs_points", "reference_lcfs_points",
            "I_PF",
        ],
    )

    obs, info = env.reset(seed=42)
    print("action_space shape:", env.action_space.shape)  # (7,)
    action_7d = np.random.uniform(env.action_space.low, env.action_space.high)
    obs, reward, term, trunc, info = env.step(action_7d)
    print("step ok.")

    env.close()


if __name__ == "__main__":
    main()
