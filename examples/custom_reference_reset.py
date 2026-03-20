"""Reset with custom initial parameters and a custom reference trajectory."""

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
    options = {
        "reset_params": {
            "signeo": 7.2e6,
            "bp": 0.20,
            "q0": 1.6,
        },
        "reference_mode": "trajectory",
        "reference": {
            "Ip": np.linspace(4.95e5, 5.05e5, config["max_steps"]),
            "R": np.linspace(0.79, 0.81, config["max_steps"]),
            "Z": np.zeros(config["max_steps"]),
        },
    }
    obs, info = env.reset(seed=0, options=options)
    print("reset ok, shot_id:", info["shot_id"])
    print("reference at step 0:", float(obs["reference_Ip"][0]), float(obs["reference_R"][0]))
    env.close()


if __name__ == "__main__":
    main()
