"""
Example reward: optional reward_fn for HFMSimulator.
Use as: HFMSimulator(config, reward_fn=example_reward_fn).
"""

import numpy as np


def _rel_err(x: float, target: float, eps: float = 1e-6) -> float:
    return abs(x - target) / max(abs(target), eps)


def example_reward_fn(
    observation: dict,
    action: np.ndarray,
    terminated: bool = False,
    truncated: bool = False,
    info: dict | None = None,
) -> float:
    """Example standardized reward using current-step reference.

    Current starter controlled variables:
    - Ip
    - R
    - Z
    LCFS points interface is exposed but not scored in this placeholder reward.
    """
    if terminated:
        return -10.0

    err_ip = _rel_err(
        float(np.asarray(observation.get("Ip", 0.0)).ravel()[0]),
        float(np.asarray(observation.get("reference_Ip", 0.0)).ravel()[0]),
        eps=1e6,
    )
    err_r = _rel_err(
        float(np.asarray(observation.get("R", 0.0)).ravel()[0]),
        float(np.asarray(observation.get("reference_R", 0.0)).ravel()[0]),
    )
    err_z = _rel_err(
        float(np.asarray(observation.get("Z", 0.0)).ravel()[0]),
        float(np.asarray(observation.get("reference_Z", 0.0)).ravel()[0]),
    )
    return float(-(err_ip + err_r + err_z) / 3.0)
