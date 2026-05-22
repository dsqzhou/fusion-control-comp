"""
示例：用 ``extract_xpt_observation_pack`` 构造简单逐步惩罚（需仿真器）。

等磁通 scheme2 接口已移除；请用 pack 中的 ``x_flux_diff``、``x_dpsi_dr/dz``、打击点坐标等自行设计 reward。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if str(EXAMPLES) not in sys.path:
    sys.path.append(str(EXAMPLES))

from environment.xpt_utils import (  # noqa: E402
    extract_target_strike_points,
    extract_target_xpoints,
    extract_xpt_observation_pack,
)
from xpt_example_common import DEFAULT_CONFIG_PATH, connection_hint, load_initial_and_current_observations  # noqa: E402


def simple_xpt_penalty(pack) -> float:
    """演示：位置/磁通/梯度/打击点槽位的粗惩罚（非官方评测）。"""
    v = pack["x_valid"]
    pos = np.mean(np.abs(pack["x_r"]) + np.abs(pack["x_z"])) * np.mean(v) if np.any(v) else 1.0
    flux = float(np.mean(pack["x_flux_diff"][v > 0.5])) if np.any(v) else 1.0
    grad = float(np.mean(np.abs(pack["x_dpsi_dr"][v > 0.5]) + np.abs(pack["x_dpsi_dz"][v > 0.5]))) if np.any(v) else 1.0
    strike = float(np.mean(pack["strike_valid"])) if pack["strike_n_use"] else 0.5
    return 0.3 * pos + 0.4 * flux + 0.2 * grad + 0.1 * (1.0 - strike)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    try:
        initial_obs, _ = load_initial_and_current_observations(args.config, seed=0, steps_after_reset=0)
    except (ConnectionError, OSError, TimeoutError) as exc:
        print(connection_hint(args.config))
        print(exc)
        return 1

    ref_rX, ref_zX, _ = extract_target_xpoints(initial_obs, slots=4)
    ref_rS, ref_zS, _, _ = extract_target_strike_points(initial_obs, strike_slots=8)

    import yaml
    from environment import HFMSimulator

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env = HFMSimulator(cfg)
    obs, _ = env.reset(seed=0)
    for t in range(args.steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        pack = extract_xpt_observation_pack(obs, ref_rX, ref_zX, ref_rS, ref_zS, strike_slots=8)
        pen = simple_xpt_penalty(pack)
        print(f"step {t}: env_reward={reward:.4f} demo_xpt_penalty={pen:.4f} nX={pack['nX']}")
        if term or trunc:
            break
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
