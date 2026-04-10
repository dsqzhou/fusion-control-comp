"""
示例：在 HFMSimulator 上使用基于 XPT 等磁通残差的 reward_fn（本地训练）。

**全程与真实环境交互**（reset + 多步随机动作）；观测与 reward 均来自仿真器。

用法：
  cd fusion-control-comp
  python tools/start_simulator.py -n 1 -y
  python examples/example_xpt_reward.py
  python examples/example_xpt_reward.py --steps 5 --config configs/env_default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if str(EXAMPLES) not in sys.path:
    sys.path.append(str(EXAMPLES))

from environment import HFMSimulator  # noqa: E402
from environment.xpt_utils import extract_target_xpoints, isoflux_residuals_scheme2  # noqa: E402
from xpt_example_common import DEFAULT_CONFIG_PATH, connection_hint  # noqa: E402


class XptIsofluxReward:
    """把 reset 时刻提取的目标 X 点位固定下来，供后续 step 计算 reward。"""

    def __init__(self) -> None:
        self.target_rX: np.ndarray | None = None
        self.target_zX: np.ndarray | None = None

    def set_targets_from_initial_obs(self, initial_obs: dict, slots: int = 4) -> None:
        self.target_rX, self.target_zX, _ = extract_target_xpoints(initial_obs, slots=slots)

    def __call__(
        self,
        observation: dict,
        action: np.ndarray,
        terminated: bool,
        truncated: bool,
        info: dict | None = None,
    ) -> float:
        if terminated:
            return -10.0
        if self.target_rX is None or self.target_zX is None:
            raise RuntimeError("XptIsofluxReward targets are not initialized from reset observation")
        res, _ = isoflux_residuals_scheme2(
            observation,
            target_rX=self.target_rX,
            target_zX=self.target_zX,
        )
        mask = np.isfinite(res)
        if not np.any(mask):
            return -1.0
        return float(-np.mean(np.abs(res[mask])))


def main() -> int:
    parser = argparse.ArgumentParser(description="XPT 等磁通 reward 与真实环境交互")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=3, help="reset 后执行的步数")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reward_fn = XptIsofluxReward()
    env = HFMSimulator(config, reward_fn=reward_fn)
    try:
        obs, _ = env.reset(seed=args.seed)
        reward_fn.set_targets_from_initial_obs(obs, slots=4)
        print("OK: reset 成功，开始 step。")
        print("target X points from reset:")
        print("  target_rX:", reward_fn.target_rX)
        print("  target_zX:", reward_fn.target_zX)
        for i in range(args.steps):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            print(f"  step {i + 1}: reward = {r:.6f}, terminated = {term}, truncated = {trunc}")
            if term or trunc:
                break
    except (ConnectionError, OSError, TimeoutError) as exc:
        print(connection_hint(args.config))
        print(f"[错误] {type(exc).__name__}: {exc}")
        return 1
    finally:
        env.close()

    print("OK: 与环境交互完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
