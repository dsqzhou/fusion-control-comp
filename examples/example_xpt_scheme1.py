"""
示例：方案一 XPT 特征（仅 4×X 点 valid/dr/dz/dFX）。

**仅使用 HFMSimulator 与仿真器交互后的真实 observation**：
- `reset` 得到初始平衡，用它固定 4 个目标 X 点位
- 再随机 `step` 若干步，对当前观测计算方案一 XPT 特征

用法（需先启动 HFM，且端口与 configs/env_default.yaml 一致）：
  cd fusion-control-comp
  python tools/start_simulator.py -n 1 -y   # 另开终端
  python examples/example_xpt_scheme1.py
  python examples/example_xpt_scheme1.py --config configs/env_default.yaml
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
    extract_sorted_xpoints,
    extract_target_xpoints,
    scheme1_xpoint_features,
)
from xpt_example_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    connection_hint,
    load_initial_and_current_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="XPT 方案一：从真实环境观测构造 16 维 X 点特征")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="环境 YAML（含 predictor.host/port/shot_id）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="reset 之后额外执行的随机步数（默认 1）",
    )
    args = parser.parse_args()

    try:
        initial_obs, obs = load_initial_and_current_observations(
            args.config,
            seed=0,
            steps_after_reset=args.steps,
        )
    except (ConnectionError, OSError, TimeoutError) as exc:
        print(connection_hint(args.config))
        print(f"[错误] {type(exc).__name__}: {exc}")
        return 1

    target_rX, target_zX, target_valid = extract_target_xpoints(initial_obs, slots=4)
    r_x, z_x, _, valid, nx, fb = extract_sorted_xpoints(obs, slots=4)
    print("OK: 已从 HFM 获取观测。")
    print("target X points from reset:")
    print("  valid:", target_valid)
    print("  target_rX:", target_rX)
    print("  target_zX:", target_zX)
    print("extract_sorted_xpoints: nx =", nx, "fb =", fb)
    print("  valid:", valid)
    print("  rX (sorted slots):", r_x)
    print("  zX (sorted slots):", z_x)

    vec = scheme1_xpoint_features(
        obs,
        target_rX=target_rX.tolist(),
        target_zX=target_zX.tolist(),
        slots=4,
    )
    print("scheme1_xpoint_features dim =", vec.size, "(expected 16)")
    print("  4x4 xpt features:", vec.reshape(4, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
