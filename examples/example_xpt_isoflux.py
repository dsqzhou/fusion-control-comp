"""
示例：方案二等磁通残差（8 个 LCFS 点 + 4 个目标 X 点位）与 X 点极向场模。

**仅使用 HFMSimulator 与仿真器交互后的真实 observation**：
- `reset` 观测用于提取 8 个等磁通目标点位和 4 个目标 X 点位
- 之后对当前 step 的观测做等磁通与极向场计算

用法（需先启动 HFM）：
  cd fusion-control-comp
  python tools/start_simulator.py -n 1 -y
  python examples/example_xpt_isoflux.py
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
    br_bz_at_point,
    extract_target_isoflux_points,
    extract_target_xpoints,
    get_psi_grid,
    infer_fx_reshape_order,
    isoflux_residuals_scheme2,
    xpoint_poloidal_field_magnitude,
)
from xpt_example_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    connection_hint,
    load_initial_and_current_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="XPT 方案二：等磁通残差与极向场")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=1, help="reset 后随机步数")
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

    print("OK: 已从 HFM 获取观测。")
    target_rB, target_zB, target_idx = extract_target_isoflux_points(initial_obs, lcfs_step=4)
    target_rX, target_zX, target_valid = extract_target_xpoints(initial_obs, slots=4)
    print("target LCFS isoflux points from reset:")
    print("  indices:", target_idx)
    print("  target_rB:", target_rB)
    print("  target_zB:", target_zB)
    print("target X points from reset:")
    print("  valid:", target_valid)
    print("  target_rX:", target_rX)
    print("  target_zX:", target_zX)

    order = infer_fx_reshape_order(obs)
    print("infer_fx_reshape_order:", order)

    rx, zx, psi, ord_used = get_psi_grid(obs, order=None)
    print("get_psi_grid: psi shape", psi.shape, "order", ord_used)

    r0, z0 = float(target_rX[0]), float(target_zX[0])
    br, bz = br_bz_at_point(obs, r0, z0, order=None)
    print(f"br_bz_at_point at target X0 (R,Z)=({r0:.4f},{z0:.4f}) -> Br={br:.6e}, Bz={bz:.6e}")

    res, meta = isoflux_residuals_scheme2(
        obs,
        target_rB=target_rB,
        target_zB=target_zB,
        target_rX=target_rX,
        target_zX=target_zX,
        lcfs_step=4,
    )
    print("isoflux_residuals_scheme2 shape:", res.shape, "fb=", meta["fb"])
    print("  lcfs residuals (8):", res[: target_rB.size])
    print("  target X residuals (4):", res[target_rB.size :])
    print("  abs mean residual:", float(np.nanmean(np.abs(res))))

    brs, bzs, mag = xpoint_poloidal_field_magnitude(
        obs,
        target_rX=target_rX,
        target_zX=target_zX,
        slots=4,
    )
    print("xpoint Br:", brs)
    print("xpoint Bz:", bzs)
    print("xpoint |B_pol|:", mag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
