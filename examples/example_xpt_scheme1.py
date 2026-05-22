"""
示例：复赛 XPT 任务 — 使用 ``extract_xpt_observation_pack`` 从真实 HFM 观测提取特征。

  python tools/start_simulator.py -n 1 -y
  python examples/example_xpt_scheme1.py
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
    pack_to_vector,
)
from xpt_example_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    connection_hint,
    load_initial_and_current_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="XPT pack 特征示例")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    try:
        initial_obs, obs = load_initial_and_current_observations(
            args.config, seed=0, steps_after_reset=args.steps
        )
    except (ConnectionError, OSError, TimeoutError) as exc:
        print(connection_hint(args.config))
        print(f"[错误] {type(exc).__name__}: {exc}")
        return 1

    ref_rX, ref_zX, _ = extract_target_xpoints(initial_obs, slots=4)
    ref_rS, ref_zS, _, _ = extract_target_strike_points(initial_obs, strike_slots=8)
    pack = extract_xpt_observation_pack(obs, ref_rX, ref_zX, ref_rS, ref_zS, strike_slots=8)
    vec = pack_to_vector(pack)

    print("OK: 已从 HFM 获取观测。")
    print("reference X (reset):", np.round(ref_rX, 5), np.round(ref_zX, 5))
    print("pack nX =", pack["nX"], "fb =", pack["fb"], "fx_order =", pack["fx_order"])
    print("x slots valid =", pack["x_valid"].astype(int))
    print("x_r, x_z =", np.round(pack["x_r"], 5), np.round(pack["x_z"], 5))
    print("x_flux_diff =", np.round(pack["x_flux_diff"], 6))
    print("x_dpsi_dr, x_dpsi_dz =", np.round(pack["x_dpsi_dr"], 6), np.round(pack["x_dpsi_dz"], 6))
    print("strike_n_use / n_actual =", pack["strike_n_use"], pack["strike_n_actual"])
    print("pack_to_vector dim =", vec.size, "(expected 56 with strike_slots=8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
