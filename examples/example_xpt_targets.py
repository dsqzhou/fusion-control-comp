"""
从 ``configs/env_default.yaml`` 的 ``shot_id``（默认 13906_500）reset 一次，
打印 XPT 任务二的 **reference X 点** 与 **reference 打击点**（供选手对照目标位形）。

需先启动 HFM 仿真器，端口与 env_default.yaml 一致。

  python examples/example_xpt_targets.py
  python examples/example_xpt_targets.py --config configs/env_default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if str(EXAMPLES) not in sys.path:
    sys.path.append(str(EXAMPLES))

from environment.xpt_utils import extract_target_strike_points, extract_target_xpoints  # noqa: E402
from xpt_example_common import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    connection_hint,
    load_initial_and_current_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="打印 XPT reference X 点与打击点目标")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "configs" / "xpt_reference_targets.json")
    args = parser.parse_args()

    try:
        initial_obs, obs = load_initial_and_current_observations(args.config, seed=0, steps_after_reset=0)
    except (ConnectionError, OSError, TimeoutError) as exc:
        print(connection_hint(args.config))
        print("连接失败:", exc)
        return 1

    ref_rX, ref_zX, x_valid = extract_target_xpoints(initial_obs, slots=4)
    ref_rS, ref_zS, s_valid, n_strike = extract_target_strike_points(initial_obs, strike_slots=8)

    payload = {
        "note": "reference 来自 reset 初态；X 槽按 z 降序；打击点 CCW，z_exclude=0.5",
        "reference_rX": ref_rX.tolist(),
        "reference_zX": ref_zX.tolist(),
        "reference_x_valid": x_valid.tolist(),
        "reference_strike_r": ref_rS.tolist(),
        "reference_strike_z": ref_zS.tolist(),
        "reference_strike_valid": s_valid.tolist(),
        "reference_n_strike_use": int(n_strike),
        "initial_nX": int(np.asarray(initial_obs.get("nX", 0)).ravel()[0]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== XPT reference（reset 初态）===")
    print("reference_rX =", np.round(ref_rX, 5).tolist())
    print("reference_zX =", np.round(ref_zX, 5).tolist())
    print("reference_strike_r =", np.round(ref_rS, 5).tolist())
    print("reference_strike_z =", np.round(ref_zS, 5).tolist())
    print("n_strike_use =", n_strike, "valid =", s_valid.astype(int).tolist())
    print("已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
