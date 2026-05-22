"""
13906_500 reset 初态：绘制极向磁通、LCFS（粗线）、主/次级 X 点、打击点。

需 HFM 容器；若默认 port 5558 不通，可指定宿主机映射端口（如 2223）：

  python examples/example_xpt_plot_13906.py --port 2223
  python examples/example_xpt_plot_13906.py --port 2223 --out docs/13906_xpt_flux.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment import HFMSimulator
from environment.xpt_utils import (
    extract_target_strike_points,
    extract_target_xpoints,
    plot_xpt_flux_diagnostic,
)

DEFAULT_CONFIG = ROOT / "configs" / "env_default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", type=int, default=None, help="覆盖 predictor.port（如 Docker 映射 2223）")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "13906_xpt_flux.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.port is not None:
        cfg.setdefault("predictor", {})["port"] = args.port
    if args.host is not None:
        cfg.setdefault("predictor", {})["host"] = args.host

    shot = cfg.get("predictor", {}).get("shot_id", "?")
    port = cfg.get("predictor", {}).get("port")
    host = cfg.get("predictor", {}).get("host", "127.0.0.1")
    print(f"连接 HFM {host}:{port} shot_id={shot}")

    env = HFMSimulator(cfg)
    try:
        initial_obs, _ = env.reset(seed=0)
    except Exception as exc:
        print(f"连接失败: {exc}")
        print("提示: docker ps 查看 0.0.0.0:PORT->5558，用 --port PORT")
        return 1
    finally:
        env.close()

    ref_rX, ref_zX, _ = extract_target_xpoints(initial_obs, slots=4)
    ref_rS, ref_zS, s_valid, n_s = extract_target_strike_points(initial_obs, strike_slots=8)

    payload = {
        "shot_id": shot,
        "reference_rX": ref_rX.tolist(),
        "reference_zX": ref_zX.tolist(),
        "reference_strike_r": ref_rS.tolist(),
        "reference_strike_z": ref_zS.tolist(),
        "reference_strike_valid": s_valid.astype(int).tolist(),
        "n_strike_ref": int(n_s),
        "nX_initial": int(np.asarray(initial_obs.get("nX", 0)).ravel()[0]),
    }
    ref_json = ROOT / "configs" / "xpt_reference_targets.json"
    ref_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("reference_rX =", np.round(ref_rX, 5).tolist())
    print("reference_zX =", np.round(ref_zX, 5).tolist())
    print("reference_strike (valid slots) =", int(n_s))
    print("写入", ref_json)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_xpt_flux_diagnostic(
        initial_obs,
        reference_rX=ref_rX,
        reference_zX=ref_zX,
        reference_rS=ref_rS,
        reference_zS=ref_zS,
        save_path=str(args.out),
        title="XPT 13906_500",
        show=not args.no_show,
    )
    print("已保存", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
