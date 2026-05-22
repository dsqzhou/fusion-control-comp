#!/usr/bin/env python3
# coding=utf-8
"""
生成复赛评分脚本的虚拟 target / infer_result JSON，用于单元测试。

提供多种场景：
  perfect          : 完全跟踪目标，所有约束合规，期望总分 ≈ 100
  ip_offset_3pct   : 电流恒偏 3%（位形完美）
  ip_fuse          : 电流恒偏 60 kA（每步整体清零）
  nx_violation     : F2a 后半段 nX=3（XPT 专属指标当步 0，电流 / LCFS 不受影响）
  wrong_config     : F1 lX 全为 1（η=0.5），其余完美
  coil_violation   : F2b 中段起 Icoil[0]>45 kA（该步及之后全部清零）
  early_term       : F1 在 200 步处截断（后 100 步 0 分）
  lcfs_shift_3cm   : LCFS 整体平移 3 cm（LCFS 分项=0，且证明不再对齐中心）

用法:
  python gen_mock_data.py --scenario perfect --out ./out/perfect
  python gen_mock_data.py --scenario all --out ./out
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# 同时支持 import: python -m tests.gen_mock_data 与直接 python gen_mock_data.py
HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from score import config  # noqa: E402


# -----------------------------------------------------------------------------
# 几何/数据构造
# -----------------------------------------------------------------------------

N_LCFS = 32  # 边界采样点数


def make_lcfs_limiter(R0: float = 1.0, a: float = 0.3, kappa: float = 1.6) -> np.ndarray:
    """生成一个限制器型椭圆 LCFS（用于 F1 目标位形）。"""
    th = np.linspace(0.0, 2.0 * np.pi, N_LCFS, endpoint=False)
    r = R0 + a * np.cos(th)
    z = a * kappa * np.sin(th)
    return np.stack([r, z], axis=1)


def make_lcfs_xpt(R0: float = 1.0, a: float = 0.35, kappa: float = 1.8) -> np.ndarray:
    """生成一个偏滤器/XPT 形状 LCFS（带轻微下方突出）。"""
    th = np.linspace(0.0, 2.0 * np.pi, N_LCFS, endpoint=False)
    r = R0 + a * np.cos(th)
    z = a * kappa * np.sin(th) - 0.05 * np.cos(2.0 * th)
    return np.stack([r, z], axis=1)


def make_xpt_main() -> np.ndarray:
    """主 X 点：上下各一，(r,z)。"""
    return np.array([[0.95, 0.55], [0.95, -0.55]], dtype=float)


def make_xpt_sec() -> np.ndarray:
    return np.array([[1.05, 0.75], [1.05, -0.75]], dtype=float)


def make_strikes() -> np.ndarray:
    """8 个打击点，上下各 4 个，顺时针编号约定下的虚拟坐标。"""
    pts = []
    for sign in [1.0, -1.0]:  # 上、下
        pts.extend([
            [0.40, sign * 0.95],
            [0.55, sign * 1.00],
            [0.70, sign * 0.97],
            [0.85, sign * 0.92],
        ])
    return np.array(pts, dtype=float)


# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------

def make_target() -> Dict[str, Any]:
    target = {
        "F1": {
            # Ip 由脚本按公式生成，target.Ip 这里写出来仅作为参考冗余
            "Ip": config.build_Ip_ref("F1").tolist(),
            "lcfs_points": make_lcfs_limiter().tolist(),
        },
        "F2a": {
            "Ip": config.build_Ip_ref("F2a").tolist(),
            "lcfs_points": make_lcfs_xpt().tolist(),
            "Xpt_main": make_xpt_main().tolist(),
            "Xpt_sec": make_xpt_sec().tolist(),
            "strike": make_strikes().tolist(),
        },
        # F2b 与 F2a 共享，故省略；评测脚本会 fallback
    }
    return target


# -----------------------------------------------------------------------------
# Infer trajectory 构造
# -----------------------------------------------------------------------------

def _perfect_trajectory(task_id: str, target: Dict[str, Any]) -> Dict[str, Any]:
    """构造完美跟踪目标的轨迹。lX 与任务目标拓扑一致，nX=4 仅 F2，
    Icoil 全合规，psi 偏差 0。"""
    N = config.get_total_steps(task_id)
    ref = target.get(task_id) or target.get(config.TARGET_FALLBACK.get(task_id))
    Ip_ref = config.build_Ip_ref(task_id)

    # lX
    if config.TASK_TARGET_TOPOLOGY[task_id] == "limiter":
        lX = [0] * N
    else:
        lX = [1] * N

    # nX：F2 任务期望 4，F1 不评估 XPT 专属指标，填 0 即可
    nX = [4 if task_id != "F1" else 0] * N

    # 线圈电流：远低于上限
    Icoil_step = [10_000.0] + [5_000.0] * 10 + [1_000.0]
    Icoil = [Icoil_step[:] for _ in range(N)]

    # psia/psib：选一组保证分母不为 0
    psia = [0.0] * N
    psib = [1.0] * N

    traj: Dict[str, Any] = {
        "Ip": Ip_ref.tolist(),
        "lX": lX,
        "nX": nX,
        "Icoil": Icoil,
        "psia": psia,
        "psib": psib,
    }

    # LCFS：每步都等于 target.lcfs_points
    lcfs = np.asarray(ref["lcfs_points"], dtype=float)
    traj["lcfs_per_step"] = [lcfs.tolist() for _ in range(N)]

    if task_id in ("F2a", "F2b"):
        xm = np.asarray(ref["Xpt_main"], dtype=float)
        xs = np.asarray(ref["Xpt_sec"], dtype=float)
        st = np.asarray(ref["strike"], dtype=float)
        traj["Xpt_main"] = [xm.tolist() for _ in range(N)]
        traj["Xpt_sec"] = [xs.tolist() for _ in range(N)]
        traj["strike"] = [st.tolist() for _ in range(N)]
        # psib==1, psia==0 → 分母 1；psiX==psib==1 → 偏差 0
        traj["psiX_main"] = [[1.0, 1.0] for _ in range(N)]
        traj["psiX_sec"] = [[1.0, 1.0] for _ in range(N)]
    else:
        # F1 不评估这些指标，但生成脚本一般也会提供，给空/零值都行
        traj["Xpt_main"] = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(N)]
        traj["Xpt_sec"] = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(N)]
        traj["strike"] = [[[0.0, 0.0]] * 8 for _ in range(N)]
        traj["psiX_main"] = [[1.0, 1.0] for _ in range(N)]
        traj["psiX_sec"] = [[1.0, 1.0] for _ in range(N)]

    return traj


def make_infer(scenario: str, target: Dict[str, Any]) -> Dict[str, Any]:
    """根据场景生成 infer_result（含 F1/F2a/F2b 三个子任务）。"""
    infer = {}
    for tid in config.TASK_IDS:
        traj = _perfect_trajectory(tid, target)
        infer[tid] = {"trajectory": traj, "timeout": False}

    if scenario == "perfect":
        pass

    elif scenario == "ip_offset_3pct":
        # 所有任务的 Ip 全程乘 1.03（误差 3%，< 50 kA & < 5%）
        for tid in config.TASK_IDS:
            Ip_ref = np.asarray(infer[tid]["trajectory"]["Ip"], dtype=float)
            infer[tid]["trajectory"]["Ip"] = (Ip_ref * 1.03).tolist()

    elif scenario == "ip_fuse":
        # 电流恒偏 60 kA，超出 50 kA 熔断阈值
        for tid in config.TASK_IDS:
            Ip_ref = np.asarray(infer[tid]["trajectory"]["Ip"], dtype=float)
            infer[tid]["trajectory"]["Ip"] = (Ip_ref + 60_000.0).tolist()

    elif scenario == "nx_violation":
        # 仅修改 F2a：后半段 nX=3（XPT 专属指标当步 0）
        N = config.get_total_steps("F2a")
        nX = [4] * (N // 2) + [3] * (N - N // 2)
        infer["F2a"]["trajectory"]["nX"] = nX

    elif scenario == "wrong_config":
        # F1: lX 全为 1（错误位形 → η=0.5），其余 OK
        infer["F1"]["trajectory"]["lX"] = [1] * config.get_total_steps("F1")

    elif scenario == "coil_violation":
        # F2b: 从第 200 步开始 CS 电流 50 kA，超过 45 kA 上限
        N = config.get_total_steps("F2b")
        K = 200
        Icoil = [list(infer["F2b"]["trajectory"]["Icoil"][i]) for i in range(N)]
        for k in range(K, N):
            Icoil[k][0] = 50_000.0
        infer["F2b"]["trajectory"]["Icoil"] = Icoil

    elif scenario == "early_term":
        # F1: 截断到 200 步
        traj = infer["F1"]["trajectory"]
        K = 200
        for key in ["Ip", "lX", "nX", "lcfs_per_step", "Xpt_main", "Xpt_sec",
                    "strike", "psiX_main", "psiX_sec", "Icoil", "psia", "psib"]:
            if key in traj:
                traj[key] = list(traj[key])[:K]

    elif scenario == "lcfs_shift_3cm":
        # LCFS 整体在 R 方向平移 0.03 m → 形状误差恰好 3 cm（=零分阈值）
        for tid in config.TASK_IDS:
            steps = infer[tid]["trajectory"]["lcfs_per_step"]
            shifted = []
            for c in steps:
                arr = np.asarray(c, dtype=float)
                arr[:, 0] = arr[:, 0] + 0.03
                shifted.append(arr.tolist())
            infer[tid]["trajectory"]["lcfs_per_step"] = shifted

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return infer


SCENARIOS = [
    "perfect",
    "ip_offset_3pct",
    "ip_fuse",
    "nx_violation",
    "wrong_config",
    "coil_violation",
    "early_term",
    "lcfs_shift_3cm",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=SCENARIOS + ["all"])
    parser.add_argument("--out", default=str(HERE / "mock"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = make_target()
    target_path = out_dir / "target.json"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False)

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        infer = make_infer(sc, target)
        p = out_dir / f"infer_{sc}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(infer, f, ensure_ascii=False)
        print(f"Wrote {p}")
    print(f"Target file: {target_path}")


if __name__ == "__main__":
    main()
