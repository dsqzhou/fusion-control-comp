#!/usr/bin/env python3
# coding=utf-8
"""
复赛评分脚本的回归测试：为每个场景断言期望总分与分项分。

运行：
  python test_evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from score import config, evaluate  # noqa: E402

import gen_mock_data as mock  # noqa: E402


# 期望分数：(total, {tid: task_score, ...})
# 推导见 gen_mock_data 注释，以及评分规则文档。
EXPECTED = {
    "perfect": {
        "total": 100.0,
        "tasks": {"F1": 40.0, "F2a": 30.0, "F2b": 30.0},
    },
    "ip_offset_3pct": {
        # Ip 误差 3% / 5% → score factor 0.4
        # F1:   12*0.4 + 28 = 32.8
        # F2a:  9*0.4 + 7 + 4 + 4 + 2 + 2 + 2 = 24.6
        # F2b:  同 F2a
        "total": 32.8 + 24.6 + 24.6,
        "tasks": {"F1": 32.8, "F2a": 24.6, "F2b": 24.6},
    },
    "ip_fuse": {
        # |ΔIp|=60kA > 50kA → mu=0 整步清零，所有任务总分 0
        "total": 0.0,
        "tasks": {"F1": 0.0, "F2a": 0.0, "F2b": 0.0},
    },
    "nx_violation": {
        # F2a: 前 250 步全分 (30), 后 250 步 XPT 专属 0 → 30 - (4+4+2+2+2)=16 per step
        # S_F2a = (250*30 + 250*(9+7))/500 = (7500+4000)/500 = 23.0
        "total": 40.0 + 23.0 + 30.0,
        "tasks": {"F1": 40.0, "F2a": 23.0, "F2b": 30.0},
    },
    "wrong_config": {
        # F1 lX 全为 1 → η=0.5 → F1 = 40*0.5 = 20
        "total": 20.0 + 30.0 + 30.0,
        "tasks": {"F1": 20.0, "F2a": 30.0, "F2b": 30.0},
    },
    "coil_violation": {
        # F2b 从 k=200 起线圈超限 → ρ=0；前 200 步满分
        # S_F2b = 30 * 200 / 500 = 12.0
        "total": 40.0 + 30.0 + 12.0,
        "tasks": {"F1": 40.0, "F2a": 30.0, "F2b": 12.0},
    },
    "early_term": {
        # F1 截断到 K=200 → S_F1 = 40*200/300 ≈ 26.6667
        "total": 40.0 * 200.0 / 300.0 + 30.0 + 30.0,
        "tasks": {"F1": 40.0 * 200.0 / 300.0, "F2a": 30.0, "F2b": 30.0},
    },
    "lcfs_shift_3cm": {
        # LCFS RMS = 3cm = 零分阈值 → LCFS 分项全 0
        # F1: 12 + 0 = 12
        # F2a/F2b: 9 + 0 + 4 + 4 + 2 + 2 + 2 = 23
        "total": 12.0 + 23.0 + 23.0,
        "tasks": {"F1": 12.0, "F2a": 23.0, "F2b": 23.0},
    },
}


# 容差：场景越复杂越放宽
TOL = 1e-3


def run_scenario(scenario: str):
    target = mock.make_target()
    infer = mock.make_infer(scenario, target)
    result = evaluate(infer, target)
    return result


def assert_close(actual, expected, name: str, tol: float = TOL):
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"FAIL {name}: actual={actual:.6f}, expected={expected:.6f}, "
            f"diff={actual - expected:+.6f}"
        )


def test_one(scenario: str) -> None:
    exp = EXPECTED[scenario]
    result = run_scenario(scenario)
    total = float(result["total_score"])
    tasks = {k: float(v) for k, v in result["task_scores"].items()}

    print(f"--- {scenario} ---")
    print(f"  total       = {total:.4f}   expected = {exp['total']:.4f}")
    for tid in config.TASK_IDS:
        ts = tasks.get(tid, 0.0)
        print(f"  {tid:<3}        = {ts:.4f}   expected = {exp['tasks'][tid]:.4f}")
        # 指标分项
        ms = result["task_metric_scores"].get(tid, {})
        for m, s in ms.items():
            print(f"    {m:<7}    = {float(s):.4f}")

    assert_close(total, exp["total"], f"{scenario} total")
    for tid, ts in tasks.items():
        assert_close(ts, exp["tasks"][tid], f"{scenario} {tid}")


def main() -> int:
    fails = []
    for sc in mock.SCENARIOS:
        try:
            test_one(sc)
        except AssertionError as e:
            fails.append((sc, str(e)))
            print(str(e))

    print("\n========== SUMMARY ==========")
    if fails:
        print(f"{len(fails)} / {len(mock.SCENARIOS)} scenarios failed:")
        for sc, msg in fails:
            print(f"  - {sc}: {msg}")
        return 1
    print(f"All {len(mock.SCENARIOS)} scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
