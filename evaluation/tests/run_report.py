#!/usr/bin/env python3
# coding=utf-8
"""
执行所有测试场景，落档 JSON 数据 + 生成诊断图 + 输出 per-scenario 得分明细。

输出目录：
  data/    每个场景的 target.json + infer_<scenario>.json
  figures/ 每个场景的诊断 PNG
  REPORT_data.json   汇总各场景实际与期望得分，供文档使用
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 字体：图里全部使用英文标签，避免依赖中文字体；中文描述写在 markdown 文档中。
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from score import config, evaluate  # noqa: E402

import gen_mock_data as mock  # noqa: E402
import test_evaluate as te  # noqa: E402


DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
REPORT_JSON = HERE / "REPORT_data.json"


SCENARIO_DESC = {
    "perfect":         "完美跟踪：电流按公式、LCFS/X 点/打击点等与目标一致、所有约束合规。预期满分 100。",
    "ip_offset_3pct":  "电流恒偏 +3%（≈ 3–15 kA，未触发熔断），位形完美。预期 Ip 项按线性插值得 40% 分，其它满分。",
    "ip_fuse":         "电流恒偏 +60 kA，超出 50 kA 熔断阈值。每步 μ=0，整步得分归零，总分 0。",
    "nx_violation":    "F2a 后半段 nX=3（拓扑约束破坏），其余完美。XPT 专属指标在违规步σ清零，电流/LCFS 不受影响。",
    "wrong_config":    "F1 全程 lX=1（应为限制器却报偏滤器），其余完美。每步 η=0.5，F1 得分减半。",
    "coil_violation":  "F2b 从第 200 步开始 CS=50 kA 超 45 kA。该步及之后 ρ=0 全部清零，前 200 步满分。",
    "early_term":      "F1 在第 200 步提前终止（数组长度 200）。后 100 步 σ=0，N=300 求平均。",
    "lcfs_shift_3cm":  "LCFS 整体在 R 方向平移 +3 cm（恰好等于零分阈值）。LCFS 分项=0；用于演示复赛取消中心对齐。",
}


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _clip_eps(arr: np.ndarray) -> np.ndarray:
    """图表用：把 inf / nan 处理成可见值。"""
    out = np.asarray(arr, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return out


# 任务画图时关心的指标排列与颜色
TASK_METRIC_COLORS = {
    "Ip":     "#1f77b4",
    "LCFS":   "#ff7f0e",
    "X":      "#2ca02c",
    "strike": "#d62728",
    "psiX":   "#9467bd",
    "X2":     "#8c564b",
    "psiX2":  "#e377c2",
}


def _plot_scenario(scenario: str, target: Dict[str, Any], infer: Dict[str, Any], result: Dict[str, Any]) -> Path:
    """每个场景画一张 3 行 × 3 列大图（每行一个任务，每列一个视图）。"""
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(f"Scenario: {scenario}    total={result['total_score']:.4f}", fontsize=14, fontweight="bold")

    per_task = {r.task_id: r for r in result["per_task_results"]}

    for row, tid in enumerate(config.TASK_IDS):
        N = config.get_total_steps(tid)
        tr = per_task[tid]
        weights = config.get_task_metrics_and_weights(tid)
        eps_dict = tr.per_step_epsilons

        # Col 0: Ip(t) 实际 vs 目标 + 熔断阈值带
        ax = axes[row, 0]
        Ip_actual = np.asarray(infer[tid]["trajectory"].get("Ip", []), dtype=float)
        Ip_ref = tr.Ip_ref
        t = np.arange(N) + 1  # ms 等同于步索引
        ax.plot(t, Ip_ref / 1e3, color="black", lw=2, label="Ip_ref")
        K_eff = tr.K_eff
        if Ip_actual.size:
            ax.plot(t[:len(Ip_actual)], Ip_actual / 1e3, color="#1f77b4", lw=1.2, label="Ip_actual")
        # 熔断阈值带
        ax.fill_between(t, (Ip_ref - config.CURRENT_FUSE_A) / 1e3, (Ip_ref + config.CURRENT_FUSE_A) / 1e3,
                        color="orange", alpha=0.12, label="±50 kA fuse band")
        ax.set_title(f"{tid}: Ip(t)   (W=12 / 9)")
        ax.set_xlabel("step (ms)")
        ax.set_ylabel("Ip [kA]")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Col 1: 归一化各指标 ε(t) / ε_max   (1.0 = zero-score threshold)
        # 各指标 ε_max 量级差异巨大（5% / 3cm / 5cm / 10%），同轴显示必须先归一化。
        ax = axes[row, 1]
        any_drawn = False
        norm_max = {}
        for name in weights.keys():
            eps_arr = eps_dict.get(name)
            if eps_arr is None:
                continue
            eps_plot = _clip_eps(eps_arr[:K_eff]) if K_eff > 0 else _clip_eps(eps_arr)
            em = config.EPSILON_MAX[name]
            norm = eps_plot / em
            color = TASK_METRIC_COLORS.get(name, None)
            ax.plot(np.arange(len(norm)) + 1, norm,
                    label=f"{name} (W={weights[name]}, eps_max={em:g})",
                    color=color, lw=1.3)
            any_drawn = True
            finite = norm[np.isfinite(norm)]
            if finite.size:
                norm_max[name] = float(np.nanmax(finite))
        # 零分阈值（统一在 y=1）
        ax.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.55, label="zero-score (=1)")
        ax.set_ylim(-0.05, 1.5)
        ax.set_title(f"{tid}: epsilon(t) / eps_max   (1.0 = zero-score)")
        ax.set_xlabel("step (ms)")
        ax.set_ylabel("normalized eps")
        if any_drawn:
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        # 如果有曲线超出视图（>1.5），在左上角标注峰值，避免被裁切误读
        over = {k: v for k, v in norm_max.items() if v > 1.5}
        if over:
            txt = "out-of-view peaks:\n" + "\n".join(f"  {k}: {v:.2f}" for k, v in over.items())
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=7,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"))

        # Col 2: 系数 + 单步合成得分
        ax = axes[row, 2]
        steps_full = np.arange(N) + 1
        eta = tr.eta if tr.eta is not None else np.ones(N)
        mu = tr.mu if tr.mu is not None else np.ones(N)
        rho = tr.rho if tr.rho is not None else np.ones(N)
        topo = tr.topo_mask if tr.topo_mask is not None else np.ones(N)
        ax.plot(steps_full, eta, label="eta", color="#1f77b4", lw=1.0)
        ax.plot(steps_full, mu, label="mu", color="#d62728", lw=1.0)
        ax.plot(steps_full, rho, label="rho", color="#2ca02c", lw=1.0)
        if tid != "F1":
            ax.plot(steps_full, topo, label="topo(nX=4)", color="#9467bd", lw=1.0, alpha=0.7)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("step (ms)")
        ax.set_ylabel("coefficient")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 双 Y 轴叠加单步合成得分
        ax2 = ax.twinx()
        step_total = tr.per_step_score if tr.per_step_score is not None else np.zeros(N)
        ax2.plot(steps_full, step_total, color="black", lw=0.8, alpha=0.7, label="step_score")
        ax2.set_ylabel("step score (sum of σ_i × η μ ρ)")
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{tid}: coefficients & step score   S={tr.task_score:.4f}")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f"{scenario}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _plot_lcfs_shape(scenario: str, target: Dict[str, Any], infer: Dict[str, Any]) -> Path:
    """LCFS 几何对比图：每个任务取 K_eff/2 步的 LCFS 与目标叠加。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(f"Scenario: {scenario}   LCFS shape comparison (mid step)")

    for col, tid in enumerate(config.TASK_IDS):
        ax = axes[col]
        ref = target.get(tid) or target.get(config.TARGET_FALLBACK.get(tid))
        lcfs_target = np.asarray(ref["lcfs_points"], dtype=float) if ref else None
        steps = infer[tid]["trajectory"].get("lcfs_per_step", [])
        if not steps:
            ax.set_title(f"{tid}: (no LCFS)")
            continue
        idx = min(len(steps) - 1, len(steps) // 2)
        lcfs_actual = np.asarray(steps[idx], dtype=float)

        if lcfs_target is not None:
            ring = np.vstack([lcfs_target, lcfs_target[:1]])
            ax.plot(ring[:, 0], ring[:, 1], color="black", lw=2.2, label="target")
        ring = np.vstack([lcfs_actual, lcfs_actual[:1]])
        ax.plot(ring[:, 0], ring[:, 1], color="#1f77b4", lw=1.2, label=f"actual (k={idx + 1})")
        ax.set_aspect("equal")
        ax.set_title(f"{tid}")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / f"{scenario}_lcfs_shape.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    tasks = {}
    for r in result["per_task_results"]:
        tasks[r.task_id] = {
            "task_score": float(r.task_score),
            "K_eff": int(r.K_eff),
            "gamma": float(r.gamma),
            "metric_scores": {k: float(v) for k, v in r.metric_scores.items()},
            "metric_max": dict(config.get_task_metrics_and_weights(r.task_id)),
        }
    return {
        "total_score": float(result["total_score"]),
        "task_scores": {k: float(v) for k, v in result["task_scores"].items()},
        "tasks": tasks,
    }


def main() -> int:
    _ensure_dirs()
    target = mock.make_target()
    _save_json(DATA_DIR / "target.json", target)
    print(f"[data] {DATA_DIR / 'target.json'}")

    summary = {}
    fails = []

    for scenario in mock.SCENARIOS:
        infer = mock.make_infer(scenario, target)
        infer_path = DATA_DIR / f"infer_{scenario}.json"
        _save_json(infer_path, infer)
        print(f"[data] {infer_path}")

        result = evaluate(infer, target)

        fig_path = _plot_scenario(scenario, target, infer, result)
        print(f"[fig]  {fig_path}")
        if scenario in ("perfect", "lcfs_shift_3cm"):
            fig2 = _plot_lcfs_shape(scenario, target, infer)
            print(f"[fig]  {fig2}")

        actual = _summarize(result)
        expected = te.EXPECTED[scenario]

        # 断言
        passed = abs(actual["total_score"] - expected["total"]) <= te.TOL
        for tid, ts in actual["task_scores"].items():
            if abs(ts - expected["tasks"][tid]) > te.TOL:
                passed = False

        summary[scenario] = {
            "description": SCENARIO_DESC[scenario],
            "expected_total": expected["total"],
            "expected_tasks": expected["tasks"],
            "actual": actual,
            "passed": bool(passed),
        }
        if not passed:
            fails.append(scenario)

        print(f"[run]  {scenario}: total={actual['total_score']:.4f}  expected={expected['total']:.4f}"
              f"  {'OK' if passed else 'FAIL'}")

    _save_json(REPORT_JSON, summary)
    print(f"\n[summary] -> {REPORT_JSON}")
    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("All scenarios passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
