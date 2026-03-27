# Copyright @2025 ENN Energy(enn.cn)
# 逐步评分、子任务分、总分。

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from . import config


@dataclass
class TaskResult:
    task_id: str
    per_step_epsilons: Dict[str, np.ndarray] = field(default_factory=dict)
    n_actual_steps: int = 0
    timeout: bool = False


def score_single_step(epsilon: np.ndarray, epsilon_max: float) -> np.ndarray:
    """对逐步误差数组计算逐步得分（未乘权重），返回 [0,1] 数组。"""
    if epsilon_max <= 0:
        return np.zeros_like(epsilon)
    return np.maximum(0.0, 1.0 - epsilon / epsilon_max)


def per_metric_scores(
    per_step_epsilons: Dict[str, np.ndarray],
    weights: Dict[str, float],
    epsilon_max_dict: Optional[Dict[str, float]] = None,
    timeout: bool = False,
) -> Dict[str, float]:
    """与 task_score 同一逻辑，但返回各指标的分项得分。"""
    if timeout:
        return {name: 0.0 for name in weights}
    N = config.TOTAL_STEPS
    eps_max = epsilon_max_dict or config.EPSILON_MAX
    out: Dict[str, float] = {}
    for name, w in weights.items():
        eps_arr = per_step_epsilons.get(name)
        if eps_arr is None or len(eps_arr) == 0:
            out[name] = 0.0
            continue
        em = eps_max.get(name, config.get_epsilon_max(name))
        step_scores = w * score_single_step(eps_arr, em)
        out[name] = float(np.sum(step_scores)) / N
    return out


def task_score(
    per_step_epsilons: Dict[str, np.ndarray],
    weights: Dict[str, float],
    epsilon_max_dict: Optional[Dict[str, float]] = None,
    timeout: bool = False,
) -> float:
    """按新标准：逐步评分后按固定 N=500 求平均。timeout 时直接返回 0。"""
    return sum(per_metric_scores(per_step_epsilons, weights, epsilon_max_dict, timeout).values())


def total_score(per_task_results: List[TaskResult]) -> float:
    s = 0.0
    for r in per_task_results:
        w = config.get_task_metrics_and_weights(r.task_id)
        if not w:
            continue
        s += task_score(
            r.per_step_epsilons,
            w,
            config.EPSILON_MAX,
            r.timeout,
        )
    return s


def total_score_with_tie_break(
    per_task_results: List[TaskResult],
) -> tuple:
    score = total_score(per_task_results)
    by_cfg: Dict[str, float] = {"A": 0.0, "B": 0.0}
    for r in per_task_results:
        cfg = r.task_id[0]
        w = config.get_task_metrics_and_weights(r.task_id)
        if not w:
            continue
        by_cfg[cfg] = by_cfg.get(cfg, 0.0) + task_score(
            r.per_step_epsilons, w, config.EPSILON_MAX, r.timeout
        )
    tie_key = (by_cfg["B"], by_cfg["A"])
    return score, tie_key
