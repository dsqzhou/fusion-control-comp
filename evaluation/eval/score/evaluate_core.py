# Copyright @2025 ENN Energy(enn.cn)
# 评估入口：从 trajectory + ref 计算各子任务逐步误差与总分。

import numpy as np
from typing import Dict, List, Any, Optional

from . import config
from . import tasks
from . import scoring


def evaluate(
    trajectory: Dict[str, Any],
    ref: Dict[str, Any],
    *,
    task_ids: Optional[List[str]] = None,
    timeouts: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    trajectory: Ip, R0, Z0 数组；可选 lcfs_per_step。
    ref: Ip, R0, Z0, lcfs_points（scoring_ref 格式）。
    """
    task_ids = task_ids or config.TASK_IDS
    timeouts = timeouts or {}

    per_task_results: List[scoring.TaskResult] = []
    task_scores: Dict[str, float] = {}
    task_epsilons: Dict[str, Dict[str, float]] = {}

    for tid in task_ids:
        if tid not in config.TASK_WEIGHTS:
            continue
        eps = tasks.compute_epsilons_for_task(trajectory, ref, tid)
        n_actual = max((len(v) for v in eps.values()), default=0)
        to = timeouts.get(tid, False)
        res = scoring.TaskResult(
            task_id=tid,
            per_step_epsilons=eps,
            n_actual_steps=n_actual,
            timeout=to,
        )
        per_task_results.append(res)
        w = config.get_task_metrics_and_weights(tid)
        task_scores[tid] = scoring.task_score(eps, w, config.EPSILON_MAX, to)
        task_epsilons[tid] = {
            k: float(np.mean(v)) for k, v in eps.items() if len(v) > 0
        }

    total = scoring.total_score(per_task_results)

    return {
        "total_score": total,
        "task_scores": task_scores,
        "task_epsilons": task_epsilons,
        "per_task_results": per_task_results,
    }
