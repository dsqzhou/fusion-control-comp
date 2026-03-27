# Copyright @2025 ENN Energy(enn.cn)
# 5 个子任务定义、ref 选取与指标计算串联。

import numpy as np
from typing import Dict, List, Any, Optional, Union

from . import config
from . import metrics

TASK_TYPE_BALANCE = "balance"
TASK_TYPE_CURRENT = "current"
TASK_TYPE_POSITION = "position"

TASK_TYPE: Dict[str, str] = {
    "A1": TASK_TYPE_BALANCE,
    "A2": TASK_TYPE_CURRENT,
    "B1": TASK_TYPE_BALANCE,
    "B2": TASK_TYPE_CURRENT,
    "B3": TASK_TYPE_POSITION,
}


def get_ref_series_for_task(
    ref: Dict[str, Any],
    task_id: str,
    n_steps: int,
) -> Dict[str, Any]:
    task_type = TASK_TYPE.get(task_id, TASK_TYPE_BALANCE)
    out = {}

    def to_arr(x, n: int):
        if x is None:
            return None
        a = np.asarray(x, dtype=float)
        if a.ndim == 0:
            return np.full(n, float(a))
        if len(a) >= n:
            return a[:n]
        return np.resize(a, n)

    Ip = ref.get("Ip")
    R0 = ref.get("R0")
    Z0 = ref.get("Z0")
    if Ip is not None:
        Ip = to_arr(Ip, n_steps)
    if R0 is not None:
        R0 = to_arr(R0, n_steps)
    if Z0 is not None:
        Z0 = to_arr(Z0, n_steps)

    if task_type == TASK_TYPE_BALANCE:
        if Ip is not None:
            out["Ip"] = np.full(n_steps, float(Ip[0]))
        if R0 is not None:
            out["R0"] = np.full(n_steps, float(R0[0]))
        if Z0 is not None:
            out["Z0"] = np.full(n_steps, float(Z0[0]))
    elif task_type == TASK_TYPE_CURRENT:
        if Ip is not None:
            out["Ip"] = Ip
        if R0 is not None:
            out["R0"] = np.full(n_steps, float(R0[0]))
        if Z0 is not None:
            out["Z0"] = np.full(n_steps, float(Z0[0]))
    elif task_type == TASK_TYPE_POSITION:
        if Ip is not None:
            out["Ip"] = np.full(n_steps, float(Ip[0]))
        if R0 is not None:
            out["R0"] = R0
        if Z0 is not None:
            out["Z0"] = np.full(n_steps, float(Z0[0]))
    else:
        out["Ip"] = Ip
        out["R0"] = R0
        out["Z0"] = Z0

    if "lcfs_points" in ref:
        out["lcfs_points"] = ref["lcfs_points"]
    if "lcfs_per_step" in ref:
        out["lcfs_per_step"] = ref["lcfs_per_step"]
    return out


def compute_epsilons_for_task(
    trajectory: Dict[str, Any],
    ref: Dict[str, Any],
    task_id: str,
) -> Dict[str, np.ndarray]:
    """返回各指标的逐步误差数组 Dict[metric_name, np.ndarray]。"""
    Ip_a = np.asarray(trajectory.get("Ip", []), dtype=float).flatten()
    R0_a = np.asarray(trajectory.get("R0", []), dtype=float).flatten()
    Z0_a = np.asarray(trajectory.get("Z0", []), dtype=float).flatten()
    n_steps = min(len(Ip_a), len(R0_a), len(Z0_a))
    if n_steps == 0:
        return {}

    ref_sel = get_ref_series_for_task(ref, task_id, n_steps)
    Ip_r = ref_sel.get("Ip")
    R0_r = ref_sel.get("R0")
    Z0_r = ref_sel.get("Z0")
    if Ip_r is None:
        Ip_r = np.full(n_steps, float(Ip_a[0]))
    if R0_r is None:
        R0_r = np.full(n_steps, float(R0_a[0]))
    if Z0_r is None:
        Z0_r = np.full(n_steps, float(Z0_a[0]))

    eps: Dict[str, np.ndarray] = {}
    weights = config.get_task_metrics_and_weights(task_id)
    if not weights:
        return eps

    if config.METRIC_IP in weights:
        eps[config.METRIC_IP] = metrics.epsilon_Ip(Ip_a[:n_steps], Ip_r)
    if config.METRIC_POS in weights:
        eps[config.METRIC_POS] = metrics.epsilon_pos(
            R0_a[:n_steps], Z0_a[:n_steps],
            R0_r, Z0_r,
        )
    if config.METRIC_LCFS in weights:
        lcfs_pts = ref.get("lcfs_points")
        if lcfs_pts is None:
            lcfs_pts = ref_sel.get("lcfs_points")
        lcfs_actual = trajectory.get("lcfs_per_step", [])
        if lcfs_pts is not None and lcfs_actual:
            lcfs_list = []
            for k in range(n_steps):
                if k < len(lcfs_actual):
                    lcfs_list.append(np.asarray(lcfs_actual[k], dtype=float))
                elif lcfs_actual:
                    lcfs_list.append(np.asarray(lcfs_actual[-1], dtype=float))
            if lcfs_list:
                R0_init = float(R0_r[0])
                Z0_init = float(Z0_r[0])
                eps[config.METRIC_LCFS] = metrics.epsilon_LCFS(
                    np.asarray(lcfs_pts), lcfs_list,
                    R0_actual=R0_a[:n_steps],
                    Z0_actual=Z0_a[:n_steps],
                    R0_init=R0_init,
                    Z0_init=Z0_init,
                )
    return eps
