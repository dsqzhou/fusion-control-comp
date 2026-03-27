# Copyright @2025 ENN Energy(enn.cn)
# 基础计算指标。

import numpy as np
from typing import List, Union, Sequence

M_TO_CM = 100.0


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ap = p - a
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq <= 1e-20:
        return float(np.linalg.norm(ap))
    t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _point_to_polyline_dist(p: np.ndarray, poly: np.ndarray) -> float:
    if poly is None or len(poly) < 2:
        return float(np.linalg.norm(p)) if poly is not None else np.inf
    p = np.asarray(p, dtype=float)
    poly = np.asarray(poly, dtype=float)
    d_min = np.inf
    for i in range(len(poly) - 1):
        d = _point_to_segment_dist(p, poly[i], poly[i + 1])
        if d < d_min:
            d_min = d
    return float(d_min)


def epsilon_Ip(
    Ip_actual: Union[np.ndarray, Sequence[float]],
    Ip_ref: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """逐步电流误差: |Ip(t_k) - Ip_ref(t_k)| / |Ip_ref(t_k)|"""
    Ip_actual = np.asarray(Ip_actual, dtype=float).flatten()
    Ip_ref = np.asarray(Ip_ref, dtype=float).flatten()
    n = min(len(Ip_actual), len(Ip_ref))
    if n == 0:
        return np.array([], dtype=float)
    ref_abs = np.abs(Ip_ref[:n])
    safe_ref = np.where(ref_abs > 1e-12, ref_abs, 1.0)
    return np.abs(Ip_actual[:n] - Ip_ref[:n]) / safe_ref


def epsilon_pos(
    R0_actual: Union[np.ndarray, Sequence[float]],
    Z0_actual: Union[np.ndarray, Sequence[float]],
    R0_ref: Union[np.ndarray, Sequence[float]],
    Z0_ref: Union[np.ndarray, Sequence[float]],
) -> np.ndarray:
    """逐步位置误差 (cm)"""
    R0_actual = np.asarray(R0_actual, dtype=float).flatten()
    Z0_actual = np.asarray(Z0_actual, dtype=float).flatten()
    R0_ref = np.asarray(R0_ref, dtype=float).flatten()
    Z0_ref = np.asarray(Z0_ref, dtype=float).flatten()
    n = min(len(R0_actual), len(Z0_actual), len(R0_ref), len(Z0_ref))
    if n == 0:
        return np.array([], dtype=float)
    dr = R0_actual[:n] - R0_ref[:n]
    dz = Z0_actual[:n] - Z0_ref[:n]
    return np.sqrt(dr * dr + dz * dz) * M_TO_CM


def epsilon_LCFS(
    target_lcfs_points: np.ndarray,
    actual_lcfs_per_step: List[np.ndarray],
    R0_actual: Union[np.ndarray, Sequence[float], None] = None,
    Z0_actual: Union[np.ndarray, Sequence[float], None] = None,
    R0_init: float = None,
    Z0_init: float = None,
) -> np.ndarray:
    """逐步 LCFS 形状误差 (cm)。

    将实际 LCFS 平移对齐到初始中心后，按对应极向角采样点逐点计算
    与目标 LCFS 的欧氏距离，返回每步的 RMS 值。
    """
    target = np.asarray(target_lcfs_points, dtype=float)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("target_lcfs_points must be (N_b, 2)")
    N_b = target.shape[0]
    if N_b == 0:
        return np.array([], dtype=float)

    has_shift = (R0_actual is not None and Z0_actual is not None
                 and R0_init is not None and Z0_init is not None)
    if has_shift:
        R0_a = np.asarray(R0_actual, dtype=float).flatten()
        Z0_a = np.asarray(Z0_actual, dtype=float).flatten()

    step_rms = []
    for k, contour in enumerate(actual_lcfs_per_step):
        contour = np.asarray(contour, dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 2:
            step_rms.append(0.0)
            continue

        if has_shift and k < len(R0_a):
            shift = np.array([R0_init - R0_a[k], Z0_init - Z0_a[k]])
            contour = contour + shift

        n_pts = min(N_b, len(contour))
        d_sq = (target[:n_pts, 0] - contour[:n_pts, 0]) ** 2 + \
               (target[:n_pts, 1] - contour[:n_pts, 1]) ** 2
        rms_m = np.sqrt(np.mean(d_sq))
        step_rms.append(rms_m * M_TO_CM)

    return np.array(step_rms, dtype=float) if step_rms else np.array([], dtype=float)


def epsilon_div(
    x2_actual: np.ndarray,
    x2_ref: np.ndarray,
    surface_per_step: List[np.ndarray],
) -> float:
    x2_actual = np.asarray(x2_actual, dtype=float)
    x2_ref = np.asarray(x2_ref, dtype=float)
    if x2_actual.ndim == 1:
        x2_actual = x2_actual.reshape(1, -1)
    if x2_ref.ndim == 1:
        x2_ref = x2_ref.reshape(1, -1)
    n = min(len(x2_actual), len(x2_ref), len(surface_per_step))
    if n == 0:
        return 0.0
    terms = []
    for k in range(n):
        e_pos_m = np.linalg.norm(x2_actual[k] - x2_ref[k])
        surf = surface_per_step[k] if k < len(surface_per_step) else surface_per_step[-1]
        surf = np.asarray(surf, dtype=float)
        if surf.ndim != 2 or surf.shape[1] != 2 or len(surf) < 2:
            e_topo_m = 0.0
        else:
            e_topo_m = _point_to_polyline_dist(x2_actual[k], surf)
        terms.append(np.sqrt(e_pos_m * e_pos_m + e_topo_m * e_topo_m))
    return float(np.mean(terms) * M_TO_CM)
