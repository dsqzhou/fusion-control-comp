# -*- coding: utf-8 -*-
"""
XPT（先进偏滤器）观测提取与磁通/极向场辅助函数。

与 FusionControl 中 `hfm_control_xpt/reward_ops_xpt.py` 及 RZIP isoflux 分析脚本中的
约定一致：极向磁通 ψ 在 (R,Z) 平面上，极向场分量满足

    Br = -(1/R) * ∂ψ/∂Z
    Bz =  (1/R) * ∂ψ/∂R

其中 R 为大半径 (m)，ψ 为 HFM 输出的 `Fx` 网格上的极向磁通量（与 `FA`/`FB`/`FX` 同量纲）。
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

# HFM 固定网格
_N_RX = 66
_N_ZX = 65
_FX_SIZE = _N_RX * _N_ZX

_FLUX_DIFF_SCALE = 1.0


def _safe_scalar(value: Any, default: float = 0.0) -> float:
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return float(default)
    return float(arr.flat[0])


def relative_error(
    current: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """逐元素 (x - x*) / (|x*| + eps)。"""
    c = np.asarray(current, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    return (c - t) / (np.abs(t) + eps)


def flux_abs_diff(f_x: np.ndarray, fb: float, scale: float = _FLUX_DIFF_SCALE) -> np.ndarray:
    """逐槽位 |FX - FB| / scale。"""
    fb_arr = np.float64(fb)
    return np.abs(np.asarray(f_x, dtype=np.float64) - fb_arr) / np.float64(scale)


def extract_sorted_xpoints(
    obs: dict[str, Any],
    slots: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """
    从观测 dict 读取 rX, zX, FX, nX, FB；按 zX 从高到低排序后固定到 `slots` 个槽位。

    返回
    ----
    r_pad, z_pad, fx_pad : shape (slots,)
    valid : shape (slots,) 有效槽为 1.0
    nx_use : 实际使用的 X 点个数（<= slots）
    fb : 边界参考磁通 FB
    """
    r_pad = np.zeros(slots, dtype=np.float64)
    z_pad = np.zeros(slots, dtype=np.float64)
    fx_pad = np.zeros(slots, dtype=np.float64)
    valid = np.zeros(slots, dtype=np.float64)

    if not isinstance(obs, dict):
        return r_pad, z_pad, fx_pad, valid, 0, 0.0

    r_all = np.asarray(obs.get("rX", []), dtype=np.float64).ravel()
    z_all = np.asarray(obs.get("zX", []), dtype=np.float64).ravel()
    fx_all = np.asarray(obs.get("FX", []), dtype=np.float64).ravel()
    fb = _safe_scalar(obs.get("FB", 0.0), default=0.0)

    nx_raw = int(_safe_scalar(obs.get("nX", 0), default=0.0))
    nx_valid = min(max(nx_raw, 0), len(r_all), len(z_all), len(fx_all))
    if nx_valid <= 0:
        return r_pad, z_pad, fx_pad, valid, 0, fb

    order = np.argsort(z_all[:nx_valid])[::-1]
    nx_use = min(nx_valid, slots)
    r_sorted = r_all[:nx_valid][order][:nx_use]
    z_sorted = z_all[:nx_valid][order][:nx_use]
    fx_sorted = fx_all[:nx_valid][order][:nx_use]

    r_pad[:nx_use] = r_sorted
    z_pad[:nx_use] = z_sorted
    fx_pad[:nx_use] = fx_sorted
    valid[:nx_use] = 1.0
    return r_pad, z_pad, fx_pad, valid, nx_use, fb


def extract_target_xpoints(
    initial_obs: dict[str, Any],
    slots: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从**初始平衡/初始 reset 观测**中提取固定目标 X 点位。

    返回
    ----
    target_rX, target_zX : shape (slots,)
    valid : shape (slots,)
    """
    target_rX, target_zX, _, valid, _, _ = extract_sorted_xpoints(initial_obs, slots=slots)
    return target_rX, target_zX, valid


def extract_target_isoflux_points(
    initial_obs: dict[str, Any],
    lcfs_step: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从**初始平衡/初始 reset 观测**中提取固定的 LCFS 等磁通目标点位。

    默认从 32 个边界点中按 ``0, 4, 8, ..., 28`` 取 8 个点。

    返回
    ----
    target_rB, target_zB : shape (n_points,)
    indices : 选取的原始 LCFS 索引
    """
    r_b = np.asarray(initial_obs.get("rB", []), dtype=np.float64).ravel()
    z_b = np.asarray(initial_obs.get("zB", []), dtype=np.float64).ravel()
    if r_b.size < 32 or z_b.size < 32:
        lc = np.asarray(initial_obs.get("lcfs_points", np.zeros((32, 2))), dtype=np.float64)
        if lc.shape == (32, 2):
            r_b = lc[:, 0].ravel()
            z_b = lc[:, 1].ravel()

    n_lcfs = min(32, r_b.size, z_b.size)
    idx = lcfs_isoflux_indices(n_lcfs, step=lcfs_step)
    return r_b[idx].copy(), z_b[idx].copy(), idx


def lcfs_isoflux_indices(n: int = 32, step: int = 4) -> np.ndarray:
    """LCFS 上每隔 `step` 取一点：默认 0, 4, …, 28 共 8 点。"""
    idx = np.arange(0, n, step, dtype=int)
    return idx[idx < n]


def reshape_fx_to_psi(
    Fx: np.ndarray,
    order: Literal["C", "F"] = "C",
) -> np.ndarray:
    """
    将展平的 `Fx` (4290,) 还原为 (66, 65) 的 ψ 网格。

    - ``order='C'``：行主序，索引 k = i * 65 + j（i 沿 rx，j 沿 zx）。
    - ``order='F'``：列主序（与 MATLAB ``(:)`` 一致），若与 HFM 不一致可改用 ``infer_fx_reshape_order``。

    参数
    ----
    Fx : 长度 4290 的向量
    """
    fx = np.asarray(Fx, dtype=np.float64).ravel()
    if fx.size != _FX_SIZE:
        raise ValueError(f"Fx must have length {_FX_SIZE}, got {fx.size}")
    return fx.reshape(_N_RX, _N_ZX, order=order)


def infer_fx_reshape_order(obs: dict[str, Any]) -> Literal["C", "F"]:
    """
    用 X 点处 HFM 给出的 ``FX`` 与网格插值 ψ 的吻合程度，自动选择 ``'C'`` 或 ``'F'``。

    当 ``nX==0`` 或缺少数据时默认 ``'C'``。
    """
    fx = np.asarray(obs.get("Fx", []), dtype=np.float64).ravel()
    if fx.size != _FX_SIZE:
        return "C"

    rx = np.asarray(obs.get("rx", []), dtype=np.float64).ravel()
    zx = np.asarray(obs.get("zx", []), dtype=np.float64).ravel()
    if rx.size != _N_RX or zx.size != _N_ZX:
        return "C"

    r_x = np.asarray(obs.get("rX", []), dtype=np.float64).ravel()
    z_x = np.asarray(obs.get("zX", []), dtype=np.float64).ravel()
    f_x = np.asarray(obs.get("FX", []), dtype=np.float64).ravel()
    n_x = int(_safe_scalar(obs.get("nX", 0), default=0.0))
    if n_x <= 0:
        return "C"

    best: Literal["C", "F"] = "C"
    best_err = float("inf")
    for order in ("C", "F"):
        psi = reshape_fx_to_psi(fx, order=order)
        err_sum = 0.0
        count = 0
        for k in range(min(n_x, len(r_x), len(z_x), len(f_x))):
            try:
                pred = interp_psi_bilinear(rx, zx, psi, float(r_x[k]), float(z_x[k]))
            except Exception:
                err_sum = float("inf")
                break
            err_sum += abs(pred - f_x[k])
            count += 1
        if count == 0:
            continue
        err_mean = err_sum / count
        if err_mean < best_err:
            best_err = err_mean
            best = order
    return best


def interp_psi_bilinear(
    rx: np.ndarray,
    zx: np.ndarray,
    psi: np.ndarray,
    r: float,
    z: float,
) -> float:
    """在规则网格上对 ψ 做双线性插值（纯 numpy）。"""
    rx = np.asarray(rx, dtype=np.float64).ravel()
    zx = np.asarray(zx, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (rx.size, zx.size):
        raise ValueError(f"psi shape {psi.shape} != ({rx.size}, {zx.size})")

    r = float(np.clip(r, rx[0], rx[-1]))
    z = float(np.clip(z, zx[0], zx[-1]))

    ir = int(np.searchsorted(rx, r) - 1)
    iz = int(np.searchsorted(zx, z) - 1)
    ir = int(np.clip(ir, 0, rx.size - 2))
    iz = int(np.clip(iz, 0, zx.size - 2))

    r0, r1 = rx[ir], rx[ir + 1]
    z0, z1 = zx[iz], zx[iz + 1]
    q00 = psi[ir, iz]
    q10 = psi[ir + 1, iz]
    q01 = psi[ir, iz + 1]
    q11 = psi[ir + 1, iz + 1]

    tr = (r - r0) / (r1 - r0) if r1 > r0 else 0.0
    tz = (z - z0) / (z1 - z0) if z1 > z0 else 0.0
    q0 = q00 * (1.0 - tr) + q10 * tr
    q1 = q01 * (1.0 - tr) + q11 * tr
    return float(q0 * (1.0 - tz) + q1 * tz)


def get_psi_grid(
    obs: dict[str, Any],
    order: Literal["C", "F"] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Literal["C", "F"]]:
    """
    从观测得到 (rx, zx, psi_2d, order)。

    若 ``order is None``，调用 ``infer_fx_reshape_order``。
    """
    rx = np.asarray(obs.get("rx", []), dtype=np.float64).ravel()
    zx = np.asarray(obs.get("zx", []), dtype=np.float64).ravel()
    fx = np.asarray(obs.get("Fx", []), dtype=np.float64).ravel()
    if rx.size != _N_RX or zx.size != _N_ZX or fx.size != _FX_SIZE:
        raise ValueError("obs must contain valid rx (66), zx (65), Fx (4290)")
    ord_use: Literal["C", "F"]
    if order is None:
        ord_use = infer_fx_reshape_order(obs)
    else:
        ord_use = order
    psi = reshape_fx_to_psi(fx, order=ord_use)
    return rx, zx, psi, ord_use


def psi_at_points(
    obs: dict[str, Any],
    r_points: np.ndarray | list[float],
    z_points: np.ndarray | list[float],
    order: Literal["C", "F"] | None = None,
) -> np.ndarray:
    """在给定点集上对 `Fx` 还原得到的 ψ 网格做双线性插值。"""
    rx, zx, psi, _ = get_psi_grid(obs, order=order)
    r_arr = np.asarray(r_points, dtype=np.float64).ravel()
    z_arr = np.asarray(z_points, dtype=np.float64).ravel()
    if r_arr.size != z_arr.size:
        raise ValueError("r_points and z_points must have the same length")
    vals = [interp_psi_bilinear(rx, zx, psi, float(r), float(z)) for r, z in zip(r_arr, z_arr)]
    return np.asarray(vals, dtype=np.float64)


def gradient_psi_on_grid(
    rx: np.ndarray,
    zx: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算 ∂ψ/∂R 与 ∂ψ/∂Z（与 ``numpy.gradient`` 非均匀坐标约定一致）。

    返回与 ``psi`` 同形状的二维数组。
    """
    rx = np.asarray(rx, dtype=np.float64).ravel()
    zx = np.asarray(zx, dtype=np.float64).ravel()
    psi = np.asarray(psi, dtype=np.float64)
    if psi.shape != (rx.size, zx.size):
        raise ValueError("psi must align with rx, zx")
    g = np.gradient(psi, rx, zx)
    dpsi_dr = g[0]
    dpsi_dz = g[1]
    return dpsi_dr, dpsi_dz


def poloidal_br_bz(
    r: np.ndarray | float,
    dpsi_dr: np.ndarray,
    dpsi_dz: np.ndarray,
    r_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Br = -(1/R) ∂ψ/∂Z,  Bz = (1/R) ∂ψ/∂R（逐元素，与 FusionControl RZIP isoflux 脚本一致）。
    """
    r_arr = np.maximum(np.asarray(r, dtype=np.float64), r_floor)
    br = -(1.0 / r_arr) * np.asarray(dpsi_dz, dtype=np.float64)
    bz = (1.0 / r_arr) * np.asarray(dpsi_dr, dtype=np.float64)
    return br, bz


def interp_bilinear_2d_field(
    rx: np.ndarray,
    zx: np.ndarray,
    field: np.ndarray,
    r: float,
    z: float,
) -> float:
    """对任意与 ψ 同形的标量场做双线性插值。"""
    return interp_psi_bilinear(rx, zx, field, r, z)


def br_bz_at_point(
    obs: dict[str, Any],
    r: float,
    z: float,
    order: Literal["C", "F"] | None = None,
    r_floor: float = 1e-6,
) -> tuple[float, float]:
    """
    在给定 (R,Z) 处由网格 ψ 计算极向场 Br, Bz。
    """
    rx, zx, psi, _ = get_psi_grid(obs, order=order)
    dpsi_dr, dpsi_dz = gradient_psi_on_grid(rx, zx, psi)
    ddr = interp_bilinear_2d_field(rx, zx, dpsi_dr, r, z)
    ddz = interp_bilinear_2d_field(rx, zx, dpsi_dz, r, z)
    rf = max(float(r), r_floor)
    br = -(1.0 / rf) * ddz
    bz = (1.0 / rf) * ddr
    return float(br), float(bz)


def _xpoint_features_from_targets(
    obs: dict[str, Any],
    target_rX: np.ndarray | list[float],
    target_zX: np.ndarray | list[float],
    slots: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """方案一的 X 点部分：4 * [valid, dr, dz, dFX]。"""
    tr_x = np.asarray(target_rX, dtype=np.float64).ravel()[:slots]
    tz_x = np.asarray(target_zX, dtype=np.float64).ravel()[:slots]
    if len(tr_x) < slots or len(tz_x) < slots:
        raise ValueError("target_rX/target_zX must have length >= slots")

    r_x, z_x, f_x, valid, _, fb = extract_sorted_xpoints(obs, slots=slots)
    dr = relative_error(r_x, tr_x, eps=eps) * valid
    dz = relative_error(z_x, tz_x, eps=eps) * valid
    dfx = flux_abs_diff(f_x, fb) * valid
    return np.stack([valid, dr, dz, dfx], axis=1).reshape(-1)


def scheme1_xpoint_features(
    obs: dict[str, Any],
    target_rX: np.ndarray | list[float],
    target_zX: np.ndarray | list[float],
    slots: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    方案一的 XPT 特征，仅处理 **X 点位置与磁通**：

        4 * [valid, dr, dz, dFX]

    共 `4 * slots` 维，默认 16 维。
    """
    return _xpoint_features_from_targets(
        obs,
        target_rX=target_rX,
        target_zX=target_zX,
        slots=slots,
        eps=eps,
    ).astype(np.float64)


def scheme1_feature_vector(
    obs: dict[str, Any],
    target_keys: tuple[str, ...] = ("Ip", "Rmin", "Rmax", "kappa"),
    target_values: np.ndarray | list[float] | None = None,
    target_rX: np.ndarray | list[float] | None = None,
    target_zX: np.ndarray | list[float] | None = None,
    slots: int = 4,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    兼容 FusionControl ``build_xpoint_observation`` 的组合特征：

        [Ip,Rmin,Rmax,kappa 相对 target 的误差] + 4 * [valid, dr, dz, dFX]

    若只需要 XPT 部分，请优先使用 `scheme1_xpoint_features(...)`。
    """
    if target_values is None:
        return scheme1_xpoint_features(
            obs,
            target_rX=target_rX,
            target_zX=target_zX,
            slots=slots,
            eps=eps,
        )
    tv = np.asarray(target_values, dtype=np.float64).ravel()
    if len(target_keys) != len(tv):
        raise ValueError("target_keys and target_values length mismatch")

    base = np.array([_safe_scalar(obs.get(k, 0.0)) for k in target_keys], dtype=np.float64)
    base_rel = relative_error(base, tv, eps=eps)
    xfeat = _xpoint_features_from_targets(
        obs,
        target_rX=target_rX,
        target_zX=target_zX,
        slots=slots,
        eps=eps,
    )
    return np.concatenate([base_rel, xfeat]).astype(np.float64)


def isoflux_residuals_scheme2(
    obs: dict[str, Any],
    target_rB: np.ndarray | list[float],
    target_zB: np.ndarray | list[float],
    target_rX: np.ndarray | list[float],
    target_zX: np.ndarray | list[float],
    lcfs_step: int = 4,
    order: Literal["C", "F"] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    方案二：8 个**固定 LCFS 目标点位** + 4 个**目标 X 点位**，共 12 点的 **ψ - FB**（应趋于 0）。

    - LCFS 目标点位由初始平衡提取，通常来自 32 个边界点中每隔 `lcfs_step` 取 1 个。
    - X 点位置来自初始平衡固定下来的 ``target_rX, target_zX``，而不是当前重新识别出的 X 点。

    返回
    ----
    residuals : shape (12,)
    meta : 含 indices、各点 R,Z、fb 等，便于调试
    """
    rx, zx, psi, ord_use = get_psi_grid(obs, order=order)
    fb = _safe_scalar(obs.get("FB", 0.0))

    r_b = np.asarray(target_rB, dtype=np.float64).ravel()
    z_b = np.asarray(target_zB, dtype=np.float64).ravel()
    if r_b.size != z_b.size:
        raise ValueError("target_rB and target_zB must have the same length")
    idx = np.arange(r_b.size, dtype=int)
    psi_lcfs = []
    pts = []
    for i in idx:
        rr, zz = float(r_b[i]), float(z_b[i])
        pts.append((rr, zz, "target_lcfs", int(i)))
        psi_lcfs.append(interp_psi_bilinear(rx, zx, psi, rr, zz))
    psi_lcfs = np.asarray(psi_lcfs, dtype=np.float64)
    res_lcfs = psi_lcfs - fb

    r_x = np.asarray(target_rX, dtype=np.float64).ravel()[:4]
    z_x = np.asarray(target_zX, dtype=np.float64).ravel()[:4]
    if len(r_x) < 4 or len(z_x) < 4:
        raise ValueError("target_rX/target_zX must have length >= 4")
    res_x = []
    xpts = []
    for s in range(4):
        rr, zz = float(r_x[s]), float(z_x[s])
        xpts.append((rr, zz, "target_x", s))
        res_x.append(interp_psi_bilinear(rx, zx, psi, rr, zz) - fb)
    res_x = np.asarray(res_x, dtype=np.float64)

    residuals = np.concatenate([res_lcfs, res_x])
    meta = {
        "fb": fb,
        "fx_order": ord_use,
        "lcfs_indices": idx,
        "points": pts + xpts,
    }
    return residuals, meta


def xpoint_poloidal_field_magnitude(
    obs: dict[str, Any],
    target_rX: np.ndarray | list[float] | None = None,
    target_zX: np.ndarray | list[float] | None = None,
    slots: int = 4,
    order: Literal["C", "F"] | None = None,
    r_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在给定 X 点位上计算极向场模 ``sqrt(Br^2 + Bz^2)``。

    - 若提供 `target_rX/target_zX`，则按**目标 X 点位**计算。
    - 否则回退到当前观测中排序后的 X 点槽位。

    返回
    ----
    br, bz : shape (slots,) 无效槽为 nan
    mag : shape (slots,)
    """
    rx, zx, psi, _ = get_psi_grid(obs, order=order)
    dpsi_dr, dpsi_dz = gradient_psi_on_grid(rx, zx, psi)

    if target_rX is None or target_zX is None:
        r_x, z_x, _, valid, _, _ = extract_sorted_xpoints(obs, slots=slots)
    else:
        r_x = np.asarray(target_rX, dtype=np.float64).ravel()[:slots]
        z_x = np.asarray(target_zX, dtype=np.float64).ravel()[:slots]
        if len(r_x) < slots or len(z_x) < slots:
            raise ValueError("target_rX/target_zX must have length >= slots")
        valid = np.ones(slots, dtype=np.float64)
    br = np.full(slots, np.nan, dtype=np.float64)
    bz = np.full(slots, np.nan, dtype=np.float64)
    mag = np.full(slots, np.nan, dtype=np.float64)

    for s in range(slots):
        if valid[s] < 0.5:
            continue
        rr, zz = float(r_x[s]), float(z_x[s])
        ddr = interp_bilinear_2d_field(rx, zx, dpsi_dr, rr, zz)
        ddz = interp_bilinear_2d_field(rx, zx, dpsi_dz, rr, zz)
        rf = max(rr, r_floor)
        bbr = -(1.0 / rf) * ddz
        bbz = (1.0 / rf) * ddr
        br[s] = bbr
        bz[s] = bbz
        mag[s] = float(np.hypot(bbr, bbz))

    return br, bz, mag


__all__ = [
    "relative_error",
    "flux_abs_diff",
    "extract_sorted_xpoints",
    "extract_target_xpoints",
    "extract_target_isoflux_points",
    "lcfs_isoflux_indices",
    "reshape_fx_to_psi",
    "infer_fx_reshape_order",
    "interp_psi_bilinear",
    "get_psi_grid",
    "psi_at_points",
    "gradient_psi_on_grid",
    "poloidal_br_bz",
    "interp_bilinear_2d_field",
    "br_bz_at_point",
    "scheme1_xpoint_features",
    "scheme1_feature_vector",
    "isoflux_residuals_scheme2",
    "xpoint_poloidal_field_magnitude",
]
