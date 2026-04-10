"""Unit tests for environment/xpt_utils.py (no HFM simulator required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.xpt_utils import (
    br_bz_at_point,
    extract_target_isoflux_points,
    extract_target_xpoints,
    extract_sorted_xpoints,
    flux_abs_diff,
    get_psi_grid,
    gradient_psi_on_grid,
    infer_fx_reshape_order,
    interp_psi_bilinear,
    isoflux_residuals_scheme2,
    poloidal_br_bz,
    relative_error,
    reshape_fx_to_psi,
    scheme1_feature_vector,
    scheme1_xpoint_features,
    xpoint_poloidal_field_magnitude,
)


def test_extract_sorted_xpoints_order():
    obs = {
        "rX": np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
        "zX": np.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0]),
        "FX": np.array([0.1, 0.3, 0.2, 0.0, 0.0, 0.0]),
        "nX": np.array([3]),
        "FB": np.array([0.5]),
    }
    r, z, fx, valid, nx, fb = extract_sorted_xpoints(obs, slots=4)
    assert nx == 3
    assert fb == 0.5
    assert np.allclose(z[:3], [2.0, 1.0, 0.0])
    assert np.allclose(r[:3], [2.0, 3.0, 1.0])
    assert np.allclose(valid[:3], 1.0)


def test_scheme1_dim():
    obs = {
        "Ip": np.array([1.0]),
        "Rmin": np.array([0.1]),
        "Rmax": np.array([2.0]),
        "kappa": np.array([1.5]),
        "rX": np.ones(6) * 0.5,
        "zX": np.array([1.0, 0.5, -0.5, -1.0, 0.0, 0.0]),
        "FX": np.zeros(6),
        "nX": np.array([4]),
        "FB": np.array([0.0]),
    }
    v = scheme1_feature_vector(
        obs,
        target_values=[1.0, 0.1, 2.0, 1.5],
        target_rX=[0.5, 0.5, 0.5, 0.5],
        target_zX=[1.0, 0.5, -0.5, -1.0],
    )
    assert v.shape == (20,)
    vx = scheme1_xpoint_features(
        obs,
        target_rX=[0.5, 0.5, 0.5, 0.5],
        target_zX=[1.0, 0.5, -0.5, -1.0],
    )
    assert vx.shape == (16,)


def test_linear_psi_br_bz():
    """ψ = a*R + b*Z => ∂ψ/∂R=a, ∂ψ/∂Z=b => Br=-b/R, Bz=a/R."""
    rx = np.linspace(0.5, 1.2, 66)
    zx = np.linspace(-0.8, 0.8, 65)
    a, b = 0.03, -0.02
    rr, zz = np.meshgrid(rx, zx, indexing="ij")
    psi = a * rr + b * zz
    dpsi_dr, dpsi_dz = gradient_psi_on_grid(rx, zx, psi)
    assert np.allclose(dpsi_dr, a, rtol=1e-5, atol=1e-4)
    assert np.allclose(dpsi_dz, b, rtol=1e-5, atol=1e-4)
    r_test = 0.9
    z_test = 0.1
    br_e = -(1.0 / r_test) * b
    bz_e = (1.0 / r_test) * a
    br, bz = poloidal_br_bz(r_test, np.array([[a]]), np.array([[b]]))
    assert abs(br.item() - br_e) < 1e-12
    assert abs(bz.item() - bz_e) < 1e-12


def test_infer_order_matches_synthetic():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    psi = np.sin(np.outer(np.linspace(0, 1, 66), np.ones(65))) * 0.01
    fx_c = psi.reshape(-1, order="C")
    i, j = 10, 20
    obs = {
        "Fx": fx_c,
        "rx": rx,
        "zx": zx,
        "rX": np.array([rx[i], 0, 0, 0, 0, 0]),
        "zX": np.array([zx[j], 0, 0, 0, 0, 0]),
        "FX": np.array([psi[i, j], 0, 0, 0, 0, 0]),
        "nX": np.array([1]),
        "FB": np.array([0.0]),
    }
    assert infer_fx_reshape_order(obs) == "C"


def test_infer_f_order_matches_synthetic():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    rr, zz = np.meshgrid(rx, zx, indexing="ij")
    psi = 0.01 * rr - 0.02 * zz
    fx_f = psi.reshape(-1, order="F")
    i, j = 11, 21
    obs = {
        "Fx": fx_f,
        "rx": rx,
        "zx": zx,
        "rX": np.array([rx[i], 0, 0, 0, 0, 0]),
        "zX": np.array([zx[j], 0, 0, 0, 0, 0]),
        "FX": np.array([psi[i, j], 0, 0, 0, 0, 0]),
        "nX": np.array([1]),
        "FB": np.array([0.0]),
    }
    assert infer_fx_reshape_order(obs) == "F"


def test_infer_f_order_matches_synthetic():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    rr, zz = np.meshgrid(rx, zx, indexing="ij")
    psi = 0.01 * rr - 0.02 * zz
    fx_f = psi.reshape(-1, order="F")
    i, j = 11, 21
    obs = {
        "Fx": fx_f,
        "rx": rx,
        "zx": zx,
        "rX": np.array([rx[i], 0, 0, 0, 0, 0]),
        "zX": np.array([zx[j], 0, 0, 0, 0, 0]),
        "FX": np.array([psi[i, j], 0, 0, 0, 0, 0]),
        "nX": np.array([1]),
        "FB": np.array([0.0]),
    }
    assert infer_fx_reshape_order(obs) == "F"


def test_isoflux_shape():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    psi = np.random.default_rng(1).standard_normal((66, 65)) * 0.01
    obs = {
        "Fx": psi.reshape(-1, order="C"),
        "rx": rx,
        "zx": zx,
        "rB": np.linspace(0.5, 1.1, 32),
        "zB": np.zeros(32),
        "rX": np.zeros(6),
        "zX": np.zeros(6),
        "FX": np.zeros(6),
        "nX": np.array([0]),
        "FB": np.array([0.0]),
    }
    res, _ = isoflux_residuals_scheme2(
        obs,
        target_rB=[0.5 + 0.6 * i / 7.0 for i in range(8)],
        target_zB=[0.0] * 8,
        target_rX=[0.7, 0.6, 0.6, 0.7],
        target_zX=[1.2, 0.8, -0.8, -1.2],
    )
    assert res.shape == (12,)


def test_extract_target_isoflux_points_indices():
    initial_obs = {
        "rB": np.arange(32, dtype=float),
        "zB": -np.arange(32, dtype=float),
    }
    r_b, z_b, idx = extract_target_isoflux_points(initial_obs, lcfs_step=4)
    assert np.allclose(idx, [0, 4, 8, 12, 16, 20, 24, 28])
    assert np.allclose(r_b, [0, 4, 8, 12, 16, 20, 24, 28])
    assert np.allclose(z_b, [0, -4, -8, -12, -16, -20, -24, -28])


def test_scheme1_valid_mask_ignores_invalid_slots():
    obs = {
        "rX": np.array([0.9, 1.1, 99.0, -77.0, 0.0, 0.0]),
        "zX": np.array([1.0, -1.0, 88.0, -66.0, 0.0, 0.0]),
        "FX": np.array([0.2, 0.3, 123.0, -456.0, 0.0, 0.0]),
        "nX": np.array([2]),
        "FB": np.array([0.25]),
    }
    feat = scheme1_xpoint_features(
        obs,
        target_rX=[0.9, 1.1, 0.5, 0.6],
        target_zX=[1.0, -1.0, 0.5, -0.6],
        slots=4,
    ).reshape(4, 4)
    assert np.allclose(feat[0], [1.0, 0.0, 0.0, 0.05])
    assert np.allclose(feat[1], [1.0, 0.0, 0.0, 0.05])
    assert np.allclose(feat[2], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(feat[3], [0.0, 0.0, 0.0, 0.0])


def test_isoflux_residuals_scheme2_values_match_linear_psi():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    rr, zz = np.meshgrid(rx, zx, indexing="ij")
    a, b, c = 0.02, -0.01, 0.03
    psi = a * rr + b * zz + c
    target_rB = np.linspace(0.5, 1.2, 8)
    target_zB = np.linspace(-0.2, 0.2, 8)
    target_rX = np.array([0.7, 0.8, 0.9, 1.0])
    target_zX = np.array([1.0, 0.6, -0.6, -1.0])
    fb = 0.05
    obs = {
        "Fx": psi.reshape(-1, order="C"),
        "rx": rx,
        "zx": zx,
        "FB": np.array([fb]),
    }
    res, meta = isoflux_residuals_scheme2(
        obs,
        target_rB=target_rB,
        target_zB=target_zB,
        target_rX=target_rX,
        target_zX=target_zX,
    )
    expect_lcfs = a * target_rB + b * target_zB + c - fb
    expect_x = a * target_rX + b * target_zX + c - fb
    assert np.allclose(res[:8], expect_lcfs, atol=1e-8)
    assert np.allclose(res[8:], expect_x, atol=1e-8)
    assert len(meta["points"]) == 12


def test_extract_sorted_xpoints_sorts_all_valid_then_truncates():
    obs = {
        "rX": np.array([0.1, 0.2, 0.3, 0.4, 9.9, 8.8]),
        "zX": np.array([-2.0, -1.0, 0.0, 1.0, 3.0, 2.0]),
        "FX": np.arange(6, dtype=float),
        "nX": np.array([6]),
        "FB": np.array([0.0]),
    }
    r, z, fx, valid, nx, _ = extract_sorted_xpoints(obs, slots=4)
    assert nx == 4
    assert np.allclose(z, [3.0, 2.0, 1.0, 0.0])
    assert np.allclose(r, [9.9, 8.8, 0.4, 0.3])
    assert np.allclose(fx, [4.0, 5.0, 3.0, 2.0])


def test_target_xpoint_field_uses_fixed_target_positions():
    rx = np.linspace(0.4, 1.4, 66)
    zx = np.linspace(-1.4, 1.4, 65)
    rr, zz = np.meshgrid(rx, zx, indexing="ij")
    psi = 0.02 * rr - 0.01 * zz
    obs = {
        "Fx": psi.reshape(-1, order="C"),
        "rx": rx,
        "zx": zx,
        "rX": np.zeros(6),
        "zX": np.zeros(6),
        "FX": np.zeros(6),
        "nX": np.array([0]),
        "FB": np.array([0.0]),
    }
    br, bz, mag = xpoint_poloidal_field_magnitude(
        obs,
        target_rX=[0.9, 1.0, 1.1, 1.2],
        target_zX=[0.0, 0.1, -0.1, 0.2],
        slots=4,
    )
    assert np.all(np.isfinite(br))
    assert np.all(np.isfinite(bz))
    assert np.all(np.isfinite(mag))


def run_all():
    test_extract_sorted_xpoints_order()
    test_scheme1_dim()
    test_linear_psi_br_bz()
    test_infer_order_matches_synthetic()
    test_infer_f_order_matches_synthetic()
    test_isoflux_shape()
    test_extract_target_isoflux_points_indices()
    test_scheme1_valid_mask_ignores_invalid_slots()
    test_isoflux_residuals_scheme2_values_match_linear_psi()
    test_extract_sorted_xpoints_sorts_all_valid_then_truncates()
    test_target_xpoint_field_uses_fixed_target_positions()
    print("test_xpt_utils: all passed")


if __name__ == "__main__":
    run_all()
