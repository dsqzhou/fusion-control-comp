"""Unit tests for power supply model (no socket required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.power_supply import (  # noqa: E402
    DELAY_RANGE_PF,
    DELAY_RANGE_VS,
    RATE_DEG,
    UM_VALUES,
    PowerSupplyModel,
    _apply_psm,
    _load_fit_params_mat,
    action_bounds_7d,
    action_bounds_12d,
    load_psm_coefficients,
    resolve_psm_config_path,
    uout_to_urec,
)


def test_transport_delay_before_step_change():
    model = PowerSupplyModel(
        slopes=np.ones(12),
        intercepts=np.zeros(12),
        rate_deg=180.0,
        delay_s=np.full(12, 0.003),
        seed=0,
    )
    model.reset()
    u_step = np.full(12, 50.0)
    y0 = model.step(np.zeros(12))
    assert np.allclose(y0, 0.0)
    ch = 0
    d = int(model.delay_steps[ch])
    for _ in range(d):
        y = model.step(u_step)
    assert y[ch] == 0.0
    y_after = model.step(u_step)
    assert y_after[ch] == 50.0


def test_steady_state_psm():
    slopes = np.linspace(0.9, 1.1, 12)
    intercepts = np.linspace(-2, 2, 12)
    model = PowerSupplyModel(
        slopes=slopes,
        intercepts=intercepts,
        rate_deg=180.0,
        delay_s=np.full(12, 0.003),
        bypass_vs=False,
        seed=1,
    )
    u = np.clip(np.arange(12, dtype=np.float64), -UM_VALUES, UM_VALUES)
    model.reset(u_set_init=u)
    for _ in range(10):
        y = model.step(u)
    expected = slopes * u + intercepts
    assert np.allclose(y, expected)


def test_random_delay_ranges():
    model = PowerSupplyModel(seed=42, slopes=np.ones(12), intercepts=np.zeros(12))
    assert model.delay_s.shape == (12,)
    assert np.all(model.delay_s[:11] >= DELAY_RANGE_PF[0] - 1e-9)
    assert np.all(model.delay_s[:11] <= DELAY_RANGE_PF[1] + 1e-9)
    assert model.delay_s[11] >= DELAY_RANGE_VS[0] - 1e-9
    assert model.delay_s[11] <= DELAY_RANGE_VS[1] + 1e-9


def test_uout_to_urec_rate_limit():
    um = np.full(12, 100.0)
    previous = np.zeros(12)
    target = np.full(12, 100.0)
    out = uout_to_urec(target, previous, um, np.deg2rad(RATE_DEG))
    expected = 100.0 * np.sin(np.deg2rad(RATE_DEG))
    assert np.allclose(out, expected)


def test_uout_to_urec_first_step_is_clip():
    um = UM_VALUES
    uout = um * 2.0
    out = uout_to_urec(uout, None, um, np.deg2rad(RATE_DEG))
    assert np.allclose(out, um)


def test_psm_affine_bypasses_vs():
    u = np.arange(12, dtype=np.float64)
    slopes = np.full(12, 0.5)
    intercepts = np.full(12, 1.0)
    out = _apply_psm(u, slopes, intercepts, bypass_vs=True)
    assert np.allclose(out[:-1], 0.5 * u[:-1] + 1.0)
    assert out[-1] == u[-1]


def test_load_default_v4_coefficients():
    slopes, intercepts = load_psm_coefficients()
    assert slopes.shape == (12,)
    assert intercepts.shape == (12,)
    assert abs(intercepts[0] - 113.0) < 1e-6
    mat_path = resolve_psm_config_path("fitting_coefficients_v4.mat")
    mat = _load_fit_params_mat(mat_path if mat_path.suffix == ".mat" else mat_path.with_suffix(".mat"))
    assert np.allclose(mat[:, 0], slopes)
    assert np.allclose(mat[:, 1], intercepts)


def test_rampup_psm_only_changes_cs_intercept():
    v4_s, v4_i = load_psm_coefficients("fitting_coefficients_v4.mat")
    ru_s, ru_i = load_psm_coefficients("fitting_coefficients_v4_rampup.mat")
    assert np.allclose(v4_s, ru_s)
    assert abs(ru_i[0] - (-64.0)) < 1e-6
    assert np.allclose(v4_i[1:], ru_i[1:])


def test_action_bounds_match_um_values():
    low12, high12 = action_bounds_12d()
    assert np.allclose(high12, UM_VALUES)
    assert np.allclose(low12, -UM_VALUES)
    low7, high7 = action_bounds_7d()
    assert high7.tolist() == [1500.0, 231.0, 173.0, 173.0, 348.0, 348.0, 80.0]
    assert np.allclose(low7, -high7)


def test_reset_resamples_delay():
    model = PowerSupplyModel(seed=0, slopes=np.ones(12), intercepts=np.zeros(12))
    delay_before = model.delay_s.copy()
    model.reset()
    assert not np.allclose(model.delay_s, delay_before)


def test_reset_keeps_fixed_delay():
    delay_s = np.full(12, 0.004)
    delay_s[11] = 0.001
    model = PowerSupplyModel(
        delay_s=delay_s,
        seed=0,
        slopes=np.ones(12),
        intercepts=np.zeros(12),
    )
    model.reset()
    assert np.allclose(model.delay_s, delay_s)


def test_chain_clip_then_urec_then_psm():
    model = PowerSupplyModel(
        slopes=np.ones(12),
        intercepts=np.zeros(12),
        um_values=np.full(12, 100.0),
        rate_deg=RATE_DEG,
        delay_s=np.zeros(12),
        seed=0,
    )
    model.reset()
    first = model.step(np.zeros(12))
    assert np.allclose(first, 0.0)
    y1 = model.step(np.full(12, 100.0))
    expected = 100.0 * np.sin(np.deg2rad(RATE_DEG))
    assert np.allclose(y1, expected)


if __name__ == "__main__":
    test_transport_delay_before_step_change()
    test_steady_state_psm()
    test_random_delay_ranges()
    test_uout_to_urec_rate_limit()
    test_uout_to_urec_first_step_is_clip()
    test_psm_affine_bypasses_vs()
    test_load_default_v4_coefficients()
    test_rampup_psm_only_changes_cs_intercept()
    test_action_bounds_match_um_values()
    test_reset_resamples_delay()
    test_reset_keeps_fixed_delay()
    test_chain_clip_then_urec_then_psm()
    print("test_power_supply passed.")
