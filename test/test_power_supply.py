"""Unit tests for power supply model (no socket required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.power_supply import PowerSupplyModel, simulate_delay_step  # noqa: E402


def test_delay_holds_output_before_response():
    model = PowerSupplyModel(
        K=np.ones(12),
        b=np.zeros(12),
        delay_s=np.full(12, 0.003),
        seed=0,
    )
    model.reset()
    u_step = np.full(12, 50.0)
    y_before = model.step(np.zeros(12))
    assert np.allclose(y_before, 0.0)
    d = int(model.delay_steps[0])
    for _ in range(d):
        y = model.step(u_step)
    assert np.allclose(y, 0.0)
    y_after = model.step(u_step)
    assert np.allclose(y_after, 50.0)


def test_steady_state_gain():
    K = np.linspace(0.9, 1.1, 12)
    b = np.linspace(-2, 2, 12)
    model = PowerSupplyModel(K=K, b=b, delay_s=np.full(12, 0.003), seed=1)
    u = np.arange(12, dtype=np.float64)
    model.reset(u_set_init=u)
    for _ in range(10):
        y = model.step(u)
    expected = K * u + b
    assert np.allclose(y, expected)


def test_random_delay_in_range():
    model = PowerSupplyModel(seed=42)
    assert model.delay_s.shape == (12,)
    assert np.all(model.delay_s >= 0.002 - 1e-9)
    assert np.all(model.delay_s <= 0.005 + 1e-9)


def test_simulate_matches_model():
    dt = 0.001
    u = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
    y_scalar = simulate_delay_step(u, dt, K=1.2, delay_s=0.002, b=1.0)
    model = PowerSupplyModel(
        K=np.full(12, 1.2),
        b=np.full(12, 1.0),
        delay_s=np.full(12, 0.002),
    )
    model.reset()
    y_model = np.array([model.step(np.full(12, ui))[0] for ui in u])
    assert np.allclose(y_scalar, y_model)


def test_reset_resamples_delay():
    model = PowerSupplyModel(seed=0)
    delay_before = model.delay_s.copy()
    model.reset()
    assert not np.allclose(model.delay_s, delay_before)


def test_reset_keeps_fixed_delay():
    model = PowerSupplyModel(delay_s=np.full(12, 0.004), seed=0)
    model.reset()
    assert np.allclose(model.delay_s, 0.004)


if __name__ == "__main__":
    test_delay_holds_output_before_response()
    test_steady_state_gain()
    test_random_delay_in_range()
    test_simulate_matches_model()
    test_reset_resamples_delay()
    test_reset_keeps_fixed_delay()
    print("test_power_supply passed.")
