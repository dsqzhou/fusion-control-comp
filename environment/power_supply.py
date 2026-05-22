#
# Copyright @2025 ENN Energy(enn.cn)
#
# 12-channel power supply model: static gain/offset + pure delay.
#

from __future__ import annotations

from typing import Any

import numpy as np

N_CHANNELS = 12
DT = 0.001  # simulator step: 1 ms

# Placeholder static parameters until segment_best.csv is wired in.
DEFAULT_K = np.array(
    [1.0, 0.98, 1.02, 1.0, 0.99, 1.01, 1.0, 0.97, 1.03, 1.0, 0.98, 1.02],
    dtype=np.float64,
)
DEFAULT_B = np.zeros(N_CHANNELS, dtype=np.float64)

DELAY_RANGE = (0.002, 0.005)


def _as_12d(name: str, values: np.ndarray | list[float] | None, default: np.ndarray) -> np.ndarray:
    if values is None:
        return default.copy()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != N_CHANNELS:
        raise ValueError(f"{name} must be length {N_CHANNELS}, got {arr.size}")
    return arr


class PowerSupplyModel:
    """Map policy setpoint U_set to actual coil voltage U_real via static gain and delay."""

    def __init__(
        self,
        K: np.ndarray | list[float] | None = None,
        b: np.ndarray | list[float] | None = None,
        delay_s: np.ndarray | list[float] | None = None,
        dt: float = DT,
        delay_range: tuple[float, float] = DELAY_RANGE,
        seed: int | None = None,
    ):
        self.dt = float(dt)
        self._delay_range = delay_range
        self.K = _as_12d("K", K, DEFAULT_K)
        self.b = _as_12d("b", b, DEFAULT_B)

        self._rng = np.random.default_rng(seed)
        self._randomize_on_reset = delay_s is None
        if delay_s is None:
            self._resample_delay()
        else:
            self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
            self._update_delay_steps()

        self._u_history: list[np.ndarray] = []
        self._step_count = 0
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)

    def _update_delay_steps(self) -> None:
        if np.any(self.delay_s < 0):
            raise ValueError("delay_s must be non-negative for all channels")
        self.delay_steps = np.round(self.delay_s / self.dt).astype(int)

    def _resample_delay(
        self,
        *,
        delay_s: np.ndarray | list[float] | None = None,
    ) -> None:
        if delay_s is None:
            delay_s = self._rng.uniform(self._delay_range[0], self._delay_range[1], size=N_CHANNELS)
        self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
        self._update_delay_steps()

    def reset(self, *, u_set_init: np.ndarray | None = None) -> None:
        """Clear delay buffer and optionally resample per-channel delay."""
        if self._randomize_on_reset:
            self._resample_delay()
        self._u_history = []
        self._step_count = 0
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)
        if u_set_init is not None:
            u0 = np.asarray(u_set_init, dtype=np.float64).reshape(N_CHANNELS)
            self._u_history.append(u0.copy())

    def step(self, u_set: np.ndarray) -> np.ndarray:
        """
        One discrete update:

            U_real[k] = K * U_set[k - d] + b
        """
        u_set = np.asarray(u_set, dtype=np.float64).reshape(N_CHANNELS)
        self._u_history.append(u_set.copy())
        k = self._step_count
        self._step_count += 1

        u_delayed = np.empty(N_CHANNELS, dtype=np.float64)
        u_init = self._u_history[0]
        for i in range(N_CHANNELS):
            idx = k - self.delay_steps[i]
            u_delayed[i] = u_init[i] if idx < 0 else self._u_history[idx][i]

        self._y = self.K * u_delayed + self.b
        return self._y.copy()

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()

    def get_params(self) -> dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "b": self.b.tolist(),
            "delay_s": self.delay_s.tolist(),
            "delay_steps": self.delay_steps.tolist(),
            "dt": self.dt,
        }


def simulate_delay_step(
    u_set: np.ndarray,
    dt: float,
    K: np.ndarray | float,
    delay_s: float,
    b: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Scalar open-loop simulation for plotting / validation."""
    u_set = np.asarray(u_set, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = int(round(delay_s / dt))
    y = np.empty_like(u_set)
    for k in range(u_set.size):
        idx = 0 if k - d < 0 else k - d
        y[k] = K * u_set[idx] + b
    return y
