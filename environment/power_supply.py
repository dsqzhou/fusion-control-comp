#
# Copyright @2025 ENN Energy(enn.cn)
#
# 12-channel power supply:
#   clip(um) -> transport delay -> uout_to_urec -> PSM from .mat
#

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import ACTION_7D_TO_12D_INDEX

N_CHANNELS = 12
DT = 0.001
RATE_DEG = 7.2

# Full-scale voltages used by uout_to_urec and the Gym action box.
UM_VALUES = np.array(
    [1500.0, 231.0, 231.0, 173.0, 173.0, 173.0, 173.0, 348.0, 348.0, 348.0, 348.0, 80.0],
    dtype=np.float64,
)

DELAY_RANGE_PF = (0.002, 0.005)
DELAY_RANGE_VS = (0.0, 0.001)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PSM_DIR = _REPO_ROOT / "configs" / "psm"
DEFAULT_PSM_CONFIG_PATH = _PSM_DIR / "fitting_coefficients_v4.mat"

_MI_INT32, _MI_UINT32, _MI_DOUBLE = 5, 6, 9
_MI_MATRIX, _MI_COMPRESSED = 14, 15
_MX_DOUBLE = 6


def action_bounds_12d() -> tuple[np.ndarray, np.ndarray]:
    um = UM_VALUES.astype(np.float32)
    return -um, um.copy()


def action_bounds_7d() -> tuple[np.ndarray, np.ndarray]:
    first_channel = {}
    for channel, src in enumerate(ACTION_7D_TO_12D_INDEX):
        first_channel.setdefault(src, channel)
    high = UM_VALUES[[first_channel[i] for i in range(7)]].astype(np.float32)
    return -high, high.copy()


def _as_12d(name: str, values: np.ndarray | list[float] | None, default: np.ndarray) -> np.ndarray:
    if values is None:
        return default.copy()
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != N_CHANNELS:
        raise ValueError(f"{name} must be length {N_CHANNELS}, got {arr.size}")
    return arr


def resolve_psm_config_path(path: str | Path | None = None) -> Path:
    """Resolve a shot/MATLAB PSM path to a local file.

    Accepts a basename, repo-relative path, or the in-container
    ``TOK_EXL50U/.../fitting_coefficients_*.mat`` address. JSON sidecars
    are accepted when the ``.mat`` itself is missing.
    """
    if path is None:
        candidates = [DEFAULT_PSM_CONFIG_PATH, DEFAULT_PSM_CONFIG_PATH.with_suffix(".json")]
    else:
        given = Path(path)
        candidates = [
            given,
            given.with_suffix(".json"),
            _PSM_DIR / given.name,
            _PSM_DIR / Path(given.name).with_suffix(".json"),
            _REPO_ROOT / given,
            _REPO_ROOT / given.with_suffix(".json"),
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"PSM config not found: {path!s}. Looked in {_PSM_DIR}"
    )


def _read_mat_data_element(buf: bytes, offset: int) -> tuple[int, bytes, int]:
    if offset + 4 > len(buf):
        raise ValueError("truncated MAT data element")
    small = struct.unpack_from("<I", buf, offset)[0]
    if small >> 16:  # small data element
        dtype = small & 0xFFFF
        nbytes = small >> 16
        data = buf[offset + 4 : offset + 4 + nbytes]
        return dtype, data, offset + 8
    if offset + 8 > len(buf):
        raise ValueError("truncated MAT data element")
    dtype, nbytes = struct.unpack_from("<II", buf, offset)
    start = offset + 8
    end = start + nbytes
    data = buf[start:end]
    next_off = end + ((8 - (nbytes % 8)) % 8)
    return dtype, data, next_off


def _parse_numeric_matrix(payload: bytes) -> np.ndarray:
    dtype, body, _ = _read_mat_data_element(payload, 0)
    if dtype != _MI_MATRIX:
        raise ValueError(f"expected miMATRIX, got type {dtype}")

    off = 0
    flag_t, flag_data, off = _read_mat_data_element(body, off)
    if flag_t not in (_MI_INT32, _MI_UINT32) or len(flag_data) < 4:
        raise ValueError("invalid MAT array flags")
    array_class = struct.unpack_from("<I", flag_data, 0)[0] & 0xFF
    if array_class != _MX_DOUBLE:
        raise ValueError(f"only mxDOUBLE is supported, got class {array_class}")

    dim_t, dim_data, off = _read_mat_data_element(body, off)
    if dim_t != _MI_INT32:
        raise ValueError("invalid MAT dimensions")
    dims = np.frombuffer(dim_data, dtype="<i4")
    _name_t, _name, off = _read_mat_data_element(body, off)
    real_t, real_data, _off = _read_mat_data_element(body, off)
    if real_t != _MI_DOUBLE:
        raise ValueError(f"expected miDOUBLE payload, got type {real_t}")
    values = np.frombuffer(real_data, dtype="<f8").reshape(tuple(int(d) for d in dims), order="F")
    return np.asarray(values, dtype=np.float64)


def _load_fit_params_mat(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 136 or raw[126:128] != b"IM":
        raise ValueError(f"unsupported or truncated MAT file: {path}")
    dtype, nbytes = struct.unpack_from("<II", raw, 128)
    blob = raw[136 : 136 + nbytes]
    if dtype == _MI_COMPRESSED:
        blob = zlib.decompress(blob)
    return _parse_numeric_matrix(blob)


def load_psm_coefficients(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load 12-channel slope/intercept from ``.json`` sidecar or ``.mat``."""
    resolved = resolve_psm_config_path(path)
    if resolved.suffix.lower() == ".json":
        data = json.loads(resolved.read_text(encoding="utf-8"))
        slopes = np.asarray(data["slopes"], dtype=np.float64).reshape(-1)
        intercepts = np.asarray(data["intercepts"], dtype=np.float64).reshape(-1)
    else:
        json_sidecar = resolved.with_suffix(".json")
        if json_sidecar.is_file():
            return load_psm_coefficients(json_sidecar)
        fit_params = _load_fit_params_mat(resolved)
        if fit_params.ndim != 2 or fit_params.shape[1] < 2:
            raise ValueError(f"fit_params must be (N, 2), got {fit_params.shape}")
        slopes = fit_params[:, 0].reshape(-1)
        intercepts = fit_params[:, 1].reshape(-1)
    if slopes.size != N_CHANNELS or intercepts.size != N_CHANNELS:
        raise ValueError(
            f"PSM coefficients must have {N_CHANNELS} channels, "
            f"got slopes={slopes.size}, intercepts={intercepts.size}"
        )
    return slopes, intercepts


def uout_to_urec(
    uout: np.ndarray,
    previous_urec: np.ndarray | None,
    um_values: np.ndarray,
    rate_rad: float,
) -> np.ndarray:
    """Phase-angle slew limit used by FusionControl ``UoutToUrecOperator``."""
    uout = np.clip(np.asarray(uout, dtype=np.float64), -um_values, um_values)
    if previous_urec is None:
        return uout.copy()
    previous = np.clip(np.asarray(previous_urec, dtype=np.float64), -um_values, um_values)
    theta_target = np.arcsin(np.clip(uout / um_values, -1.0, 1.0))
    theta_previous = np.arcsin(np.clip(previous / um_values, -1.0, 1.0))
    delta = np.clip(theta_target - theta_previous, -rate_rad, rate_rad)
    return um_values * np.sin(theta_previous + delta)


def _apply_psm(
    u_in: np.ndarray,
    slopes: np.ndarray,
    intercepts: np.ndarray,
    *,
    bypass_vs: bool = True,
) -> np.ndarray:
    output = slopes * u_in + intercepts
    if bypass_vs and output.size:
        output = output.copy()
        output[-1] = u_in[-1]
    return output


class PowerSupplyModel:
    """Map policy setpoint U_set to coil voltage U_real sent to HFM.

    ``delay_s=None`` (default) resamples a random delay every step.
    Pass an explicit 12-vector to freeze delay, or zeros to disable it.
    """

    def __init__(
        self,
        psm_config_path: str | Path | None = None,
        slopes: np.ndarray | list[float] | None = None,
        intercepts: np.ndarray | list[float] | None = None,
        um_values: np.ndarray | list[float] | None = None,
        rate_deg: float = RATE_DEG,
        delay_s: np.ndarray | list[float] | None = None,
        dt: float = DT,
        use_psm: bool = True,
        bypass_vs: bool = True,
        seed: int | None = None,
    ):
        self.dt = float(dt)
        self.use_psm = bool(use_psm)
        self.bypass_vs = bool(bypass_vs)
        self.um_values = _as_12d("um_values", um_values, UM_VALUES)
        self.rate_deg = float(rate_deg)
        if self.rate_deg <= 0:
            raise ValueError("rate_deg must be positive")
        self.rate_rad = np.deg2rad(self.rate_deg)
        self.psm_config_path = None if psm_config_path is None else str(psm_config_path)

        if slopes is not None or intercepts is not None:
            if slopes is None or intercepts is None:
                raise ValueError("slopes and intercepts must be provided together")
            loaded_slopes = _as_12d("slopes", slopes, np.ones(N_CHANNELS))
            loaded_intercepts = _as_12d("intercepts", intercepts, np.zeros(N_CHANNELS))
        elif self.use_psm:
            loaded_slopes, loaded_intercepts = load_psm_coefficients(psm_config_path)
        else:
            loaded_slopes = np.ones(N_CHANNELS, dtype=np.float64)
            loaded_intercepts = np.zeros(N_CHANNELS, dtype=np.float64)

        self.slopes = loaded_slopes
        self.intercepts = loaded_intercepts

        self._rng = np.random.default_rng(seed)
        self._randomize_delay_each_step = delay_s is None
        if delay_s is None:
            self._resample_delay()
        else:
            self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
            self._update_delay_steps()

        self._u_history: list[np.ndarray] = []
        self._step_count = 0
        self._last_urec: np.ndarray | None = None
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)

    def _update_delay_steps(self) -> None:
        if np.any(self.delay_s < 0):
            raise ValueError("delay_s must be non-negative for all channels")
        self.delay_steps = np.round(self.delay_s / self.dt).astype(int)

    def _resample_delay(self, *, delay_s: np.ndarray | list[float] | None = None) -> None:
        if delay_s is None:
            delay_s = np.empty(N_CHANNELS, dtype=np.float64)
            delay_s[:11] = self._rng.uniform(DELAY_RANGE_PF[0], DELAY_RANGE_PF[1], size=11)
            delay_s[11] = self._rng.uniform(DELAY_RANGE_VS[0], DELAY_RANGE_VS[1])
        self.delay_s = _as_12d("delay_s", delay_s, np.full(N_CHANNELS, 0.0035))
        self._update_delay_steps()

    def _transport_delay(self, u_set: np.ndarray) -> np.ndarray:
        self._u_history.append(u_set.copy())
        k = self._step_count
        self._step_count += 1
        if self._randomize_delay_each_step:
            self._resample_delay()

        u_delayed = np.empty(N_CHANNELS, dtype=np.float64)
        u_init = self._u_history[0]
        for i in range(N_CHANNELS):
            idx = k - self.delay_steps[i]
            u_delayed[i] = u_init[i] if idx < 0 else self._u_history[idx][i]
        return u_delayed

    def reset(self, *, u_set_init: np.ndarray | None = None) -> None:
        """Clear buffers; random delay is resampled on every step when enabled."""
        if self._randomize_delay_each_step:
            self._resample_delay()
        self._u_history = []
        self._step_count = 0
        self._last_urec = None
        self._y = np.zeros(N_CHANNELS, dtype=np.float64)
        if u_set_init is not None:
            u0 = np.clip(
                np.asarray(u_set_init, dtype=np.float64).reshape(N_CHANNELS),
                -self.um_values,
                self.um_values,
            )
            self._u_history.append(u0.copy())

    def step(self, u_set: np.ndarray) -> np.ndarray:
        """
        U_set[k] -> clip -> delay -> uout_to_urec -> PSM -> U_real[k]
        """
        u_set = np.clip(
            np.asarray(u_set, dtype=np.float64).reshape(N_CHANNELS),
            -self.um_values,
            self.um_values,
        )
        u_delayed = self._transport_delay(u_set)
        u_urec = uout_to_urec(u_delayed, self._last_urec, self.um_values, self.rate_rad)
        self._last_urec = u_urec.copy()
        if self.use_psm:
            self._y = _apply_psm(
                u_urec, self.slopes, self.intercepts, bypass_vs=self.bypass_vs
            )
        else:
            self._y = u_urec.copy()
        return self._y.copy()

    @property
    def y(self) -> np.ndarray:
        return self._y.copy()

    def get_params(self) -> dict[str, Any]:
        return {
            "psm_config_path": self.psm_config_path,
            "slopes": self.slopes.tolist(),
            "intercepts": self.intercepts.tolist(),
            "um_values": self.um_values.tolist(),
            "rate_deg": self.rate_deg,
            "delay_s": self.delay_s.tolist(),
            "delay_steps": self.delay_steps.tolist(),
            "dt": self.dt,
            "use_psm": self.use_psm,
            "bypass_vs": self.bypass_vs,
        }


def simulate_power_supply_step(
    u_set: np.ndarray,
    dt: float,
    delay_s: float | np.ndarray,
    um_value: float = 1500.0,
    rate_deg: float = RATE_DEG,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> np.ndarray:
    """Scalar open-loop simulation for plotting / validation."""
    u_set = np.asarray(u_set, dtype=np.float64)
    delay_steps = int(round(float(np.asarray(delay_s, dtype=np.float64).reshape(-1)[0]) / dt))
    um = float(um_value)
    rate_rad = np.deg2rad(float(rate_deg))
    y = np.empty_like(u_set)
    history: list[float] = []
    last_urec: float | None = None
    for k, u in enumerate(u_set):
        u = float(np.clip(u, -um, um))
        history.append(u)
        idx = 0 if k - delay_steps < 0 else k - delay_steps
        u_delayed = history[idx]
        urec = uout_to_urec(
            np.array([u_delayed]),
            None if last_urec is None else np.array([last_urec]),
            np.array([um]),
            rate_rad,
        )[0]
        last_urec = float(urec)
        y[k] = float(slope) * urec + float(intercept)
    return y
