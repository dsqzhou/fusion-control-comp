#
# Copyright @2025 ENN Energy(enn.cn)
#
# HFM Simulator: Gymnasium Env, dict observation, 12D action, optional reward_fn.
# Reference is trajectory-based: hold-initial-equilibrium or user-defined trajectory.
#

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from .case_references import CASE_SHOT_IDS, build_case_reference
from .hfm_predictor import HFMSocketPredictor
from .power_supply import action_bounds_12d
from .shot_registry import REFERENCE_KEYS

RAW_OBSERVATION_SPECS: dict[str, tuple[int, ...]] = {
    "Ip": (1,),
    "R": (1,),
    "Z": (1,),
    "I_PF": (12,),
    "Rmax": (1,),
    "Rmin": (1,),
    "aminor": (1,),
    "deltal": (1,),
    "deltau": (1,),
    "kappa": (1,),
    "rc": (1,),
    "zc": (1,),
    "FA": (1,),
    "FB": (1,),
    "Bm": (164,),
    "Ff": (47,),
    "nrx": (1,),
    "nzx": (1,),
    "Fx": (4290,),
    "rx": (66,),
    "zx": (65,),
    "rX": (6,),
    "zX": (6,),
    "nX": (1,),
    "lX": (1,),
    "rB": (32,),
    "zB": (32,),
    "FX": (6,),
}

SCALAR_REFERENCE_KEYS = tuple(REFERENCE_KEYS)
XPT_REFERENCE_SPECS: dict[str, tuple[int, ...]] = {
    "rX": (4,),
    "zX": (4,),
    "x_valid": (4,),
    "strike_r": (8,),
    "strike_z": (8,),
    "strike_valid": (8,),
    "nX": (1,),
    "n_strike": (1,),
}


def _default_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    return action_bounds_12d()


def _box(shape: tuple[int, ...]) -> gym.spaces.Box:
    return gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float64)


def _zeros(shape: tuple[int, ...], dtype=np.float64) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def _coerce_raw_value(key: str, value: Any) -> np.ndarray:
    shape = RAW_OBSERVATION_SPECS[key]
    if value is None:
        return _zeros(shape)

    arr = np.asarray(value, dtype=np.float64)
    expected_size = int(np.prod(shape))
    if arr.size != expected_size:
        raise ValueError(
            f"Raw observation key {key} expects shape {shape} (size={expected_size}), got {arr.shape}"
        )
    return arr.reshape(shape)


def _build_observation_space() -> gym.Space:
    spaces: dict[str, gym.Space] = {key: _box(shape) for key, shape in RAW_OBSERVATION_SPECS.items()}
    for key in SCALAR_REFERENCE_KEYS:
        spaces[f"reference_{key}"] = _box((1,))
    for key, shape in XPT_REFERENCE_SPECS.items():
        spaces[f"reference_{key}"] = _box(shape)
    spaces["failure"] = gym.spaces.MultiBinary(1)
    return gym.spaces.Dict(spaces)


def _coerce_scalar_series(values: Any, max_steps: int, default: float) -> np.ndarray:
    if values is None:
        arr = np.full((max_steps,), default, dtype=np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.full((max_steps,), float(arr[0]), dtype=np.float64)
        elif arr.size != max_steps:
            raise ValueError(f"Reference length must equal max_steps={max_steps}, got {arr.size}")
    return arr


def _coerce_vector_series(
    values: Any,
    max_steps: int,
    shape: tuple[int, ...],
    default: np.ndarray | None = None,
) -> np.ndarray:
    base = np.zeros(shape, dtype=np.float64) if default is None else np.asarray(default, dtype=np.float64)
    if base.shape != shape:
        base = np.zeros(shape, dtype=np.float64)
    if values is None:
        return np.repeat(base[None, ...], max_steps, axis=0)

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == shape:
        return np.repeat(arr[None, ...], max_steps, axis=0)
    if arr.shape == (max_steps,) + shape:
        return arr
    if shape == (1,) and arr.size == 1:
        return np.full((max_steps, 1), float(arr.reshape(-1)[0]), dtype=np.float64)
    raise ValueError(
        f"reference vector must have shape {shape} or ({max_steps}, {', '.join(map(str, shape))}), "
        f"got {arr.shape}"
    )


def _build_hold_reference(raw: dict[str, Any], max_steps: int) -> dict[str, np.ndarray]:
    reference = {
        key: np.full(
            (max_steps,),
            float(_coerce_raw_value(key, raw.get(key)).reshape(-1)[0]),
            dtype=np.float64,
        )
        for key in SCALAR_REFERENCE_KEYS
    }
    for key, shape in XPT_REFERENCE_SPECS.items():
        reference[key] = np.zeros((max_steps,) + shape, dtype=np.float64)
    return reference


def _build_reference_trajectory(
    reference_mode: str,
    reference_spec: dict[str, Any] | None,
    raw: dict[str, Any],
    max_steps: int,
) -> dict[str, np.ndarray]:
    hold_ref = _build_hold_reference(raw, max_steps)
    if reference_mode == "hold":
        return hold_ref

    if reference_mode != "trajectory":
        raise ValueError(f"Unsupported reference_mode: {reference_mode}")

    reference_spec = reference_spec or {}
    reference = {
        key: _coerce_scalar_series(reference_spec.get(key), max_steps, float(hold_ref[key][0]))
        for key in SCALAR_REFERENCE_KEYS
    }
    for key, shape in XPT_REFERENCE_SPECS.items():
        reference[key] = _coerce_vector_series(
            reference_spec.get(key),
            max_steps,
            shape,
            hold_ref[key][0],
        )
    return reference


def _obs_dict_from_raw(
    raw: dict[str, Any],
    reference: dict[str, np.ndarray],
    reference_index: int,
) -> dict[str, Any]:
    out = {key: _coerce_raw_value(key, raw.get(key)) for key in RAW_OBSERVATION_SPECS}

    ref_idx = int(min(max(reference_index, 0), len(reference["Ip"]) - 1))
    for key in SCALAR_REFERENCE_KEYS:
        out[f"reference_{key}"] = np.array([reference[key][ref_idx]], dtype=np.float64)
    for key in XPT_REFERENCE_SPECS:
        out[f"reference_{key}"] = np.asarray(reference[key][ref_idx], dtype=np.float64)
    out["failure"] = np.array([1 if raw.get("failure", False) else 0], dtype=np.uint8)
    return out


class HFMSimulator(gym.Env):
    """Gymnasium environment for HFM."""

    def __init__(
        self,
        config: dict[str, Any],
        reward_fn: Callable[[dict, np.ndarray], float] | None = None,
    ):
        super().__init__()
        self.config = config
        self.reward_fn = reward_fn
        predictor_config = config.get("predictor", config)
        self.engine = HFMSocketPredictor(config=predictor_config)

        self.max_steps = int(config.get("max_steps", 100))
        self.step_count = 0
        self.shot_id = predictor_config.get("shot_id", "13844")

        reference_cfg = config.get("reference", {})
        self.reference_mode_default = reference_cfg.get("mode", "hold")
        self.reference: dict[str, np.ndarray] | None = None

        action_low, action_high = _default_action_bounds()
        cfg_low = config.get("action_low")
        cfg_high = config.get("action_high")
        if cfg_low is not None:
            action_low = np.asarray(cfg_low, dtype=np.float32)
        if cfg_high is not None:
            action_high = np.asarray(cfg_high, dtype=np.float32)

        self.action_space = gym.spaces.Box(
            low=action_low, high=action_high, shape=(12,), dtype=np.float32
        )
        self.observation_space = _build_observation_space()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        reset_params = options.get("reset_params", {})
        reference_mode = options.get("reference_mode", self.reference_mode_default)
        reference_spec = options.get("reference")
        if (
            reference_spec is None
            and reference_mode == "trajectory"
            and self.shot_id in CASE_SHOT_IDS
        ):
            reference_spec = build_case_reference(self.shot_id, self.max_steps)

        signeo = reset_params.get("signeo")
        bp = reset_params.get("bp")
        q0 = reset_params.get("q0")

        raw = self.engine.reset(signeo=signeo, bp=bp, q0=q0)
        self.step_count = 0
        self.reference = _build_reference_trajectory(
            reference_mode=reference_mode,
            reference_spec=reference_spec,
            raw=raw,
            max_steps=self.max_steps,
        )

        obs = _obs_dict_from_raw(
            raw=raw,
            reference=self.reference,
            reference_index=self.step_count,
        )
        info = {
            "reset_params": reset_params,
            "shot_id": self.shot_id,
            "reference_mode": reference_mode,
            "reference_keys": list(REFERENCE_KEYS) + list(XPT_REFERENCE_SPECS),
            "reference_length": self.max_steps,
            "reference_trajectory": self.reference,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,):
            action = np.broadcast_to(action, (12,)).copy()

        raw = self.engine.step(action)
        self.step_count += 1

        if self.reference is None:
            raise RuntimeError("Environment must be reset before calling step().")

        obs = _obs_dict_from_raw(
            raw=raw,
            reference=self.reference,
            reference_index=self.step_count,
        )
        terminated = bool(raw.get("failure", False))
        truncated = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "failure": terminated,
            "shot_id": self.shot_id,
            "reference_keys": list(REFERENCE_KEYS) + list(XPT_REFERENCE_SPECS),
            "reference_index": min(self.step_count, self.max_steps - 1),
            "reference_length": self.max_steps,
            "raw_observation": raw,
        }

        reward = 0.0
        if self.reward_fn is not None:
            reward = float(
                self.reward_fn(
                    obs,
                    action,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
            )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.engine.close()
