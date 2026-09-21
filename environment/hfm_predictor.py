#
# Copyright @2025 ENN Energy(enn.cn)
#
# HFM Socket Predictor. L_addr, LX_addr fixed at init.
# reset() only accepts signeo, bp, q0 in the public interface.
#

from typing import Any

import numpy as np

from .docker_socket_predictor import DockerSocketPredictor
from .power_supply import PowerSupplyModel
from .shot_registry import (
    SHOT_CONFIG_PATH,
    SHOT_REGISTRY,
    get_fge_init_config_for_shot,
    get_shot_psm_config_path,
)

VECTOR_OBSERVATION_LENGTHS: dict[str, int] = {
    "I_PF": 12,
    "Fx": 4290,
    "rx": 66,
    "zx": 65,
    "Bm": 164,
    "Ff": 47,
    "rX": 6,
    "zX": 6,
    "rB": 32,
    "zB": 32,
    "FX": 6,
}


class HFMSocketPredictor(DockerSocketPredictor):
    """HFM socket predictor with fixed shot files and minimal reset interface."""

    def __init__(
        self, name: str = "HFMSocketPredictor", config: dict[str, Any] | None = None
    ):
        config = config or {}
        super().__init__(name, config)
        self._power_supply = self._build_power_supply(config)

    @staticmethod
    def _build_power_supply(config: dict[str, Any]) -> PowerSupplyModel:
        ps_cfg = dict(config.get("power_supply") or {})
        psm_path = ps_cfg.get("psm_config_path")
        if psm_path is None:
            psm_path = get_shot_psm_config_path(config.get("shot_id"))
        return PowerSupplyModel(
            psm_config_path=psm_path,
            slopes=ps_cfg.get("slopes"),
            intercepts=ps_cfg.get("intercepts"),
            um_values=ps_cfg.get("um_values"),
            rate_deg=float(ps_cfg.get("rate_deg", 7.2)),
            delay_s=ps_cfg.get("delay_s"),
            use_psm=ps_cfg.get("use_psm", True),
            seed=ps_cfg.get("seed"),
        )

    def _get_init_config(self, config: dict[str, Any]) -> dict[str, Any]:
        if "fge_init_config" in config:
            fge = dict(config.get("fge_init_config", {}))
        elif "FGE_init_dict" in config:
            fge = dict(config.get("FGE_init_dict", {}))
        else:
            fge = {}

        shot_id = config.get("shot_id")
        if shot_id and shot_id in SHOT_REGISTRY:
            base = get_fge_init_config_for_shot(
                shot_id=shot_id,
                signeo=fge.get("signeo"),
                bp=fge.get("bp"),
                q0=fge.get("q0"),
            )
            base.update(fge)
            fge = base
        elif shot_id:
            raise KeyError(
                f"shot_id={shot_id!r} not in shot registry ({SHOT_CONFIG_PATH}). "
                f"Available: {sorted(SHOT_REGISTRY.keys())}"
            )

        if hasattr(fge, "to_dict"):
            fge = fge.to_dict()
        return {key: value for key, value in dict(fge).items() if value is not None}

    def _parse_observation(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        for key, size in VECTOR_OBSERVATION_LENGTHS.items():
            obs_dict[key] = np.asarray(obs_dict.get(key, [0] * size), dtype=np.float64).reshape(size)
        if "lX" not in obs_dict:
            nx = np.asarray(obs_dict.get("nX", 0.0), dtype=np.float64).reshape(-1)
            obs_dict["lX"] = float(nx[0] > 0.5) if nx.size else 0.0
        obs_dict["failure"] = obs_dict.get("is_failure", False)
        return obs_dict

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.ensure_connected()
        action = np.asarray(action, dtype=float)
        if action.size != 12:
            raise ValueError(f"action must be 12-dimensional, got shape {action.shape}")
        action = self._power_supply.step(action)
        raw = self._protocol_step(action)
        return self._parse_observation(raw)

    def reset(
        self,
        signeo: float | None = None,
        bp: float | None = None,
        q0: float | None = None,
    ) -> dict[str, Any]:
        """Reset with optional signeo, bp, q0 only. L_addr/LX_addr unchanged."""
        reset_params = {}
        if signeo is not None:
            reset_params["signeo"] = signeo
        if bp is not None:
            reset_params["bp"] = bp
        if q0 is not None:
            reset_params["q0"] = q0

        raw = super().reset(**reset_params) if reset_params else super().reset()
        self._power_supply.reset()
        return self._parse_observation(raw)

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info["action_dim"] = 12
        return info
