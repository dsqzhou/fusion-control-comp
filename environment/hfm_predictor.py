#
# Copyright @2025 ENN Energy(enn.cn)
#
# HFM Socket Predictor. L_addr, LX_addr fixed at init.
# reset() only accepts signeo, bp, q0 in the public interface.
#

from typing import Any

import numpy as np

from .docker_socket_predictor import DockerSocketPredictor
from .shot_registry import SHOT_CONFIG_PATH, SHOT_REGISTRY, get_fge_init_config_for_shot

VECTOR_OBSERVATION_LENGTHS: dict[str, int] = {
    "I_PF": 12,
    "Fx": 4290,
    "rx": 66,
    "zx": 65,
    "Bm": 100,
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
        return dict(fge)

    def _parse_observation(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        for key, size in VECTOR_OBSERVATION_LENGTHS.items():
            obs_dict[key] = np.asarray(obs_dict.get(key, [0] * size), dtype=np.float64).reshape(size)
        obs_dict["failure"] = obs_dict.get("is_failure", False)
        return obs_dict

    def _load_psm_params(self, config: dict):
        self.max_change_per_step=np.array([1499, 175, 175, 175, 175, 175, 80])*0.16
        self.psm_slopes = np.array([
            0.8580659942, 0.6072767812, 0.6072767812, 0.8035000000,
            0.8035000000, 0.5528314158, 0.5528314158, 0.7901962762,
            0.7901962762, 0.8659211512, 0.8659211512, 1.0,
        ])
        self.psm_intercepts = np.array([
            245.0691566557, -40.2756786822, -44.2045503841, -48.0000000000,
            -48.0000000000, -21.9107201644, -22.5016982305, 72.2624890457,
            73.9466700254, 47.0431299209, 51.8093723495, 0.0,
        ])

    def _power_model(self, input_voltage):
        """电源模型：修正输入电压为输出电压
        
        应用两个步骤：
        1. 速率限制（限制每步最大变化量，在PSM变换前）
        2. PSM模型变换（slope、intercept、bias）
        
        """
        # 确保输入是numpy数组，并转换为float类型以保持精度
        if not isinstance(input_voltage, np.ndarray):
            input_voltage = np.array(input_voltage, dtype=float)
        else:
            input_voltage = input_voltage.astype(float)
        
        # ============================================================
        # 第一步：应用速率限制（Rate Limiting）
        # ============================================================
        # 限制每步的最大变化量，阻止电源突变
        if self.last_command_voltage is None:
            # 第一步，没有历史数据，直接使用延迟后的指令
            limited_voltage = input_voltage.copy()
        else:
            # 计算指令电压变化量
            voltage_change = input_voltage - self.last_command_voltage
            
            # 对每个通道进行速率限制
            limited_voltage = self.last_command_voltage.copy()
            for i in range(len(input_voltage)):
                # 限制变化量
                if abs(voltage_change[i]) > self.max_change_per_step[i]:
                    # 超过限制，截断到最大变化范围
                    if voltage_change[i] > 0:
                        limited_voltage[i] = self.last_command_voltage[i] + self.max_change_per_step[i]
                    else:
                        limited_voltage[i] = self.last_command_voltage[i] - self.max_change_per_step[i]
                else:
                    # 未超过限制，使用延迟后的指令
                    limited_voltage[i] = input_voltage[i]
        
        # 更新上一步的指令电压（PSM变换前）
        self.last_command_voltage = limited_voltage.copy()
        
        # ============================================================
        # 第二步：应用PSM模型变换（slope、intercept、bias）
        # ============================================================
        
        # 创建输出电压数组（float类型）
        output_voltage = limited_voltage.copy()
        
        # 应用电源模型变换
        voltage_length = len(limited_voltage)
        
        for i in range(voltage_length):
            if i < len(self.psm_slopes):
                # 使用对应的斜率、截距和偏置进行线性变换
                output_voltage[i] = limited_voltage[i] * self.psm_slopes[i] + self.psm_intercepts[i]
            else:
                # 如果通道索引超出了配置数组范围，保持原值
                output_voltage[i] = limited_voltage[i]
        return output_voltage

    def step(self, action: np.ndarray) -> dict[str, Any]:
        self.ensure_connected()
        action = np.asarray(action, dtype=float)
        if action.size != 12:
            raise ValueError(f"action must be 12-dimensional, got shape {action.shape}")
        action = self._power_model(action)
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
        return self._parse_observation(raw)

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info["action_dim"] = 12
        return info
