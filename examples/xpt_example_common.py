# -*- coding: utf-8 -*-
"""
XPT 示例脚本共用：从 HFMSimulator 拉取真实观测（reset + 若干随机步）。

使用前须已启动与 `configs/env_default.yaml` 中 `predictor.host` / `port` 一致的 HFM 服务。
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environment import HFMSimulator

DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "env_default.yaml"


def load_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def connection_hint(config_path: Path | str | None = None) -> str:
    cfg = load_config(config_path)
    pred = cfg.get("predictor", {})
    host = pred.get("host", "127.0.0.1")
    port = pred.get("port", "?")
    return (
        f"无法连接 HFM 服务（{host}:{port}）。\n"
        "请先在本机启动仿真器，例如：\n"
        "  cd tools && python start_simulator.py -n 1 -y\n"
        "并确认 `configs/env_default.yaml` 中 `predictor.port` 与容器端口一致。"
    )


def load_observation_from_hfmsimulator(
    config_path: Path | str | None = None,
    *,
    seed: int = 0,
    steps_after_reset: int = 1,
) -> dict:
    """
    连接环境、reset、再执行 `steps_after_reset` 次随机动作，返回**最后一步**的 observation dict。

    环境在函数内创建并关闭；适用于只读观测的示例。
    """
    cfg = load_config(config_path)
    env = HFMSimulator(cfg)
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max(0, steps_after_reset)):
            a = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        return obs
    finally:
        env.close()


def load_initial_and_current_observations(
    config_path: Path | str | None = None,
    *,
    seed: int = 0,
    steps_after_reset: int = 1,
) -> tuple[dict, dict]:
    """
    返回 `(initial_obs, current_obs)`：

    - `initial_obs`：reset 后初始平衡观测
    - `current_obs`：继续随机 step `steps_after_reset` 次后的当前观测
    """
    cfg = load_config(config_path)
    env = HFMSimulator(cfg)
    try:
        initial_obs, _ = env.reset(seed=seed)
        current_obs = initial_obs
        for _ in range(max(0, steps_after_reset)):
            a = env.action_space.sample()
            current_obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        return initial_obs, current_obs
    finally:
        env.close()

