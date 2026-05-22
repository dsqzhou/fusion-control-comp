"""
并行 PPO 示例：一般偏滤器(13844_500) -> XPT(13906_500)。

目标参考复赛 F-2a/F-2b 的 XPT 评分项：
- Ip: 0~100 ms 从 500 kA 线性降到 400 kA，之后维持 400 kA
- 形状: 0~100 ms 保持初始位形，之后逐步过渡到 13906_500 XPT 目标
- XPT: 2 个主 X 点、2 个次级 X 点、8 个打击点、X 点磁通接近 FB

默认使用 24 个已有 HFM 容器端口 2223-2246 训练，2247 提取目标。

示例：
  python examples/train_xpt_f2a_ppo.py --ports 2223-2246 --target-port 2247 --total-timesteps 24000
  python examples/train_xpt_f2a_ppo.py --ports 2223-2246 --target-port 2247 --total-timesteps 2000000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment import HFMSimulator, action_7d_to_12d  # noqa: E402
from environment.xpt_utils import (  # noqa: E402
    X_PRIMARY_SLOTS,
    X_SECONDARY_SLOTS,
    extract_target_strike_points,
    extract_target_xpoints,
    extract_xpt_observation_pack,
)

ACTION_LOW_NORM_7D = np.full(7, -1.0, dtype=np.float32)
ACTION_HIGH_NORM_7D = np.full(7, 1.0, dtype=np.float32)
ACTION_LOW_PHYS_7D = np.array([-1499, -230, -172, -172, -348, -348, -270], dtype=np.float32)
ACTION_HIGH_PHYS_7D = np.array([1499, 230, 172, 172, 348, 348, 270], dtype=np.float32)

SURVIVAL_REWARD = 0.08
TRACKING_REWARD_WEIGHT = 0.92
TERMINATION_PENALTY = -2.0
SMOOTH_MIN_ALPHA = -1.0
SHAPE_RAMP_START_STEP = 100


def _scalar(obs: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(np.asarray(obs.get(key, default), dtype=np.float64).ravel()[0])


def _parse_ports(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def _ip_reference(max_steps: int) -> np.ndarray:
    """F-2a/F-2b: 500 kA -> 400 kA in first 100 ms, then hold 400 kA."""
    t = np.arange(max_steps, dtype=np.float64)
    return np.where(t <= 100.0, 500_000.0 - 1_000.0 * t, 400_000.0)


def _score(err: float, threshold: float) -> float:
    return float(np.clip(1.0 - err / max(threshold, 1e-12), 0.0, 1.0))


def _clip(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return low
    return float(max(low, min(high, value)))


def _scale(value: float, src_low: float, src_high: float, dst_low: float, dst_high: float) -> float:
    ratio = (value - src_low) / (src_high - src_low + 1e-12)
    return float(dst_low + ratio * (dst_high - dst_low))


def _logistic(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-_clip(value, -50.0, 50.0))))


def _soft_score(error: float, good: float, bad: float) -> float:
    high = float(np.log(19.0))
    scaled = _scale(error, bad, good, -high, high)
    return _logistic(scaled)


def _smooth_min_score(scores: list[float], weights: list[float], alpha: float = SMOOTH_MIN_ALPHA) -> float:
    values = np.asarray(scores, dtype=np.float64)
    weight_arr = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not np.any(valid):
        return 0.0
    values = values[valid]
    weight_arr = weight_arr[valid]
    logits = np.log(weight_arr) + alpha * values
    logits -= np.max(logits)
    soft_weights = np.exp(logits)
    soft_weights /= np.sum(soft_weights)
    return float(np.sum(values * soft_weights))


@dataclass
class XptTarget:
    rX: np.ndarray
    zX: np.ndarray
    rS: np.ndarray
    zS: np.ndarray
    lcfs: np.ndarray
    R: float
    Z: float
    Ip0: float
    FB: float
    FA: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "rX": self.rX.tolist(),
            "zX": self.zX.tolist(),
            "rS": self.rS.tolist(),
            "zS": self.zS.tolist(),
            "lcfs": self.lcfs.tolist(),
            "R": self.R,
            "Z": self.Z,
            "Ip0": self.Ip0,
            "FB": self.FB,
            "FA": self.FA,
        }


@dataclass
class ShapeReference:
    lcfs: np.ndarray
    R: float
    Z: float


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_for_port(
    base: dict[str, Any],
    *,
    shot_id: str,
    port: int,
    max_steps: int,
    timeout: float | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["max_steps"] = int(max_steps)
    cfg.setdefault("predictor", {})
    cfg["predictor"]["shot_id"] = shot_id
    cfg["predictor"]["port"] = int(port)
    if timeout is not None:
        cfg["predictor"]["timeout"] = float(timeout)
    cfg.setdefault("reference", {})
    cfg["reference"]["mode"] = "trajectory"
    return cfg


def extract_shape_reference(
    base_cfg: dict[str, Any],
    *,
    shot_id: str,
    port: int,
    max_steps: int,
    timeout: float | None,
) -> ShapeReference:
    cfg = config_for_port(base_cfg, shot_id=shot_id, port=port, max_steps=max_steps, timeout=timeout)
    env = HFMSimulator(cfg)
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()
    return ShapeReference(
        lcfs=np.asarray(obs["lcfs_points"], dtype=np.float64),
        R=_scalar(obs, "R"),
        Z=_scalar(obs, "Z"),
    )


def extract_xpt_target(base_cfg: dict[str, Any], *, target_port: int, max_steps: int, timeout: float | None) -> XptTarget:
    cfg = config_for_port(base_cfg, shot_id="13906_500", port=target_port, max_steps=max_steps, timeout=timeout)
    env = HFMSimulator(cfg)
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()

    rX, zX, _ = extract_target_xpoints(obs, slots=4)
    rS, zS, _, _ = extract_target_strike_points(obs, strike_slots=8)
    target = XptTarget(
        rX=np.asarray(rX, dtype=np.float64),
        zX=np.asarray(zX, dtype=np.float64),
        rS=np.asarray(rS, dtype=np.float64),
        zS=np.asarray(zS, dtype=np.float64),
        lcfs=np.asarray(obs["lcfs_points"], dtype=np.float64),
        R=_scalar(obs, "R"),
        Z=_scalar(obs, "Z"),
        Ip0=_scalar(obs, "Ip"),
        FB=_scalar(obs, "FB"),
        FA=_scalar(obs, "FA"),
    )
    return target


def _shape_ramp(max_steps: int, ramp_start: int = SHAPE_RAMP_START_STEP) -> np.ndarray:
    if max_steps <= ramp_start:
        return np.zeros((max_steps,), dtype=np.float64)
    t = np.arange(max_steps, dtype=np.float64)
    denom = max(float(max_steps - 1 - ramp_start), 1.0)
    return np.clip((t - float(ramp_start)) / denom, 0.0, 1.0)


def build_shape_reference_trajectory(
    start: ShapeReference,
    target: XptTarget,
    max_steps: int,
) -> dict[str, Any]:
    ramp = _shape_ramp(max_steps)
    r_ref = (1.0 - ramp) * start.R + ramp * target.R
    z_ref = (1.0 - ramp) * start.Z + ramp * target.Z
    lcfs_ref = (1.0 - ramp[:, None, None]) * start.lcfs[None, :, :] + ramp[:, None, None] * target.lcfs[None, :, :]
    return {
        "Ip": _ip_reference(max_steps).tolist(),
        "R": r_ref.tolist(),
        "Z": z_ref.tolist(),
        "lcfs_points": lcfs_ref.tolist(),
    }


class F2XptReward:
    """近似复赛 F-2a/F-2b 单步评分的 dense reward。"""

    def __init__(self, target: XptTarget):
        self.target = target
        self.curriculum_progress = 0.0

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def __call__(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> float:
        if terminated:
            return TERMINATION_PENALTY

        ref_ip = _scalar(observation, "reference_Ip", 400_000.0)
        ip = _scalar(observation, "Ip", 0.0)
        ip_abs_err = abs(ip - ref_ip)

        ip_err = ip_abs_err / max(abs(ref_ip), 1.0)
        ip_score = _soft_score(ip_err, 0.005, 0.05)

        r_error = _scalar(observation, "R") - _scalar(observation, "reference_R", _scalar(observation, "R"))
        z_error = _scalar(observation, "Z") - _scalar(observation, "reference_Z", _scalar(observation, "Z"))
        position_cm = float(np.hypot(r_error, z_error) * 100.0)
        position_score = _soft_score(position_cm, 0.5, 3.0)

        lcfs = np.asarray(observation.get("lcfs_points", np.zeros((32, 2))), dtype=np.float64)
        reference_lcfs = np.asarray(observation.get("reference_lcfs_points", self.target.lcfs), dtype=np.float64)
        n = min(len(lcfs), len(reference_lcfs))
        lcfs_cm = float(np.sqrt(np.mean(np.sum((lcfs[:n] - reference_lcfs[:n]) ** 2, axis=1))) * 100.0)
        lcfs_score = _soft_score(lcfs_cm, 0.5, 3.0)

        pack = extract_xpt_observation_pack(
            observation,
            self.target.rX,
            self.target.zX,
            self.target.rS,
            self.target.zS,
            fx_order="C",
        )

        xpt_ok = pack["nX"] == 4
        if xpt_ok:
            primary = list(X_PRIMARY_SLOTS)
            secondary = list(X_SECONDARY_SLOTS)
            dist_x_cm = np.sqrt((pack["x_r"] - self.target.rX) ** 2 + (pack["x_z"] - self.target.zX) ** 2) * 100.0
            x_score = _soft_score(float(np.sqrt(np.mean(dist_x_cm[primary] ** 2))), 0.5, 3.0)
            x2_score = _soft_score(float(np.sqrt(np.mean(dist_x_cm[secondary] ** 2))), 1.0, 5.0)

            denom = abs(_scalar(observation, "FB") - _scalar(observation, "FA"))
            denom = max(denom, 1e-8)
            flux_norm = pack["x_flux_diff"] / denom
            x_flux_score = _soft_score(float(np.sqrt(np.mean(flux_norm[primary] ** 2))), 0.005, 0.05)
            x2_flux_score = _soft_score(float(np.sqrt(np.mean(flux_norm[secondary] ** 2))), 0.01, 0.10)

            strike_valid = pack["strike_valid"] > 0.5
            if np.sum(strike_valid) >= 1:
                strike_cm = np.sqrt((pack["strike_r"] - self.target.rS) ** 2 + (pack["strike_z"] - self.target.zS) ** 2) * 100.0
                strike_score = _soft_score(float(np.sqrt(np.mean(strike_cm[strike_valid] ** 2))), 0.5, 3.0)
            else:
                strike_score = 0.0
        else:
            # 评分里 nX!=4 时 XPT 专属指标为 0；训练 reward 仍给很小的可学习信号。
            x_score = x2_score = x_flux_score = x2_flux_score = strike_score = 0.02

        # 课程学习：先把变 Ip 和粗位形参考练稳，XPT 拓扑项一直纳入输入/奖励，但早期权重更小。
        xpt_weight_scale = 0.1 + 0.9 * self.curriculum_progress
        tracking_score = _smooth_min_score(
            [ip_score, position_score, lcfs_score, x_score, strike_score, x_flux_score, x2_score, x2_flux_score],
            [9.0, 3.0, 7.0, 4.0 * xpt_weight_scale, 4.0 * xpt_weight_scale, 2.0 * xpt_weight_scale, 2.0 * xpt_weight_scale, 2.0 * xpt_weight_scale],
        )
        reward = SURVIVAL_REWARD + TRACKING_REWARD_WEIGHT * tracking_score

        if ip_abs_err > 50_000.0:
            reward -= 0.5

        ipf = np.asarray(observation.get("I_PF", np.zeros(12)), dtype=np.float64).ravel()
        limits = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)
        if ipf.size == 12 and np.any(np.abs(ipf) > limits):
            reward -= 1.0

        action = np.asarray(action, dtype=np.float64).ravel()
        reward -= 0.002 * float(np.mean((action / 500.0) ** 2))
        return float(reward)


class XptFeatureWrapper(gym.Wrapper):
    """把 dict obs 转成适合 MLP 的 F-2 XPT 特征。"""

    def __init__(self, env: gym.Env, target: XptTarget):
        super().__init__(env)
        self.target = target
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(132,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._features(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._features(obs), reward, terminated, truncated, info

    def set_curriculum_progress(self, progress: float) -> None:
        current = self.env
        while True:
            reward_fn = getattr(current, "reward_fn", None)
            if hasattr(reward_fn, "set_curriculum_progress"):
                reward_fn.set_curriculum_progress(progress)
                return
            if not hasattr(current, "env"):
                return
            current = current.env

    def _features(self, obs: dict[str, Any]) -> np.ndarray:
        ref_ip = _scalar(obs, "reference_Ip", 400_000.0)
        ref_r = _scalar(obs, "reference_R", self.target.R)
        ref_z = _scalar(obs, "reference_Z", self.target.Z)
        base = np.array(
            [
                (_scalar(obs, "Ip") - ref_ip) / 50_000.0,
                (_scalar(obs, "R") - ref_r) / 0.05,
                (_scalar(obs, "Z") - ref_z) / 0.05,
                (_scalar(obs, "FB") - self.target.FB) / max(abs(self.target.FB), 1e-6),
                (_scalar(obs, "FA") - self.target.FA) / max(abs(self.target.FA), 1e-6),
            ],
            dtype=np.float64,
        )

        lcfs = np.asarray(obs.get("lcfs_points", np.zeros((32, 2))), dtype=np.float64)
        reference_lcfs = np.asarray(obs.get("reference_lcfs_points", self.target.lcfs), dtype=np.float64)
        lcfs_diff = ((lcfs[:32] - reference_lcfs[:32]) / 0.03).reshape(-1)

        ipf = np.asarray(obs.get("I_PF", np.zeros(12)), dtype=np.float64).ravel()
        ipf_limits = np.asarray([45_000.0] + [14_000.0] * 10 + [4_000.0], dtype=np.float64)
        ipf_feat = ipf[:12] / ipf_limits

        pack = extract_xpt_observation_pack(
            obs,
            self.target.rX,
            self.target.zX,
            self.target.rS,
            self.target.zS,
            fx_order="C",
        )
        denom = max(abs(_scalar(obs, "FB") - _scalar(obs, "FA")), 1e-8)
        x_feat = np.concatenate(
            [
                [pack["nX"] / 4.0],
                pack["x_valid"],
                (pack["x_r"] - self.target.rX) / 0.05,
                (pack["x_z"] - self.target.zX) / 0.05,
                pack["x_flux_diff"] / denom,
                pack["x_dpsi_dr"] / 0.1,
                pack["x_dpsi_dz"] / 0.1,
                pack["strike_valid"],
                (pack["strike_r"] - self.target.rS) / 0.05,
                (pack["strike_z"] - self.target.zS) / 0.05,
                [pack["strike_n_use"] / 8.0, pack["strike_n_actual"] / 8.0],
            ]
        )
        return np.concatenate([base, lcfs_diff, ipf_feat, x_feat]).astype(np.float32)


class Normalized7DTo12DActionWrapper(gym.ActionWrapper):
    """参考 use_7d_action.py / first_test：策略输出 [-1,1]^7，再映射为物理 12D 电压。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=ACTION_LOW_NORM_7D,
            high=ACTION_HIGH_NORM_7D,
            shape=(7,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, ACTION_LOW_NORM_7D, ACTION_HIGH_NORM_7D)
        norm = (action - ACTION_LOW_NORM_7D) / (ACTION_HIGH_NORM_7D - ACTION_LOW_NORM_7D)
        action_phys_7d = norm * (ACTION_HIGH_PHYS_7D - ACTION_LOW_PHYS_7D) + ACTION_LOW_PHYS_7D
        return action_7d_to_12d(action_phys_7d)


def make_env(
    *,
    base_cfg: dict[str, Any],
    port: int,
    start_reference: ShapeReference,
    target: XptTarget,
    max_steps: int,
    timeout: float | None,
    use_12d_action: bool,
):
    def _init():
        cfg = config_for_port(base_cfg, shot_id="13844_500", port=port, max_steps=max_steps, timeout=timeout)
        cfg["reference"] = {
            "mode": "trajectory",
            "reference_keys": ["Ip", "R", "Z", "lcfs_points"],
            "reference": build_shape_reference_trajectory(start_reference, target, max_steps),
        }
        env: gym.Env = HFMSimulator(cfg, reward_fn=F2XptReward(target))
        if not use_12d_action:
            env = Normalized7DTo12DActionWrapper(env)
        env = XptFeatureWrapper(env, target)
        return env

    return _init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "env_default.yaml")
    parser.add_argument("--ports", type=str, default="2223-2246")
    parser.add_argument("--target-port", type=int, default=2247)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--hfm-timeout", type=float, default=300.0)
    parser.add_argument("--total-timesteps", type=int, default=24_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Global timesteps between checkpoints")
    parser.add_argument("--save-dir", type=Path, default=ROOT / "results" / "xpt_f2a_ppo")
    parser.add_argument("--use-12d-action", action="store_true", help="Debug only; default uses normalized 7D action.")
    parser.add_argument("--curriculum-timesteps", type=int, default=100_000)
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging if installed")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ImportError:
        print("缺少依赖：pip install stable-baselines3 torch")
        return 1

    base_cfg = load_config(args.config)
    ports = _parse_ports(args.ports)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] reset 13844_500 on port {args.target_port}")
    start_reference = extract_shape_reference(
        base_cfg,
        shot_id="13844_500",
        port=args.target_port,
        max_steps=args.max_steps,
        timeout=args.hfm_timeout,
    )

    print(f"[target] reset 13906_500 on port {args.target_port}")
    target = extract_xpt_target(
        base_cfg,
        target_port=args.target_port,
        max_steps=args.max_steps,
        timeout=args.hfm_timeout,
    )
    with open(args.save_dir / "xpt_target_13906_500.json", "w", encoding="utf-8") as f:
        json.dump(target.to_jsonable(), f, indent=2)
    print("target rX:", np.round(target.rX, 5).tolist())
    print("target zX:", np.round(target.zX, 5).tolist())
    print("target strike r:", np.round(target.rS, 5).tolist())
    print("target strike z:", np.round(target.zS, 5).tolist())

    print(f"[train] 13844_500 -> 13906_500, envs={len(ports)}, ports={ports}")
    env = SubprocVecEnv(
        [
            make_env(
                base_cfg=base_cfg,
                port=port,
                start_reference=start_reference,
                target=target,
                max_steps=args.max_steps,
                timeout=args.hfm_timeout,
                use_12d_action=args.use_12d_action,
            )
            for port in ports
        ],
        start_method="spawn",
    )
    env = VecMonitor(env, filename=str(args.save_dir / "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=100,
        batch_size=400,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=1,
        tensorboard_log=str(args.save_dir / "tb") if args.tensorboard else None,
        device="cpu",
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_freq) // max(1, len(ports))),
        save_path=str(args.save_dir / "checkpoints"),
        name_prefix="ppo_xpt_f2a",
    )

    class CurriculumCallback(BaseCallback):
        def __init__(self, curriculum_timesteps: int):
            super().__init__()
            self.curriculum_timesteps = max(1, int(curriculum_timesteps))
            self._last_progress = -1.0

        def _on_training_start(self) -> None:
            self.training_env.env_method("set_curriculum_progress", 0.0)

        def _on_step(self) -> bool:
            progress = float(np.clip(self.num_timesteps / self.curriculum_timesteps, 0.0, 1.0))
            if progress - self._last_progress >= 0.01 or progress >= 1.0:
                self.training_env.env_method("set_curriculum_progress", progress)
                self._last_progress = progress
            return True

    callbacks = CallbackList([checkpoint_cb, CurriculumCallback(args.curriculum_timesteps)])
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
    model.save(str(args.save_dir / "ppo_xpt_f2a_latest"))
    env.close()
    print(f"saved: {args.save_dir / 'ppo_xpt_f2a_latest.zip'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
