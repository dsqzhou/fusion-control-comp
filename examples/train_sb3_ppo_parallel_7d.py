"""Train a parallel SB3 PPO baseline with 7D symmetric actions.

This baseline intentionally keeps the task small:
- observation: scaled tracking errors for Ip, R, Z only
- action: normalized 7D actions rescaled to physical voltages, then mapped to 12D
- vectorization: one HFM simulator socket port per SubprocVecEnv worker
- domain randomization: implemented but disabled by default

Example:
    # Start N simulator instances first, listening on consecutive ports.
    # Then train with the same start port.
    python examples/train_sb3_ppo_parallel_7d.py --num-envs 4 --start-port 5558
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
import gymnasium as gym

from environment import HFMSimulator, action_7d_to_12d

try:
    import onnx  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    HAS_TRAIN_DEPS = True
except ImportError:
    HAS_TRAIN_DEPS = False


ERROR_KEYS = ("Ip", "R", "Z")
ERROR_SCALES = np.array([1.0e6, 0.1, 0.1], dtype=np.float32)
ACTION_LOW_NORM_7D = np.full(7, -1.0, dtype=np.float32)
ACTION_HIGH_NORM_7D = np.full(7, 1.0, dtype=np.float32)
ACTION_LOW_PHYS_7D = np.array([-1499, -230, -172, -172, -348, -348, -270], dtype=np.float32)
ACTION_HIGH_PHYS_7D = np.array([100, 230, 172, 172, 348, 348, 270], dtype=np.float32)


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0


def _tracking_errors(observation: dict[str, Any]) -> np.ndarray:
    errors = []
    for key in ERROR_KEYS:
        value = _scalar(observation.get(key, 0.0))
        reference = _scalar(observation.get(f"reference_{key}", value))
        errors.append(value - reference)
    return np.asarray(errors, dtype=np.float32)


def ip_r_z_reward_fn(
    observation: dict[str, Any],
    action: np.ndarray,
    terminated: bool = False,
    truncated: bool = False,
    info: dict | None = None,
) -> float:
    """Small baseline reward: penalize scaled Ip/R/Z tracking error."""
    del action, truncated, info
    if terminated:
        return -10.0
    scaled_abs_error = np.abs(_tracking_errors(observation) / ERROR_SCALES)
    return float(-np.mean(scaled_abs_error))


class IpRZErrorObservationWrapper(gym.ObservationWrapper):
    """Expose only scaled Ip/R/Z tracking errors to the policy."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(ERROR_KEYS),),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        return (_tracking_errors(observation) / ERROR_SCALES).astype(np.float32)


class Normalized7DTo12DActionWrapper(gym.ActionWrapper):
    """Rescale normalized 7D actions to old-config physical voltages, then map to 12D."""

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


class ResetDomainRandomizationWrapper(gym.Wrapper):
    """Optional reset-parameter randomization, disabled for the first baseline."""

    def __init__(self, env: gym.Env, enabled: bool = False, seed: int = 0):
        super().__init__(env)
        self.enabled = enabled
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = copy.deepcopy(options) if options is not None else {}
        if self.enabled and "reset_params" not in options:
            options["reset_params"] = {
                "signeo": float(self.rng.uniform(4.5e6, 1.05e7)),
                "bp": float(self.rng.uniform(0.189, 0.231)),
                "q0": float(self.rng.uniform(1.35, 1.65)),
            }
        return self.env.reset(seed=seed, options=options)


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.register_buffer("action_low", torch.as_tensor(ACTION_LOW_NORM_7D, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(ACTION_HIGH_NORM_7D, dtype=torch.float32))

    def forward(self, observation):
        actions, _, _ = self.policy(observation, deterministic=True)
        return torch.clamp(actions, self.action_low, self.action_high)


def make_env(
    base_config: dict[str, Any],
    rank: int,
    start_port: int,
    seed: int,
    domain_randomization: bool,
    startup_retries: int,
    startup_retry_delay: float,
    init_stagger: float,
):
    def _init():
        if init_stagger > 0:
            time.sleep(rank * init_stagger)

        config = copy.deepcopy(base_config)
        config.setdefault("predictor", {})
        config["predictor"]["port"] = int(start_port + rank)

        env = HFMSimulator(config, reward_fn=ip_r_z_reward_fn)
        env = ResetDomainRandomizationWrapper(
            env,
            enabled=domain_randomization,
            seed=seed + rank,
        )
        env = Normalized7DTo12DActionWrapper(env)
        env = IpRZErrorObservationWrapper(env)
        env = Monitor(env)
        last_exc: Exception | None = None
        for attempt in range(1, startup_retries + 1):
            try:
                env.reset(seed=seed + rank)
                break
            except Exception as exc:
                last_exc = exc
                print(
                    f"[worker {rank}] initial reset failed "
                    f"({attempt}/{startup_retries}): {exc}",
                    file=sys.stderr,
                )
                if attempt == startup_retries:
                    env.close()
                    raise
                time.sleep(startup_retry_delay)
        if last_exc is not None:
            print(f"[worker {rank}] initial reset recovered.", file=sys.stderr)
        return env

    return _init


def _largest_divisor_at_most(value: int, limit: int) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def resolve_batch_size(n_steps: int, num_envs: int, batch_size: int) -> int:
    rollout_size = n_steps * num_envs
    if rollout_size <= 1:
        raise ValueError("n_steps * num_envs must be greater than 1 for PPO.")
    if batch_size > rollout_size:
        print(
            f"batch_size={batch_size} is larger than rollout buffer "
            f"({n_steps} * {num_envs} = {rollout_size}); using {rollout_size}."
        )
        batch_size = rollout_size
    if rollout_size % batch_size != 0:
        adjusted = _largest_divisor_at_most(rollout_size, batch_size)
        print(
            f"batch_size={batch_size} does not divide rollout buffer "
            f"{rollout_size}; using {adjusted}."
        )
        batch_size = adjusted
    return batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=root / "configs" / "env_default.yaml")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-range-vf", type=float, default=10.0)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--start-method", default="forkserver")
    parser.add_argument("--startup-retries", type=int, default=5)
    parser.add_argument("--startup-retry-delay", type=float, default=5.0)
    parser.add_argument("--init-stagger", type=float, default=0.25)
    parser.add_argument("--load-path", type=Path, default=None)
    parser.add_argument("--reset-num-timesteps", action="store_true")
    parser.add_argument("--checkpoint-timesteps", type=int, default=100_000)
    parser.add_argument("--save-dir", type=Path, default=root / "runs" / "sb3_ppo_parallel_7d")
    parser.add_argument(
        "--export-path",
        type=Path,
        default=root / "submission_sb3_ppo_7d_iprz" / "model" / "policy.onnx",
    )
    return parser.parse_args()


def main():
    if not HAS_TRAIN_DEPS:
        print("Install dependencies first: pip install stable-baselines3 onnx torch")
        return

    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    start_port = args.start_port
    if start_port is None:
        start_port = int(config.get("predictor", {}).get("port", 5558))

    batch_size = resolve_batch_size(args.n_steps, args.num_envs, args.batch_size)

    env = SubprocVecEnv(
        [
            make_env(
                base_config=config,
                rank=rank,
                start_port=start_port,
                seed=args.seed,
                domain_randomization=args.domain_randomization,
                startup_retries=args.startup_retries,
                startup_retry_delay=args.startup_retry_delay,
                init_stagger=args.init_stagger,
            )
            for rank in range(args.num_envs)
        ],
        start_method=args.start_method,
    )

    checkpoint_save_freq = max(args.checkpoint_timesteps // args.num_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_path=str(args.save_dir / "checkpoints"),
        name_prefix="ppo_7d_iprz",
    )
    print(
        f"Saving checkpoints every ~{checkpoint_save_freq * args.num_envs} "
        f"timesteps to {args.save_dir / 'checkpoints'}."
    )

    if args.load_path is not None:
        print(f"Loading model from {args.load_path}.")
        model = PPO.load(
            args.load_path,
            env=env,
            device=args.device,
            print_system_info=False,
        )
        print(f"Loaded model with num_timesteps={model.num_timesteps}.")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            policy_kwargs={"net_arch": [64, 64], "activation_fn": torch.nn.Tanh},
            verbose=1,
            device=args.device,
            seed=args.seed,
        )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=args.load_path is None or args.reset_num_timesteps,
    )

    model_path = args.save_dir / "ppo_7d_iprz_final"
    model.save(model_path)

    sample_obs = env.reset()
    sample_obs_tensor = torch.as_tensor(sample_obs, dtype=torch.float32)
    onnxable_policy = OnnxablePolicy(model.policy)
    torch.onnx.export(
        onnxable_policy,
        sample_obs_tensor,
        args.export_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    env.close()
    print(f"Saved SB3 model to {model_path}.zip")
    print(f"Saved ONNX policy to {args.export_path}")
    print("ONNX action output is normalized 7D; submission must rescale to physical 7D, then map 7D -> 12D.")


if __name__ == "__main__":
    main()
