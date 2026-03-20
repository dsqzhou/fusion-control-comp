"""Train a small SB3 PPO baseline and export ONNX for submission testing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from environment import DictObsFlattenWrapper, HFMSimulator
from examples.example_reward import example_reward_fn

try:
    import onnx  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    HAS_EXPORT_DEPS = True
except ImportError:
    HAS_EXPORT_DEPS = False


class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        actions, _, _ = self.policy(observation, deterministic=True)
        return actions


def make_env(config):
    def _init():
        env = HFMSimulator(config, reward_fn=example_reward_fn)
        env = DictObsFlattenWrapper(env)
        return env

    return _init


def main():
    if not HAS_EXPORT_DEPS:
        print("Install dependencies first: pip install stable-baselines3 onnx torch")
        return

    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "env_default.yaml"
    export_path = root / "submission" / "model" / "policy.onnx"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env = DummyVecEnv([make_env(config)])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=500)

    sample_obs = env.reset()
    sample_obs_tensor = torch.as_tensor(sample_obs, dtype=torch.float32)
    onnxable_policy = OnnxablePolicy(model.policy)

    torch.onnx.export(
        onnxable_policy,
        sample_obs_tensor,
        export_path,
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
    print(f"Saved ONNX policy to {export_path}")


if __name__ == "__main__":
    main()
