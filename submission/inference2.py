"""Submission inference for policy2 (F2a/F2b: to XPT)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
ENVIRONMENT_DIR = PROJECT_ROOT / "environment"
if str(ENVIRONMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_DIR))

_preprocessing = importlib.import_module("preprocessing")
action_7d_to_12d = _preprocessing.action_7d_to_12d
flatten_dict_observation = _preprocessing.flatten_dict_observation

MODEL_DIR = ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "policy2.onnx"
ACTION_DIM = 12
ACTION_DIM_7 = 7


class Policy:
    """Policy interface used by service2."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        self.session = None
        self.input_name = None
        self.output_name = None
        self._try_load_default_model()

    def _try_load_default_model(self) -> None:
        model_path = self._resolve_model_path()
        if model_path is None:
            return
        try:
            import onnxruntime as ort  # noqa: PLC0415

            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception:
            self.session = None
            self.input_name = None
            self.output_name = None

    def _resolve_model_path(self) -> Path | None:
        if DEFAULT_MODEL_PATH.exists():
            return DEFAULT_MODEL_PATH
        fallback = MODEL_DIR / "policy.onnx"
        if fallback.exists():
            return fallback
        candidates = sorted(self.model_dir.glob("*.onnx"))
        return candidates[0] if candidates else None

    def reset(self) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self.session is None or self.input_name is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)

        flat = flatten_dict_observation(observation).reshape(1, -1)
        try:
            action = self.session.run([self.output_name], {self.input_name: flat})[0]
        except Exception:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size == ACTION_DIM_7:
            action = action_7d_to_12d(action)
        if action.size != ACTION_DIM:
            raise ValueError(f"Policy must return 12D or 7D action, got size {action.size}")
        return action.astype(np.float32)
