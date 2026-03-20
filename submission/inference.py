"""Submission inference template.

Participants should only need to modify this file.
The official submission service imports `Policy` from here.
"""
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
DEFAULT_MODEL_PATH = MODEL_DIR / "policy.onnx"
ACTION_DIM = 12
ACTION_DIM_7 = 7


class Policy:
    """Standard policy interface used by the official submission service.

    Required methods:
    - reset(): clear episode-level hidden state / caches
    - act(observation): return action in 12D physical voltage units
    """

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
            import onnxruntime as ort  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

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
        candidates = sorted(self.model_dir.glob("*.onnx"))
        if candidates:
            return candidates[0]
        return None

    def reset(self) -> None:
        """Clear policy state at the beginning of each episode.

        Override this if your model uses recurrent state, history windows,
        custom filters, or any episode-level cache.
        """
        return None

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a 12D physical-voltage action for the current observation."""
        if self.session is None or self.input_name is None:
            # Minimal safe fallback: zero action in physical units.
            return np.zeros(ACTION_DIM, dtype=np.float32)

        # This shared flatten helper is only a starter example, not a required rule.
        flat = flatten_dict_observation(observation).reshape(1, -1)
        try:
            action = self.session.run([self.output_name], {self.input_name: flat})[0]
        except Exception:
            # Keep the template usable even if a stale/mismatched ONNX file is present locally.
            return np.zeros(ACTION_DIM, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size == ACTION_DIM_7:
            action = action_7d_to_12d(action)
        if action.size != ACTION_DIM:
            raise ValueError(f"Policy must return 12D or 7D action, got size {action.size}")
        return action.astype(np.float32)
