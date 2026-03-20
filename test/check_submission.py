"""Build image, run container (sleep entrypoint), exec service.py, probe /health /reset /act."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
submission_dir = root / "submission"
model_dir = submission_dir / "model"
inference_script = submission_dir / "inference.py"
service_script = submission_dir / "service.py"
dockerfile = submission_dir / "Dockerfile"
requirements = submission_dir / "requirements.txt"
image_name = "fusion-control-comp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if submission/model/ does not contain an ONNX file.",
    )
    return parser.parse_args()


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"health check failed: {last_error}")


def main():
    args = parse_args()
    if not submission_dir.is_dir():
        print("Missing submission/")
        sys.exit(1)
    if not inference_script.is_file():
        print("Missing submission/inference.py")
        sys.exit(1)
    if not service_script.is_file():
        print("Missing submission/service.py")
        sys.exit(1)
    if not dockerfile.is_file():
        print("Missing submission/Dockerfile")
        sys.exit(1)
    if not requirements.is_file():
        print("Missing submission/requirements.txt")
        sys.exit(1)
    if not model_dir.is_dir():
        print("Missing submission/model/")
        sys.exit(1)
    if args.require_model and not any(model_dir.glob("*.onnx")):
        print("Missing ONNX model in submission/model/ (expected *.onnx)")
        sys.exit(1)
    if not any(model_dir.glob("*.onnx")):
        print("Warning: no ONNX model found, fallback zero-action policy will be checked.")

    example_obs = {
        "Ip": [0.0],
        "R": [0.0],
        "Z": [0.0],
        "reference_Ip": [0.0],
        "reference_R": [0.0],
        "reference_Z": [0.0],
        "lcfs_points": [[0.0, 0.0]] * 32,
        "reference_lcfs_points": [[0.0, 0.0]] * 32,
        "I_PF": [0.0] * 12,
    }
    host_port = args.port
    base_url = f"http://127.0.0.1:{host_port}"
    container_id = None

    try:
        subprocess.run(
            ["docker", "build", "-f", "submission/Dockerfile", "-t", image_name, "."],
            cwd=str(root),
            check=True,
        )
        container_id = (
            subprocess.check_output(
                [
                    "docker",
                    "run",
                    "-d",
                    "--rm",
                    "-p",
                    f"{host_port}:8000",
                    "-v",
                    f"{root}:/saisdata/4/standalone-env:ro",
                    image_name,
                ],
                cwd=str(root),
                text=True,
            )
            .strip()
        )
        subprocess.run(
            ["docker", "exec", "-d", container_id, "python3", "/app/submission/service.py"],
            check=True,
        )

        health = _wait_for_health(base_url)
        if not health.get("ok", False):
            raise RuntimeError(f"/health returned {health}")

        reset_request = urllib.request.Request(
            f"{base_url}/reset",
            data=json.dumps({"episode_id": "check-submission"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(reset_request, timeout=5) as resp:
            reset_payload = json.loads(resp.read().decode("utf-8"))
        if not reset_payload.get("ok", False):
            raise RuntimeError(f"/reset returned {reset_payload}")

        act_request = urllib.request.Request(
            f"{base_url}/act",
            data=json.dumps({"observation": example_obs}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(act_request, timeout=5) as resp:
            act_payload = json.loads(resp.read().decode("utf-8"))
        if not act_payload.get("ok", False):
            raise RuntimeError(f"/act returned {act_payload}")
        if len(act_payload.get("action", [])) != 12:
            raise RuntimeError(f"/act must return 12D action, got {act_payload}")

        print("Submission docker build and API check passed.")
    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], check=False, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
