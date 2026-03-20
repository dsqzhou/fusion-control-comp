"""Local integration test between env and submission service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-url", default="http://127.0.0.1:18002")
    parser.add_argument("--config", default=str(ROOT / "configs" / "env_default.yaml"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--launch-service",
        choices=["none", "local", "docker"],
        default="docker",
        help="How to launch submission service before testing.",
    )
    return parser.parse_args()


def start_service(mode: str, service_url: str) -> tuple[subprocess.Popen[str] | None, str | None]:
    parsed = urlparse(service_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    if mode == "none":
        return None, None
    if mode == "local":
        env = dict(os.environ)
        env["SUBMISSION_HOST"] = host
        env["SUBMISSION_PORT"] = str(port)
        return (
            subprocess.Popen(
                [sys.executable, str(ROOT / "submission" / "service.py")],
                cwd=str(ROOT / "submission"),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            ),
            None,
        )
    if mode == "docker":
        image_name = "fusion-control-comp"
        subprocess.run(
            ["docker", "build", "-f", "submission/Dockerfile", "-t", image_name, "."],
            cwd=str(ROOT),
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
                    f"{port}:8000",
                    "-v",
                    f"{ROOT}:/saisdata/4/standalone-env:ro",
                    image_name,
                ],
                cwd=str(ROOT),
                text=True,
            )
            .strip()
        )
        subprocess.run(
            ["docker", "exec", "-d", container_id, "python3", "/app/submission/service.py"],
            check=True,
        )
        return None, container_id
    raise ValueError(f"Unsupported launch mode: {mode}")


def wait_for_health(service_url: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{service_url.rstrip('/')}/health", timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Submission service health check failed: {last_error}")


def main() -> None:
    from environment import HFMSimulator

    args = parse_args()
    process = None
    container_id = None

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base = args.service_url.rstrip("/")
    env = HFMSimulator(config)

    try:
        process, container_id = start_service(args.launch_service, args.service_url)
        health = wait_for_health(base)
        print("health:", health)

        reset_request = urllib.request.Request(
            f"{base}/reset",
            data=json.dumps(
                {"episode_id": "local-test", "options": {"reference_mode": "hold"}}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(reset_request, timeout=10) as resp:
            reset_payload = json.loads(resp.read().decode("utf-8"))
        print("reset:", reset_payload)

        obs, info = env.reset(options={"reference_mode": "hold"})
        print("env reset shot_id:", info["shot_id"])

        for step in range(args.steps):
            act_request = urllib.request.Request(
                f"{base}/act",
                data=json.dumps({"observation": {k: v.tolist() for k, v in obs.items()}}).encode(
                    "utf-8"
                ),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(act_request, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            print(f"step {step} act:", payload)
            if not payload.get("ok", False):
                raise RuntimeError(payload)
            action = payload["action"]
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                f"step {step} env: reward={reward:.4f}, terminated={terminated}, truncated={truncated}"
            )
            if terminated or truncated:
                break

        print("integration test passed")
    finally:
        env.close()
        if container_id is not None:
            subprocess.run(["docker", "stop", container_id], check=False, stdout=subprocess.DEVNULL)
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
