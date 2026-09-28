#
# Piecewise-linear reference trajectories for the 21311 / 21316 training cases.
# Times are absolute shot time in milliseconds.
#

from __future__ import annotations

from typing import Any

import numpy as np

from .shot_registry import get_shot_spec

CASE_SHOT_IDS = (
    "21311_150",
    "21311_300",
    "21311_timevar",
    "21316_150",
    "21316_300",
    "21316_timevar",
    "21710_200",
    "21710_300",
    "21743_200",
    "21743_300",
    "21743_400",
)

# Shared 150 ms shape / current program. 300 ms cases hold the terminal values.
_RAMP_WAYPOINTS = {
    "Rmax": [(150.0, 1.13), (320.0, 1.26)],
    "Rmin": [(150.0, 0.27), (200.0, 0.27), (300.0, 0.29)],
    "Ip": [(150.0, 320e3), (270.0, 475e3), (300.0, 500e3)],
    "kappa": [(150.0, 1.60), (200.0, 1.60), (240.0, 1.85), (300.0, 1.85)],
}

# Flat targets. *_150 stays on the ramp unless hold_initial=True.
_HOLD_BY_SHOT = {
    "21311_150": {"Rmax": 1.13, "Rmin": 0.27, "Ip": 320e3, "kappa": 1.60},
    "21316_150": {"Rmax": 1.13, "Rmin": 0.27, "Ip": 320e3, "kappa": 1.60},
    "21311_300": {"Rmax": 1.26, "Rmin": 0.29, "Ip": 500e3, "kappa": 1.85},
    "21316_300": {"Rmax": 1.26, "Rmin": 0.29, "Ip": 500e3, "kappa": 1.85},
    "21311_timevar": {"Rmax": 1.26, "Rmin": 0.29, "Ip": 500e3, "kappa": 1.85},
    "21316_timevar": {"Rmax": 1.26, "Rmin": 0.29, "Ip": 500e3, "kappa": 1.85},
    "21743_200": {"Rmax": 1.24, "Rmin": 0.31, "Ip": 400e3, "kappa": 1.85},
    "21743_300": {"Rmax": 1.24, "Rmin": 0.31, "Ip": 400e3, "kappa": 1.85},
    "21743_400": {"Rmax": 1.24, "Rmin": 0.31, "Ip": 400e3, "kappa": 1.85},
    "21710_200": {"Rmax": 1.25, "Rmin": 0.29, "Ip": 400e3, "kappa": 1.85},
    "21710_300": {"Rmax": 1.27, "Rmin": 0.29, "Ip": 400e3, "kappa": 1.85},
}

CASE_NOTES = {
    "21311_300": "Rmax 扰动不超过 3 cm，Rmin 不超过 1.5 cm；主频率可取 3 Hz 与 30 Hz 附近。",
    "21316_300": "Rmax 扰动不超过 4 cm，Rmin 不超过 1.5 cm；主频率可取 3 Hz 与 30 Hz 附近。",
    "21311_timevar": "时变 bp/q0 写在 LX v4.1 里；INIT 不要传 bp/q0。150 ms 时变把 LX 改成 ini_21311_150_v4.1.mat 即可。",
    "21316_timevar": "时变 bp/q0 写在 LX v4.1 里；INIT 不要传 bp/q0。150 ms 时变把 LX 改成 ini_21316_150_v4.1.mat 即可。",
}


def _interp_waypoints(waypoints: list[tuple[float, float]], times_ms: np.ndarray) -> np.ndarray:
    xs = np.asarray([p[0] for p in waypoints], dtype=np.float64)
    ys = np.asarray([p[1] for p in waypoints], dtype=np.float64)
    return np.interp(times_ms, xs, ys)


def build_case_reference(
    shot_id: str,
    max_steps: int,
    dt: float = 0.001,
    hold_initial: bool = False,
) -> dict[str, Any]:
    """Build a trajectory reference dict for ``HFMSimulator.reset(options=...)``.

    ``*_150`` defaults to the ramp program. ``hold_initial=True`` keeps that
    shot's t=0 state instead, which is the 300 kA flattop-hold task.
    """
    spec = get_shot_spec(shot_id)
    t0_ms = float(spec.get("start_time_ms", 300))
    times_ms = t0_ms + np.arange(int(max_steps), dtype=np.float64) * float(dt) * 1000.0
    z = np.full(times_ms.shape, -0.015, dtype=np.float64)

    if shot_id.endswith("_150") and not hold_initial:
        series = {
            key: _interp_waypoints(points, times_ms) for key, points in _RAMP_WAYPOINTS.items()
        }
    else:
        values = _HOLD_BY_SHOT[shot_id]
        series = {
            key: np.full(times_ms.shape, float(value), dtype=np.float64)
            for key, value in values.items()
        }

    reference: dict[str, Any] = {
        "Ip": series["Ip"].tolist(),
        "Z": z.tolist(),
        "Rmax": series["Rmax"].tolist(),
        "Rmin": series["Rmin"].tolist(),
        "kappa": series["kappa"].tolist(),
    }
    return reference
