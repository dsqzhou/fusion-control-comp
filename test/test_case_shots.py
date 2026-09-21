"""Shot registry and 21311/21316 case reference tests (no socket)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from environment.case_references import CASE_SHOT_IDS, build_case_reference  # noqa: E402
from environment.shot_registry import (  # noqa: E402
    SHOT_REGISTRY,
    get_fge_init_config_for_shot,
    get_shot_psm_config_path,
)


def test_case_shots_registered():
    for shot_id in CASE_SHOT_IDS:
        assert shot_id in SHOT_REGISTRY


def test_150_uses_rampup_psm():
    assert "rampup" in get_shot_psm_config_path("21316_150")
    assert "rampup" in get_shot_psm_config_path("21311_150")
    assert "rampup" not in get_shot_psm_config_path("21316_300")


def test_timevar_omits_bp_q0():
    cfg = get_fge_init_config_for_shot("21316_timevar")
    assert "bp" not in cfg
    assert "q0" not in cfg
    assert cfg["LX_addr"].endswith("ini_21316_300_v4.1.mat")
    assert cfg["z_target"] == -0.015


def test_non_timevar_keeps_bp_q0():
    cfg = get_fge_init_config_for_shot("21316_150")
    assert cfg["bp"] == 0.04
    assert cfg["q0"] == 2.62
    assert cfg["LX_addr"].endswith("ini_21316_150_v4.mat")


def test_150_reference_ramps():
    ref = build_case_reference("21316_150", max_steps=171, dt=0.001)
    ip = np.asarray(ref["Ip"])
    rmax = np.asarray(ref["Rmax"])
    assert abs(ip[0] - 320e3) < 1.0
    assert abs(ip[150] - 500e3) < 1.0  # t=300 ms
    assert abs(rmax[0] - 1.13) < 1e-6
    assert abs(rmax[-1] - 1.26) < 1e-6
    assert np.allclose(ref["Z"], -0.015)


def test_300_reference_holds():
    ref = build_case_reference("21311_300", max_steps=20)
    assert np.allclose(ref["Ip"], 500e3)
    assert np.allclose(ref["Rmax"], 1.26)
    assert np.allclose(ref["kappa"], 1.85)


if __name__ == "__main__":
    test_case_shots_registered()
    test_150_uses_rampup_psm()
    test_timevar_omits_bp_q0()
    test_non_timevar_keeps_bp_q0()
    test_150_reference_ramps()
    test_300_reference_holds()
    print("test_case_shots passed.")
