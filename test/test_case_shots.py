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


def test_150_hold_keeps_initial():
    ref = build_case_reference("21311_150", max_steps=20, hold_initial=True)
    assert np.allclose(ref["Ip"], 320e3)
    assert np.allclose(ref["Rmax"], 1.13)
    assert np.allclose(ref["kappa"], 1.60)


def test_400ka_references():
    ref_21743 = build_case_reference("21743_300", max_steps=10)
    ref_21710 = build_case_reference("21710_300", max_steps=10)
    ref_21710_200 = build_case_reference("21710_200", max_steps=10)
    assert np.allclose(ref_21743["Ip"], 400e3)
    assert np.allclose(ref_21743["Rmax"], 1.24)
    assert np.allclose(ref_21743["Rmin"], 0.31)
    assert np.allclose(ref_21710["Rmax"], 1.27)
    assert np.allclose(ref_21710_200["Rmax"], 1.25)
    assert "rampup" not in get_shot_psm_config_path("21743_200")
    cfg = get_fge_init_config_for_shot("21743_400")
    assert cfg["bp"] == 0.28
    assert cfg["q0"] == 1.84
    assert cfg["LX_addr"].endswith("ini_21743_400_v4.mat")


if __name__ == "__main__":
    test_case_shots_registered()
    test_150_uses_rampup_psm()
    test_timevar_omits_bp_q0()
    test_non_timevar_keeps_bp_q0()
    test_150_reference_ramps()
    test_300_reference_holds()
    test_150_hold_keeps_initial()
    test_400ka_references()
    print("test_case_shots passed.")
