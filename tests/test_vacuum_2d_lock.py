"""Candado y pruebas del vacío 2D (brazos conversion_current y
tilt_rotation en el reloj S; scripts/run_vacuum_2d.py): la corriente de
conversión crea φ_E y su trabajo entra en la identidad; la rotación de la
inclinación respeta f de referencia y publica W_tilt; el analizador lee
las reglas del JSON y falla cerrado; la preinscripción, si existe, tiene
los campos congelados y las corridas citan su sha."""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core.s_clock import DECLARED_FORMS, ClockConfig, SClock, diagonal_crossing_S

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-22_vacuum_2d"
PREREG = OUT / "preregistration.json"


def test_conversion_current_creates_space_and_keeps_identity():
    base = SClock(ClockConfig(delta0=0.01)).run()
    conv = SClock(ClockConfig(delta0=0.01, kappa_conv=0.5)).run()
    fb, fc = base["checks"]["diagonal"]["flow"], conv["checks"]["diagonal"]["flow"]
    assert fc["theta_max"] > fb["theta_max"] + 0.1                      # la corriente mueve θ hacia la diagonal
    v2 = conv["checks"]["diagonal"]["vacuum_2d"]
    assert v2["kappa_conv"] == 0.5 and v2["kappa"] > 0.0 and v2["W_conv_over_T0"] != 0.0
    assert conv["checks"]["descent"]["S_equals_f_identity"]["max_abs_diff"] < 1e-9   # S = f − f₀ + (W_J + W_conv)/T₀
    assert conv["checks"]["descent"]["monotonia_4_5"]["pass"]
    assert "mp_ep_panel" in fc and len(fc["mp_ep_panel"]["cos2_theta"]) > 10
    assert "vacuum_2d" in DECLARED_FORMS and "κ = κ̂·ρ₊/T₀" in DECLARED_FORMS["vacuum_2d"]
    with pytest.raises(ValueError):
        SClock(ClockConfig(delta0=0.01, kappa_conv=-1.0))


def test_tilt_rotation_publishes_W_tilt_and_stops_on_effective_landscape():
    r = diagonal_crossing_S(0.01, 0.1, tilt_cross_f=0.5)
    assert r["finished"] and r["W_tilt_over_T0"] != 0.0
    assert r["S_equals_f_max_diff"] < 1e-6                              # identidad con W_tilt incluido
    assert r["theta_max"] > 0.5                                         # el polo de espacio pasa a ser el mínimo tras f_×
    with pytest.raises(ValueError):
        SClock(ClockConfig(delta0=0.01, tilt_cross_f=1.5))
    fixed = SClock(ClockConfig(delta0=0.01)).run()["checks"]["diagonal"]["vacuum_2d"]
    assert fixed["W_tilt_over_T0"] == 0.0 and fixed["eta_eff_final"] == fixed["eta0"]


def test_theta_max_drop_is_zero_for_monotone_and_positive_otherwise():
    conv = SClock(ClockConfig(delta0=0.01, kappa_conv=1.0)).run()["checks"]["diagonal"]["flow"]
    assert conv["theta_max_drop"] > 0.1                                 # el piloto: θ sube y vuelve al polo de masa
    th = np.array([0.0, 0.1, 0.2, 0.15, 0.3])
    assert np.max(np.maximum.accumulate(th) - th) == pytest.approx(0.05)


def test_analyzer_reads_rules_from_json_and_fails_closed(tmp_path):
    body = (REPO / "scripts" / "run_vacuum_2d.py").read_text(encoding="utf-8").split("def _letter_for_arm(")[1]
    assert 'R["S_window"]' in body and 'R["width_ratio_A"]' in body and 'R["tol_theta_drop_rad"]' in body
    assert re.search(r">=\s*1\.2\b", body) is None and re.search(r"0\.95\s*<=", body) is None
    code = ("import sys, pathlib; sys.path.insert(0, %r); import importlib.util as u; "
            "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "m.PREREG = pathlib.Path(%r) / 'nope.json'; m.cmd_analyze(None)"
            % (str(REPO), str(REPO / "scripts" / "run_vacuum_2d.py"), str(tmp_path)))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert p.returncode != 0 and "FALLO CERRADO" in (p.stderr + p.stdout)


@pytest.mark.skipif(not PREREG.exists(), reason="preinscripción del vacío 2D no congelada")
def test_prereg_frozen_fields_and_runs_cite_sha():
    d = json.loads(PREREG.read_text(encoding="utf-8"))
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert sha in (OUT / "preregistration.md").read_text(encoding="utf-8")
    R = d["rules"]
    assert R["S_window"] == [0.95, 1.05] and R["width_ratio_A"] == 1.2 and R["tol_theta_drop_rad"] == 0.01 and R["tol_identity"] == 1e-6
    assert d["declared"]["J_push_arm_i"] == 0.1 and len(d["declared"]["kappa_hat_grid"]) == 25 and len(d["declared"]["f_cross_grid"]) == 24
    assert set(d["declared"]["delta0_grid"]) == {"0.003", "0.01", "0.03", "delta_H_full"}
    assert d["prohibitions"]["J_circulation_results_untouched"] is True
    runs = OUT / "runs"
    if runs.exists():
        for p in runs.glob("*.json"):
            assert json.loads(p.read_text(encoding="utf-8"))["preregistration_sha256"] == sha, p.name
