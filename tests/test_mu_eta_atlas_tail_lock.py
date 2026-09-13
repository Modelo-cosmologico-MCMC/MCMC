"""Candado de la preinscripción E3_Atlas (cola física del canal Atlas):
campos congelados fijados, formas cerradas esperadas coherentes con el
módulo, desenlaces enumerados sin retoque de umbrales, ejecutor tras la
barrera, y artefacto (si existe) consistente con las reglas preinscritas.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.mu_eta_atlas import ATLAS_STATUS, tail_pole_residues

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-13_mu_eta_atlas_tail"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "mu_eta_atlas_tail.json"
RULES = {"E3a_residue_rel_tol": 1e-4, "E3b_slope_rel_tol": 0.10,
         "E3b_loglog_exponent_range": [1.9, 2.1], "E3b_purity_std_max": 0.01}


def _prereg():
    assert PREREG.exists(), "preinscripción E3_Atlas ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    doc = _prereg()
    assert doc["assumptions_frozen"]["xi"] == 1.0
    a = doc["E3a_ladder"]
    assert a["tower"] == {"nlow": -4, "nhigh": 16, "prof": 12}
    assert [1.05, 0.30] in a["grid_lambda_alpha"] and [1.012, 0.012] in a["grid_lambda_alpha"]
    assert a["residues"]["alphas"] == [0.10, 0.30, 0.60]
    assert a["residues"]["h_steps"] == [1 / 200, 1 / 100, 1 / 50, 1 / 25]
    b = doc["E3b_adiabatic_harness"]
    assert b["point"] == {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
    assert b["k_over_H0"] == [66.667, 133.333, 200.0]
    assert b["a_start"] == 0.1 and b["e_fit_max"] == 0.02 and b["ic_tower_nmax"] == 8
    assert doc["outcomes_enumerated"]["rules"] == RULES


def test_expected_closed_forms_match_module():
    doc = _prereg()
    ex = doc["E3a_ladder"]["residues"]["expected_numbers"]
    for al in (0.1, 0.3, 0.6):
        r = tail_pole_residues(al)
        assert ex[str(al)]["P_eta"] == pytest.approx(r["P_eta"], rel=1e-12)
        assert ex[str(al)]["P_mu_local"] == pytest.approx(r["P_mu_local"], rel=1e-12)
    hp = doc["E3a_ladder"]["expected_numbers_harness_point"]
    assert hp["author_prediction_coef_eta"] == 17.07
    assert hp["eta_qs_truncated_coefficient"] == pytest.approx(46.2, abs=0.1)


def test_outcomes_frontiers_prohibitions_declared():
    doc = _prereg()
    oc = doc["outcomes_enumerated"]
    assert "Nunca se ajustan los umbrales" in oc["rule"]
    assert set(oc["E3b"]) == {"A_confirmed", "B_qs_wins", "C_open"}
    assert "SOSPECHA DE ERROR EN LA DERIVACIÓN PRIMERO" in oc["E3b"]["B_qs_wins"]
    fr = " ".join(doc["declared_frontiers_not_derived"])
    assert "Q_η" in fr and "ξ ≠ 1" in fr and "PPN" in fr
    assert doc["prohibitions"]["no_data"] and doc["prohibitions"]["no_threshold_tuning"]
    assert "NO gobierna el desenlace" in \
        doc["E3b_adiabatic_harness"]["control_non_gating"]


def test_runner_loads_prereg_before_computing():
    src = (REPO / "scripts" / "run_mu_eta_atlas_tail.py").read_text(encoding="utf-8")
    assert src.index("load_prereg()") < src.index("build_quadratic_action()")
    assert "FALLO CERRADO" in src


def test_artifact_if_present_matches_rules_and_status():
    if not ARTIFACT.exists():
        pytest.skip("E3_Atlas aún no ejecutado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()).hexdigest()
    rules = doc["rules"]
    a = doc["E3a"]
    exp_a = all(g["leading_pass"] for g in a["grid"]) and all(
        r["P_eta_rel_err"] <= rules["E3a_residue_rel_tol"]
        and r["P_mu_rel_err"] <= rules["E3a_residue_rel_tol"] for r in a["residues"])
    assert a["pass"] == exp_a and doc["outcome_E3a"] == ("A" if exp_a else "B")
    b = doc["E3b"]
    lo, hi = rules["E3b_loglog_exponent_range"]
    ce, cm = b["ladder_coefficients"]["coef_eta_e2"], b["ladder_coefficients"]["coef_mu_local_e2"]
    for r in b["rows"]:
        ml = (abs(r["slope_eta_e2"] / ce - 1) <= rules["E3b_slope_rel_tol"]
              and abs(r["slope_mu_local_e2"] / cm - 1) <= rules["E3b_slope_rel_tol"])
        exp_pass = (ml and lo <= r["loglog_exponent_eta"] <= hi
                    and r["p_spread_std"] <= rules["E3b_purity_std_max"]
                    and r["solver_success"])
        assert r["pass"] == exp_pass
    if all(r["pass"] for r in b["rows"]):
        expected_b = "A"
    elif all(r["match_qs"] for r in b["rows"]):
        expected_b = "B"
    else:
        expected_b = "C"
    assert doc["outcome_E3b"] == expected_b
    # el estatuto del módulo debe reflejar el desenlace publicado
    assert "DERIVADO-NULO" in ATLAS_STATUS
    if expected_b == "A":
        assert "confirmad" in ATLAS_STATUS
    elif expected_b == "C":
        assert "pendiente de confirmación numérica" in ATLAS_STATUS
    # la escalera de E3a reproduce la predicción independiente del autor
    if doc["outcome_E3a"] == "A":
        assert ce == pytest.approx(17.07, abs=0.01)
