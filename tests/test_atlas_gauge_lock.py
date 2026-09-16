"""Candado de E5_Atlas: preinscripción congelada (transformación de gauge,
teorema a fijar, control externo declarado, reglas, desenlaces), ejecutor
tras la barrera y artefacto (si existe) coherente con las reglas y con
el estatuto del módulo.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.mu_eta_atlas import ATLAS_STATUS

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-15_mu_eta_atlas_gauge"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "mu_eta_atlas_gauge.json"
RULES_A = {"etaN_e2_coefficient_exact_zero": True,
           "muD_closed_form_rel_tol": 0.05, "muD_closed_form_alpha_max": 0.012,
           "muD_at_bound_k002_leq": 1e-5, "tail_k_h_over_Mpc": 0.02}
RULES_B = {"etaN_abs_max": 1e-6, "muD_slope_rel_tol": 0.10,
           "unitary_match_rel_tol": 0.10, "purity_std_max": 0.01}


def _prereg():
    assert PREREG.exists(), "preinscripción E5_Atlas ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    doc = _prereg()
    gt = doc["gauge_transformation"]
    assert gt["invariants"] == {"Psi_N": "ψ + ḃ", "Phi_N": "φ − Hb", "Delta": "δ − 3H l1 (m = 1)"}
    assert gt["rules"]["psi"] == "ψ − Ṫ" and gt["rules"]["b"] == "b + T"
    assert "estrés anisótropo" in doc["theorem_to_lock"]
    a = doc["E5a_ladder"]
    assert a["rules"] == RULES_A
    assert [1.012, 1e-6] in a["points"] and [1.05, 0.3] in a["points"] and [1.0001, 1e-4] in a["points"]
    assert a["tower"] == {"nlow": -4, "nhigh": 16, "prof": 12, "reliable_n_max": 4}
    assert a["deep_identity"]["prof"] == 16 and a["deep_identity"]["n_check"] == [0, 2, 4, 6, 8]
    assert a["expected_at_bound"]["point"] == [1.012, 1e-6]
    b = doc["E5b_harness"]
    assert b["point"] == {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
    assert b["k_over_H0"] == [133.333, 200.0, 300.0, 400.0]
    assert b["a_fit_min"] == 0.3 and b["e_fit_max"] == 0.02 and b["a_start"] == 0.1
    assert b["rules"] == RULES_B and "preinscripción NUEVA" in b["note"]


def test_external_control_declared_and_outcomes():
    doc = _prereg()
    ec = doc["external_control"]
    assert "ANTES" in ec["when"] and "no fija umbrales" in ec["role"]
    assert ec["muD_minus_one_e2_rel_GB"]["(1.05, 0.3)"] == -1.396
    oa = doc["E5a_ladder"]["outcomes"]
    assert set(oa) >= {"A_closed", "B_tail_survives", "C_theorem_fails", "INDETERMINADO"}
    assert "Nunca se ajustan los umbrales" in oa["rule"]
    ob = doc["E5b_harness"]["outcomes"]
    assert set(ob) >= {"A_confirmed", "B_gauge_wrong", "C_open"}
    assert doc["prohibitions"]["no_data"] and doc["prohibitions"]["no_threshold_tuning"]
    assert doc["prohibitions"]["no_change_to_E1_E2_E3_E4_artifacts"]


def test_runner_loads_prereg_first():
    src = (REPO / "scripts" / "run_atlas_gauge.py").read_text(encoding="utf-8")
    assert src.index("load_prereg()") < src.index("build_quadratic_action()")
    assert "require_available(" not in src        # sin datos


def test_artifact_if_present_matches_rules_and_status():
    if not ARTIFACT.exists():
        pytest.skip("E5_Atlas aún no ejecutado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    a = doc["E5a"]
    r = a["rules"]
    clean = all(s["coef_etaN_e2_is_zero"] and s["etaN0_is_1"] for s in a["ladders"])
    closed = all(abs(s["ratio_muD_to_closed"] - 1) <= r["muD_closed_form_rel_tol"]
                 for s in a["ladders"] if s["alpha_a"] <= r["muD_closed_form_alpha_max"])
    bound = abs(a["at_bound"]["muD_minus_one_at_k002"]) <= r["muD_at_bound_k002_leq"]
    exp = "C" if not clean else ("A" if (closed and bound) else "B")
    assert doc["outcome_E5a"] == exp
    b = doc["E5b"]
    rb = b["rules"]
    cmD = b["ladder_coefficients"]["coef_muD_e2"]
    for row in b["rows"]:
        ok = (row["etaN_minus_one_max_abs"] <= rb["etaN_abs_max"]
              and abs(row["slope_muD_e2"] / cmD - 1) <= rb["muD_slope_rel_tol"]
              and row["p_spread_std"] <= rb["purity_std_max"] and row["solver_success"])
        assert row["pass"] == bool(ok)
    if all(row["pass"] for row in b["rows"]):
        expb = "A"
    elif all(row["etaN_matches_unitary"] for row in b["rows"]):
        expb = "B"
    else:
        expb = "C"
    assert doc["outcome_E5b"] == expb
    # el estatuto del módulo refleja lo publicado
    assert "gauge" in ATLAS_STATUS.lower()
    if exp == "A":
        assert "η_N ≡ 1" in ATLAS_STATUS or "η_N ≡ 1" in ATLAS_STATUS.replace("≡", "≡")
