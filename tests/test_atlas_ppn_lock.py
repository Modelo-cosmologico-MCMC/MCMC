"""Candado de E4_Atlas: preinscripción congelada (identidades, reglas,
desenlaces, dataset de cotas con aviso de procedencia, E4b como
preinscripción nueva), ejecutor tras la barrera y artefacto (si existe)
coherente con las reglas y con el estatuto del módulo.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.mu_eta_atlas import ATLAS_STATUS, alpha_a_max_from_ppn

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-15_mu_eta_atlas_ppn"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "mu_eta_atlas_ppn.json"
MANIFEST = REPO / "data" / "manifests" / "ppn_bounds.json"
RULES_A = {"alpha_a_max_leq": 1e-5, "tail_unobservable_leq": 1e-5,
           "tail_observable_geq": 1e-3, "residues_cleanliness_leq": 1e-3,
           "tail_k_h_over_Mpc": 0.02}
RULES_B = {"slope_rel_tol": 0.10, "purity_std_max": 0.01}


def _prereg():
    assert PREREG.exists(), "preinscripción E4_Atlas ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    doc = _prereg()
    a = doc["E4a_ppn"]
    assert len(a["identities_to_lock"]) == 10
    assert a["rules"] == RULES_A and a["lambda_nominal"] == 0.012
    assert a["tails_at_bound"]["ladder_point"] == {"lambda_K": 1.012, "alpha_a": 1e-6}
    assert a["tails_at_bound"]["k_h_over_Mpc"] == [0.02, 0.05, 0.1]
    assert a["bounds_dataset"] == "ppn_bounds" and "TRANSCRITAS" in a["provenance_caveat"]
    b = doc["E4b_e3b_closure"]
    assert b["point"] == {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
    assert b["k_over_H0"] == [66.667, 133.333, 200.0]
    assert b["rules"] == RULES_B and b["e_fit_max"] == 0.02 and b["a_start"] == 0.1
    assert "preinscripción NUEVA" in b["note"]
    assert b["expected_numbers"]["ladder_coef_eta_from_E3"] == 17.0735


def test_outcomes_prohibitions_frontiers():
    doc = _prereg()
    oa = doc["E4a_ppn"]["outcomes"]
    assert set(oa) >= {"A_focused", "B_tails_alive", "C_mapping_broken", "INDETERMINADO"}
    assert "Nunca se ajustan los umbrales" in oa["rule"]
    assert "HALLAZGO ESTRUCTURAL" in oa["B_tails_alive"]
    ob = doc["E4b_e3b_closure"]["outcomes"]
    assert set(ob) >= {"A_confirmed", "B_qs_wins", "C_open"}
    assert doc["prohibitions"]["no_data"] and doc["prohibitions"]["no_threshold_tuning"]
    fr = " ".join(doc["declared_frontiers"])
    assert "Λ_sc" in fr and "sombrero" in fr and "transcripción" in fr
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man["state"] == "AVAILABLE"
    assert any("NO verificados" in t for t in man["transformations"])


def test_runner_loads_prereg_first():
    src = (REPO / "scripts" / "run_atlas_ppn.py").read_text(encoding="utf-8")
    assert src.index("load_prereg()") < src.index("require_available(")
    assert src.index("load_prereg()") < src.index("build_quadratic_action()")


def test_artifact_if_present_matches_rules_and_status():
    if not ARTIFACT.exists():
        pytest.skip("E4_Atlas aún no ejecutado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert doc["ppn_bounds_manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    a = doc["E4a"]
    r = a["rules"]
    b = alpha_a_max_from_ppn(a["bounds_used"]["alpha1"], a["bounds_used"]["alpha2"], 1.012)
    assert a["alpha_a_bound_nominal"]["alpha_a_max"] == pytest.approx(b["alpha_a_max"], rel=1e-9)
    amax, tail, clean = (a["alpha_a_bound_nominal"]["alpha_a_max"],
                         a["eta_minus_one_at_bound_k002"], a["residues_cleanliness"])
    if not a["identities_pass"]:
        exp = "C"
    elif amax <= r["alpha_a_max_leq"] and tail <= r["tail_unobservable_leq"] \
            and clean <= r["residues_cleanliness_leq"]:
        exp = "A"
    elif amax <= r["alpha_a_max_leq"] and tail >= r["tail_observable_geq"]:
        exp = "B"
    else:
        exp = "INDETERMINADO"
    assert doc["outcome_E4a"] == exp
    e4b = doc["E4b"]
    ce, cm = e4b["ladder_coefficients"]["coef_eta_e2"], e4b["ladder_coefficients"]["coef_mu_local_e2"]
    tol = e4b["rules"]["slope_rel_tol"]
    for row in e4b["rows"]:
        ml = (abs(row["slope_eta_e2_quartic"] / ce - 1) <= tol
              and abs(row["slope_mu_local_e2_quartic"] / cm - 1) <= tol)
        assert row["pass"] == bool(ml and row["p_spread_std"] <= e4b["rules"]["purity_std_max"]
                                   and row["solver_success"])
    if all(row["pass"] for row in e4b["rows"]):
        expb = "A"
    elif all(row["match_qs"] for row in e4b["rows"]):
        expb = "B"
    else:
        expb = "C"
    assert doc["outcome_E4b"] == expb
    # el estatuto del módulo refleja lo publicado
    assert "PPN" in ATLAS_STATUS
    if expb == "A":
        assert "confirmada" in ATLAS_STATUS.lower()
