"""Candado de la preinscripción µ/η/Σ (canal Cronos): campos congelados
fijados, prohibición ejecutable y artefacto (si existe) consistente con
la regla preinscrita.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-13_mu_eta_cronos"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "mu_eta_cronos.json"


def _prereg():
    assert PREREG.exists(), "preinscripción µ/η ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_parameters_pinned():
    doc = _prereg()
    par = doc["parameters_from_treatise_only"]
    assert par["alpha0_inv"]["value"] == 1e-6
    assert "límite superior" in par["alpha0_inv"]["role"]
    assert par["rho_c_over_mean"] == 200.0
    assert par["epsilon_K"]["value"] == pytest.approx(0.012)
    tr = par["background_theta_ref"]
    assert tr["H0"] == pytest.approx(67.87, abs=0.05)
    assert tr["Omega_m"] == pytest.approx(0.3263, abs=0.002)
    assert len(tr["chains_sha256"]) == 64
    cl = doc["closures_declared"]
    assert cl["2_rho_c"]["modes"] == ["comoving", "physical"]
    assert cl["3_validity_window"]["k_max_hMpc"] == 0.2
    assert doc["grids"]["k_hMpc"] == [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
    assert doc["grids"]["z"] == [0.0, 0.5, 1.0, 2.0, 3.0]


def test_expectations_and_prohibitions_declared():
    doc = _prereg()
    ex = doc["expectations"]
    e1 = ex["E1_boring_consistency_control"]
    assert "NO es fracaso" in e1["statement"]
    assert "≤ 0.001" in e1["rule"]
    assert e1["scope"] == "cierre 'comoving' únicamente"
    assert len(ex["E2_distinctive_signature_identities"]["rules"]) == 6
    assert "NO EJECUTADA" in ex["E3_out_of_sample_chain"]["status"]
    assert "lensing" in doc["prohibitions"]["no_mu_eta_from_lensing"]
    assert "PENDIENTE (frente 3)" in doc["atlas_channel"]["status"]
    assert len(doc["note_estimates_to_verify"]["rows"]) == 3
    assert len(doc["what_this_does_not_claim"]) == 4


def test_artifact_if_present_cites_prereg_and_matches_rules():
    if not ARTIFACT.exists():
        pytest.skip("cómputo µ/η aún no ejecutado")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()).hexdigest()
    assert doc["alpha0_inv_used"] == 1e-6
    assert doc["E2_pass"] is True
    e1 = doc["fsigma8_ratio_E1"]
    assert e1["E1_pass_comoving"] == (e1["comoving_max_abs"]
                                     <= e1["threshold"])
    assert doc["atlas_channel"]["contribution_computed"] is False
    # cada estimación de la nota lleva su cifra calculada y su cociente
    for row in doc["note_table_verification"]:
        for key in ("z0_comoving", "z1_physical", "z3_physical"):
            assert "computed" in row[key]
            assert "ratio_computed_over_note" in row[key]
