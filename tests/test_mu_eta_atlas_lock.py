"""Candado de la preinscripción del canal Atlas (frente 3): campos
congelados fijados, fronteras declaradas, y artefacto (si existe)
consistente con las reglas E1/E2 preinscritas y con las formas cerradas.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.mu_eta_atlas import (
    ATLAS_STATUS,
    growth_index_matter_era,
    mu_atlas_subhorizon,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-13_mu_eta_atlas"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "mu_eta_atlas.json"


def _prereg():
    assert PREREG.exists(), "preinscripción Atlas ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_harness_and_thresholds_pinned():
    doc = _prereg()
    e2 = doc["E2_numerical_integration"]
    assert e2["harness_point"] == {"lambda_K": 1.05, "xi": 1.02,
                                   "alpha_a": 0.30}
    assert e2["k_over_H0"] == [66.667, 200.0, 600.0]
    assert e2["a_start"] == 1e-3
    assert e2["window_a"] == [0.3, 1.0]
    assert "≤ 0.01" in e2["rules"]["mu"]
    assert "≤ 0.01" in e2["rules"]["eta"]
    assert "≤ 0.01" in e2["rules"]["growth_index"]
    assert "≤ 0.01" in e2["rules"]["mode_purity"]
    ex = doc["E1_identities_to_lock"]["expected_numbers"]
    assert ex["mu_subhorizon_harness"] == pytest.approx(
        mu_atlas_subhorizon(1.05, 1.02, 0.30), rel=1e-12)
    assert ex["p_qs_harness"] == pytest.approx(
        growth_index_matter_era(1.05, 1.02, 0.30), rel=1e-12)
    assert len(doc["E1_identities_to_lock"]["identities"]) == 10


def test_frontiers_erratum_and_prohibitions_declared():
    doc = _prereg()
    fr = " ".join(doc["declared_frontiers_not_derived"])
    assert "1/(λ_K−1)" in fr and "superhorizonte" in fr and "PPN" in fr
    assert "H.2.2" in doc["erratum_candidate_v36_H22"]
    assert "decisión del autor" in doc["erratum_candidate_v36_H22"]
    assert "α_a/2" in doc["residues_refinement"]
    assert doc["prohibitions"]["no_data"]
    assert "B_contrary" in doc["outcomes_enumerated"]
    assert "Nunca se ajustan los umbrales" in \
        doc["outcomes_enumerated"]["B_contrary"]


def test_artifact_if_present_cites_prereg_and_matches_rules():
    if not ARTIFACT.exists():
        pytest.skip("derivación/arnés Atlas aún no ejecutados")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()).hexdigest()
    assert doc["outcome"] in ("A", "B")
    e1, e2 = doc["E1"], doc["E2"]
    assert e1["pass"] == all(e1["flags"].values())
    tol = e2["tolerances"]
    for r in e2["rows"]:
        expected = (abs(r["mu_ratio_minus_one"]) <= tol["mu"]
                    and abs(r["eta_num_minus_one"]) <= tol["eta"]
                    and abs(r["p_rel_err"]) <= tol["p"]
                    and r["p_spread_std"] <= tol["purity_std"]
                    and r["delta_sign_flips_in_window"] == 0
                    and r["solver_success"])
        assert r["pass"] == expected
    assert e2["pass"] == all(r["pass"] for r in e2["rows"])
    assert doc["outcome"] == ("A" if (e1["pass"] and e2["pass"]) else "B")
    if doc["outcome"] == "A":
        # el estatuto del módulo debe reflejar el desenlace publicado
        assert "DERIVADO-NULO" in ATLAS_STATUS
