"""Candado de la preinscripción del sector perturbativo: los campos
congelados quedan fijados y el artefacto (si existe) la cita exacta.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.growth_prediction import Z_BAND

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-12_perturbations_fsigma8"
PREREG = OUT / "preregistration.json"
PREDICTION = OUT / "prediction_band.json"


def test_prereg_exists_and_pins_method():
    assert PREREG.exists(), "preinscripción ausente"
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    mf = doc["method_frozen"]
    assert mf["n_draws"] == 1000
    assert mf["draw_seed"] == 20260912
    assert mf["sigma8_external"]["value"] == pytest.approx(0.805)
    assert len(mf["chains_sha256"]) == 64
    assert "NO la aproximación γ de Linder" in mf["growth"]
    assert "SOLO CC+BAO+SNe" in mf["posterior"]


def test_prereg_chains_sha_matches_tree():
    """Las cadenas congeladas SON las del árbol: cambiarlas sin nueva
    preinscripción rompe la suite."""
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    chains = (REPO / "results" / "2026-08-10_production_fit"
              / "chains_mcmc.npz")
    assert doc["method_frozen"]["chains_sha256"] == \
        hashlib.sha256(chains.read_bytes()).hexdigest()


def test_prereg_declares_boring_outcome_and_prohibition():
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert "NO fracaso" in doc["expected_outcome_A"]["statement"]
    assert "Δχ²" in doc["expected_outcome_A"]["classification_rule"]
    assert "discrimination_procedure" in doc["contrary_outcome_B"]
    epi = doc["epistemic_order_declared_not_executed"]
    assert "lensing" in epi["prohibition"]
    assert "ε_c/Atlas" in epi["order"]
    assert "publication_commitment" in doc


def test_prediction_if_present_cites_prereg_and_matches_rule():
    if not PREDICTION.exists():
        pytest.skip("banda aún no computada")
    doc = json.loads(PREDICTION.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()).hexdigest()
    assert doc["outcome"] in ("A", "B")
    assert len(doc["band"]["ratio_p50"]) == len(Z_BAND)
    # la clasificación publicada es la de la regla congelada:
    expected = (doc["ratio_within_prior_envelope"]
                and abs(doc["out_of_sample"]["delta_chi2"]) <= 2.0)
    assert doc["outcome"] == ("A" if expected else "B")
