"""Candado de la primera aplicación a Dovekie real: preinscripción
congelada (priors, umbrales, sampler, referencia oficial, precondición
de mocks), doble barrera en el ejecutor y artefacto consistente con la
regla preinscrita.
"""

import hashlib
import json
from pathlib import Path

import pytest

from cosmology.dovekie_real_fit import classify_outcome

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-09-13_dovekie_real"
PREREG = OUT / "preregistration.json"
ARTIFACT = OUT / "dovekie_real.json"
MOCKVAL = REPO / "results" / "2026-09-12_dovekie_mocks" / "mock_validation.json"


def _prereg():
    assert PREREG.exists(), "preinscripción Dovekie real ausente"
    return json.loads(PREREG.read_text(encoding="utf-8"))


def test_frozen_fields_pinned():
    doc = _prereg()
    pr = doc["priors_frozen"]
    assert pr["H0"]["support"] == [60.0, 80.0] and pr["H0"]["gauss"] == [67.4, 5.0]
    assert pr["Omega_m"]["support"] == [0.10, 0.50]
    assert pr["epsilon"]["support"] == [-0.05, 0.10]
    assert pr["epsilon"]["gauss"] == [0.012, 0.05]
    assert pr["z_trans"]["support"] == [1.0, 20.0]
    assert pr["k"] == {"lcdm": 2, "mcmc": 4}
    assert doc["sampler"] == {"nwalkers": 32, "nsteps": 3000, "seed": 42,
                              "burn": "nsteps//2", "thin": 4}
    rules = doc["outcomes_enumerated"]["rules"]
    assert rules == {"C_bench_sigma": 3.0, "C_chi2nu_max": 1.30,
                     "B_dBIC_pro_mcmc": 2.0, "A_dBIC_min": 0.0,
                     "A_eps_sd_min": 0.04, "A_bench_sigma": 1.0}
    d = doc["data_and_scope"]
    assert d["sn"]["n_sn"] == 1820 and d["cc"]["n"] == 31 and d["bao"]["n"] == 6
    assert d["n_total_for_BIC"] == 1857
    off = doc["official_reference"]["Omega_m_weighted"]
    assert off["mean"] == pytest.approx(0.3306, abs=0.002)
    assert off["sd"] == pytest.approx(0.0154, abs=0.002)


def test_precondition_cites_current_mock_pass():
    doc = _prereg()
    pre = doc["precondition_mock_validation"]
    assert pre["status"] == "PASS"
    assert pre["mock_validation_sha256"] == hashlib.sha256(
        MOCKVAL.read_bytes()).hexdigest()


def test_wording_roles_and_prohibitions_declared():
    doc = _prereg()
    A = doc["outcomes_enumerated"]["A_expected_boring"]
    assert "no identifican ε_Λ" in A["mandatory_wording"]
    assert "NUNCA" in A["mandatory_wording"]
    assert "SOSPECHA DE ERROR PRIMERO" in \
        doc["outcomes_enumerated"]["B_preference"]["treatment"]
    assert "PROHIBIDO reparametrizar" in \
        doc["outcomes_enumerated"]["C_tension"]["treatment"]
    joined = " ".join(doc["prohibitions"])
    assert "Unite" in joined and "NO independiente" in joined
    assert "zHD > 0" in joined


def test_runner_has_double_barrier():
    """El ejecutor pasa por load_real_data (que exige el PASS de mocks) y
    por load_prereg (que exige esta preinscripción), y cita ambos hashes."""
    src = (REPO / "scripts" / "run_dovekie_real.py").read_text(encoding="utf-8")
    assert "load_prereg()" in src and "load_real_data(" in src
    assert "mock_validation_sha256" in src and "prereg_sha256()" in src
    assert src.index("load_prereg()") < src.index("load_real_data(")


def test_artifact_if_present_matches_rule_and_cites_hashes():
    if not ARTIFACT.exists():
        pytest.skip("primera aplicación aún no ejecutada")
    doc = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert doc["preregistration_sha256"] == hashlib.sha256(
        PREREG.read_bytes()).hexdigest()
    assert doc["mock_validation_sha256"] == hashlib.sha256(
        MOCKVAL.read_bytes()).hexdigest()
    c = doc["contrast_statsys"]
    b = doc["benchmark_sn_only_lcdm"]
    expected = classify_outcome(b["delta_sigma_official"], c["chi2_nu_lcdm"],
                                c["dBIC_mcmc_minus_lcdm"], c["eps_sd"],
                                c["eps_ci95_contains_zero"], doc["rules"])
    assert doc["outcome"] == expected
    assert doc["n_points"]["total"] == 1857
    for tag in ("lcdm_statsys", "mcmc_statsys", "lcdm_statonly", "mcmc_statonly"):
        assert (OUT / f"chains_{tag}.npz").exists()
