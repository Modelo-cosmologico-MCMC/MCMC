"""Candados del benchmark DESI DR2 (6A.2/6A.4).

El test de equivalencia contra Cobaya corre solo si cobaya y su
packages path están presentes (skip declarado si no — la unit suite
no requiere red; la instalación de Cobaya está documentada en el
informe del frente).
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "results" / "2026-08-18_desi_dr2_background" / "benchmark.json"


def test_benchmark_artifact_locks():
    """CANDADO de los números publicados del benchmark: equivalencia
    PASS (1.85e-13 < 1e-10), ΛCDM Ω_m = 0.2968 a 0.08σ del oficial
    0.2975 ± 0.0086 (arXiv:2503.14738), χ²_min = 10.28 (n=13, k=2),
    puerta PASS. Si el fondo o la likelihood cambian, esto obliga a
    regenerar el benchmark."""
    doc = json.loads(BENCH.read_text(encoding="utf-8"))
    eq = doc["equivalence"]
    assert eq["status"] == "PASS"
    assert eq["max_abs_diff"] < 1e-10
    b = doc["lcdm_benchmark"]
    assert b["gate"] == "PASS"
    assert b["chi2_min"] == pytest.approx(10.284, abs=0.02)
    assert b["Omega_m"]["median"] == pytest.approx(0.2968, abs=0.002)
    assert abs(b["Omega_m"]["median"] - b["official"]["Omega_m"]) \
        < b["official"]["sigma"]
    assert b["H0rd_kms"]["median"] == pytest.approx(10167.0, abs=30.0)
    assert b["converged"] is True


def test_equivalence_against_cobaya_if_available():
    """6A.2 re-ejecutable: χ² propio ≡ Cobaya 3.6.2 sobre vectores
    sintéticos (tolerancia predeclarada 1e-10)."""
    cobaya = pytest.importorskip("cobaya")
    assert cobaya.__version__ == "3.6.2", \
        "la referencia externa de esta ronda está fijada a 3.6.2"
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from run_desi_dr2_benchmark import COBAYA_PACKAGES, equivalence_check
    if not COBAYA_PACKAGES.exists():
        pytest.skip("packages path de Cobaya ausente (instalación "
                    "manual documentada en el informe)")
    from cosmology.desi_bao import load_desi_dr2_all
    eq = equivalence_check(load_desi_dr2_all())
    assert eq["status"] == "PASS"
    assert eq["max_abs_diff"] < 1e-10
    # la logpdf de Cobaya en m = d es χ² = 0 (sin constante extra):
    assert abs(eq["logp_at_data"]) < 1e-20


def test_rd_contract_documented_and_enforced():
    """El contrato r_d: un solo H0·rd común — el módulo de ajuste no
    contiene ningún parámetro r_d separado por modelo (escaneo), y el
    docstring lo declara."""
    src = (REPO / "cosmology" / "desi_background_fit.py").read_text(
        encoding="utf-8")
    assert "CALIBRACIÓN COMÚN" in src
    assert "rd_lcdm" not in src and "rd_mcmc" not in src
    # ambos log_prob usan la MISMA log_prior_common sobre H0rd:
    assert src.count("log_prior_common(Om, H0rd)") == 2
