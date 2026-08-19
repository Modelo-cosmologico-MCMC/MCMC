"""Candados del benchmark y contraste DESI DR2 (6A.2/6A.4/6A.5).

Tras la revisión adversarial de la ronda, estos candados no solo leen
los artefactos publicados: los ATAN al código vigente recomputando
offline el χ² en cada argmin publicado (si el fondo o la likelihood
cambian, estos tests FALLAN — antes un cambio del fondo dejaba la
suite en verde), fijan el ancla oficial de la puerta como literal, y
recomputan el bloque estructural y los percentiles del prior truncado.

El test de equivalencia contra Cobaya corre solo si cobaya == 3.6.2 y
su packages path están presentes (skip declarado si no — la unit suite
no requiere red; el path se recrea con scripts/setup_cobaya_packages.py
o se relocaliza con COBAYA_PACKAGES_PATH).
"""

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "2026-08-18_desi_dr2_background"


@pytest.fixture(scope="module")
def bench():
    return json.loads((OUT / "benchmark.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contrast():
    return json.loads((OUT / "contrast.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def desi_data():
    from cosmology.desi_bao import load_desi_dr2_all
    return load_desi_dr2_all()


def test_benchmark_artifact_locks(bench):
    """CANDADO de los números publicados del benchmark (fondo
    normalizado): equivalencia PASS (1.85e-13 < 1e-10), identidad de
    datos computada 16/16, ΛCDM Ω_m = 0.2971 a 0.042σ del oficial
    0.2975 ± 0.0086 (arXiv:2503.14738 — ahora comparando la fracción
    de materia del ΛCDM plano de verdad), χ²_min = 10.284 con argmin
    interior, puerta PASS."""
    eq = bench["equivalence"]
    assert eq["status"] == "PASS"
    assert eq["max_abs_diff"] < 1e-10
    ident = eq["data_identity"]
    assert ident["status"] == "PASS"
    assert ident["n_identical"] == ident["n_files"] == 16
    assert ident["mismatches"] == []
    assert ident["packages_version"] == "v2.6"
    b = bench["lcdm_benchmark"]
    assert b["gate"] == "PASS"
    assert b["chi2_min"] == pytest.approx(10.284, abs=0.02)
    assert b["Omega_m"]["median"] == pytest.approx(0.2971, abs=0.002)
    assert b["dev_sigma"] == pytest.approx(0.042, abs=0.02)
    assert abs(b["Omega_m"]["median"] - b["official"]["Omega_m"]) \
        < b["official"]["sigma"]
    assert b["H0rd_kms"]["median"] == pytest.approx(10155.0, abs=30.0)
    assert b["argmin"]["at_boundary"] is False
    assert b["converged"] is True


def test_official_anchor_is_literal(bench):
    """El ancla oficial de la puerta NO puede regenerarse con otro
    valor sin romper este test (hallazgo: la re-verificación del gate
    usaba el bloque «official» del propio JSON)."""
    off = bench["lcdm_benchmark"]["official"]
    assert off["Omega_m"] == 0.2975
    assert off["sigma"] == 0.0086
    src = (REPO / "scripts" / "run_desi_dr2_benchmark.py").read_text(
        encoding="utf-8")
    assert '"Omega_m": 0.2975' in src and '"sigma": 0.0086' in src


def test_benchmark_argmin_recomputes_offline(bench, desi_data):
    """El candado que ata artefacto a CÓDIGO: χ² recomputado con el
    fondo vigente en el argmin publicado == χ²_min publicado. Un
    cambio en cosmology/background.py que altere E(z) rompe esto."""
    from cosmology.desi_background_fit import chi2_at
    b = bench["lcdm_benchmark"]
    assert chi2_at(b["argmin"]["theta"], "lcdm", desi_data) == \
        pytest.approx(b["chi2_min"], abs=1e-9)


def test_equivalence_against_cobaya_if_available(desi_data):
    """6A.2 re-ejecutable: χ² propio ≡ Cobaya 3.6.2 sobre vectores
    sintéticos (tolerancia predeclarada 1e-10) + identidad de datos
    computada por sha256."""
    cobaya = pytest.importorskip("cobaya")
    assert cobaya.__version__ == "3.6.2", \
        "la referencia externa de esta ronda está fijada a 3.6.2"
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from run_desi_dr2_benchmark import COBAYA_PACKAGES, equivalence_check
    if not COBAYA_PACKAGES.exists():
        pytest.skip("packages path de Cobaya ausente — recrear con "
                    "scripts/setup_cobaya_packages.py o relocalizar "
                    "con COBAYA_PACKAGES_PATH")
    eq = equivalence_check(desi_data)
    assert eq["status"] == "PASS"
    assert eq["max_abs_diff"] < 1e-10
    assert eq["data_identity"]["status"] == "PASS"
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


def test_run_fit_is_deterministic():
    """«Semilla 42» significa reproducibilidad REAL: dos ejecuciones
    con la misma semilla dan cadenas bit-idénticas (hallazgo: emcee 3
    copia su RandomState del estado global legacy — sembrar solo la
    bola inicial no reproducía nada)."""
    pytest.importorskip("emcee")
    from cosmology.desi_background_fit import run_fit

    def logp_toy(theta, data, idx):
        return -0.5 * float(np.sum(np.asarray(theta) ** 2))

    a = run_fit(logp_toy, (0.5, 0.5), 2, None, nwalkers=8, nsteps=60,
                seed=42)
    b = run_fit(logp_toy, (0.5, 0.5), 2, None, nwalkers=8, nsteps=60,
                seed=42)
    assert np.array_equal(a["flat"], b["flat"])


def test_contrast_artifact_locks(contrast):
    """CANDADO del contraste 6A.5 publicado (χ²_M = mínimos VERDADEROS
    sobre el cierre del soporte del prior, multistart acotado):
    veredicto uniforme en las 8 configuraciones (Δχ² ∈ [−0.35, 0);
    ΔBIC ∈ [+4.4, +5.0] pro-ΛCDM), ε dominado por el prior TRUNCADO
    (cociente de anchuras ≈ 1), ningún bin decisivo en ninguno de los
    dos modelos, frontera del argmin publicada."""
    rows = contrast["rows"]
    assert len(rows) == 8
    prior = contrast["prior_eps"]
    all_row = next(r for r in rows if r["config"] == "DESI_ALL")
    assert all_row["chi2_lcdm"] == pytest.approx(10.284, abs=0.02)
    assert all_row["chi2_mcmc"] == pytest.approx(10.058, abs=0.02)
    assert all_row["delta_bic"] == pytest.approx(4.904, abs=0.05)
    assert all_row["eps_median"] == pytest.approx(0.0154, abs=0.003)
    for r in rows:
        assert -0.35 < r["delta_chi2"] < 0.0, r["config"]
        assert 4.4 < r["delta_bic"] < 5.0, r["config"]
        assert abs(r["eps_median"] - prior["median"]) < 0.01, r["config"]
        assert 0.9 < r["eps_width_ratio_vs_prior"] < 1.1, r["config"]
        assert abs(r["shift_omega_m_vs_ALL"]) < 0.005, r["config"]
        assert abs(r["shift_omega_m_lcdm_vs_ALL"]) < 0.005, r["config"]
        assert r["conv_lcdm"] and r["conv_mcmc"], r["config"]
        # el mínimo del MCMC vive en la frontera (ε=−0.05, z_t=1) en
        # las 8 — publicado, no ocultado; el de ΛCDM es interior:
        assert r["at_boundary_mcmc"] is True, r["config"]
        assert r["at_boundary_lcdm"] is False, r["config"]


def test_contrast_argmins_recompute_offline(contrast, desi_data):
    """Cada χ² publicado del contraste se recomputa con el código
    vigente en su argmin publicado (ambos modelos, las 8
    configuraciones): artefacto ↔ código atados."""
    from cosmology.desi_background_fit import chi2_at
    from cosmology.desi_bao import loo_configurations
    _, _, _, bins, _ = desi_data
    configs = loo_configurations(bins)
    for r in contrast["rows"]:
        idx = configs[r["config"]]
        assert chi2_at(r["argmin_lcdm"], "lcdm", desi_data, idx=idx) \
            == pytest.approx(r["chi2_lcdm"], abs=1e-8), r["config"]
        assert chi2_at(r["argmin_mcmc"], "mcmc", desi_data, idx=idx) \
            == pytest.approx(r["chi2_mcmc"], abs=1e-8), r["config"]


def test_structural_block_recomputes(contrast, desi_data):
    """El número estructural publicado sale de una recomputación en
    vivo (hallazgo: la primera versión publicaba el valor de un solo
    punto como cota del rango, y un «≤ 3e-5» que no salía de ningún
    cómputo): máximo de |ΔE/E| sobre z ∈ [0, 2.33] y distancia χ² a
    parámetros fijos, contra el artefacto."""
    from cosmology.desi_background_fit import (
        E_of_z,
        predict_desi_dr2_vector,
    )
    from cosmology.desi_bao import chi2_bao
    st = contrast["structural"]
    zg = np.linspace(0.0, 2.33, 4001)
    rel = np.abs(
        E_of_z(zg, st["omega_m_ref"], eps=st["eps_probe"],
               z_trans=st["z_trans"])
        / E_of_z(zg, st["omega_m_ref"], eps=0.0) - 1.0)
    assert float(rel.max()) == pytest.approx(st["dE_rel_max_range"],
                                             rel=1e-9)
    assert float(rel[-1]) == pytest.approx(st["dE_rel_at_z233"],
                                           rel=1e-9)
    _, _, _, _, C = desi_data
    v1 = predict_desi_dr2_vector(st["omega_m_ref"], st["H0rd_ref"],
                                 eps=st["eps_probe"],
                                 z_trans=st["z_trans"], data=desi_data)
    v0 = predict_desi_dr2_vector(st["omega_m_ref"], st["H0rd_ref"],
                                 data=desi_data)
    assert chi2_bao(v0, v1, C) == pytest.approx(
        st["chi2_distance_fixed_params"], rel=1e-6)
    # consistencia: el residuo tras reabsorber nunca supera la
    # distancia bruta
    assert st["chi2_residual_after_reabsorbing"] \
        <= st["chi2_distance_fixed_params"]


def test_prior_eps_percentiles_recompute(contrast):
    """La referencia de la dominación por el prior es el prior
    TRUNCADO realmente muestreado (mediana ≠ 0.012 por la asimetría
    del truncamiento) — recomputada contra el artefacto."""
    truncnorm = pytest.importorskip("scipy.stats").truncnorm
    pe = contrast["prior_eps"]
    a = (-0.05 - 0.012) / 0.05
    b = (0.10 - 0.012) / 0.05
    p16, p50, p84 = truncnorm.ppf([0.16, 0.50, 0.84], a, b,
                                  loc=0.012, scale=0.05)
    assert pe["median"] == pytest.approx(float(p50), abs=1e-12)
    assert pe["minus"] == pytest.approx(float(p50 - p16), abs=1e-12)
    assert pe["plus"] == pytest.approx(float(p84 - p50), abs=1e-12)
    assert pe["width_68"] == pytest.approx(float(p84 - p16), abs=1e-12)
