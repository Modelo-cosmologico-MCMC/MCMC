"""Candados del crosscheck JAX ↔ NumPy (6A, prioridad 3).

Dos capas: los candados del artefacto (sin jax — leen crosscheck.json
y verifican las puertas y los números publicados) y la recomputación
(importorskip jax, declarado: jax NO es dependencia del proyecto; el
CI sin jax hace skip y la suite sigue sin red)."""

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "results" / "2026-08-19_jax_crosscheck" / "crosscheck.json"


@pytest.fixture(scope="module")
def doc():
    if not ART.exists():
        pytest.skip("crosscheck.json llega con la regeneración "
                    "post-fix del integrador (el registro de la "
                    "primera ejecución es crosscheck_pre_fix.json)")
    return json.loads(ART.read_text(encoding="utf-8"))


def test_gates_all_pass(doc):
    """CANDADO: las puertas predeclaradas (max < 1e-8, RMS < 1e-10,
    |Δχ²| < 1e-8) pasan TODAS con el integrador corregido."""
    assert doc["all_gates_pass"] is True
    assert all(doc["gates"].values())
    s = doc["summary"]
    assert s["block1_max"] < 1e-8 and s["block1_rms"] < 1e-10
    assert s["block2_max"] < 1e-8 and s["block2_rms"] < 1e-10
    assert s["block3_max"] < 1e-8 and s["block3_rms"] < 1e-10
    assert s["block4_max_chi2_diff"] < 1e-8


def test_block1_2_at_machine_precision(doc):
    """El álgebra pura (E/H/D_H y S↔z) coincide a precisión de
    máquina — no solo bajo la puerta."""
    assert doc["summary"]["block1_max"] < 1e-13
    assert doc["summary"]["block2_max"] < 1e-13


def test_chi2_at_published_argmins(doc):
    """Los χ² NumPy del bloque 4 en los argmin publicados COINCIDEN
    con los de los artefactos DESI (misma cadena de números)."""
    bench = json.loads((REPO / "results"
                        / "2026-08-18_desi_dr2_background"
                        / "benchmark.json").read_text("utf-8"))
    contr = json.loads((REPO / "results"
                        / "2026-08-18_desi_dr2_background"
                        / "contrast.json").read_text("utf-8"))
    all_row = next(r for r in contr["rows"]
                   if r["config"] == "DESI_ALL")
    b4 = doc["block4_chi2"]
    assert b4["benchmark_argmin_lcdm"]["chi2_numpy"] == pytest.approx(
        bench["lcdm_benchmark"]["chi2_min"], abs=1e-9)
    assert b4["contrast_argmin_lcdm"]["chi2_numpy"] == pytest.approx(
        all_row["chi2_lcdm"], abs=1e-9)
    assert b4["contrast_argmin_mcmc"]["chi2_numpy"] == pytest.approx(
        all_row["chi2_mcmc"], abs=1e-9)


def test_sne_block_published_without_gate(doc):
    """El bloque 5 (integrador SNe de producción) se publica como
    medición: error del trapecio de 2048 puntos ~1e-6 relativo,
    ≤ 1e-5 mag en μ — despreciable frente a σ_μ ~ 0.1 y declarado."""
    b5 = doc["block5_sne_production"]
    assert 1e-8 < b5["comoving_distance"]["max_rel"] < 1e-5
    assert b5["distance_modulus_max_abs_mag"] < 1e-4
    assert "sin puerta" in b5["note"]


def test_independence_no_reference_imports():
    """La implementación JAX no importa NADA de las implementaciones
    NumPy de referencia (regla de independencia de la ronda): solo
    fórmulas declaradas + SHARED_CONSTANTS."""
    src = (REPO / "validation" / "jax_background.py").read_text(
        encoding="utf-8")
    for banned in ("from cosmology", "import cosmology",
                   "from mcmc_ontology", "import mcmc_ontology"):
        assert banned not in src
    assert "SHARED_CONSTANTS" in src


def test_recompute_with_jax_if_available(doc):
    """Recomputación (jax presente): E y el vector DESI vuelven a
    coincidir con la referencia bajo las puertas — ata el artefacto
    al código vigente de AMBAS implementaciones."""
    pytest.importorskip("jax")
    from cosmology.background import H_of_z
    from cosmology.desi_background_fit import predict_desi_dr2_vector
    from cosmology.desi_bao import load_desi_dr2_all
    from validation.jax_background import E_jax, predict_desi_vector_jax
    zg = np.linspace(0.0, 3.0, 101)
    for om, eps, zt in [(0.2971, 0.0, 8.9), (0.296, -0.05, 1.0)]:
        E_np = np.asarray(H_of_z(zg, H0=1.0, Omega_m=om, eps=eps,
                                 z_trans=zt, dz=1.5))
        E_jx = np.asarray(E_jax(zg, om, eps, zt, 1.5))
        assert np.max(np.abs(E_jx / E_np - 1.0)) < 1e-13
    data = load_desi_dr2_all()
    z_eff, quant, _, _, _ = data
    th = doc["block4_chi2"]["contrast_argmin_mcmc"]
    om, h0rd, eps, zt = (
        json.loads((REPO / "results" / "2026-08-18_desi_dr2_background"
                    / "contrast.json").read_text("utf-8"))
        ["rows"][0]["argmin_mcmc"])
    v_np = predict_desi_dr2_vector(om, h0rd, eps=eps, z_trans=zt,
                                   data=data)
    v_jx = np.asarray(predict_desi_vector_jax(om, h0rd, eps, zt, 1.5,
                                              z_eff, quant))
    assert np.max(np.abs(v_jx / v_np - 1.0)) < 1e-8
    assert th["abs_diff"] < 1e-8
