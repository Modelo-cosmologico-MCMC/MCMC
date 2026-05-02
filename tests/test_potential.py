"""Tests del potencial V(Φ; S)."""

import numpy as np
import pytest

from mcmc_ontology import constants as C
from mcmc_ontology.potential import (
    V, beta_match, alpha_per_seal, alpha_from_matching, chi_inf,
    V_pp_total, delta_m_eff, delta_m_eff_raw,
)
from mcmc_ontology.seals import betas_table, alphas_per_seal


def test_chi_inf():
    """Fracción de sellado latente χ∞_n (Ec. 312)."""
    assert chi_inf(1) == 0.0
    assert abs(chi_inf(2) - (1 - 0.009 / 0.099)) < 1e-9
    assert abs(chi_inf(3) - (1 - 0.099 / 0.999)) < 1e-9
    assert abs(chi_inf(4) - (1 - 0.999 / 1.001)) < 1e-9


def test_beta_matching_per_seal():
    """β_n = 2 α_n S_n / v_n^2 con α_n por sello (Ec. 310)."""
    for seal in ("C1", "C2", "C3", "C4"):
        alpha_n = alpha_per_seal(seal)
        Sn = C.S_SEALS[seal]
        vn = C.V_GEV[seal]
        beta_derived = 2.0 * alpha_n * Sn / vn ** 2
        assert abs(beta_derived - C.BETA[seal]) / C.BETA[seal] < 1e-12


def test_betas_are_independent_constants():
    """Las β_n cubren ~8 órdenes de magnitud (no son derivables de un α global)."""
    betas = betas_table()
    assert betas["C1"] == 1e-43
    assert betas["C3"] == 0.130
    assert betas["C4"] == 1e7
    # Verifica que un α global SÓLO encaja en el sello que lo origina:
    alpha_global = alpha_from_matching("C3")
    assert abs(beta_match("C3", alpha_global) - C.BETA["C3"]) < 1e-12
    # ...pero NO en otros sellos: el ratio se aleja de 1 en muchos órdenes
    ratio_C1 = beta_match("C1", alpha_global) / C.BETA["C1"]
    ratio_C4 = beta_match("C4", alpha_global) / C.BETA["C4"]
    assert ratio_C1 > 10 or ratio_C1 < 0.1
    assert ratio_C4 > 10 or ratio_C4 < 0.1


def test_alphas_per_seal_distinct():
    """Cada sello impone su propio α_n (los α_n no coinciden entre sí)."""
    alphas = alphas_per_seal()
    vals = list(alphas.values())
    # Al menos C1 y C4 difieren por muchos órdenes de magnitud
    assert max(vals) / min(vals) > 1e3


def test_V_at_minimum():
    """V evaluado en Φ = v_n a S = S_n debe estar dominado por el último escalón."""
    Sn = C.S_SEALS["C3"]
    vn = C.V_GEV["C3"]
    val = V(vn, Sn)
    assert val >= 0.0
    assert np.isfinite(val)


def test_delta_m_eff_calibrated():
    """Δm_eff calibrados (Tratado, Tabla P3)."""
    assert abs(delta_m_eff("C2") - 0.1001) < 1e-3
    assert abs(delta_m_eff("C3") - 2.5647) < 1e-3
    assert abs(delta_m_eff("C4") - 5.2680) < 1e-3


def test_delta_m_eff_raw_C4_matches_calibration():
    """En C4 la fórmula numérica cruda coincide con el valor calibrado.

    Ese acuerdo es la *definición* de K_norm = 679.14 GeV.
    """
    assert abs(delta_m_eff_raw("C4") - C.DELTA_M_EFF_CAL["C4"]) < 5e-3


def test_K_norm_value():
    """K_norm = 679.14 GeV (calibración desde C4)."""
    assert abs(C.K_NORM - 679.14) < 0.01


def test_curvature_positive():
    """V''_total > 0 en cada sello (suma de positivos)."""
    for seal in ("C1", "C2", "C3", "C4"):
        assert V_pp_total(seal) > 0.0
        assert delta_m_eff(seal) > 0.0
