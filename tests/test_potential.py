"""Tests del potencial V(Φ; S) y de la Ley de la Década."""

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.potential import (
    V, beta_match, alpha_from_matching, chi_inf,
    V_pp_total, delta_m_eff,
)


def test_decade_thresholds():
    """Ley de la Década (v35, Prop. 8.1): S_k = λ^(k−2) − ΔS y S_F = 1 + ΔS.

    Los cuatro umbrales se reducen a tres datos (λ, ΔS, regla ±1 cuanto)
    y deben coincidir con los sellos C1, C2, C3, C4 de S_SEALS.
    """
    thresholds = C.decade_thresholds()
    expected = [C.S_SEALS[k] for k in ("C1", "C2", "C3", "C4")]
    assert np.allclose(thresholds, expected, rtol=0.0, atol=1e-12)
    assert np.allclose(thresholds, [0.009, 0.099, 0.999, 1.001],
                       rtol=0.0, atol=1e-12)


def test_s0_victoria():
    """s0 = π/ln(10) ≈ 1.3644 (derivado para λ=10, v35 F.4)."""
    assert abs(C.S0_VICTORIA - 1.3644) < 1e-4


def test_chi_inf():
    """Fracción de sellado latente χ∞_n (Ec. 312, v32)."""
    assert chi_inf(1) == 0.0
    assert abs(chi_inf(2) - (1 - 0.009 / 0.099)) < 1e-9
    assert abs(chi_inf(3) - (1 - 0.099 / 0.999)) < 1e-9
    assert abs(chi_inf(4) - (1 - 0.999 / 1.001)) < 1e-9


def test_beta_matching():
    """β_n = 2 α S_n / v_n^2 — verifica el matching que ancla a C3."""
    alpha = alpha_from_matching("C3")
    assert abs(beta_match("C3", alpha) - C.BETA["C3"]) < 1e-12


def test_V_at_minimum():
    """V evaluado en Φ = v_n a S = S_n debe estar dominado por el último escalón."""
    Sn = C.S_SEALS["C3"]
    vn = C.V_GEV["C3"]
    val = V(vn, Sn)
    # Cuártico se anula en Φ=v_n; sólo queda α S Φ^2 ≥ 0
    assert val >= 0.0
    assert np.isfinite(val)


def test_curvature_positive():
    """V''_total > 0 en cada sello."""
    for seal in ("C1", "C2", "C3", "C4"):
        assert V_pp_total(seal) > 0.0
        assert delta_m_eff(seal) > 0.0
