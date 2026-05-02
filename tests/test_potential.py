"""Tests del potencial V(Φ; S)."""

import numpy as np
import pytest

from mcmc_ontology import constants as C
from mcmc_ontology.potential import (
    V, beta_match, alpha_from_matching, chi_inf,
    V_pp_total, delta_m_eff,
)


def test_chi_inf():
    """Fracción de sellado latente χ∞_n (Ec. 312)."""
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
