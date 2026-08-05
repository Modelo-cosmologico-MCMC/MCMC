"""Tests del frente 2 instrumentado (ec. 14.2): espectro ⟺ walking."""

import numpy as np
import pytest

from core.decade import lambda_from_s0
from core.victoria_exponent import (
    S0_TARGET,
    STATUS_FRENTE2,
    has_complex_pair,
    o1_ansatz_scan,
    routes_coincide,
    s0_spectral,
)

S0 = 1.3644
M2 = np.array([[0.3, -S0], [S0, 0.3]])          # par 0.3 ± i·1.3644


def _m3(coupling: float = 0.4) -> np.ndarray:
    M = np.zeros((3, 3))
    M[:2, :2] = M2
    M[2, 2] = -0.7
    M[0, 2] = coupling
    return M


def test_spectral_route():
    """Ruta 1: s0 = |Im μ(M)| exacto; sin par complejo se rechaza."""
    assert abs(s0_spectral(M2) - S0) < 1e-12
    with pytest.raises(ValueError):
        s0_spectral(np.diag([1.0, -2.0, 0.5]))


def test_routes_coincide_2x2_and_3x3():
    """Las dos rutas al exponente de Victoria coinciden (espectro ⟺
    periodo del walking medido en el flujo integrado), también con un
    modo real acoplado al plano."""
    for M in (M2, _m3()):
        r = routes_coincide(M)
        assert r["coincide"]
        assert r["rel_error"] < 1e-6
        assert abs(r["lambda_spectral"] - 10.0) < 1e-2   # λ = e^{π/s0}


def test_lambda_of_target_is_ten():
    """s0 = π/ln10 ⟺ λ = 10 (Def. 8.3), el número de la Década."""
    assert abs(lambda_from_s0(S0_TARGET) - 10.0) < 1e-12


def test_dsi_generic_and_scale_invariant():
    """Con ansatz O(1), la cascada DSI (par complejo) es GENÉRICA
    (~2/3) e independiente de la escala del ansatz."""
    fracs = [o1_ansatz_scan(n=4000, scale=a, seed=13)["frac_dsi"]
             for a in (1.0, 2.0)]
    for f in fracs:
        assert 0.60 < f < 0.75
    assert abs(fracs[0] - fracs[1]) < 0.03


def test_band_fraction_is_selection_not_consequence():
    """λ = 10 es una SELECCIÓN: la fracción en banda ±10% de
    s0 = π/ln10 nunca es genérica (≪ 1) y además depende de la escala
    del ansatz (advertencia declarada del módulo)."""
    f1 = o1_ansatz_scan(n=4000, scale=1.0, seed=17)["frac_band"]
    f2 = o1_ansatz_scan(n=4000, scale=2.0, seed=17)["frac_band"]
    assert f1 < 0.3 and f2 < 0.3          # nunca genérica
    assert f2 > f1                         # y depende de la escala


def test_has_complex_pair():
    assert has_complex_pair(M2)
    assert not has_complex_pair(np.diag([1.0, 2.0]))


def test_status_declares_gap():
    """El hueco del frente (las β de Fokker-Planck) queda declarado."""
    assert "Fokker-Planck" in STATUS_FRENTE2
    assert "frente abierto nº 2" in STATUS_FRENTE2
