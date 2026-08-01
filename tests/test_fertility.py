"""Tests del diagrama de fertilidad (frente nº 4, H.2.3)."""

import numpy as np

from core.fertility_map import (
    sample_shapes, fertility_fraction, delta_saturation,
)
from core.basal import quasi_cancellation_ok


def test_sampled_shapes_are_viable():
    """Todas las muestras cumplen la casi-cancelación (3.3)."""
    s = sample_shapes(200, seed=1)
    assert len(s["m_bar"]) == 200
    for m, b, c0 in zip(s["m_bar"], s["b_bar"], s["C0"]):
        assert quasi_cancellation_ok(m, b, c0)


def test_fertility_fraction_bounds_and_frontier():
    """La fracción es una probabilidad; la frontera coincide con la
    condición muestra a muestra (γR > m̄²/ē ⟺ fértil)."""
    res = fertility_fraction(2000, seed=2, gamma_max=3.0)
    assert 0.0 <= res["fraction_fertile"] <= 1.0
    manual = res["gamma_R"] > res["gamma_frontier"]
    assert np.array_equal(manual, res["fertile"])
    # y la ganancia A > 1 exactamente en las fértiles (ec. H.7):
    assert np.array_equal(res["gains"] > 1.0, res["fertile"])


def test_fertility_deterministic():
    """Misma semilla → misma cartografía (reproducible)."""
    a = fertility_fraction(500, seed=3)
    b = fertility_fraction(500, seed=3)
    assert a["fraction_fertile"] == b["fraction_fertile"]


def test_delta_saturation_ceiling():
    """El Techo: δ_sat = (W_max/c̄)^{1/3} crece con W_max (Lema 10.3)."""
    d1 = delta_saturation(1.0, 1.0, 3.0, 1.0)
    d2 = delta_saturation(8.0, 1.0, 3.0, 1.0)
    assert abs(d2 / d1 - 2.0) < 1e-12  # (8/1)^{1/3} = 2
