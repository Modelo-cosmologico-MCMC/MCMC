"""Predicción cinética del criterio (cronos/cronos_jeans_kinetic.py) y
el instrumento de siembra silenciosa (cronos/cronos_jeans_1d.py)."""

import numpy as np
import pytest

from cronos.cronos_jeans_1d import fit_growth, run_sheets
from cronos.cronos_jeans_kinetic import (
    F_of_y,
    gamma_over_k_fluid,
    gamma_over_k_kinetic,
    prediction_table,
    y_of_q,
)


def test_F_limits_and_monotone():
    assert F_of_y(0.0) == pytest.approx(1.0)
    y = np.linspace(0.0, 6.0, 200)
    F = F_of_y(y)
    assert np.all(np.diff(F) < 0.0) and F[-1] < 0.02
    # asintótica F ≈ 1/(2y²)
    assert F_of_y(20.0) == pytest.approx(1.0 / (2.0 * 400.0), rel=0.01)


def test_threshold_exactly_at_q_one_and_fluid_limit():
    assert y_of_q(1.0) == 0.0 and y_of_q(0.5) == 0.0
    assert y_of_q(1.0 + 1e-6) > 0.0
    # q ≫ 1: γ/k → σ√q (fluido a orden dominante)
    assert gamma_over_k_kinetic(400.0) == pytest.approx(np.sqrt(400.0), rel=0.05)
    assert gamma_over_k_kinetic(1.2) < gamma_over_k_fluid(1.2)       # la fluida sobreestima cerca del umbral
    assert gamma_over_k_kinetic(1.2) == pytest.approx(0.1492, abs=2e-3)
    assert gamma_over_k_kinetic(2.0) == pytest.approx(0.6120, abs=2e-3)


def test_transfer_function_and_instrument_prediction():
    from cronos.cronos_jeans_kinetic import (
        gamma_over_k_instrument,
        transfer_function_cic,
    )
    assert transfer_function_cic(1.0, 1e-9) == pytest.approx(1.0, abs=1e-12)
    W = transfer_function_cic(2.0 * np.pi * 16, 1.0 / 256)
    assert 0.94 < W < 0.96
    # cerca del umbral, una corrección del 5 % en q baja γ/k ~30 %
    g_c, g_i = gamma_over_k_kinetic(1.2), gamma_over_k_instrument(1.2, 2.0 * np.pi * 16, 1.0 / 256)
    assert g_i < g_c and 0.6 < g_i / g_c < 0.85
    assert gamma_over_k_instrument(0.8, 10.0, 1e-3) == 0.0


def test_prediction_table_shape():
    t = prediction_table()
    assert [r["unstable"] for r in t] == [q > 1.0 for q in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0)]


def test_quiet_start_and_seeded_mode():
    res = run_sheets(1.2, N=100_000, ng=128, T=0.02, dt=1e-3, sample_every=5, quiet_start=True, seed_mode=4,
                     seed_amp=0.005, n_beams=64)
    d0 = res["samples"][0]["delta_k"]
    assert d0[3] == pytest.approx(0.0025, rel=0.05)          # |δ_k| = A/2
    # arranque silencioso: sin ruido en t = 0 salvo el armónico 2k de la siembra (O(A²))
    assert all(a < 1e-6 for i, a in enumerate(d0) if i not in (3, 7))
    assert d0[7] < 1e-4
    fit = fit_growth(res, 4, 1e-9, 1.0)
    assert fit["n_points"] >= 3 and np.isfinite(fit["gamma"])


def test_eigenmode_beam_amplitudes_satisfy_dispersion_relation():
    """Ronda 2: la perturbación por haz del modo propio creciente tiene
    módulo |v|/√(v² + y'²) y fase arg(v + iy') dependientes de la
    velocidad, y su suma ponderada reproduce δρ̂ — es la relación de
    dispersión F(y) = 1/q con la cuadratura de los haces; el residuo
    decrece con el número de haces."""
    import numpy as np
    from scipy.special import erfinv

    from cronos.cronos_jeans_kinetic import (
        eigenmode_beam_amplitudes,
        eigenmode_dispersion_residual,
        gamma_over_k_kinetic,
    )
    q = 2.0
    g = gamma_over_k_kinetic(q)
    res = []
    for nb in (256, 1024, 4096):
        u = (np.arange(nb) + 0.5) / nb
        vb = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)
        res.append(eigenmode_dispersion_residual(vb, q, g))
    assert res[-1] < 1e-3 and res[0] > res[-1]
    c = eigenmode_beam_amplitudes(np.array([-1.0, 0.0, 1.0]), q, g, 1.0)
    assert abs(c[1]) == 0.0                                  # el haz en reposo no participa
    assert np.angle(c[2]) > 0.0 and np.angle(c[0]) < 0.0      # fases opuestas: arg(v + iy')
    assert abs(abs(c[2]) - abs(c[0])) < 1e-15
