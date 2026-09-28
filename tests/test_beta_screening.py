"""Cribado de candidatos a β (PR-6): la Δβ declarada del candidato 1 bajo sus dos cierres, la proyección, el
gancho del reloj en modo emergente (publicación, escalón del potencial, regla de parada en la frontera) y los
tres criterios del cribado. Nada aquí decide λ."""

import numpy as np
import pytest

from core.beta_candidates import (
    CANDIDATES,
    CLOSURES,
    _project_halfinteger_series,
    delta_beta,
    screening_report,
)
from core.fokker_planck_beta import beta_functions
from core.s_clock import ClockConfig, FPClosure, SClock


def test_projection_matches_least_squares_on_fine_grid():
    rng = np.random.default_rng(3)
    coefs = rng.normal(size=7); xp = 0.03
    d = _project_halfinteger_series(coefs, xp)
    x = np.linspace(0.0, xp, 20001)
    f = sum(c * x ** (k + 1.5) for k, c in enumerate(coefs))
    A = np.stack([x, x ** 2, x ** 3], axis=1)
    ref = np.linalg.lstsq(A, f, rcond=None)[0]
    assert np.allclose(d, ref, rtol=1e-3, atol=1e-6 * np.abs(ref).max())


def test_candidate_closures_plane_is_zero_and_quadrant_is_linear_in_kappa():
    lam = (1e-4, 0.03, 1.0); xp = 0.0264
    assert np.all(delta_beta("conversion_current", *lam, kappa=1e5, x_plus=xp, closure="plane") == 0.0)
    d1 = delta_beta("conversion_current", *lam, kappa=1e5, x_plus=xp, closure="quadrant")
    d2 = delta_beta("conversion_current", *lam, kappa=2e5, x_plus=xp, closure="quadrant")
    assert np.allclose(d2, 2.0 * d1) and d1[2] > 0.0                      # C0 crece: D baja
    assert np.all(delta_beta("conversion_current", *lam, kappa=0.0, x_plus=xp) == 0.0)
    with pytest.raises(ValueError):
        delta_beta("otro", *lam, kappa=1.0, x_plus=xp)
    with pytest.raises(ValueError):
        delta_beta("conversion_current", *lam, kappa=1.0, x_plus=xp, closure="disco")
    assert "conversion_current" in CANDIDATES and set(CLOSURES) == {"plane", "quadrant"}


def test_clock_rejects_screening_outside_emergent_mode():
    with pytest.raises(ValueError, match="emergent"):
        SClock(ClockConfig(delta0=0.01, beta_extra="conversion_current", couplings_flow=True))
    with pytest.raises(ValueError, match="desconocido"):
        SClock(ClockConfig(delta0=0.01, thresholds="emergent", couplings_flow=True, beta_extra="x"))
    with pytest.raises(ValueError, match="emergent"):
        SClock(ClockConfig(delta0=0.01, reset_mass_on_collapse=True))


def test_clock_publishes_screening_and_stops_at_boundary():
    base = {"delta0": 0.01, "thresholds": "emergent", "couplings_flow": True, "fp": FPClosure(tau=0.1)}
    r0 = SClock(ClockConfig(**base)).run()
    r1 = SClock(ClockConfig(**base, beta_extra="conversion_current", beta_extra_kappa_hat=1000.0,
                            reset_mass_on_collapse=True)).run()
    assert r0["beta_extra"] is None and r1["beta_extra"]["candidate"] == "conversion_current"
    assert r1["beta_extra"]["kappa_hat"] == 1000.0 and "E13" in r1["beta_extra"]["status"]
    # el candidato entra en el flujo: C0 se mueve respecto del canónico (Δβ_C0 > 0), poco a κ̂ = 1000
    assert r0["trajectory"]["C0"][-1] == pytest.approx(1.0, abs=1e-4)       # canónica: β_C0 ≈ −3e-3, apenas se mueve
    assert 1.0 < r1["trajectory"]["C0"][-1] < 1.2 and r1["trajectory"]["C0"][-1] > r0["trajectory"]["C0"][-1]
    # regla de parada en la frontera (antes: bucle hasta max_steps con S parado)
    assert r0["stop_reason"].startswith("descenso detenido en la frontera") and r0["steps"] < 50_000
    lam0 = SClock(ClockConfig(**base)).lam0
    assert np.allclose(beta_functions(*lam0), [-0.12, -12.0, -3.3e-3], rtol=1e-3)


def test_screening_report_criteria():
    S = np.linspace(0.0, 1.0, 1001); D = 1.0 - 2.0 * S; M = np.ones_like(S)
    rep = screening_report([0.5], D, S, M, [0.009, 0.099, 0.999])
    assert rep["criterion_i"] and not rep["criterion_ii"] and rep["n_crossings"] == 1
    rep3 = screening_report([0.009, 0.09, 0.9], D, S, M, [0.009, 0.099, 0.999])
    assert rep3["passes_i_and_ii"] and rep3["ratios_successive"] == pytest.approx([10.0, 10.0])
    assert rep3["decade_would_be"] == 10.0 and "E13" in rep3["status"]
    rep_m = screening_report([0.5], D, S, np.where(S < 0.4, 1.0, -1.0), [0.009, 0.099, 0.999])
    assert not rep_m["criterion_i"]                                         # M0² ya negativo en el cruce
    assert not screening_report([], D, S, M, [0.009])["criterion_i"]
