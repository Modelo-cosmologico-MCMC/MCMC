"""Candidato 2 del cribado de β: las β a un lazo (LPA) del potencial séxtico O(N) en dimensión d (core/beta_one_loop.py) y el
criterio (i) en forma integral (core/beta_candidates.integral_criterion). Se fija la derivación (límite canónico, coeficientes
de Taylor del lado derecho exacto, coeficiente de libro en d = 4), la invariancia de R y el teorema de signo; nada aquí es
demostración física (E8)."""

import math

import numpy as np
import pytest

from core.basal import B_BAR, C0_DEFAULT, M_BAR
from core.beta_candidates import integral_criterion, screening_report
from core.beta_one_loop import (
    F_wetterich,
    R_c_wetterich,
    R_star_polchinski,
    beta_polchinski,
    beta_wetterich,
    c_d,
    integrate_flow,
    lnR_flow,
    lnR_flow_polchinski_closed,
    ratio_R,
    rhs_potential,
    wetterich_window,
)
from core.fokker_planck_beta import beta_functions

RNG = np.random.default_rng(20261018)


def _taylor_123(f, h=0.04, deg=12):
    """Coeficientes de x, x², x³ de una función analítica en 0, por ajuste de Chebyshev en [−h, h]."""
    nodes = h * np.cos(np.pi * (np.arange(4 * deg) + 0.5) / (4 * deg))
    cheb = np.polynomial.chebyshev.Chebyshev.fit(nodes, f(nodes), deg, domain=[-h, h])
    power = cheb.convert(kind=np.polynomial.Polynomial, domain=[-h, h], window=[-h, h]).coef
    return power[1], power[2], power[3]


def test_polchinski_N2_without_scaling_is_the_canonical_clock_flow():
    for _ in range(20):
        M, B, C = RNG.uniform(-1, 1), RNG.uniform(-1, 1), RNG.uniform(0.1, 2)
        assert np.allclose(beta_polchinski(M, B, C, d=RNG.uniform(0, 4), N=2, scaling=False), beta_functions(M, B, C, 0.5, 0.5),
                           rtol=0, atol=1e-14)


@pytest.mark.parametrize("scheme", ["wetterich_litim", "polchinski"])
def test_betas_are_the_taylor_coefficients_of_the_exact_rhs(scheme):
    """dM/ds = 2·[x], dB/ds = −4·[x²], dC/ds = 6·[x³] del lado derecho EXACTO de la LPA (sin truncar)."""
    for N in (1, 2, 3):
        for d in (1.0, 2.0, 3.0, 4.0):
            M, B, C = RNG.uniform(0.0, 0.3), RNG.uniform(-0.2, 0.2), RNG.uniform(0.05, 0.5)
            a, b = RNG.uniform(0.2, 1.0), RNG.uniform(0.2, 1.0)
            t1, t2, t3 = _taylor_123(lambda x: rhs_potential(x, M, B, C, d, N, scheme, a, b))
            beta = beta_wetterich(M, B, C, d, N) if scheme == "wetterich_litim" else beta_polchinski(M, B, C, d, N, a, b)
            assert np.allclose([2 * t1, -4 * t2, 6 * t3], beta, rtol=1e-7, atol=1e-9), (N, d)


def test_d4_one_loop_textbook_coefficient():
    """d = 4, N = 1, V = λφ⁴/4! (λ = −6B en la convención del reloj): ∂_t λ = 6λ² normalizado, 6·c_4 = 3/(16π²) físico."""
    lam = 0.37
    dB_ds = beta_wetterich(0.0, -lam / 6.0, 0.0, d=4.0, N=1)[1]
    dlam_dt = -6.0 * (-dB_ds)                       # t = ln k = −s
    assert dlam_dt == pytest.approx(6.0 * lam ** 2, rel=1e-12)
    assert 6.0 * c_d(4.0) == pytest.approx(3.0 / (16.0 * math.pi ** 2), rel=1e-12)


def test_R_flow_closed_forms_and_d_independence():
    """d ln R/ds no depende de d (los términos de escala se cancelan) y coincide con las formas cerradas."""
    for _ in range(50):
        M, B, C, N = RNG.uniform(1e-3, 0.5), RNG.uniform(1e-3, 0.5), RNG.uniform(0.05, 2.0), int(RNG.integers(1, 4))
        a, b = RNG.uniform(0.1, 2.0), RNG.uniform(0.1, 2.0)
        fw = [lnR_flow(beta_wetterich(M, B, C, d, N), M, B, C) for d in (0.0, 1.0, 2.0, 3.0)]
        fp = [lnR_flow(beta_polchinski(M, B, C, d, N, a, b), M, B, C) for d in (0.0, 1.0, 2.0, 3.0)]
        assert np.ptp(fw) < 1e-9 * max(1.0, abs(fw[0])) and np.ptp(fp) < 1e-9 * max(1.0, abs(fp[0]))
        R = float(ratio_R(M, B, C))
        assert fw[0] == pytest.approx(8.0 * C / (B * (1.0 + M) ** 4) * F_wetterich(R, M, N), rel=1e-9)
        assert fp[0] == pytest.approx(lnR_flow_polchinski_closed(M, B, C, N, a, b), rel=1e-9)


def test_sign_theorem_and_thresholds():
    """Wetterich: R > R_c(N) = (13N + 74)/(12(N + 2)) ⟹ F > 0 para todo m̃²; Polchinski: R > (N+4)/(N+2), M > 0 ⟹ crece."""
    assert R_c_wetterich(2) == pytest.approx(25.0 / 12.0) and R_c_wetterich(1) == pytest.approx(29.0 / 12.0)
    assert R_star_polchinski(2) == pytest.approx(1.5)
    Mg = np.concatenate([np.linspace(-0.99, 2.0, 300001), np.geomspace(2.0, 1e5, 4001)])
    for N in (1, 2, 3, 8):
        assert np.min(F_wetterich(R_c_wetterich(N) * (1 + 1e-6), Mg, N)) > 0.0
        assert np.min(F_wetterich(R_c_wetterich(N) * (1 - 1e-2), Mg, N)) < 0.0
        assert wetterich_window(R_c_wetterich(N) * 1.001, N) is None and wetterich_window(R_c_wetterich(N) * 0.99, N) is not None
    for _ in range(200):
        N = int(RNG.integers(1, 5))
        M, C = RNG.uniform(1e-6, 1.0), RNG.uniform(0.01, 3.0)
        R = R_star_polchinski(N) * (1.0 + RNG.uniform(1e-6, 3.0))
        B = math.sqrt(4.0 * R * C * M)
        assert lnR_flow_polchinski_closed(M, B, C, N, RNG.uniform(0.01, 3), RNG.uniform(0.01, 3)) > 0.0
    R0 = B_BAR ** 2 / (4.0 * C0_DEFAULT * M_BAR ** 2)
    assert R0 == 2.25 and R0 > R_c_wetterich(2)                  # formas por defecto: por encima del umbral con N = 2
    lo, hi = wetterich_window(R0, 1)                            # un escalar: ventana de masa en la que R decrece
    assert lo == pytest.approx(0.0200, abs=2e-4) and hi == pytest.approx(0.0577, abs=2e-4)


@pytest.mark.parametrize("scheme", ["wetterich_litim", "polchinski"])
def test_default_forms_reach_mass_instability_before_any_crossing(scheme):
    """Desde el escalado (3.2) (M₀² = m̄²δ₀², B = b̄δ₀, C₀): en toda d y N ∈ {1, 2}, M₀² llega a cero antes que D, y la
    integral del criterio (i) no se acerca al valor exigido."""
    for delta0 in (0.01, 0.03):
        lam0 = (M_BAR ** 2 * delta0 ** 2, B_BAR * delta0, C0_DEFAULT)
        for N in (1, 2):
            for d in (1.0, 2.0, 3.0):
                r = integrate_flow(scheme, lam0, d, N, s_max=40.0)
                assert r["s_D_zero"] is None and r["s_mass_zero"] is not None, (scheme, delta0, N, d)
                assert r["integral_criterion"]["fraction_of_required"] < 0.5


def test_integral_criterion_on_synthetic_trajectories():
    S = np.linspace(0.0, 1.0, 1001)
    C = np.ones_like(S)
    M = np.full_like(S, 1e-4)
    B_down = np.sqrt(4.0 * C * M * (2.25 - 1.5 * S))         # R: 2.25 → 0.75 (cruza 4/3 y 1)
    r = integral_criterion(S, M, B_down, C)
    assert r["required_integral"] == pytest.approx(-math.log(2.25)) and r["reached_before_mass_zero"]
    assert r["S_T0_zero"] == pytest.approx((2.25 - 4.0 / 3.0) / 1.5, abs=2e-3) and r["S_D_zero"] == pytest.approx(1.25 / 1.5, abs=2e-3)
    assert r["fraction_of_required"] > 1.0
    B_up = np.sqrt(4.0 * C * M * (2.25 + S))                 # R crece: la integral nunca baja de 0
    r = integral_criterion(S, M, B_up, C)
    assert not r["reached_before_mass_zero"] and r["fraction_of_required"] == pytest.approx(0.0) and r["integral_at_end_of_positive_mass"] > 0
    M_flip = 1e-4 * (1.0 - 2.0 * S)                           # M₀² se anula en S = 0.5: el tramo se corta ahí
    r = integral_criterion(S, M_flip, B_down, C)
    assert r["S_mass_zero"] == pytest.approx(0.5, abs=2e-3)
    with pytest.raises(ValueError):
        integral_criterion(S, -M, B_down, C)


def test_screening_report_keeps_pr6_keys_and_adds_integral_block():
    S = np.linspace(0.0, 1.0, 11)
    D = np.linspace(1.0, 0.5, 11)
    M = np.full(11, 1e-4)
    old = screening_report([], D, S, M, [0.009, 0.099, 0.999])
    assert "criterion_i_integral" not in old and old["criterion_i"] is False
    new = screening_report([], D, S, M, [0.009, 0.099, 0.999], B=np.full(11, 0.03), C0=np.ones(11))
    assert {k: new[k] for k in old} == old and new["criterion_i_integral"]["R_initial"] == pytest.approx(2.25)
