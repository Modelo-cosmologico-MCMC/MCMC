"""Tests del 5C estructural, la curva ρ_id y el comparador binado.

Comprobación interna de la implementación, no demostración física
(v35.1, E8).
"""

import numpy as np
import pytest

from dynamics.binned_profile import (
    chi2_profile,
    load_binned_profile,
    synthetic_selftest,
)
from dynamics.disc_cronos import (
    outer_decline_ratio,
    rho_midplane,
    sigma_exponential,
    v_bar_sq_freeman,
    v_cronos_sq,
    x_peak_v_cronos,
)
from dynamics.rho_id_target import (
    mass_cored,
    rho0_required,
    sculptor_rho_id_curve,
)
from dynamics.weak_field import G_PC

SIGMA0, R_D, ZETA = 500.0, 2000.0, 0.15   # M⊙/pc², pc, —


def test_freeman_curve_matches_numeric_integration():
    """ANCLAJE: la forma cerrada de Freeman coincide con la integral
    numérica del potencial del disco delgado (anillos; kernel de
    Bessel integrado en k): v² = 4πGΣ0R_d·y²[I0K0−I1K1]."""
    from scipy.integrate import quad
    from scipy.special import j1

    def v_sq_numeric(R):
        # v²(R) = R·∫ dk S(k)·J1(kR)·k, S(k) = 2πGΣ0/(1+(kR_d)²)^(3/2)
        def integrand(k):
            S = 2.0 * np.pi * G_PC * SIGMA0 * R_D ** 2 \
                / (1.0 + (k * R_D) ** 2) ** 1.5
            return S * j1(k * R) * k
        val, _ = quad(integrand, 0.0, 50.0 / R_D, limit=400)
        return R * val

    for R in (1000.0, 2000.0, 4000.0):
        ana = float(v_bar_sq_freeman(R, SIGMA0, R_D))
        num = v_sq_numeric(R)
        assert abs(num / ana - 1.0) < 1e-3, (R, ana, num)


def test_freeman_peak_location():
    """El máximo de la curva de Freeman está en R ≈ 2.15·R_d
    (propiedad conocida del disco exponencial)."""
    R = np.linspace(0.1 * R_D, 6.0 * R_D, 2000)
    v2 = v_bar_sq_freeman(R, SIGMA0, R_D)
    assert abs(R[np.argmax(v2)] / R_D - 2.15) < 0.05


def test_v_cronos_shape_peak_and_decline():
    """Identidades de FORMA (independientes de A, Σ0, R_d, ζ):
    v_cronos² ∝ x·e^(−3x/2) alcanza el máximo en x = 2/3 y en x = 4
    ha caído a ~4 % de su pico."""
    R = np.linspace(0.01 * R_D, 8.0 * R_D, 4000)
    v2 = v_cronos_sq(R, SIGMA0, R_D, ZETA, A=1e-9)
    x_max = R[np.argmax(v2)] / R_D
    assert abs(x_max - x_peak_v_cronos()) < 0.01
    ratio = outer_decline_ratio(4.0)
    assert v2[np.argmin(np.abs(R - 4.0 * R_D))] / v2.max() \
        == pytest.approx(ratio, rel=1e-2)
    assert ratio < 0.05


def test_v_cronos_linear_in_A_and_zero_recovery():
    """Recuperación A = 0 ⟹ término nulo exacto; linealidad en A."""
    R = np.linspace(100.0, 8000.0, 50)
    assert np.all(v_cronos_sq(R, SIGMA0, R_D, ZETA, 0.0) == 0.0)
    v1 = v_cronos_sq(R, SIGMA0, R_D, ZETA, 1e-9)
    v2 = v_cronos_sq(R, SIGMA0, R_D, ZETA, 2e-9)
    assert np.allclose(v2, 2.0 * v1, rtol=1e-12)


def test_rho_midplane_units_and_guard():
    """ρ(0) = Σ0/(2ζR_d) y ζ ≤ 0 rechazado."""
    assert rho_midplane(0.0, SIGMA0, R_D, ZETA) \
        == pytest.approx(SIGMA0 / (2.0 * ZETA * R_D), rel=1e-12)
    with pytest.raises(ValueError):
        rho_midplane(0.0, SIGMA0, R_D, 0.0)
    assert sigma_exponential(R_D, SIGMA0, R_D) \
        == pytest.approx(SIGMA0 / np.e, rel=1e-12)


def test_mass_cored_matches_numeric_integral():
    """M_id(<r) cerrada = integral numérica de 4πr²ρ_id(r)."""
    rho0, r_c = 0.05, 400.0
    r = np.geomspace(1.0, 2000.0, 4000)
    integrand = 4.0 * np.pi * r ** 2 * rho0 / (1.0 + (r / r_c) ** 2)
    m_num = np.trapezoid(integrand, r)
    m_ana = float(mass_cored(2000.0, rho0, r_c)) \
        - float(mass_cored(1.0, rho0, r_c))
    assert abs(m_num / m_ana - 1.0) < 1e-3


def test_rho0_required_closes_the_mass():
    """La curva de degeneración cierra la masa objetivo exactamente."""
    M_target, r_enc = 1.6e7, 347.0
    for r_c in (100.0, 300.0, 600.0):
        rho0 = float(rho0_required(r_c, M_target, r_enc))
        assert float(mass_cored(r_enc, rho0, r_c)) \
            == pytest.approx(M_target, rel=1e-12)


def test_sculptor_curve_subtracts_stellar_part():
    """El objetivo ρ_id es M_1/2 MENOS la parte estelar dentro de
    r_1/2 (Plummer), y la curva lo cierra."""
    m_id, rho0 = sculptor_rho_id_curve(
        np.array([300.0]), M_half=2.05e7, M_star=5.2e6,
        a_plummer=260.0, r_half_3d=347.0)
    assert 0.0 < m_id < 2.05e7
    assert float(mass_cored(347.0, float(rho0[0]), 300.0)) \
        == pytest.approx(m_id, rel=1e-12)
    with pytest.raises(ValueError):
        sculptor_rho_id_curve(np.array([300.0]), M_half=1e6,
                              M_star=5.2e6, a_plummer=260.0,
                              r_half_3d=347.0)


def test_binned_machinery_selftest_and_declared_absence():
    """El comparador binado: autoprueba sintética coherente (χ² del
    modelo verdadero ~ n_bins; el desplazado lo empeora) y la AUSENCIA
    del fichero de datos es el estado declarado (mensaje de estatuto)."""
    st = synthetic_selftest()
    assert st["chi2_true"] < 3.0 * st["n_bins"]
    assert st["chi2_shifted"] > 4.0 * st["chi2_true"]
    assert chi2_profile(np.array([9.0]), np.array([9.0]),
                        np.array([1.0])) == 0.0
    with pytest.raises(FileNotFoundError, match="pendiente"):
        load_binned_profile()
