"""Tests del frente 5 (5A/5B/5D): potencial débil y Jeans esférico.

Comprobación interna de la implementación, no demostración física
(v35.1, E8): identidades analíticas, límites de recuperación y las
premisas del problema inverso de la amplitud de Cronos.
"""

import numpy as np
import pytest

from dynamics.jeans import (
    sigma_los_sq,
    sigma_los_sq_lum_avg,
)
from dynamics.weak_field import (
    bound_saturation_ratio,
    cronos_amplitude,
    g_cronos_plummer,
    g_eff_plummer,
    plummer_density,
    plummer_g_newton,
    plummer_mass,
    plummer_sigma_los_sq_isotropic,
    plummer_surface_density,
)

M_STAR = 5.2e6      # M_sol (Υ⋆ = 2 × L_V = 2.6e6)
A_PL = 260.0        # pc
R_GRID = np.geomspace(0.05, 3e4, 800)


def _nu():
    return plummer_density(R_GRID, M_STAR, A_PL)


def test_plummer_mass_and_density_consistent():
    """M(<r) es la integral de ρ (identidad del perfil)."""
    r = np.geomspace(0.1, 1e4, 2000)
    rho = plummer_density(r, M_STAR, A_PL)
    m_num = np.concatenate(
        ([0.0], np.cumsum(0.5 * (rho[:-1] * r[:-1] ** 2
                                 + rho[1:] * r[1:] ** 2)
                          * np.diff(r)) * 4.0 * np.pi))
    m_ana = plummer_mass(r, M_STAR, A_PL)
    assert abs(m_num[-1] / m_ana[-1] - 1.0) < 1e-3


def test_sigma_los_matches_analytic_plummer():
    """ANCLAJE EXACTO: el solucionador numérico (Jeans + proyección)
    clava la identidad del Plummer isótropo autoconsistente
    σ_los²(R) = (3π/64)·(GM/a)·(1+R²/a²)^(−1/2)."""
    g = plummer_g_newton(R_GRID, M_STAR, A_PL)
    R_eval = np.array([0.0, 130.0, 260.0, 520.0, 1040.0])
    s2_num = sigma_los_sq(R_eval, R_GRID, _nu(), g, beta=0.0,
                          u_max=3e4)
    s2_ana = plummer_sigma_los_sq_isotropic(R_eval, M_STAR, A_PL)
    assert np.max(np.abs(s2_num / s2_ana - 1.0)) < 2e-3


def test_recovery_newtonian_at_zero_amplitude():
    """Recuperación (propiedad ontológica): con A = 0 (ε_c → 0),
    g_eff ≡ g_N EXACTAMENTE — el límite newtoniano intacto."""
    g0 = g_eff_plummer(R_GRID, M_STAR, A_PL, A=0.0)
    gn = plummer_g_newton(R_GRID, M_STAR, A_PL)
    assert np.array_equal(g0, gn)


def test_cronos_term_is_inward_and_grows_sigma():
    """El término −c²∇ε_c es atracción (g_cronos ≥ 0, Cor. 11.3c) y
    sube la dispersión: σ_los²(A>0) > σ_los²(0)."""
    A = 1e-12
    gc = g_cronos_plummer(R_GRID, M_STAR, A_PL, A)
    assert np.all(gc >= 0.0)
    s2_0 = sigma_los_sq_lum_avg(R_GRID, _nu(),
                                g_eff_plummer(R_GRID, M_STAR, A_PL, 0.0),
                                R_max=2e3)
    s2_A = sigma_los_sq_lum_avg(R_GRID, _nu(),
                                g_eff_plummer(R_GRID, M_STAR, A_PL, A),
                                R_max=2e3)
    assert s2_A > s2_0


def test_linearity_in_amplitude():
    """PREMISA DEL PROBLEMA INVERSO: σ_los² es lineal en la amplitud A
    (Jeans y proyección son lineales en g)."""
    A1, A2 = 4e-13, 8e-13
    base = sigma_los_sq_lum_avg(R_GRID, _nu(),
                                g_eff_plummer(R_GRID, M_STAR, A_PL, 0.0),
                                R_max=2e3)
    d1 = sigma_los_sq_lum_avg(R_GRID, _nu(),
                              g_eff_plummer(R_GRID, M_STAR, A_PL, A1),
                              R_max=2e3) - base
    d2 = sigma_los_sq_lum_avg(R_GRID, _nu(),
                              g_eff_plummer(R_GRID, M_STAR, A_PL, A2),
                              R_max=2e3) - base
    assert d1 > 0.0
    assert abs(d2 / d1 - 2.0) < 1e-6


@pytest.mark.parametrize("beta", [-0.5, 0.0, 0.3])
def test_projected_average_is_anisotropy_invariant(beta):
    """CONTROL ESTRUCTURAL del proyector: el promedio total pesado por
    luminosidad ⟨σ_los²⟩ es independiente de β (esfericidad: cada
    estrella proyecta en promedio 1/3 de su rapidez cuadrática)."""
    g = plummer_g_newton(R_GRID, M_STAR, A_PL)
    ref = sigma_los_sq_lum_avg(R_GRID, _nu(), g, beta=0.0,
                               R_max=2e4, u_max=3e4)
    val = sigma_los_sq_lum_avg(R_GRID, _nu(), g, beta=beta,
                               R_max=2e4, u_max=3e4)
    assert abs(val / ref - 1.0) < 5e-3


def test_anisotropy_changes_the_profile_shape():
    """Control negativo del anterior: aunque el promedio total es
    invariante, el PERFIL σ_los(R) sí depende de β (si no dependiera,
    el test de invariancia sería vacuo)."""
    g = plummer_g_newton(R_GRID, M_STAR, A_PL)
    R_eval = np.array([26.0, 1040.0])
    s2_iso = sigma_los_sq(R_eval, R_GRID, _nu(), g, beta=0.0, u_max=3e4)
    s2_rad = sigma_los_sq(R_eval, R_GRID, _nu(), g, beta=0.3, u_max=3e4)
    assert np.max(np.abs(s2_rad / s2_iso - 1.0)) > 0.01


def test_bound_guard_via_cronos_module():
    """La cota dura (11.5) llega desde cronos.cronos_v3 — una sola
    fuente: α₀⁻¹ > 1e-6 se rechaza también aquí."""
    with pytest.raises(ValueError):
        cronos_amplitude(2e-6, rho_c=1.0)


def test_amplitude_combination_is_degenerate():
    """(α₀⁻¹, ρ_c) solo entran por A = α₀⁻¹/ρ_c^(3/2): dos pares con
    la misma A dan ε_c y g_eff idénticos."""
    A1 = cronos_amplitude(1e-6, rho_c=4.0)
    A2 = cronos_amplitude(1.25e-7, rho_c=1.0)
    assert abs(A1 / A2 - 1.0) < 1e-12
    g1 = g_eff_plummer(R_GRID, M_STAR, A_PL, A1)
    g2 = g_eff_plummer(R_GRID, M_STAR, A_PL, A2)
    assert np.allclose(g1, g2, rtol=1e-12)


def test_bound_saturation_ratio_scales_with_A():
    """El cociente max c²ε_c/|Φ_N| (la lectura ejecutable de la
    ec. 11.5) es lineal en A y positivo."""
    r = np.geomspace(0.1, 5e3, 400)
    q1 = bound_saturation_ratio(r, M_STAR, A_PL, 1e-13)
    q2 = bound_saturation_ratio(r, M_STAR, A_PL, 2e-13)
    assert q1 > 0.0
    assert abs(q2 / q1 - 2.0) < 1e-9


def test_surface_density_normalizes_to_M():
    """∫ Σ(R)·2πR dR = M (proyección conserva la masa)."""
    R = np.geomspace(0.01, 1e5, 4000)
    Sig = plummer_surface_density(R, M_STAR, A_PL)
    tot = np.trapezoid(Sig * 2.0 * np.pi * R, R)
    assert abs(tot / M_STAR - 1.0) < 1e-3


def test_newtonian_dispersion_scale_for_sculptor_stars():
    """Regresión del contraste (control): con solo las estrellas de
    Sculptor (Υ⋆ = 2), la dispersión luminosa media queda en pocos
    km/s — el déficit clásico frente a σ_obs ≈ 9.2 (el número exacto
    lo publica el informe; aquí se fija la escala)."""
    g = plummer_g_newton(R_GRID, M_STAR, A_PL)
    s2 = sigma_los_sq_lum_avg(R_GRID, _nu(), g, R_max=2e3)
    assert 2.0 < np.sqrt(s2) < 6.0
    assert np.sqrt(s2) < 9.2 - 1.1   # el déficit no es marginal


def test_published_verdict_regression():
    """CANDADO de los números centrales publicados en
    results/2026-08-10_jeans_dsph/report.md (Υ⋆ = 2, β = 0), con el
    mismo montaje del script: σ_N ≈ 2.93 km/s, dominancia exigida
    c²ε_c/|Φ_N| ≈ ×12, ρ_c máximo compatible (α₀⁻¹ = 1e-6)
    ≈ 1.38 M⊙/pc³. Si el solucionador o los datos cambian, este test
    obliga a regenerar el informe."""
    from cronos.cronos_v3 import ALPHA0_INV_MAX

    a0, sigma_obs, A_unit = 260.0, 9.2, 1e-13
    r = np.geomspace(0.05, 120.0 * a0, 800)
    nu = plummer_density(r, M_STAR, a0)
    gN = plummer_g_newton(r, M_STAR, a0)
    kw = {"R_max": 8.0 * a0, "u_max": 120.0 * a0}
    s2_N = sigma_los_sq_lum_avg(r, nu, gN, **kw)
    dS2 = sigma_los_sq_lum_avg(
        r, nu, g_eff_plummer(r, M_STAR, a0, A_unit), **kw) - s2_N
    A_req = (sigma_obs ** 2 - s2_N) / dS2 * A_unit
    A_bound = A_unit / bound_saturation_ratio(r, M_STAR, a0, A_unit)
    assert np.sqrt(s2_N) == pytest.approx(2.93, abs=0.02)
    assert A_req / A_bound == pytest.approx(12.2, abs=0.5)
    assert (ALPHA0_INV_MAX / A_req) ** (2.0 / 3.0) \
        == pytest.approx(1.38, abs=0.05)
