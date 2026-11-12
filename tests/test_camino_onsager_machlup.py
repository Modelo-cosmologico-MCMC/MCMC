"""La positividad por reflexión del propio Camino con difusión (core/reflection_positivity.py; revisión del 10-nov-2026,
hallazgo 3; PR R4 del plan del 11-nov).

El Camino es de primer orden con difusión Θ: su medida de trayectorias es exp(−A) con la acción de Onsager–Machlup y su
generador de Fokker–Planck, simetrizado con e^{−V/2Θ}, es H_FP = (Θ/G)·a†a ≥ 0. Se comprueba en el corte del polo de
masa: autovalor mínimo nulo, modo cero e^{−V/2Θ}, núcleo de transferencia semidefinido positivo, brecha igual a la
tasa de Kramers, pozos de U_FP en los tres puntos críticos y el control negativo con el balance detallado roto.
Comprobación interna (E8) en una dimensión, con Θ declarada: no es el Teorema 7.1 ni decide sobre qué acción se
enuncia (decisión 3 del autor)."""

import numpy as np
import pytest

from core.nucleation import kramers_escape
from core.reflection_positivity import (
    STATUS_CAMINO_FP,
    camino_fp_report,
    camino_transfer_kernel,
    detailed_balance_control,
    fp_factor,
    fp_rates,
    fp_site_potential,
    fp_symmetrize,
    fp_symmetrized_generator,
    fp_zero_mode,
    kramers_rate_integral,
    pole_cut,
)

DELTA_H = 0.0554
BARRIER_OVER_THETA = (6.0, 3.0, 2.0, 1.0)          # Θ entre ΔV_b/6 y ΔV_b


@pytest.fixture(scope="module")
def reports():
    return {k: camino_fp_report(DELTA_H, k) for k in BARRIER_OVER_THETA}


def test_lowest_eigenvalue_is_zero_with_the_equilibrium_zero_mode(reports):
    """Autovalor mínimo nulo a 1e-10 (en unidades de δ0²/G) y modo cero e^{−V/2Θ} con residuo relativo < 1e-12."""
    for r in reports.values():
        assert abs(r["lambda_min_scaled"]) < 1e-10 and r["eigenvalues_scaled"][1] > 1e-5
        assert r["zero_mode_residual_relative"] < 1e-12 and r["factor_zero_mode_residual_relative"] < 1e-12
    # H_FP = BᵀB con B de n − 1 filas y n columnas: ≥ 0 y con un autovalor nulo por construcción
    cut = pole_cut(DELTA_H, n=60)
    Theta = cut["landscape"]["barrier_height"] / 6.0
    b_diag, b_up = fp_factor(cut["V"], cut["h"], Theta)
    B = np.zeros((59, 60))
    B[np.arange(59), np.arange(59)] = b_diag
    B[np.arange(59), np.arange(1, 60)] = b_up
    diag, off = fp_symmetrized_generator(cut["V"], cut["h"], Theta)
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    assert np.allclose(B.T @ B, H, rtol=1e-12, atol=1e-12 * np.abs(H).max())
    assert np.all(off == pytest.approx(-Theta / cut["h"] ** 2, rel=1e-12))           # fuera de la diagonal: −Θ/(G·h²)
    assert np.linalg.norm(B @ fp_zero_mode(cut["V"], Theta)) < 1e-12 * np.abs(B).max()
    # y es la ecuación maestra con balance detallado, simetrizada
    L = fp_rates(cut["V"], cut["h"], Theta)
    assert np.allclose(L.sum(axis=0), 0.0, atol=1e-12 * np.abs(L).max())             # conserva la probabilidad
    assert np.allclose(fp_symmetrize(L, cut["V"], Theta), H, rtol=1e-10, atol=1e-12 * np.abs(H).max())


def test_transfer_kernel_is_symmetric_and_positive_semidefinite(reports):
    """exp(−a·H_FP) es simétrico y semidefinido positivo: ⟨ϑ(F̄)F⟩ = fᵀKf ≥ 0 sin el término cinético de (7.1)."""
    for r in reports.values():
        assert r["kernel"]["positive_semidefinite"] and r["kernel"]["asymmetry"] < 1e-12
        assert r["kernel"]["min_eigenvalue_over_max"] >= -1e-12
    cut = pole_cut(DELTA_H, n=200)
    Theta = cut["landscape"]["barrier_height"] / 3.0
    a = 1.0 / kramers_escape(DELTA_H, Theta)["V_pp_fv"]
    K = camino_transfer_kernel(cut["V"], cut["h"], Theta, a)
    rng = np.random.default_rng(20261112)
    for _ in range(20):
        f = rng.standard_normal(200)
        assert f @ K @ f >= -1e-12 * np.abs(K).max() * (f @ f)
    psi0 = fp_zero_mode(cut["V"], Theta)
    assert np.allclose(K @ psi0, psi0, atol=1e-10)                                   # el equilibrio es estacionario
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).max() == pytest.approx(1.0, abs=1e-10)


def test_gap_is_the_kramers_rate(reports):
    """Brecha dentro del 15 % de la tasa de Kramers para Θ entre ΔV_b/6 y ΔV_b (0,94 / 0,91 / 0,93 / 1,09)."""
    ratios = [reports[k]["gap_over_kramers_gaussian"] for k in BARRIER_OVER_THETA]
    assert all(abs(x - 1.0) < 0.15 for x in ratios)
    assert ratios == pytest.approx([0.944, 0.914, 0.932, 1.088], abs=3e-3)
    # la Θ de la comparación es la D_ent declarada de la nucleación
    r6 = reports[6.0]
    assert r6["kramers_gaussian_scaled"] == pytest.approx(kramers_escape(DELTA_H, r6["Theta"])["Gamma_K"] / DELTA_H ** 2, rel=1e-12)
    # el nivel siguiente es del orden de V″ del falso vacío (0,54–0,89 frente a 0,70): también una tasa
    nxt = [reports[k]["next_level_scaled"] for k in BARRIER_OVER_THETA]
    assert min(nxt) == pytest.approx(0.54, abs=5e-3) and max(nxt) == pytest.approx(0.89, abs=5e-3)
    assert r6["V_pp_false_vacuum_scaled"] == pytest.approx(0.70, abs=5e-3)
    # la desviación de la fórmula gaussiana viene del falso vacío junto a la frontera: con la forma integral, que cuenta
    # la pared, la brecha coincide al 0,5 % con barrera de 6Θ también donde la gaussiana se aparta un 19 %
    for d0, gauss in ((0.0554, 0.944), (0.03, 0.973), (0.01, 1.190)):
        r = r6 if d0 == DELTA_H else camino_fp_report(d0, 6.0)
        assert r["gap_over_kramers_gaussian"] == pytest.approx(gauss, abs=3e-3)
        assert r["gap_over_kramers_integral"] == pytest.approx(1.0, abs=5e-3)
        cut = pole_cut(d0)
        assert r["kramers_integral_scaled"] == pytest.approx(kramers_rate_integral(cut, r["Theta"]) / d0 ** 2, rel=1e-9)


def test_site_potential_has_a_well_at_each_of_the_three_critical_points(reports):
    """U_FP = −V″/2G en los puntos críticos (−0,348 / +0,432 / −6,243 en falso vacío, barrera y vacío) y, con Θ =
    ΔV_b/6, un mínimo local junto a cada uno: también en la cima de la barrera."""
    r6 = reports[6.0]
    u = r6["U_FP_at_critical_points_scaled"]
    assert [u["false_vacuum"], u["barrier"], u["true_vacuum"]] == pytest.approx([-0.348, 0.432, -6.243], abs=5e-4)
    crit, wells = r6["critical_points_rho"], r6["U_FP_wells_rho"]
    assert len(wells) == 3
    span = crit["true_vacuum"] - crit["false_vacuum"]
    for w, key in zip(wells, ("false_vacuum", "barrier", "true_vacuum")):
        assert abs(w - crit[key]) < 0.02 * span
    # con Θ mayor el pozo de la cima se pierde (la corrección −V‴·Δρ/2 lo borra): se publica, no se oculta
    assert len(reports[3.0]["U_FP_wells_rho"]) == 2 and len(reports[1.0]["U_FP_wells_rho"]) == 2
    assert len(camino_fp_report(DELTA_H, 12.0)["U_FP_wells_rho"]) == 3
    # la forma de U_FP y el límite continuo del generador en malla: el error baja como h²
    cut = pole_cut(DELTA_H, n=400)
    Theta = cut["landscape"]["barrier_height"] / 6.0
    assert fp_site_potential(cut["dV"], cut["d2V"], Theta, G=2.0) == pytest.approx(
        cut["dV"] ** 2 / (8.0 * Theta) - cut["d2V"] / 4.0, rel=1e-12, abs=1e-30)
    err = [camino_fp_report(DELTA_H, 6.0, n=n)["U_FP_grid_vs_continuum_relative"] for n in (1000, 2000)]
    assert err[1] < 0.02 and err[0] / err[1] == pytest.approx(4.0, rel=0.1)


def test_levels_are_rates_that_scale_as_delta0_squared():
    """La escala de masas cambia: la brecha y los niveles de H_FP son TASAS, ∝ δ0² a Θ/ΔV_b fija (con la acción de
    (7.1) la brecha es una frecuencia, ∝ √V″ ∝ δ0). Sin inclinación el espectro escalado no depende de δ0."""
    a = camino_fp_report(0.01, 6.0, e_bar=0.0)
    b = camino_fp_report(0.0554, 6.0, e_bar=0.0)
    assert a["eigenvalues_scaled"][1:] == pytest.approx(b["eigenvalues_scaled"][1:], rel=1e-6)
    assert a["gap_scaled"] * 0.01 ** 2 / (b["gap_scaled"] * 0.0554 ** 2) == pytest.approx((0.01 / 0.0554) ** 2, rel=1e-6)
    assert a["V_pp_false_vacuum_scaled"] == pytest.approx(1.0, rel=1e-6)             # V″(0) = M0² = m̄²δ0²


def test_negative_control_broken_detailed_balance():
    """CONTROL NEGATIVO: con una fuerza no conservativa sobre un corte periódico la simetrización falla."""
    c = detailed_balance_control()
    rev, brk = c["reversible"], c["broken"]
    assert rev["symmetric"] and rev["asymmetry"] < 1e-12 and rev["kernel_asymmetry"] < 1e-10
    assert rev["max_abs_imag_eigenvalue"] == 0.0 and rev["min_real_eigenvalue"] > -1e-6 * rev["gap"]
    assert rev["zero_mode_residual"] < 1e-12 and abs(rev["kolmogorov_cycle_affinity"]) < 1e-10
    assert not brk["symmetric"] and brk["asymmetry"] > 1e-3 and brk["kernel_asymmetry"] > 0.05
    assert brk["max_abs_imag_eigenvalue"] > 1.0 and brk["zero_mode_residual"] > 1e-6
    assert brk["kolmogorov_cycle_affinity"] == pytest.approx(c["affinity"], rel=1e-9)   # criterio de Kolmogórov roto
    # con fronteras reflectantes una fuerza constante ES un gradiente: se simetriza con V − f·x (no hay control ahí)
    cut = pole_cut(DELTA_H, n=80)
    Theta = cut["landscape"]["barrier_height"] / 3.0
    f = 0.3 * Theta / cut["h"] / 80.0
    H = fp_symmetrize(fp_rates(cut["V"], cut["h"], Theta, force=f), cut["V"] - f * cut["rho"], Theta)
    off = np.linalg.norm(H - np.diag(np.diag(H)))
    assert np.linalg.norm(H - H.T) < 1e-10 * off
    Hbad = fp_symmetrize(fp_rates(cut["V"], cut["h"], Theta, force=f), cut["V"], Theta)   # con el peso equivocado
    assert np.linalg.norm(Hbad - Hbad.T) > 1e-3 * off


def test_theta_is_declared_and_the_statement_is_scoped():
    assert "DECLARADA" in STATUS_CAMINO_FP and "D_ent" in STATUS_CAMINO_FP and "decisión 3 del autor" in STATUS_CAMINO_FP
    assert "no es el Teorema 7.1" in STATUS_CAMINO_FP
    r = camino_fp_report(DELTA_H, 6.0, n=600, n_kernel=120)
    assert r["status"] == STATUS_CAMINO_FP and r["unit"] == "δ0²/G"
    assert r["Theta"] == pytest.approx(pole_cut(DELTA_H)["landscape"]["barrier_height"] / 6.0, rel=1e-12)
    with pytest.raises(ValueError, match="Θ > 0"):
        fp_factor(np.zeros(4), 0.1, 0.0)
    with pytest.raises(ValueError, match="Sin falso vacío"):
        pole_cut(0.2)
