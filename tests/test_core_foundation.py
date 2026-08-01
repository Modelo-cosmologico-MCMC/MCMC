"""Tests del corazón de la cadena: caps. 2-4 (Plano Dual, Basal, Flujo).

Cada test verifica un teorema o definición del tratado; los controles
negativos comprueban el caso que debe fallar y falla.
"""

import numpy as np
import pytest

from core.dual_plane import (
    to_dual, from_polar, dual_reflection, is_physical, on_diagonal,
    THETA_DIAGONAL,
)
from core.basal import (
    V0, scaled_params, quasi_cancellation_ok, discriminant, kappa_plus,
    c_bar, T0_analytic, T0_numeric, chi_residue,
)
from core.path_flow import grad_V, flow, exits_to_mass_pole


# ------------------------------ Cap. 2 -----------------------------------

def test_dual_roundtrip():
    """Ida y vuelta (φM,φE) ↔ (ρ,θ) exacta (Def. 2.1)."""
    rng = np.random.default_rng(5)
    phi_M, phi_E = rng.uniform(0.1, 3.0, size=2)
    c = to_dual(phi_M, phi_E)
    back_M, back_E = from_polar(c["rho"], c["theta"])
    assert abs(back_M - phi_M) < 1e-12 and abs(back_E - phi_E) < 1e-12
    # χ y ς reconstruyen las proyecciones:
    assert abs((c["sigma"] + c["chi"]) / np.sqrt(2.0) - phi_M) < 1e-12
    assert abs((c["sigma"] - c["chi"]) / np.sqrt(2.0) - phi_E) < 1e-12


def test_dual_reflection_maps():
    """Def. 2.2: la reflexión Z₂ lleva χ → −χ y θ → π/2 − θ."""
    c1 = to_dual(2.0, 0.5)
    rM, rE = dual_reflection(2.0, 0.5)
    c2 = to_dual(rM, rE)
    assert abs(c2["chi"] + c1["chi"]) < 1e-12
    assert abs(c2["theta"] - (np.pi / 2.0 - c1["theta"])) < 1e-12
    assert abs(c2["rho"] - c1["rho"]) < 1e-12  # ρ invariante


def test_diagonal_is_fixed_locus():
    """La diagonal dual θ=π/4 (χ=0) es el lugar fijo del intercambio."""
    assert on_diagonal(1.3, 1.3)
    assert not on_diagonal(1.3, 0.7)
    c = to_dual(1.3, 1.3)
    assert abs(c["theta"] - THETA_DIAGONAL) < 1e-12


def test_physical_domain_first_quadrant():
    assert is_physical(1.0, 0.0)
    assert not is_physical(-0.1, 1.0)


# ------------------------------ Cap. 3 -----------------------------------

def test_scaled_params_and_marginal_rigidity():
    """(3.2): M0² ∝ δ0² — «la perfección apenas se sostiene»."""
    p1, p2 = scaled_params(1e-2), scaled_params(2e-2)
    assert abs(p2["M0_sq"] / p1["M0_sq"] - 4.0) < 1e-12   # δ0² exacto
    assert abs(p2["B"] / p1["B"] - 2.0) < 1e-12           # δ0
    assert abs(p2["eta"] / p1["eta"] - 8.0) < 1e-12       # δ0³


def test_T0_scaling_law_exponent_3():
    """Prop. 3.4: T0 = c̄·δ0³ — barrido de δ0 y ajuste del exponente."""
    deltas = np.logspace(-4, -1, 12)
    T0s = np.array([T0_numeric(d) for d in deltas])
    slope = np.polyfit(np.log(deltas), np.log(T0s), 1)[0]
    assert abs(slope - 3.0) < 1e-6
    # Y coincide con la analítica c̄·δ0³:
    for d in (1e-3, 1e-2):
        assert abs(T0_numeric(d) - T0_analytic(d)) / T0_analytic(d) < 1e-10


def test_T0_zero_iff_delta_zero():
    """Axioma 3 / Def. 3.3: T0 = 0 ⟺ δ0 = 0."""
    assert T0_numeric(0.0) == 0.0
    assert T0_numeric(1e-3) > 0.0


def test_negative_control_flat_at_delta_zero():
    """Control negativo (§3.2): con δ0=0 el pozo es plano hasta sexto
    orden (V0 = C0ρ⁶/6 exactamente) y no hay salida (∇V(0) = 0)."""
    rho = np.linspace(0.0, 0.5, 20)
    v = V0(rho, 0.0, delta0=0.0)
    assert np.allclose(v, rho ** 6 / 6.0, atol=1e-15)
    assert np.allclose(grad_V(np.array([0.0, 0.0]), 0.0), 0.0)


def test_quasi_cancellation_and_kappa():
    """(3.3) con los defaults; κ+ y c̄ son O(1) y positivos."""
    assert quasi_cancellation_ok()
    assert 0.1 < kappa_plus() < 10.0
    assert c_bar() > 0.0
    # Control: violar (3.3) (C0 grande) → sin vacío más profundo
    assert not quasi_cancellation_ok(C0=2.0)
    assert T0_numeric(1e-2, C0=2.5) <= 0.0 or discriminant(1e-2, C0=2.5) < 0


def test_chi_residue_linear():
    """⟨χ⟩ ≃ (ē/m̄²)·δ0 (Prop. 3.4) — la semilla de la memoria."""
    assert abs(chi_residue(2e-3) / chi_residue(1e-3) - 2.0) < 1e-12


# ------------------------------ Cap. 4 -----------------------------------

def test_monotonia_del_camino():
    """Teo. 4.5: V no creciente y producción entrópica ≥ 0 en toda la
    trayectoria — la flecha del tiempo como teorema."""
    res = flow(np.array([1e-4, 1e-4]), delta0=0.05, n_steps=5000)
    dV = np.diff(res["V"])
    assert np.all(dV <= 1e-12)
    assert np.all(res["S_production_rate"] >= 0.0)


def test_monotonia_identity():
    """dV/dσ = −(∇V)ᵀG⁻¹(∇V): la identidad del Teo. 4.5, numérica."""
    phi = np.array([0.3, 0.1])
    g = grad_V(phi, delta0=0.05)
    # En el interior (sin proyección), un paso de Euler:
    d_sigma = 1e-6
    from core.dual_plane import to_dual
    from core.basal import V0 as V
    c0 = to_dual(phi[0], phi[1])
    v0 = V(c0["rho"], c0["chi"], 0.05)
    phi1 = phi - d_sigma * g
    c1 = to_dual(phi1[0], phi1[1])
    v1 = V(c1["rho"], c1["chi"], 0.05)
    assert abs((v1 - v0) / d_sigma + g @ g) < 1e-4 * max(g @ g, 1.0)


def test_exclusion_no_return():
    """Lema 4.7: sin órbitas periódicas — V estrictamente decreciente
    mientras el gradiente no se anula, sobre un barrido de condiciones
    iniciales; el flujo no vuelve al punto de partida."""
    rng = np.random.default_rng(9)
    for _ in range(5):
        phi0 = rng.uniform(0.05, 1.5, size=2)
        res = flow(phi0, delta0=0.05, n_steps=4000)
        V = res["V"]
        assert np.all(np.diff(V) <= 1e-12)
        # no-retorno: el punto final está lejos del inicial o el
        # potencial ha bajado estrictamente (nunca órbita cerrada)
        moved = np.linalg.norm(res["final"] - phi0)
        assert moved > 1e-6 or V[-1] < V[0] - 1e-15 or \
            np.linalg.norm(grad_V(phi0, 0.05)) < 1e-10


def test_falso_vacio_es_metaestable():
    """Prop. 3.5: la salida de S0 es NUCLEACIÓN (Coleman), no rodadura
    clásica — el flujo desde el entorno del origen queda atrapado en el
    falso vacío (el origen es mínimo local, §3.2 (iii))."""
    res = flow(np.array([1e-6, 1e-6]), delta0=0.05, n_steps=5000)
    rho_barrier = float(np.sqrt(0.382 * 0.05))  # x− = κ−·δ0, κ− ≈ 0.382
    assert float(np.linalg.norm(res["final"])) < 0.5 * rho_barrier


def test_salida_hacia_polo_de_masa():
    """Prop. 3.5: tras la nucleación (más allá de la barrera x−), la
    caída ocurre hacia el polo de masa (θ ≈ 0) y termina en el anillo de
    vacíos ρ+ = √(κ+·δ0): el máximo de χ sobre el anillo está en φE=0."""
    delta0 = 0.05
    # punto post-nucleación: justo tras la barrera, con componente Ep
    rho_barrier = float(np.sqrt(0.382 * delta0))
    phi0 = np.array([1.2 * rho_barrier, 0.5 * rho_barrier])
    res = flow(phi0, delta0=delta0, d_sigma=0.05, n_steps=40000)
    assert exits_to_mass_pole(res)
    rho_final = float(np.linalg.norm(res["final"]))
    rho_plus = float(np.sqrt(kappa_plus() * delta0))
    assert abs(rho_final - rho_plus) / rho_plus < 0.05


def test_inercia_eterna_delta_zero():
    """Límite de recuperación (§13.5): δ0 → 0 devuelve la inercia eterna
    — el flujo desde el origen no va a ninguna parte."""
    res = flow(np.array([0.0, 0.0]), delta0=0.0, n_steps=100)
    assert np.linalg.norm(res["final"]) < 1e-15


def test_rigidez_must_be_positive():
    """La rigidez G ≻ 0 es hipótesis del teorema: G indefinida se rechaza."""
    with pytest.raises(ValueError):
        flow(np.array([0.1, 0.1]), delta0=0.05,
             G=np.array([[1.0, 0.0], [0.0, -1.0]]), n_steps=10)
