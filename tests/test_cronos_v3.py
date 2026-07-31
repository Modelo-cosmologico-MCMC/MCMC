"""Tests de Cronos v3 (Tratado de Fundamentos v35, cap. 11).

Incluye la firma falsable de la fig. 11.1: la fricción de compuerta está
activa durante el colapso (ρ̇ > 0) y se apaga exactamente al virializar
(ρ̇ ≤ 0), a diferencia del esquema v32 que la mantenía siempre activa.
"""

import numpy as np
import pytest

from cronos.cronos_v3 import (
    ALPHA0_INV_MAX, check_alpha0_inv, epsilon_c, lapse,
    gate_friction, gate_friction_freefall, extra_force, kdk_step_v3,
)


ALPHA = 1e-7   # dentro de la cota
RHO_C = 200.0


def test_alpha0_bound_enforced():
    """La cota dura α0⁻¹ ≲ 1e-6 (ec. 11.5) se rechaza si se viola."""
    check_alpha0_inv(ALPHA0_INV_MAX)  # el límite exacto pasa
    with pytest.raises(ValueError):
        check_alpha0_inv(1e-5)
    with pytest.raises(ValueError):
        epsilon_c(np.array([100.0]), 1e-3, RHO_C)


def test_lapse_below_one_in_dense_regions():
    """En regiones densas sin potencial, N < 1 (la descarga se estanca)."""
    rho = np.array([0.0, RHO_C, 100.0 * RHO_C])
    N = lapse(np.zeros(3), rho, ALPHA, RHO_C)
    assert N[0] == 1.0
    assert N[1] < 1.0
    assert N[2] < N[1]


def test_gate_friction_collapse_history():
    """Firma falsable (fig. 11.1): Γ > 0 durante el colapso, Γ = 0 tras
    virializar."""
    # Historia sintética: ρ crece (colapso) y luego se estabiliza
    rho = np.array([10.0, 50.0, 200.0, 500.0, 500.0, 500.0])
    rho_dot = np.array([5.0, 20.0, 40.0, 10.0, 0.0, -1.0])
    Gamma = gate_friction(rho, rho_dot, ALPHA, RHO_C)
    assert np.all(Gamma[:4] > 0.0)      # colapsando: fricción activa
    assert np.all(Gamma[4:] == 0.0)     # virializado (ρ̇ ≤ 0): apagada


def test_gate_friction_freefall_exponent_two():
    """Cor. 11.3b: la forma cerrada escala como (ρ/ρ_c)² (exponente 2)."""
    g1 = gate_friction_freefall(np.array([RHO_C]), ALPHA, RHO_C)
    g2 = gate_friction_freefall(np.array([2.0 * RHO_C]), ALPHA, RHO_C)
    assert abs(g2[0] / g1[0] - 4.0) < 1e-12


def test_extra_force_attracts_to_peaks():
    """Cor. 11.3c: F_extra = +c²∇ε_c apunta hacia el gradiente de ε_c."""
    grad = np.array([[0.5, 0.0, -0.25]])
    F = extra_force(grad, c=2.0)
    assert np.allclose(F, 4.0 * grad)


def test_kdk_recovers_newtonian_when_alpha_zero():
    """Con α0⁻¹ = 0, un paso de Cronos v3 == leapfrog KDK newtoniano."""
    rng = np.random.default_rng(3)
    n = 8
    x0 = rng.normal(size=(n, 3))
    u0 = rng.normal(size=(n, 3))
    g = rng.normal(size=(n, 3))          # −∇Φ_N constante durante el paso
    dt = 1e-3
    x1, u1 = kdk_step_v3(
        x0.copy(), u0.copy(), dt,
        Phi_N=np.zeros(n), grad_Phi_N=-g, rho=np.full(n, 100.0),
        rho_dot=np.full(n, 10.0), grad_eps_c=np.zeros((n, 3)),
        alpha0_inv=0.0, rho_c=RHO_C,
    )
    # Leapfrog newtoniano de referencia (N=1, Γ=0, sin fuerza extra):
    u_half = u0 + g * dt / 2.0
    x_ref = x0 + u_half * dt
    u_ref = u_half + g * dt / 2.0
    assert np.allclose(x1, x_ref, atol=1e-12)
    assert np.allclose(u1, u_ref, atol=1e-12)


def test_kdk_friction_damps_velocity():
    """Durante el colapso, la fricción reduce |u| frente al caso sin ella."""
    n = 4
    x0 = np.zeros((n, 3))
    u0 = np.ones((n, 3))
    kwargs = dict(
        Phi_N=np.zeros(n), grad_Phi_N=np.zeros((n, 3)),
        rho=np.full(n, 100.0 * RHO_C), grad_eps_c=np.zeros((n, 3)),
        rho_c=RHO_C,
    )
    _, u_col = kdk_step_v3(x0.copy(), u0.copy(), 1.0,
                           rho_dot=np.full(n, 100.0), alpha0_inv=ALPHA,
                           **kwargs)
    _, u_vir = kdk_step_v3(x0.copy(), u0.copy(), 1.0,
                           rho_dot=np.zeros(n), alpha0_inv=ALPHA,
                           **kwargs)
    assert np.all(np.linalg.norm(u_col, axis=1)
                  < np.linalg.norm(u_vir, axis=1))
