"""Invariantes físicos del fondo — el nivel 3 del CI (propuesta v36 §XI).

Un resultado no puede estar «verde» solo porque el programa corre: debe
conservar las relaciones que la ontología declara invariantes. El caso
que motivó esta suite es real: el CI estaba completamente verde
mientras validate_all imprimía H(0) = 70.09 con H0 = 69.8 — el fondo
violaba su propia definición (+0.4 % en el punto fiducial, y clausura
no plana al variar Ω_m). Corrección: ago-2026, rama
fix/background-normalization.
"""

import numpy as np
import pytest

from cosmology.background import OMEGA_L0, OMEGA_R0, H_of_z, Lambda_rel

OMEGA_M_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
EPS_GRID = [0.0, 0.01, 0.05]


@pytest.mark.parametrize("Omega_m", OMEGA_M_GRID)
@pytest.mark.parametrize("eps", EPS_GRID)
def test_H0_is_definitionally_H0(Omega_m, eps):
    """H(z=0; θ) = H0 para TODO θ admisible — por definición, no por
    coincidencia fiducial (el invariante que el CI verde violaba)."""
    assert float(H_of_z(0.0, H0=67.4, Omega_m=Omega_m, eps=eps)) \
        == pytest.approx(67.4, rel=1e-12)


@pytest.mark.parametrize("eps", EPS_GRID)
def test_lambda_rel_today_is_omega_L0(eps):
    """Ω_Λ_rel(0) = Ω_Λ0 exacto para todo ε (transición normalizada:
    F(z)/F(0) con F(0) ≡ F(0))."""
    assert float(Lambda_rel(0.0, eps=eps)) \
        == pytest.approx(OMEGA_L0, rel=1e-15)


@pytest.mark.parametrize("Omega_m", OMEGA_M_GRID)
def test_flat_closure_per_call(Omega_m):
    """Clausura plana por llamada: H²(0)/H0² = Ω_m + Ω_r + Ω_DE,0 = 1
    exacto para cualquier Ω_m (antes solo se cumplía en 0.300)."""
    h_ratio_sq = (float(H_of_z(0.0, H0=100.0, Omega_m=Omega_m)) / 100.0) ** 2
    assert h_ratio_sq == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("Omega_m", [0.25, 0.30, 0.35])
def test_recovery_lcdm_for_any_omega_m(Omega_m):
    """Prop. A.1 sobre una MALLA de parámetros (no solo el punto
    fiducial): con ε = 0, H(z) es ΛCDM plano exacto para todo Ω_m."""
    z = np.linspace(0.0, 1100.0, 300)
    mcmc = np.asarray(H_of_z(z, H0=67.4, Omega_m=Omega_m, eps=0.0))
    OmDE = 1.0 - Omega_m - OMEGA_R0
    lcdm = 67.4 * np.sqrt(Omega_m * (1 + z) ** 3
                          + OMEGA_R0 * (1 + z) ** 4 + OmDE)
    assert np.max(np.abs(mcmc - lcdm)) == pytest.approx(0.0, abs=1e-10)


def test_transition_amplitude_preserved():
    """La normalización no mata la física: la transición sigue ahí —
    a z ≫ z_trans, Ω_Λ_rel/Ω_Λ0 → (1−ε)/F(0) < 1 (la meseta baja), y
    la amplitud total del salto sigue gobernada por ε."""
    eps = 0.05
    deep = float(Lambda_rel(100.0, eps=eps)) / OMEGA_L0
    today = float(Lambda_rel(0.0, eps=eps)) / OMEGA_L0
    F0 = 1.0 + eps * np.tanh(8.9 / 1.5)
    assert today == pytest.approx(1.0, rel=1e-15)
    assert deep == pytest.approx((1.0 - eps) / F0, rel=1e-10)
    assert deep < 1.0


def test_H_still_monotone_with_transition():
    """H(z) sigue siendo creciente con la transición activa (sanidad)."""
    z = np.linspace(0.0, 50.0, 500)
    H = np.asarray(H_of_z(z, eps=0.05))
    assert np.all(np.diff(H) > 0.0)