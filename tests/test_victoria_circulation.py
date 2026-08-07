"""Tests de la Circulación de Victoria (F2A–F2D, propuesta v36)."""

import numpy as np
import pytest

from core.victoria_circulation import (
    S0_TARGET,
    STATUS_CIRCULATION,
    J_plane,
    alpha_c_closed_form_2d,
    alpha_threshold,
    alpha_victoria,
    alpha_victoria_closed_form_2d,
    circulation_flow_matrix,
    gradient_flow_matrix,
    max_im_spectrum,
    obstruction_holds,
    s0_signal,
    spinodal_normal,
    tangential_part,
    tangential_projector,
)

H2 = np.diag([1.0, 3.0])
I2 = np.eye(2)


def test_obstruction_theorem_over_random_draws():
    """F2A: gradiente puro (J = 0) ⟹ espectro real, para CUALQUIER
    G ≻ 0 y H simétrica — el control negativo del frente 2: sin
    circulación no hay DSI."""
    rng = np.random.default_rng(20260807)
    for _ in range(200):
        A = rng.standard_normal((3, 3))
        G = A @ A.T + 3.0 * np.eye(3)          # SPD
        B = rng.standard_normal((3, 3))
        H = 0.5 * (B + B.T)                     # simétrica (indefinida ok)
        assert obstruction_holds(G, H)


def test_circulation_breaks_obstruction():
    """Con J ≠ 0 la misma (G, H) puede complejificar: la implicación
    s0 ≠ 0 ⟹ J ≠ 0 tiene contenido (existe el caso con giro)."""
    M = circulation_flow_matrix(I2, H2, J_plane(2), alpha=2.0)
    assert max_im_spectrum(M) > 0.1


def test_alpha_c_matches_closed_form():
    """F2B: α_c medido por bisección clava la forma cerrada 2D
    α_c = |h1−h2|/(2√(h1h2))."""
    a_num = alpha_threshold(I2, H2, J_plane(2))
    a_cf = alpha_c_closed_form_2d(1.0, 3.0)
    assert abs(a_num - a_cf) / a_cf < 1e-6


def test_alpha_c_vanishes_isotropic():
    """H isótropa: cualquier circulación complejifica (α_c = 0) —
    el umbral finito lo crea la ANISOTROPÍA del paisaje."""
    assert alpha_threshold(I2, np.eye(2), J_plane(2)) < 1e-9


def test_alpha_victoria_matches_closed_form_and_target():
    """F2B: α_V medido clava la forma cerrada y s0(α_V) = π/ln10."""
    a_num = alpha_victoria(I2, H2, J_plane(2))
    a_cf = alpha_victoria_closed_form_2d(1.0, 3.0)
    assert abs(a_num - a_cf) / a_cf < 1e-6
    s0 = max_im_spectrum(circulation_flow_matrix(I2, H2, J_plane(2), a_num))
    assert abs(s0 - S0_TARGET) < 1e-9


def test_alpha_victoria_unreachable_raises():
    with pytest.raises(ValueError):
        alpha_victoria(I2, H2, J_plane(2), alpha_max=alpha_c_closed_form_2d(
            1.0, 3.0) * 1.0001)


def test_spinodal_normal_and_projector():
    """F2C: n unitario en el cuello; P_T proyector con P_T·n = 0."""
    n = spinodal_normal()
    P = tangential_projector(n)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-12
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.allclose(P @ n, 0.0, atol=1e-12)


def test_tangential_separates_crossing_from_walking():
    """F2C: circulación construida EN el plano tangente sobrevive a
    P_T·M·P_T; la construida en un plano que contiene a n, no."""
    n = spinodal_normal()
    # base ortonormal (n, t1, t2):
    a = np.array([1.0, 0.0, 0.0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    J_tang = np.outer(t1, t2) - np.outer(t2, t1)
    J_norm = np.outer(n, t1) - np.outer(t1, n)
    for J, survives in ((J_tang, True), (J_norm, False)):
        M = circulation_flow_matrix(np.eye(3), np.eye(3), J, alpha=2.0)
        s_full = max_im_spectrum(M)
        s_tang = max_im_spectrum(tangential_part(M, n))
        assert s_full > 0.1
        assert (s_tang > 0.1) is survives


def test_signal_route_matches_spectral():
    """F2D: la ruta de señal (cruces por cero, sin espectro de M)
    coincide con la espectral sobre la matriz fiducial del frente
    (par 0.3 ± i·π/ln10 con modo real acoplado)."""
    M = np.zeros((3, 3))
    M[:2, :2] = np.array([[0.3, -S0_TARGET], [S0_TARGET, 0.3]])
    M[2, 2] = -0.7
    M[0, 2] = 0.4
    s_sig = s0_signal(M)
    assert abs(s_sig - S0_TARGET) / S0_TARGET < 1e-3


def test_signal_route_zero_for_gradient():
    """F2D, control negativo: sobre un gradiente puro la señal no
    oscila y el estimador devuelve 0."""
    M = gradient_flow_matrix(np.eye(3), np.diag([1.0, 2.0, 3.0]))
    assert s0_signal(M) == 0.0


def test_status_declares_scope():
    """El alcance queda declarado: clase β = (−G⁻¹+J)∇C, mapa no
    constante, β reales pendientes."""
    assert "clase declarada" in STATUS_CIRCULATION
    assert "Fokker-Planck" in STATUS_CIRCULATION
    assert "no demuestra" in STATUS_CIRCULATION
