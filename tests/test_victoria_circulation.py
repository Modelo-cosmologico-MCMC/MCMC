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


def _random_in_class(rng, dim=3):
    """Un sorteo de la clase declarada: G ≻ 0, H simétrica, J₀ antisim."""
    A = rng.standard_normal((dim, dim))
    B = rng.standard_normal((dim, dim))
    C = rng.standard_normal((dim, dim))
    return (A @ A.T + 0.5 * np.eye(dim), 0.5 * (B + B.T), 0.5 * (C - C.T))


def _find_reentrant_draw(seed=7, n_draws=400, alpha_max=50.0, n_grid=200):
    """Primer sorteo 3D cuya complejificación es RE-ENTRANTE: s0 > 0 en
    un tramo acotado pero s0(α_max) = 0 (existe — ~20 % de los sorteos)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, alpha_max, n_grid)
    for _ in range(n_draws):
        G, H, J0 = _random_in_class(rng)
        s0s = np.array([max_im_spectrum(circulation_flow_matrix(G, H, J0, a))
                        for a in grid])
        if s0s[-1] < 1e-10 and (s0s > 1e-8).any():
            return G, H, J0
    raise RuntimeError("sin sorteo re-entrante en el presupuesto")


def test_alpha_threshold_reentrant_first_onset():
    """El conjunto {α : s0(α) > 0} puede ser re-entrante en 3D: la
    versión que solo miraba s0(α_max) declaraba «sin complejificación»
    donde sí la hay. El barrido debe devolver el PRIMER arranque:
    s0 = 0 en toda la malla por debajo y s0 > 0 justo por encima."""
    G, H, J0 = _find_reentrant_draw()
    a_c = alpha_threshold(G, H, J0)          # antes: ValueError falso
    below = np.linspace(0.0, a_c * 0.98, 50)
    s_below = [max_im_spectrum(circulation_flow_matrix(G, H, J0, a))
               for a in below]
    assert max(s_below) <= 1e-10
    s_above = max_im_spectrum(
        circulation_flow_matrix(G, H, J0, a_c * 1.02))
    assert s_above > 1e-10


def test_alpha_victoria_first_crossing_not_endpoint():
    """α_Victoria debe ser el PRIMER cruce del objetivo aunque s0(α)
    recaiga después: en la malla no existe ningún α < α_V con
    s0(α) ≥ objetivo, y s0(α_V) = objetivo."""
    rng = np.random.default_rng(7)
    grid = np.linspace(0.0, 50.0, 200)
    for _ in range(400):
        G, H, J0 = _random_in_class(rng)
        s0s = np.array([max_im_spectrum(circulation_flow_matrix(G, H, J0, a))
                        for a in grid])
        if (s0s >= S0_TARGET).any() and s0s[-1] < S0_TARGET:
            break
    else:
        pytest.skip("sin sorteo con recaída bajo el objetivo")
    a_v = alpha_victoria(G, H, J0)
    s_at = max_im_spectrum(circulation_flow_matrix(G, H, J0, a_v))
    assert abs(s_at - S0_TARGET) < 1e-6
    below = grid[grid < a_v * 0.99]
    s_below = np.array([max_im_spectrum(
        circulation_flow_matrix(G, H, J0, a)) for a in below])
    assert np.all(s_below < S0_TARGET)


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


def test_signal_route_zero_for_random_gradient_flows():
    """F2D, control negativo sobre SORTEOS de la clase (no solo el caso
    diagonal): sin las guardias de amplitud/semiperiodo, el iterado
    convergido de un gradiente puro puede caer en un 2-ciclo de coma
    flotante (~1 ulp) y registrar un falso s0 = π/dt — el sorteo nº 4
    de esta semilla lo exhibía. El estimador debe devolver 0.0 en
    todos."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        A = rng.standard_normal((3, 3))
        B = rng.standard_normal((3, 3))
        G = A @ A.T + 0.5 * np.eye(3)
        H = 0.5 * (B + B.T)
        assert obstruction_holds(G, H)
        assert s0_signal(gradient_flow_matrix(G, H)) == 0.0


def test_spinodal_point_and_gradient_pinned():
    """F2C: el cuello declarado está SOBRE el discriminante del
    tratado (D = B² − 4·C0·M0², Def. 8.4 — la misma forma que usa
    core/decade.py) y la ∇D cableada coincide con el gradiente
    numérico de esa D."""
    from core.victoria_circulation import SPINODAL_POINT

    def D(lam):
        m0_sq, b, c0 = lam
        return b ** 2 - 4.0 * c0 * m0_sq

    assert D(SPINODAL_POINT) == 0.0
    h = 1e-7
    num = np.array([
        (D(SPINODAL_POINT + h * e) - D(SPINODAL_POINT - h * e)) / (2 * h)
        for e in np.eye(3)])
    n = spinodal_normal()
    assert np.allclose(n, num / np.linalg.norm(num), atol=1e-7)


def test_tangential_separation_is_example_bound():
    """F2C, limitación ejecutable: con H genérica la circulación del
    plano de CRUCE puede sobrevivir a P_T (P_T·J_cruce·H·P_T ≠ 0 si H
    acopla n al tangente) — la separación exacta es propiedad del
    ejemplo construido G = H = I, no de P_T en general."""
    n = spinodal_normal()
    a = np.array([1.0, 0.0, 0.0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1)
    J_norm = np.outer(n, t1) - np.outer(t1, n)
    rng = np.random.default_rng(1)
    survived = 0.0
    for _ in range(200):
        B = rng.standard_normal((3, 3))
        H = 2.5 * np.eye(3) + 0.5 * (B + B.T)
        M = circulation_flow_matrix(np.eye(3), H, J_norm, alpha=2.0)
        survived = max(survived, max_im_spectrum(tangential_part(M, n)))
    assert survived > 0.1


def test_status_declares_scope():
    """El alcance queda declarado: clase β = (−G⁻¹+J)∇C, mapa no
    constante, β reales pendientes."""
    assert "clase declarada" in STATUS_CIRCULATION
    assert "Fokker-Planck" in STATUS_CIRCULATION
    assert "no demuestra" in STATUS_CIRCULATION
