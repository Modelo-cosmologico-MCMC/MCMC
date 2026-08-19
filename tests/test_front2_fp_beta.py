"""Suite del frente 2 — β de Fokker-Planck (Def. 4.4).

Valida la DERIVACIÓN (no el desenlace): álgebra analítica contra
diferencias finitas, el a-término contra el suavizado gaussiano exacto,
el b-término contra la multiplicación polinómica directa, y la
ecuación completa contra la solución EXACTA de la FP vía Hopf-Cole
(e^{−V_t} = e^{tΔ/2}·e^{−V_0}) en el régimen declarado de validez del
truncamiento. Todo interno (E8), sin datos, sin red.
"""

import numpy as np
import pytest

from core.fokker_planck_beta import (
    beta_functions,
    beta_functions_quartic,
    canonical_closure,
    spinodal_canonical_point,
    stability_matrix,
    stability_matrix_quartic,
)


def _c_from_phys(M0_sq, B, C0):
    return np.array([M0_sq / 2.0, -B / 4.0, C0 / 6.0])


def test_stability_matrix_is_jacobian():
    """M_ij = ∂βi/∂λj analítica == jacobiano numérico de las β."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        lam = rng.uniform(0.2, 3.0, size=3)
        a, b = rng.uniform(0.1, 2.0, size=2)
        M = stability_matrix(*lam, a=a, b=b)
        eps = 1e-7
        for j in range(3):
            lp, lm = lam.copy(), lam.copy()
            lp[j] += eps
            lm[j] -= eps
            col = (beta_functions(*lp, a=a, b=b)
                   - beta_functions(*lm, a=a, b=b)) / (2 * eps)
            assert np.allclose(M[:, j], col, rtol=1e-6, atol=1e-6)


def test_stability_matrix_quartic_is_jacobian():
    """Ídem para la variante cuártica (4×4), incluida E4 ≠ 0."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        lam = rng.uniform(0.2, 3.0, size=4)
        a, b = rng.uniform(0.1, 2.0, size=2)
        M = stability_matrix_quartic(*lam, a=a, b=b)
        eps = 1e-7
        for j in range(4):
            lp, lm = lam.copy(), lam.copy()
            lp[j] += eps
            lm[j] -= eps
            col = (beta_functions_quartic(*lp, a=a, b=b)
                   - beta_functions_quartic(*lm, a=a, b=b)) / (2 * eps)
            assert np.allclose(M[:, j], col, rtol=1e-6, atol=1e-6)


def test_diffusion_term_exact_on_polynomials():
    """El a-término es el suavizado gaussiano EXACTO: V_t(φ) =
    E[V(φ + √t·ξ)] con ξ ~ N(0, I₂), comparado con los acoplos
    evolucionados por las EDO lineales del a-término (b = 0):
        c3(t) = c3;  c2(t) = c2 + 36a·c3·t;
        c1(t) = c1 + 16a·c2·t + 288a²·c3·t²;
        c0(t) = 4a·c1·t + 32a²·c2·t² + 384a³·c3·t³  (identidad exacta,
    verificada por cuadratura de Gauss-Hermite, sin Monte Carlo)."""
    a = 0.5                      # ⟺ varianza del kernel = t exacta
    c1, c2, c3 = 0.7, -0.3, 0.2
    t = 0.37

    # cuadratura Gauss-Hermite 2D para E[V(φ+√t·ξ)]
    nodes, weights = np.polynomial.hermite_e.hermegauss(24)
    W = np.outer(weights, weights) / (2 * np.pi)

    def V(pm, pe):
        x = pm ** 2 + pe ** 2
        return c1 * x + c2 * x ** 2 + c3 * x ** 3

    # acoplos evolucionados (integración exacta de las EDO lineales):
    c3t = c3
    c2t = c2 + 36 * a * c3 * t
    c1t = c1 + 16 * a * c2 * t + 288 * a ** 2 * c3 * t ** 2
    c0t = (4 * a * c1 * t + 32 * a ** 2 * c2 * t ** 2
           + 384 * a ** 3 * c3 * t ** 3)

    rng = np.random.default_rng(3)
    for _ in range(6):
        pm, pe = rng.uniform(-1.2, 1.2, size=2)
        gm = pm + np.sqrt(t) * nodes[:, None]
        ge = pe + np.sqrt(t) * nodes[None, :]
        smeared = float(np.sum(W * V(gm, ge)))
        x = pm ** 2 + pe ** 2
        predicted = c0t + c1t * x + c2t * x ** 2 + c3t * x ** 3
        assert smeared == pytest.approx(predicted, rel=1e-9, abs=1e-9)


def test_drift_term_matches_polynomial_multiplication():
    """El b-término: los coeficientes truncados de −b·4x·(V'(x))²
    contra la multiplicación polinómica directa (numpy.polynomial)."""
    rng = np.random.default_rng(4)
    for _ in range(10):
        M0_sq, B, C0 = rng.uniform(0.2, 3.0, size=3)
        b = float(rng.uniform(0.1, 2.0))
        c1, c2, c3 = _c_from_phys(M0_sq, B, C0)
        vp = np.array([c1, 2 * c2, 3 * c3])          # V'(x)
        sq = np.polynomial.polynomial.polymul(vp, vp)
        grad2 = 4.0 * np.concatenate([[0.0], sq])     # 4x·(V')²
        # dc/dt del b-término, truncado a grado 3 en x:
        dc = -b * grad2[1:4]
        beta_b_only = beta_functions(M0_sq, B, C0, a=0.0, b=b)
        assert beta_b_only[0] == pytest.approx(2 * dc[0], rel=1e-12)
        assert beta_b_only[1] == pytest.approx(-4 * dc[1], rel=1e-12)
        assert beta_b_only[2] == pytest.approx(6 * dc[2], rel=1e-12)


def test_full_equation_against_exact_fp_hopf_cole():
    """Control de la ECUACIÓN COMPLETA contra la FP exacta: con
    Θ = 1, W = e^{−V} satisface dW/dt = ½ΔW, así que
    V_t = −ln(e^{tΔ/2} e^{−V0}) EXACTO (suavizado por Gauss-Hermite).
    En el régimen declarado (acoplos pequeños, t pequeño — donde la
    generación de x⁴⁺ es subdominante), la deriva medida de (c1,c2,c3)
    por ajuste local debe coincidir con las β truncadas (a = b = ½).
    Éste es el control que la preinscripción promete."""
    a = b = 0.5
    M0_sq, B, C0 = 0.10, 0.12, 0.08     # régimen declarado
    c = _c_from_phys(M0_sq, B, C0)

    nodes, weights = np.polynomial.hermite_e.hermegauss(32)
    W2 = np.outer(weights, weights) / (2 * np.pi)

    def V0(pm, pe):
        x = pm ** 2 + pe ** 2
        return c[0] * x + c[1] * x ** 2 + c[2] * x ** 3

    # malla de ajuste local (ventana declarada x < 1.0); la base sube
    # hasta x⁵ para que el contenido x⁴⁺ REAL que la FP exacta genera
    # no se filtre (aliasing) en c1..c3:
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1.0, 1.0, size=(800, 2))
    xs = np.sum(pts ** 2, axis=1)
    keep = xs < 1.0
    pts, xs = pts[keep], xs[keep]
    A = np.stack([xs ** k for k in range(6)], axis=1)

    def dc_at(t):
        Vt = np.empty(len(pts))
        for i, (pm, pe) in enumerate(pts):
            gm = pm + np.sqrt(t) * nodes[:, None]
            ge = pe + np.sqrt(t) * nodes[None, :]
            Vt[i] = -np.log(float(np.sum(W2 * np.exp(-V0(gm, ge)))))
        coef, *_ = np.linalg.lstsq(A, Vt, rcond=None)
        return (coef[1:4] - c) / t

    # extrapolación de Richardson t → 0 (elimina el O(t)):
    t = 0.02
    dc_measured = 2.0 * dc_at(t / 2) - dc_at(t)

    dl_pred = beta_functions(M0_sq, B, C0, a=a, b=b)
    dc_pred = np.array([dl_pred[0] / 2, -dl_pred[1] / 4, dl_pred[2] / 6])

    # tolerancia del régimen declarado (truncamiento + t finito):
    scale = np.max(np.abs(dc_pred))
    assert np.allclose(dc_measured, dc_pred, atol=0.02 * scale), \
        f"medido {dc_measured} vs predicho {dc_pred}"


def test_closure_is_frozen_dataclass():
    """El cierre es inmutable: el análisis lo recibe, no lo ajusta."""
    cl = canonical_closure()
    with pytest.raises(Exception):
        cl.a = 1.0
    assert cl.a == cl.b == 0.5
    assert cl.truncation == "cubic"
    assert cl.s0_band == 0.10
    assert cl.s0_target == pytest.approx(np.pi / np.log(10.0))


def test_spinodal_point_is_on_spinodal():
    """El punto preinscrito cumple D = B² − 4·C0·M0² = 0 exacto."""
    M0_sq, B, C0 = spinodal_canonical_point()
    assert B ** 2 - 4.0 * C0 * M0_sq == 0.0
