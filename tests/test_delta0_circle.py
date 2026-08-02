"""Tests del círculo de δ₀ (ronda 5): atractor de Victoria vs. empalme C¹."""

import numpy as np

import mass_program.B7_empalme as B7
from core.basal import c_bar
from core.delta0_circle import (
    LAMBDA_H_SEAL, delta0_H, W_max_required, attractor_analytic,
    attractor_numeric, consistency_factors, closure_test, landscape_scan,
)

D_MIN = 1e-4          # suelo de activación para los tests
GAMMA_FERTILE = 1.5   # A = 1.5 > 1 con las formas fiduciales (fértil)


def test_delta0_H_matches_B7():
    """δ_H del círculo == δ0* del empalme (misma ecuación, H.8):
    0.130/√5 ≈ 0.0581 con las formas fiduciales."""
    assert delta0_H() == B7.delta0_required()
    assert abs(delta0_H() - 0.0581) < 5e-4


def test_W_max_required_fiducial():
    """La ecuación de consistencia: W_max = c̄·δ_H³ ≈ 1.65e-4."""
    w = W_max_required(delta0_H())
    assert abs(w - c_bar() * delta0_H() ** 3) < 1e-18
    assert abs(w - 1.652e-4) < 2e-6


def test_attractor_is_the_ceiling_universally():
    """En toda la región fértil el atractor es el Techo: independiente
    de γR (mientras A > 1) y del δ inicial sobre el suelo (Teo. 10.6)."""
    W = W_max_required(delta0_H())
    targets = []
    for g in (1.1, 1.5, 2.7):
        for d0 in (2e-4, 0.012, 0.03):
            r = attractor_numeric(g, W, d0, delta_min=D_MIN)
            targets.append(r["attractor"])
    assert np.allclose(targets, targets[0], rtol=1e-12)
    assert abs(targets[0] - delta0_H()) < 1e-12


def test_analytic_equals_numeric():
    """La forma cerrada del atractor coincide con la iteración real."""
    for g, W in ((1.5, 1e-4), (2.0, 5e-4), (0.8, 1e-4)):
        ana = attractor_analytic(g, W, delta_min=D_MIN)
        num = attractor_numeric(g, W, 0.012, delta_min=D_MIN)["attractor"]
        assert abs(ana - num) < 1e-15


def test_circle_closes_iff_W_max_matches():
    """El círculo cierra con W_max = c̄·δ_H³ y NO cierra con W_max/2
    (control negativo: el atractor es el Techo, no un 0.058 intrínseco)."""
    W_close = W_max_required(delta0_H())
    good = closure_test(W_close, gamma_R=GAMMA_FERTILE,
                        delta_min=D_MIN, delta_init=0.012)
    assert good["closes"]
    bad = closure_test(0.5 * W_close, gamma_R=GAMMA_FERTILE,
                       delta_min=D_MIN, delta_init=0.012)
    assert not bad["closes"]
    # con la mitad de Techo, el atractor cae el factor 2^{1/3}:
    assert abs(bad["attractor"] - delta0_H() / 2.0 ** (1.0 / 3.0)) < 1e-12


def test_silence_outcomes():
    """Los dos Silencios: región infértil (A ≤ 1) y Techo bajo el suelo
    (δ_sat < δ_min) terminan ambos en 0 — desenlace real, no descartado."""
    W = W_max_required(delta0_H())
    assert attractor_analytic(0.9, W, delta_min=D_MIN) == 0.0
    infertile = attractor_numeric(0.9, W, 0.012, delta_min=D_MIN)
    assert infertile["attractor"] == 0.0
    # Techo bajo el suelo: δ_sat(W→0) < δ_min
    tiny_W = c_bar() * (0.5 * D_MIN) ** 3
    assert attractor_analytic(1.5, tiny_W, delta_min=D_MIN) == 0.0
    num = attractor_numeric(1.5, tiny_W, 0.012, delta_min=D_MIN)
    assert num["attractor"] == 0.0


def test_consistency_factors():
    """Las consecuencias de δ₀ = δ_H a verificar (análisis ronda 4):
    T₀ ×(δ_H/0.012)³ ≈ 114 y m_θ² ×(δ_H/0.012)^{5/2} ≈ 52."""
    f = consistency_factors()
    r = f["ratio"]
    assert abs(f["T0_ratio"] - r ** 3) < 1e-9 * r ** 3
    assert abs(f["m_theta_sq_ratio"] - r ** 2.5) < 1e-9 * r ** 2.5
    assert abs(f["T0_ratio"] - 113.7) < 1.0
    assert abs(f["m_theta_sq_ratio"] - 51.7) < 0.5


def test_landscape_scan_containment():
    """La contención por paisaje: fracción creciente en W_max, y todos
    los δ_H fértiles positivos y finitos (D > 0 garantizado por 3.3)."""
    scan = landscape_scan(n=2000, seed=5)
    d_H, w_req = scan["delta_H"], scan["W_required"]
    assert scan["n_fertile"] > 0
    assert np.all(np.isfinite(d_H)) and np.all(d_H > 0.0)
    assert np.all(np.isfinite(w_req)) and np.all(w_req > 0.0)
    cf = scan["containment_fraction"]
    lo, mid, hi = cf(1e-6), cf(1e-4), cf(1e-2)
    assert lo <= mid <= hi
    assert cf(float(w_req.max())) == 1.0
    # En toda la región O(1) el discriminante acota δ_H ≳ 0.033: ninguna
    # forma O(1) cierra el empalme en δ₀ = 0.012 (refuerzo independiente
    # de la regla canónica — el 0.012 de ε_Λ queda fuera de rango).
    assert d_H.min() > 0.033


def test_lambda_seal_consistent_with_B7():
    """El valor declarado en core coincide con el convenio 12.1 de B7."""
    assert LAMBDA_H_SEAL == B7.BETA3_CONVENIO_12_1
