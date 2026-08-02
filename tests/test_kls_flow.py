"""Tests del flujo KLS integrado (frente E; cap. 8, ec. 8.3-8.4)."""

import numpy as np
import pytest

from core.decade import fixed_points, walk_period
from core.kls_flow import (
    STATUS_LAMBDA,
    cruce_de_victoria,
    delay_scaling,
    divergence_exponent,
    integrate_flow,
    walking_time_analytic,
    walking_time_measured,
)

B, C0, G = 2.0, 1.0, 1.0   # xR = B/(2C0) = 1


def _M0_sq(om_rel: float) -> float:
    """M0² que deja Ω = om_rel·xR (D = −(2C0·Ω)²)."""
    return (B ** 2 + (2.0 * C0 * om_rel) ** 2) / (4.0 * C0)


def test_rk4_matches_quadrature():
    """La integración RK4 del flujo (8.3) coincide con la cuadratura
    exacta de la ODE separable — el integrador está validado."""
    M0 = _M0_sq(0.05)
    _, xR, omega = fixed_points(B, M0, C0)
    K = 10.0
    res = integrate_flow(xR + K * omega, B, M0, C0, G, d_sigma=1e-3,
                         n_steps=80000, stop_below=xR - K * omega)
    q = walking_time_measured(B, M0, C0, G, K=K)
    assert res["collapsed"]
    assert abs(res["sigma_final"] / q - 1.0) < 1e-3


def test_walking_law_quantified():
    """El «≃» de la ec. 8.4, medido: el error del prefactor x ≈ xR cae
    con Ω/xR (0.3% en 0.01; 3e-5 en 0.001), y frente a la forma π/Ω
    del tratado queda el factor de ventana 2·arctan(K)/π."""
    K = 50.0
    prev = np.inf
    for om_rel in (0.01, 0.003, 0.001):
        m = walking_time_measured(B, _M0_sq(om_rel), C0, G, K=K)
        aK = walking_time_analytic(B, _M0_sq(om_rel), C0, G, K=K)
        err = abs(m / aK - 1.0)
        assert err < 0.005
        assert err < prev + 1e-12          # el error decrece con Ω
        prev = err
    m = walking_time_measured(B, _M0_sq(0.001), C0, G, K=K)
    a_full = walking_time_analytic(B, _M0_sq(0.001), C0, G)   # ec. 8.4
    assert abs(m / a_full - 2.0 * np.arctan(K) / np.pi) < 1e-3
    # y la forma K=None es exactamente walk_period de core.decade:
    assert a_full == walk_period(B, _M0_sq(0.001), C0, G)


def test_divergence_exponent_minus_half():
    """Δσ_walk ∝ |D|^(−1/2) — el exponente de la divergencia en la
    espinodal, ajustado sobre cuatro décadas."""
    d = divergence_exponent(B, C0, G)
    assert abs(d["exponent"] + 0.5) < 0.01


def test_window_validation():
    """La ventana K·Ω no puede alcanzar el polo x = 0 (se rechaza)."""
    with pytest.raises(ValueError):
        walking_time_measured(B, _M0_sq(0.05), C0, G, K=50.0)


def test_cruce_dispara_despues_del_cruce():
    """Obs. 8.6: con D(σ) hundiéndose, el colapso ocurre DESPUÉS del
    cruce de la espinodal (σ* exacto conocido), y antes del cruce el
    flujo sigue adiabáticamente el vacío x+(σ)."""
    cv = cruce_de_victoria(B, 0.9, rate=1e-3)
    assert cv["delay"] > 0.0
    # a mitad de camino hacia σ*, x sigue el x+ instantáneo:
    i_half = int(0.5 * cv["sigma_star"] / cv["d_sigma"])
    _, x_plus_half, _ = fixed_points(B, 0.9 + 1e-3 * 0.5 * cv["sigma_star"],
                                     C0)
    assert abs(cv["x"][i_half] - x_plus_half) / x_plus_half < 0.02


def test_control_negativo_sin_hundimiento():
    """Sin hundimiento (D > 0 constante) no hay colapso: el flujo
    aparca en x+ y se queda."""
    _, x_plus, _ = fixed_points(B, 0.9, C0)
    res = integrate_flow(1.1 * x_plus, B, 0.9, C0, G, d_sigma=1e-2,
                         n_steps=5000, stop_below=0.25 * B / (2 * C0))
    assert not res["collapsed"]
    assert abs(res["x"][-1] - x_plus) / x_plus < 1e-6


def test_delay_scaling_minus_third():
    """El retraso del colapso escala como rate^(−1/3) (silla-nodo con
    deriva — resultado del programa, declarado como tal)."""
    ds = delay_scaling(B, 0.9, np.array([1e-3, 3e-3, 1e-2]))
    assert abs(ds["exponent"] + 1.0 / 3.0) < 0.04


def test_status_declares_open_front():
    """El módulo no decide λ = 10: el estatuto lo declara (frente 2)."""
    assert "condicional" in STATUS_LAMBDA and "frente 2" in STATUS_LAMBDA
