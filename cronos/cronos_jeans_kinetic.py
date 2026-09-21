"""Predicción CONVERGIDA del criterio de Cronos–Jeans: teoría cinética
lineal (Vlasov) de un medio maxwelliano con la fuerza local de Cronos.

Linealizando ∂_t f + v∂_x f − ∂_x U ∂_v f = 0 con δU = −κ·δρ, κ ≡
c²ε_c'(ρ₀) = q·σ²/ρ₀ (Def. 11.1 con ε_c = A·ρ^{3/2}), la respuesta
maxwelliana da la relación de dispersión

    1 = q·[1 + ζ·Z(ζ)],   ζ = ω/(√2·k·σ),

con Z la función de dispersión del plasma. Para un modo puramente
creciente ω = iγ, ζ = i·y y 1 + ζZ(ζ) = 1 − √π·y·e^{y²}·erfc(y) ≡ F(y),
decreciente de 1 (y = 0) a 0 (y → ∞). Por tanto:

    umbral:   F(0) = 1 ⟹ q = 1 exactamente;
    q > 1:    √π·y·e^{y²}·erfc(y) = 1 − 1/q,   γ = √2·k·σ·y(q):
              la tasa es EXACTAMENTE proporcional a k (catástrofe
              ultravioleta) y γ/(kσ) = √2·y(q) es una función universal
              de q, sin parámetros libres;
    q ≫ 1:    F(y) ≈ 1/(2y²) ⟹ γ → kσ√q, el límite fluido γ = kσ√(q−1)
              de dynamics.cronos_jeans a orden dominante.
    q < 1:    ningún modo creciente (los modos acústicos se amortiguan
              por Landau).

Esta es la predicción que el test preinscrito del frente 5 refundado
debe reproducir con un instrumento de láminas (cronos_jeans_1d) antes
de que ningún halo diga nada. Sin datos, sin ajuste (E8).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcx


def F_of_y(y):
    """F(y) = 1 − √π·y·e^{y²}·erfc(y) (usa erfcx para estabilidad)."""
    y = np.asarray(y, dtype=float)
    return 1.0 - np.sqrt(np.pi) * y * erfcx(y)


def y_of_q(q: float) -> float:
    """Solución de F(y) = 1/q para q > 1; 0 en q ≤ 1 (sin crecimiento)."""
    if q <= 1.0:
        return 0.0
    target = 1.0 / q
    return float(brentq(lambda y: F_of_y(y) - target, 0.0, 50.0, xtol=1e-14))


def gamma_over_k_kinetic(q: float, sigma: float = 1.0) -> float:
    """γ/k = √2·σ·y(q): tasa de crecimiento por unidad de k (cinética exacta)."""
    return float(np.sqrt(2.0) * sigma * y_of_q(q))


def gamma_over_k_fluid(q: float, sigma: float = 1.0) -> float:
    """γ/k = σ·√(q − 1): límite fluido (para comparar; exacto solo si q ≫ 1)."""
    return float(sigma * np.sqrt(max(q - 1.0, 0.0)))


def prediction_table(qs=(0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 4.0), sigma: float = 1.0) -> list[dict]:
    return [{"q": float(q), "y": y_of_q(q), "gamma_over_k_kinetic": gamma_over_k_kinetic(q, sigma),
             "gamma_over_k_fluid": gamma_over_k_fluid(q, sigma), "unstable": bool(q > 1.0)} for q in qs]


# ---------------------------------------------------- corrección del instrumento
def transfer_function_cic(k: float, h: float) -> float:
    """Función de transferencia de la fuerza en el instrumento de láminas:
    depósito CIC (sinc²(kh/2)) × recogida CIC (sinc²(kh/2)) × gradiente
    por diferencias centradas (sin(kh)/(kh)). Multiplica el acoplo
    efectivo: q_ef = q·W(k). DERIVADA de la discretización, no ajustada;
    W → 1 cuando kh → 0."""
    x = 0.5 * k * h
    sinc = np.sin(x) / x if x != 0.0 else 1.0
    grad = np.sin(k * h) / (k * h) if k * h != 0.0 else 1.0
    return float(sinc ** 4 * grad)


def gamma_over_k_instrument(q: float, k: float, h: float, sigma: float = 1.0) -> float:
    """Predicción cinética para el sistema DISCRETIZADO: γ/k = √2·σ·y(q·W(k)).
    Cerca del umbral la corrección es grande aunque W ≈ 1 (y ∝ √(q_ef − 1))."""
    return gamma_over_k_kinetic(q * transfer_function_cic(k, h), sigma)
