"""Acoplo de Wilson entrópico y mapa Sₙ → jₙ (Tratado de Fundamentos v35, Ap. D).

Acción de Wilson con acoplo entrópico (D.1):

    β(S) = β0 + β1·exp(−b_S·(S − S3)),   S3 = 1.000

La escala QCD emerge cuando se sella V3D (S3 = 1.000).

Mapa sellos → espines LQG (Proposición D.1), con γ* = 0.274 fijado sin
libertad por S_BH = A/4ℓ_P² + conteo Kaul–Majumdar:

    S1 = 0.009 ↔ j1 = 1/2
    S2 = 0.099 ↔ j2 = 3/2
    S3 = 0.999 ↔ j3 = 5/2
    S4 = 1.001 ↔ j4 = 7/2

Área LQG: A(j) = 8πγ·ℓ_P²·sqrt(j(j+1)).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C

S3_QCD = 1.000  # Sello V3D: emergencia de la escala QCD (v35 D.1)

# Prop. D.1 — sellos de la Década ↔ espines LQG:
SEAL_TO_SPIN = {
    0.009: 0.5,
    0.099: 1.5,
    0.999: 2.5,
    1.001: 3.5,
}


def beta_S(S: np.ndarray | float, beta0: float = 6.0, beta1: float = 1.0,
           b_S: float = 10.0) -> np.ndarray | float:
    """Acoplo de Wilson entrópico β(S) = β0 + β1·exp(−b_S·(S − S3)) (D.1).

    beta0 = 6.0 coincide con el acoplo del config lattice_su3.yaml;
    beta1 y b_S son parámetros de forma del flujo entrópico.
    """
    S = np.asarray(S, dtype=float)
    out = beta0 + beta1 * np.exp(-b_S * (S - S3_QCD))
    return float(out) if out.ndim == 0 else out


def lqg_area(j: float, gamma: float = C.GAMMA_LQG, l_P2: float = 1.0) -> float:
    """Área LQG A(j) = 8πγ·ℓ_P²·sqrt(j(j+1)), con γ* = 0.274 (Prop. D.1).

    γ está sellado por la entropía de Bekenstein-Hawking más el conteo
    de Kaul–Majumdar: no es un parámetro libre.
    """
    return float(8.0 * np.pi * gamma * l_P2 * np.sqrt(j * (j + 1.0)))


def seal_spin(S_n: float) -> float:
    """Espín jₙ asociado al sello S_n según la Prop. D.1."""
    if S_n not in SEAL_TO_SPIN:
        raise KeyError(f"Sello sin espín asignado en la Prop. D.1: {S_n}")
    return SEAL_TO_SPIN[S_n]


def seal_area_table() -> dict[float, float]:
    """Tabla S_n → A(j_n) para los cuatro sellos de la Década."""
    return {S_n: lqg_area(j) for S_n, j in SEAL_TO_SPIN.items()}


def vertex_amplitude_ratio(j_n: float, j_prev: float,
                           delta_N: float = 1.0) -> float:
    """Cociente de amplitudes de vértice EPRL-FK entre umbrales (v35, D.4):

        A_v^(n)/A_v^(n−1) ~ [(2j_n + 1)/(2j_{n−1} + 1)]^{ΔN_n}

    Añadir área (mayor j) multiplica los microestados; el cociente
    reproduce la ley de Bekenstein (elasticidad ~1e-3 según el corpus).
    """
    return float(((2.0 * j_n + 1.0) / (2.0 * j_prev + 1.0)) ** delta_N)


def partition_ratio(dS: float = C.DELTA_S, k_b: float = 1.0) -> float:
    """Z_n/Z_{n−1} ≈ e^{+ΔS/k_b} — crecimiento de la función de
    partición por salto entrópico (v35, ec. D.4)."""
    import math
    return math.exp(dS / k_b)
