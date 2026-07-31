"""Álgebras de Clifford por régimen ontológico (Tabla 4, Tratado Unificado v32).

Realiza los axiomas 5-6 (sellos, tramo euclidiano) del Tratado de
Fundamentos (v35, §1.2). Nota v35 (§5.1): el álgebra de trabajo del tramo
d-dimensional es C(d+1,0) — d generadores espaciales más el generador
entrópico γS — y la cadena es C(2,0)→C(3,0)→C(4,0)→C(3,1); el etiquetado
Cl(n,0) de este módulo procede del corpus anterior.

Régimen pre-geométrico (S < 1.001): Cl(n,0) Euclidiano, amplitudes REALES.
Régimen post-geométrico (S ≥ 1.001): Cl(1,3) Minkowski, signatura Lorentziana.
La transición se realiza por una "rotación de Wick tensional"
(Ec. 408, Tratado Unificado v32).
"""

from __future__ import annotations

import numpy as np

from . import constants as C


# --- Matrices de Pauli (base de Cl(n,0) en dim baja) ---
sigma0 = np.eye(2, dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]], dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]], dtype=complex)


def euclidean_generators(d: int) -> list[np.ndarray]:
    """Generadores Euclidianos {γ^E_i, γ^E_j} = 2 δ_ij.

    d=1: σ_1 (Cl(1,0))
    d=2: σ_1, σ_2 (Cl(2,0))
    d=3: σ_1, σ_2, σ_3 (Cl(3,0); también γ matrices 4x4 disponibles)
    """
    if d == 1:
        return [sigma1.copy()]
    if d == 2:
        return [sigma1.copy(), sigma2.copy()]
    if d == 3:
        return [sigma1.copy(), sigma2.copy(), sigma3.copy()]
    raise ValueError(f"Dimensión Euclidiana no soportada: {d}")


def dirac_gamma_minkowski() -> list[np.ndarray]:
    """Gammas en representación de Dirac, signatura {+,-,-,-} (Cl(1,3))."""
    g0 = np.block([[sigma0, np.zeros((2, 2))], [np.zeros((2, 2)), -sigma0]])
    gi = [
        np.block([[np.zeros((2, 2)),  s], [-s, np.zeros((2, 2))]])
        for s in (sigma1, sigma2, sigma3)
    ]
    return [g0.astype(complex)] + gi


def wick_angle(S: float, S_birth: float | None = None,
               width: float = C.LAMBDA_ONT) -> float:
    """θ_W(S) = (π/2) Θ_λ(S - S_{1.001}) — Ec. 408, v32.

    Función suave (tanh) con grosor `width`.
    """
    if S_birth is None:
        S_birth = C.S_SEALS["C4"]
    theta_step = 0.5 * (1.0 + np.tanh((S - S_birth) / width))
    return float((np.pi / 2.0) * theta_step)


def wick_rotated_gamma0(S: float) -> np.ndarray:
    """γ^0(S) = exp(i θ_W(S)) γ^0_E.

    En S << S_birth: ~ γ^0_E (Euclidiano).
    En S >> S_birth: i γ^0_E (rotación de Wick completa → Minkowski).
    """
    g0_E = np.block([[sigma0, np.zeros((2, 2))], [np.zeros((2, 2)), sigma0]])
    return np.exp(1j * wick_angle(S)) * g0_E


def regime(S: float) -> str:
    """Devuelve 'pre' o 'post' según el régimen geométrico de S."""
    return "post" if S >= C.S_SEALS["C4"] else "pre"
