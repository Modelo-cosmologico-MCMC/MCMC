"""Positividad por reflexión en modelo de juguete (v35, Cap. 7).

LO QUE ESTE MÓDULO ES — Y LO QUE NO. El Teorema de Positividad de
Florencia (7.1) establece ⟨ϑ(F̄)·F⟩ ≥ 0 bajo hipótesis (H1)-(H4) para la
teoría euclidiana en la loncha de Florencia, y con ello la
reconstrucción de Osterwalder-Schrader (7.3): un espacio de Hilbert de
norma positiva, Ĥ ≥ 0 y evolución unitaria — el tiempo nace unitario.
Este módulo verifica la CONCLUSIÓN en un retículo escalar 1D pequeño
con la medida e^{−S_E}: es una comprobación en modelo de juguete, NO la
demostración del teorema. El sector espinorial de Wilson (Teo. 7.4)
queda como lo que el tratado declara: CONDICIONAL (§13.4).

La verificación operativa: para un retículo con reflexión ϑ respecto de
una loncha, ⟨ϑ(F̄)·F⟩ = fᵀ·K·f con K el núcleo de transferencia

    K(φ, φ') = exp(−(φ−φ')²/2 − V(φ)/2 − V(φ')/2)
             = exp(−φ²/2 − φ'²/2 − V/2 − V'/2) · exp(J·φ·φ'),  J = +1

RP ⟺ K semidefinido positivo. El acoplo J = +1 procede del término
cinético (H1) — y el CONTROL NEGATIVO es exactamente ese: un acoplo
temporal invertido (J < 0) rompe la positividad (autovalores negativos),
que es lo que distingue una auditoría de una lista de éxitos.

El papel del término séxtico (§3.1): C0 > 0 hace la medida
normalizable (V → +∞); con C0 < 0 el paisaje se desfonda y la medida
e^{−V} no existe — segunda condición comprobada.
"""

from __future__ import annotations

import numpy as np

STATUS_WILSON = ("condicional (§13.4): la RP del sector espinorial de "
                 "Wilson (Prop. 7.4 — reclasificada de Teorema en "
                 "v35.1, E3) no se toca aquí — frente abierto nº 1 "
                 "para acoplos no estacionarios")


def transfer_kernel(phi_grid: np.ndarray, V_values: np.ndarray,
                    J: float = 1.0) -> np.ndarray:
    """Núcleo de transferencia del retículo escalar 1D.

    K(φ,φ') = exp(−φ²/2 − φ'²/2 + J·φφ' − V(φ)/2 − V(φ')/2).
    Con J = +1 equivale a exp(−(φ−φ')²/2)·e^{−V/2}·e^{−V'/2} (el término
    cinético físico). J < 0 es el control negativo (acoplo invertido).
    """
    phi = np.asarray(phi_grid, dtype=float)
    V = np.asarray(V_values, dtype=float)
    diag = np.exp(-0.5 * phi ** 2 - 0.5 * V)
    cross = np.exp(J * np.outer(phi, phi))
    return diag[:, None] * cross * diag[None, :]


def rp_min_eigenvalue(phi_grid: np.ndarray, V_values: np.ndarray,
                      J: float = 1.0) -> float:
    """Autovalor mínimo (normalizado) del núcleo simétrico.

    RP ⟺ ⟨ϑ(F̄)F⟩ = fᵀKf ≥ 0 para todo funcional f soportado en un
    lado de la loncha ⟺ min eig(K) ≥ 0.
    """
    K = transfer_kernel(phi_grid, V_values, J=J)
    eigs = np.linalg.eigvalsh(0.5 * (K + K.T))
    return float(eigs.min() / max(abs(eigs).max(), 1e-300))


def rp_holds(phi_grid: np.ndarray, V_values: np.ndarray,
             J: float = 1.0, tol: float = 1e-10) -> bool:
    """⟨ϑ(F̄)·F⟩ ≥ 0 sobre toda la familia de funcionales del grid."""
    return rp_min_eigenvalue(phi_grid, V_values, J=J) >= -tol


def measure_normalizable(delta0: float, C0: float,
                         rho_max: float = 50.0) -> bool:
    """¿Existe la medida e^{−V0}? Requiere V0 → +∞ (C0 > 0, §3.1).

    Comprobación operativa: V0 creciente y positivo en el borde lejano.
    """
    from .basal import V0
    v_far = float(V0(rho_max, 0.0, delta0, C0=C0))
    v_farther = float(V0(2.0 * rho_max, 0.0, delta0, C0=C0))
    return v_far > 0.0 and v_farther > v_far
