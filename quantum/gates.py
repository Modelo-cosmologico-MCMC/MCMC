"""Compuertas de colapso X_{n→n+1}.

Activación condicional: la compuerta se aplica solo cuando τ_n ≥ τ_crit.
"""

from __future__ import annotations

import numpy as np

from .qudit import D, basis


def X_collapse(n: int) -> np.ndarray:
    """Compuerta de transición |Sn⟩ → |S_{n+1}⟩ (unitaria sub-bloque 2x2)."""
    if not 0 <= n < D - 1:
        raise ValueError("X_collapse: 0 ≤ n < D-1")
    U = np.eye(D, dtype=complex)
    # Intercambio en el sub-espacio {|n⟩, |n+1⟩}
    U[n, n] = 0; U[n + 1, n + 1] = 0
    U[n, n + 1] = 1; U[n + 1, n] = 1
    return U


def threshold_apply(state: np.ndarray, n: int, tau: float, tau_crit: float = 1.0) -> np.ndarray:
    """Aplica X_collapse(n) si tau ≥ tau_crit; identidad en caso contrario."""
    if tau >= tau_crit:
        return X_collapse(n) @ state
    return state
