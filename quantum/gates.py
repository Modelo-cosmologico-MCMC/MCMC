"""Compuertas de colapso X_{n→n+1} (Tratado de Fundamentos v35, C.2).

Las compuertas unitarias condicionales se disparan exactamente en los
umbrales de la Década (GATE_THRESHOLDS_S):

    X̂0→1 en S = 0.009,  X̂1→2 en S = 0.099,
    X̂2→3 en S = 0.999,  X̂3→4 en S = 1.001

`threshold_apply` conserva además el mecanismo del corpus v32 basado en
la tensión acumulada τ_n ≥ τ_crit (variable distinta de S).
"""

from __future__ import annotations

import numpy as np

from .qudit import D, basis


# Umbrales de disparo en S (v35 C.2 — coinciden con la Ley de la Década):
GATE_THRESHOLDS_S = {
    0: 0.009,   # X̂0→1
    1: 0.099,   # X̂1→2
    2: 0.999,   # X̂2→3
    3: 1.001,   # X̂3→4
}


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
    """Aplica X_collapse(n) si tau ≥ tau_crit (mecanismo τ del corpus v32).

    τ es la tensión acumulada del nivel, no el índice S; para el disparo
    por umbral entrópico de la v35 usar `apply_gate_at_S`.
    """
    if tau >= tau_crit:
        return X_collapse(n) @ state
    return state


def apply_gate_at_S(state: np.ndarray, n: int, S: float) -> np.ndarray:
    """Aplica X̂n→n+1 si S ha alcanzado su umbral de la Década (v35 C.2).

    Única fuente de umbrales entrópicos de disparo: GATE_THRESHOLDS_S.
    """
    if n not in GATE_THRESHOLDS_S:
        raise ValueError(f"apply_gate_at_S: 0 ≤ n < {D - 1}")
    if S >= GATE_THRESHOLDS_S[n]:
        return X_collapse(n) @ state
    return state
