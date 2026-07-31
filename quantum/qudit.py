"""Qudit ontológico d=5 — representación analógica del MCMC (v35, C.1).

Base computacional de 5 niveles (Tratado de Fundamentos, C.1):
    |S0⟩ ↔ V0D, |S1⟩ ↔ V1D, |S2⟩ ↔ V2D, |S3⟩ ↔ V3D, |S4⟩ ↔ V3+1D

Estado general: |ψ⟩ = Σ c_n |Sn⟩,  Σ |c_n|^2 = 1. H_ten es diagonal con
E0 < … < E4; las compuertas condicionales viven en quantum.gates.
"""

from __future__ import annotations

import numpy as np


D = 5  # dimensión del qudit


def basis(n: int) -> np.ndarray:
    """Vector base |Sn⟩ en C^5."""
    if not 0 <= n < D:
        raise ValueError(f"Nivel fuera de rango: {n}")
    e = np.zeros(D, dtype=complex)
    e[n] = 1.0
    return e


def state(coeffs: np.ndarray) -> np.ndarray:
    """Construye un estado normalizado a partir de coeficientes."""
    c = np.asarray(coeffs, dtype=complex)
    if c.shape[0] != D:
        raise ValueError(f"Se requieren {D} coeficientes")
    return c / np.linalg.norm(c)


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    """Fidelidad |⟨psi|phi⟩|^2."""
    return float(np.abs(np.vdot(psi, phi)) ** 2)
