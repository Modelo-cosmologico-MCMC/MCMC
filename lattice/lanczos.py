"""Diagonalización Lanczos para extracción del espectro de la torre del gap."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh


def lowest_eigenvalues(H: np.ndarray, k: int = 6) -> np.ndarray:
    """k autovalores más bajos del operador hermitiano H."""
    H = np.asarray(H)
    # Si H es pequeño, usar np.linalg.eigh
    if H.shape[0] <= 200:
        evals = np.linalg.eigvalsh(H)
        return np.sort(evals)[:k]
    evals = eigsh(H, k=k, which="SA", return_eigenvectors=False)
    return np.sort(evals)
