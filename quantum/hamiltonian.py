"""Hamiltoniano ontológico H_MCMC.

    H_MCMC = Σ_n [α_n S_n + β_n T_n + γ_n V_n]

donde:
  S_n: operador de evolución |Sn⟩ → |S_{n+1}⟩  (raising step)
  T_n: canal tensorial (controlado por masa/espacio local)
  V_n: energía cuántica virtual sellada (ECV)
"""

from __future__ import annotations

import numpy as np

from .qudit import D, basis


def step_operator(n: int) -> np.ndarray:
    """S_n = |S_{n+1}⟩⟨S_n|  (raising)."""
    if not 0 <= n < D - 1:
        raise ValueError("step_operator: 0 ≤ n < D-1")
    return np.outer(basis(n + 1), basis(n))


def tensorial_operator(n: int) -> np.ndarray:
    """T_n: canal tensorial — operador hermitiano local en el nivel n."""
    return np.outer(basis(n), basis(n))


def virtual_energy_operator(n: int, gamma: float = 1.0) -> np.ndarray:
    """V_n: energía sellada en el nivel n (diagonal)."""
    op = np.zeros((D, D), dtype=complex)
    op[n, n] = gamma * (n + 1)
    return op


def H_MCMC(alphas: np.ndarray | None = None,
           betas: np.ndarray | None = None,
           gammas: np.ndarray | None = None) -> np.ndarray:
    """Hamiltoniano completo.

    Por defecto: alphas/betas/gammas = 1.
    """
    if alphas is None: alphas = np.ones(D - 1)
    if betas  is None: betas  = np.ones(D)
    if gammas is None: gammas = np.ones(D)
    H = np.zeros((D, D), dtype=complex)
    for n in range(D - 1):
        S = step_operator(n)
        H = H + alphas[n] * (S + S.conj().T)
    for n in range(D):
        H = H + betas[n] * tensorial_operator(n)
        H = H + virtual_energy_operator(n, gamma=gammas[n])
    return H
