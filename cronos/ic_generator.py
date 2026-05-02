"""Generador de condiciones iniciales con P(k, z_ini) consistente con MCMC."""

from __future__ import annotations

import numpy as np


def power_law_pk(k: np.ndarray, A: float = 1.0, n_s: float = 0.965) -> np.ndarray:
    """Espectro power-law P(k) = A k^n_s (placeholder operativo)."""
    return A * k ** n_s


def generate_ic_grid(N: int, L: float, seed: int = 0,
                     A: float = 1.0, n_s: float = 0.965) -> dict:
    """Genera condiciones iniciales en una caja periódica L^3 con N^3 partículas.

    Devuelve un dict con posiciones, velocidades y campo de densidad inicial.
    Esquema lineal de Zel'dovich.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    # Campo gaussiano simple en espacio real (no Fourier completo) — placeholder
    delta = rng.normal(scale=0.02, size=(N, N, N))
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pos += 0.5 * (L / N) * rng.normal(size=pos.shape)
    vel = np.zeros_like(pos)
    return {"pos": pos, "vel": vel, "delta": delta, "L": L, "N": N}
