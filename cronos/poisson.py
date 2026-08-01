"""Poisson modificado del MCMC en malla periódica (v35, Apéndice B, ec. B.2).

    ∇²Φ_tot = 4πG·a²·(ρ̄_m·δ_m + δρ_id + δρ_lat)

Resolución espectral (FFT) sobre malla cúbica periódica. El ciclo B.3
resuelve por separado Φ_N (materia) y Φ_id (canal de vacío efectivo);
ambos usan este mismo solver con la fuente que corresponda.
"""

from __future__ import annotations

import numpy as np


def solve_poisson(source: np.ndarray, box_size: float,
                  a_scale: float = 1.0, G: float = 1.0) -> np.ndarray:
    """Resuelve ∇²Φ = 4πG·a²·source en malla periódica cúbica.

    `source` es el término entre paréntesis de (B.2): ρ̄_m·δ_m + δρ_id
    + δρ_lat (o cada canal por separado para Φ_N / Φ_id del ciclo B.3).
    El modo k=0 (media) se anula: el potencial queda definido salvo
    constante, como corresponde en caja periódica.
    """
    source = np.asarray(source, dtype=float)
    n = source.shape[0]
    if source.shape != (n, n, n):
        raise ValueError("source debe ser una malla cúbica (n,n,n)")
    rhs_k = np.fft.rfftn(4.0 * np.pi * G * a_scale ** 2 * source)
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    k1r = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1r, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0  # evitar división por cero; el modo medio se anula
    phi_k = -rhs_k / k2
    phi_k[0, 0, 0] = 0.0
    return np.fft.irfftn(phi_k, s=source.shape, axes=(0, 1, 2))


def gradient(field: np.ndarray, box_size: float) -> np.ndarray:
    """Gradiente espectral de un campo en malla periódica → (3, n, n, n)."""
    field = np.asarray(field, dtype=float)
    n = field.shape[0]
    f_k = np.fft.rfftn(field)
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    k1r = 2.0 * np.pi * np.fft.rfftfreq(n, d=box_size / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1r, indexing="ij")
    out = np.empty((3, n, n, n))
    for i, k_i in enumerate((kx, ky, kz)):
        out[i] = np.fft.irfftn(1j * k_i * f_k, s=field.shape,
                               axes=(0, 1, 2))
    return out
