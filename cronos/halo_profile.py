"""Perfiles de halo del MCMC: cored (no cuspy).

r_core(M, z) = r* · (M/M*)^α_r · ((1+z)/(1+z*))^β_r

Versus el perfil NFW cuspy de ΛCDM estándar.
"""

from __future__ import annotations

import numpy as np


# Parámetros de calibración fiducial (consistentes con SPARC + halos masivos)
R_STAR = 5.0          # kpc
M_STAR = 1e10         # M_sun
Z_STAR = 0.0
ALPHA_R = 0.40
BETA_R  = -0.10


def r_core(M: float | np.ndarray, z: float | np.ndarray = 0.0) -> float | np.ndarray:
    """Radio del core en kpc."""
    M = np.asarray(M, dtype=float)
    z = np.asarray(z, dtype=float)
    return R_STAR * (M / M_STAR) ** ALPHA_R * ((1.0 + z) / (1.0 + Z_STAR)) ** BETA_R


def cored_profile(r: np.ndarray, M: float, z: float = 0.0,
                  rho0: float = 1e7) -> np.ndarray:
    """Perfil ρ(r) cored (Burkert-like).

    ρ(r) = ρ_0 · r_c^3 / [(r + r_c)(r^2 + r_c^2)]
    """
    rc = r_core(M, z)
    r = np.asarray(r, dtype=float)
    return rho0 * rc ** 3 / ((r + rc) * (r ** 2 + rc ** 2))


def nfw_profile(r: np.ndarray, M: float, c: float = 10.0,
                R_vir: float = 200.0) -> np.ndarray:
    """Perfil NFW estándar (referencia ΛCDM)."""
    rs = R_vir / c
    r = np.asarray(r, dtype=float)
    rho_s = M / (4 * np.pi * rs ** 3 * (np.log(1 + c) - c / (1 + c)))
    return rho_s / ((r / rs) * (1 + r / rs) ** 2)
