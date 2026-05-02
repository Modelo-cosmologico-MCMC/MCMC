"""Perturbaciones lineales en el MCMC.

Funciones operativas para δ(k,z), factor de crecimiento D(a; S),
fσ8(z) y los moduladores µ(a), η(a) que codifican el efecto del
Campo de Adrián tensorial sobre las ecuaciones de Boltzmann.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from mcmc_ontology import constants as C
from .background import H_of_z, OMEGA_M0


def growth_factor(a_grid: np.ndarray, Omega_m: float = OMEGA_M0,
                  H0: float = C.H0_MCMC) -> np.ndarray:
    """Factor de crecimiento D(a) integrando la ecuación lineal estándar.

    d^2 D / d ln a^2 + (2 + d ln H / d ln a) dD/d ln a - 1.5 Ω_m(a) D = 0
    """
    a_grid = np.asarray(a_grid, dtype=float)

    def _rhs(y, ln_a):
        D, Dp = y
        a = np.exp(ln_a)
        z = 1.0 / a - 1.0
        H = H_of_z(z, H0=H0, Omega_m=Omega_m) / H0
        Om_a = Omega_m * a ** -3 / H ** 2
        # d ln H / d ln a
        eps_a = 1e-4
        H_p = H_of_z(1.0 / (a * (1 + eps_a)) - 1.0, H0=H0, Omega_m=Omega_m) / H0
        dlnH = (np.log(H_p) - np.log(H)) / eps_a
        return [Dp, -(2.0 + dlnH) * Dp + 1.5 * Om_a * D]

    ln_a = np.log(a_grid)
    y0 = [a_grid[0], a_grid[0]]  # D ~ a en MD
    sol = odeint(_rhs, y0, ln_a)
    D = sol[:, 0]
    return D / D[-1]  # normalizado a hoy


def f_sigma8(z: np.ndarray, sigma8_0: float = C.SIGMA8_MCMC,
             Omega_m: float = OMEGA_M0) -> np.ndarray:
    """fσ8(z) ≈ Ω_m(z)^0.55 · σ8 · D(z) (aproximación de Linder)."""
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    a_grid = np.linspace(1e-3, 1.0, 200)
    D = growth_factor(a_grid, Omega_m=Omega_m)
    Dz = np.interp(a, a_grid, D)
    H = H_of_z(z, Omega_m=Omega_m)
    H0 = C.H0_MCMC
    Om_a = Omega_m * (1.0 + z) ** 3 / (H / H0) ** 2
    f = Om_a ** 0.55
    return f * sigma8_0 * Dz


def mu_modifier(a: np.ndarray | float, eps: float = C.EPSILON_0) -> np.ndarray | float:
    """Modulador µ(a) de la ecuación de Poisson efectiva."""
    a = np.asarray(a, dtype=float)
    return 1.0 + eps * (1.0 - a)


def eta_modifier(a: np.ndarray | float, eps: float = C.EPSILON_0) -> np.ndarray | float:
    """Modulador η(a) ≡ Ψ/Φ."""
    a = np.asarray(a, dtype=float)
    return 1.0 - 0.5 * eps * (1.0 - a)
