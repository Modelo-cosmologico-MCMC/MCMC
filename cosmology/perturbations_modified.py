"""III.C — Gravedad modificada µ(a), η(a) desde el Campo de Adrián tensorial.

Tratado §3.19.8 Etapa II→III, Cuadro 15.

Parametrización:
    µ(a) = 1 + δµ₀ · dS/d ln a
    η(a) = Φ/Ψ - 1     (deslizamiento gravitacional)

Exponente del crecimiento (Cuadro 15):
    p = 3(1 + w_T) - ξ_Cronos

| Régimen           | w_T  | ξ_Cronos | p     |
|-------------------|------|----------|-------|
| Materia dominante | 0    | 9/2      | -3/2  |
| Energía oscura    | -1   | 0        | 0     |
| Transición z~zt   | -ε/3 | ~1       | ≈-1   |
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from mcmc_ontology import constants as C


def dS_dlna(a: float | np.ndarray, eps: float = C.EPSILON_0,
            z_trans: float = C.Z_TRANS, dz: float = 1.0) -> float | np.ndarray:
    """Derivada de S respecto a ln(a) (Ley de Cronos heurística).

    Pico en z ≈ z_trans, decae rápido fuera de la transición.
    """
    a = np.asarray(a, dtype=float)
    z = 1.0 / a - 1.0
    arg = (z - z_trans) / dz
    # Guard contra overflow de cosh en redshifts altos
    with np.errstate(over="ignore"):
        sech2 = np.where(np.abs(arg) > 350.0, 0.0, 1.0 / np.cosh(arg) ** 2)
    return -3.0 * eps * sech2


def mu_effective(a: float | np.ndarray, delta_mu0: float = 0.05) -> float | np.ndarray:
    """µ(a) = 1 + δµ₀ · dS/d ln a   (gravedad modificada efectiva)."""
    return 1.0 + delta_mu0 * dS_dlna(a)


def eta_slip(a: float | np.ndarray, eta0: float = 0.02) -> float | np.ndarray:
    """η = Φ/Ψ - 1 (decae a redshift alto)."""
    a = np.asarray(a, dtype=float)
    return eta0 * (1.0 - a)


def growth_exponent(regime: str = "matter") -> float:
    """Exponente p del crecimiento según régimen (Cuadro 15)."""
    if regime == "matter":
        return -1.5
    if regime == "lambda":
        return 0.0
    if regime == "transition":
        return -1.0
    raise ValueError(f"Régimen desconocido: {regime}")


def growth_factor_modified(a_arr: np.ndarray, omega_m: float = 0.30,
                           delta_mu0: float = 0.05) -> np.ndarray:
    """Factor de crecimiento D(a) con µ(a) modificada (RGE D-V).

        D' = V/a
        V' = -[3/a + dlnH/dlna · (1/a)] V + (3 Ω_m H₀² / 2H² a²) µ(a) D
    """
    a_arr = np.asarray(a_arr, dtype=float)

    def H(a):
        return np.sqrt(omega_m / a ** 3 + (1.0 - omega_m))

    def dy_da(y, a):
        D, V = y
        if a <= 0:
            return [0.0, 0.0]
        H_a = H(a)
        # dlnH/dlna ≈ d ln H / d ln a por diferencia hacia adelante
        dH_da = -1.5 * omega_m / a ** 4 / H_a
        dlnH_dlna = dH_da * a / H_a
        mu = float(mu_effective(a, delta_mu0))
        dD = V / a
        dV = -(3.0 / a + dlnH_dlna / a) * V + (3.0 * omega_m / (2.0 * H_a ** 2 * a ** 2)) * mu * D
        return [dD, dV]

    sol = odeint(dy_da, [a_arr[0], a_arr[0]], a_arr)
    D = sol[:, 0]
    return D / D[-1]


# --- Datos de comparación ---
GRAVITY_OBS_LIMITS = {
    "KiDS_DES_combined_2024": {
        "delta_mu0_max": 0.30,
        "eta0_max":      0.30,
        "ref": "Troxel+2018, Abbott+2023",
    },
    "Planck_ISW_late": {
        "mu_minus_1_max_at_a05": 0.20,
        "ref": "Naidoo+2022 arXiv:2209.01846",
    },
    "Euclid_forecast_2030": {
        "delta_mu0_sigma": 0.01,
        "eta0_sigma":      0.03,
        "projection": True,
    },
}
