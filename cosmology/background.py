"""Cosmología de fondo del MCMC.

Descomposición de densidad energética:
    ρ_tot(z; S) = ρ_b(z) + ρ_id(z; S) + ρ_lat(z; S)

Friedmann:
    H^2(z) = (8πG/3)[ρ_b + ρ_id + ρ_lat]

Energía oscura dinámica Λ_rel(z):
    Λ_rel(z) = Λ_0 [1 + ε (z_trans - z)]   para z ≤ z_trans

Ajuste global (Tabla 17):
    ε ≡ δ₀ = 0.012 ± 0.003
    z_trans = 8.9 ± 0.4
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C


# Densidades fraccionales fiduciales (post-ajuste MCMC)
OMEGA_M0 = 0.300
OMEGA_R0 = 9.2e-5      # radiación + neutrinos relativistas
OMEGA_L0 = 1.0 - OMEGA_M0 - OMEGA_R0


def Lambda_rel(z: np.ndarray | float,
               eps: float = C.EPSILON_0,
               z_trans: float = C.Z_TRANS,
               dz: float = 1.0) -> np.ndarray | float:
    """Densidad fraccional de Λ relativa con transición suave.

        Ω_Λ_rel(z) = Ω_Λ0 · (1 + ε · tanh((z_trans - z) / dz))
    """
    z = np.asarray(z, dtype=float)
    return OMEGA_L0 * (1.0 + eps * np.tanh((z_trans - z) / dz))


def H_of_z(z: np.ndarray | float,
           H0: float = C.H0_MCMC,
           Omega_m: float = OMEGA_M0,
           Omega_r: float = OMEGA_R0,
           eps: float = C.EPSILON_0,
           z_trans: float = C.Z_TRANS) -> np.ndarray | float:
    """H(z) en km/s/Mpc con Λ_rel dinámico."""
    z = np.asarray(z, dtype=float)
    OmL = Lambda_rel(z, eps=eps, z_trans=z_trans)
    arg = Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4 + OmL
    return H0 * np.sqrt(arg)


def rho_b(z: np.ndarray | float, Omega_b: float = 0.0489) -> np.ndarray | float:
    """Densidad bariónica (en unidades ρ_crit,0)."""
    z = np.asarray(z, dtype=float)
    return Omega_b * (1.0 + z) ** 3


def rho_id(z: np.ndarray | float, S: float = C.S_SEALS["S_actual"],
           rho_id_0: float = 0.65) -> np.ndarray | float:
    """Energía cuántica virtual (ECV).

    Modelo operativo: ρ_id_0 + δρ proporcional a la conversión Mp→Ep.
    En S=S_actual → ≈ 0.65 (post-ajuste).
    """
    z = np.asarray(z, dtype=float)
    s_frac = S / C.S_SEALS["S_max"]
    return rho_id_0 * (1.0 + 0.05 * np.tanh((C.Z_TRANS - z) / 1.0)) * (1.0 + 0.0 * s_frac)


def rho_lat(z: np.ndarray | float, S: float = C.S_SEALS["S_actual"],
            rho_lat_0: float = 0.05) -> np.ndarray | float:
    """Masa cuántica virtual (MCV) — Mp residual no procesada."""
    z = np.asarray(z, dtype=float)
    s_frac = 1.0 - S / C.S_SEALS["S_max"]  # proporcional al Mp restante
    return rho_lat_0 * s_frac * (1.0 + z) ** 0.0


def w_id(z: np.ndarray | float, S: float = C.S_SEALS["S_actual"],
         rho_id_0: float = 0.65, eps_z: float = 1e-3) -> np.ndarray | float:
    """Ecuación de estado efectiva del sector ECV (Tratado, §6.7):

        w_id(z) = -1 + (1/3) · d ln ρ_id / d ln(1+z)

    Calculada por diferenciación numérica de ρ_id(z;S).
    """
    z = np.asarray(z, dtype=float)
    rp = rho_id(z + eps_z, S=S, rho_id_0=rho_id_0)
    rm = rho_id(np.maximum(z - eps_z, 0.0), S=S, rho_id_0=rho_id_0)
    # d ln ρ / d ln(1+z) = ((1+z) / ρ) · dρ/dz
    drho = (rp - rm) / (2.0 * eps_z)
    rho = rho_id(z, S=S, rho_id_0=rho_id_0)
    dlnrho_dln1pz = (1.0 + z) / rho * drho
    return -1.0 + dlnrho_dln1pz / 3.0


def cs2_id() -> float:
    """Velocidad del sonido del sector oscuro c²_s,id = 1 (Tratado, §6.7)."""
    return 1.0
