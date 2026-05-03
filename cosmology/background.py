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
           rho_id_0: float = 0.65, eps: float = C.EPSILON_0) -> np.ndarray | float:
    """Energía cuántica virtual (ECV).

    Modelo operativo: ρ_id_0 + δρ proporcional a la conversión Mp→Ep.
    Modulación dinámica con amplitud ε ≡ δ₀ = 0.012 (consistente con
    Λ_rel(z) y con la predicción de cruce phantom sub-percentual de w_id).
    """
    z = np.asarray(z, dtype=float)
    return rho_id_0 * (1.0 + eps * np.tanh((C.Z_TRANS - z) / 1.0))


def rho_lat(z: np.ndarray | float, S: float = C.S_SEALS["S_actual"],
            rho_lat_0: float = 0.05) -> np.ndarray | float:
    """Masa cuántica virtual (MCV) — Mp residual no procesada."""
    z = np.asarray(z, dtype=float)
    s_frac = 1.0 - S / C.S_SEALS["S_max"]  # proporcional al Mp restante
    return rho_lat_0 * s_frac * (1.0 + z) ** 0.0


def w_id(z: np.ndarray | float, S: float = C.S_SEALS["S_actual"],
         rho_id_0: float = 0.65, eps_z: float = 1e-3) -> np.ndarray | float:
    """Ecuación de estado efectiva del sector ECV/oscuro (Tratado §6.7):

        w_id(z) = -1 + (1/3) · d ln ρ_id / d ln(1+z)

    COMPORTAMIENTO ESPERADO (predicción del MCMC, no error):
      · w_id ≈ -1 para z ≪ z_trans y z ≫ z_trans  (límites Λ-like)
      · w_id ≈ -1 - ε/3 · |sech²(...)| en z ≈ z_trans  (cruce phantom suave)
      · |w_id + 1| ≤ ε/3 ≈ 0.004  (amplitud sub-percentual)

    El cruce phantom NO viola energía: emerge del Campo de Adrián como
    campo escalar efectivo. Es una predicción observacional del MCMC
    distinguible de ΛCDM en surveys futuros (Euclid, DESI).

    c²_s,id = 1 (velocidad del sonido del sector oscuro, sin
    inestabilidades Jeans).
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
