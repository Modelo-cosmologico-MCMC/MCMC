"""Cosmología de fondo del MCMC (Tratado de Fundamentos v35, Apéndice A).

Los dos canales de energía oscura (A.1):
    ρ_DE(S) = ρ_id(S) + ρ_lat(S)
    dρ_lat/dS = κ_lat(S) − η_lat(S)     [κ: sellado, η: liberación]

Friedmann tensional (A.2):
    H²(z) = (8πG_eff/3)·[ρ_b(1+z)³ + ρ_cdm(1+z)³ + ρ_r(1+z)⁴
            + ρ_id(z) + ρ_lat(z)] − k(1+z)²/a0²
    Λ_rel(z) ≡ 8πG_eff·[ρ_id(z) + ρ_lat(z)]

Parametrización de la transición (A.3) — forma tanh:
    Λ_rel(z) = Λ0 · [1 + ε·tanh((z_trans − z)/Δz)],  z ≤ z_trans
    ε = 0.012 ± 0.003,  z_trans = 8.9 ± 0.4,  Δz ≈ 1.5

(La forma lineal Λ0[1 + ε(z_trans − z)] que anunciaba una versión anterior
de este docstring es la parametrización superada: el código siempre
implementó la tanh, que es la vigente.)

Mapa S ↔ observables (A.7):
    d(ln a)/dS = C(S);   dt_rel/dS = T(S)·N(S),   N(S) = e^{Φ_ten(S)}

Límite de recuperación (Prop. A.1): con ε → 0 y tasas → 0 se recupera
exactamente Friedmann con Λ constante (ver tests/test_recovery_limit.py).
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C


# Densidades fraccionales fiduciales (referencia del corpus)
OMEGA_M0 = 0.300
OMEGA_R0 = 9.2e-5      # radiación + neutrinos relativistas
OMEGA_L0 = 1.0 - OMEGA_M0 - OMEGA_R0


def Lambda_rel(z: np.ndarray | float,
               eps: float = C.EPSILON_0,
               z_trans: float = C.Z_TRANS,
               dz: float = C.DZ_TRANS) -> np.ndarray | float:
    """Densidad fraccional de Λ relativa con transición suave (v35 A.3).

        Ω_Λ_rel(z) = Ω_Λ0 · (1 + ε · tanh((z_trans - z) / dz))

    dz = Δz ≈ 1.5 según el Apéndice A.3 (antes 1.0, valor del corpus v32).
    """
    z = np.asarray(z, dtype=float)
    return OMEGA_L0 * (1.0 + eps * np.tanh((z_trans - z) / dz))


def H_of_z(z: np.ndarray | float,
           H0: float = C.H0_MCMC,
           Omega_m: float = OMEGA_M0,
           Omega_r: float = OMEGA_R0,
           eps: float = C.EPSILON_0,
           z_trans: float = C.Z_TRANS,
           dz: float = C.DZ_TRANS) -> np.ndarray | float:
    """H(z) en km/s/Mpc con Λ_rel dinámico (v35 A.2/A.3)."""
    z = np.asarray(z, dtype=float)
    OmL = Lambda_rel(z, eps=eps, z_trans=z_trans, dz=dz)
    arg = Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4 + OmL
    return H0 * np.sqrt(arg)


def rho_b(z: np.ndarray | float, Omega_b: float = 0.0489) -> np.ndarray | float:
    """Densidad bariónica (en unidades ρ_crit,0)."""
    z = np.asarray(z, dtype=float)
    return Omega_b * (1.0 + z) ** 3


def rho_id(z: np.ndarray | float, S: float = 95.0,
           rho_id_0: float = 0.65) -> np.ndarray | float:
    """Energía cuántica virtual (ECV) — forma operativa v32.

    Modelo operativo: ρ_id_0 + δρ proporcional a la conversión Mp→Ep.
    Los defaults S=95.0 y el máximo 150.0 son la parametrización del
    Unificado (bloque LEGACY_V32 en constants.py); la v35 no la contiene.
    Nota: la dependencia en S está desactivada (factor 0.0) — la forma
    de dos canales de la v35 (A.1) está pendiente de implementación.
    """
    z = np.asarray(z, dtype=float)
    s_frac = S / 150.0  # normalización v32 (LEGACY_V32)
    return rho_id_0 * (1.0 + 0.05 * np.tanh((C.Z_TRANS - z) / C.DZ_TRANS)) * (1.0 + 0.0 * s_frac)


def rho_lat(z: np.ndarray | float, S: float = 95.0,
            rho_lat_0: float = 0.05) -> np.ndarray | float:
    """Masa cuántica virtual (MCV) — Mp residual no procesada (forma v32).

    Defaults de la parametrización del Unificado (LEGACY_V32).
    """
    z = np.asarray(z, dtype=float)
    s_frac = 1.0 - S / 150.0  # proporcional al Mp restante (LEGACY_V32)
    return rho_lat_0 * s_frac * (1.0 + z) ** 0.0
