"""Fricción entrópica local y factor de kick Cronos — ESQUEMA v32, SUPERADO.

    f_kick = 1 / (1 + (ρ_local / ρ_c0)^1.5 / α_cronos)
    a_fric = -(ρ_local / ρ_c0)^1.5 / α_cronos · H_a · v

La v35 (Cor. 11.3a) reemplaza esta fricción siempre activa por la
fricción con compuerta Γ = (3/2)(ρ̇/ρ)ε_c·Θ(ρ̇), que se apaga al
virializar (cronos/cronos_v3.py). α_cronos NO es la Amplitud de Cronos
α0⁻¹ de la ec. 11.5.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C


def kick_factor(rho_local: np.ndarray, alpha_cronos: float = C.ALPHA_CRONOS,
                rho_c0: float = C.RHO_C0) -> np.ndarray:
    """Factor multiplicativo del kick: 1/(1 + (ρ/ρ_c0)^1.5 / α)."""
    rho = np.asarray(rho_local, dtype=float)
    return 1.0 / (1.0 + (rho / rho_c0) ** 1.5 / alpha_cronos)


def friction_acceleration(v: np.ndarray, rho_local: np.ndarray,
                          H_a: float = 1.0,
                          alpha_cronos: float = C.ALPHA_CRONOS,
                          rho_c0: float = C.RHO_C0) -> np.ndarray:
    """Aceleración de fricción entrópica.

    a_fric = -(ρ_local/ρ_c0)^1.5 / α_cronos · H_a · v
    """
    rho = np.asarray(rho_local, dtype=float)
    coeff = -(rho / rho_c0) ** 1.5 / alpha_cronos * H_a
    return coeff[:, None] * v
