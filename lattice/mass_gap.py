"""Mass gap cuántico E_min(S) = k · ΔS.

Con k ≡ M_Pl c^2 y ΔS = 1e-3, en S = 1.001:
    E_min = 1.22e19 · 1e-3 GeV ≈ 1.22e16 GeV → reescalado por VEVs ≈ m_H
"""

from __future__ import annotations

from mcmc_ontology import constants as C


M_PL_GEV = 1.22e19


def E_min(S: float, k: float = M_PL_GEV) -> float:
    """E_min(S) = k · ΔS  (independiente de S en la versión nominal)."""
    return k * C.DELTA_S


def glueball_correction(lambda_IR: float, c: float = 0.02) -> float:
    """Δm(λ_IR) ≡ corrección MCMC al espectro glueball.

    Δm → 0 en el límite λ_IR → 0 (recupera QCD pura).
    Forma operativa: Δm = c · λ_IR (lineal a primer orden).
    """
    return c * lambda_IR
