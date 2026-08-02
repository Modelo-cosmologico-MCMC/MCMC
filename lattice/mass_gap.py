"""Mass gap cuántico y piso infrarrojo (v35, Apéndice D.4-D.5).

Escala irremovible (ec. D.5): E_min = k·ΔS, k ≡ M_Pl·c². Con ΔS = 1e-3:
E_min = 1.22e16 GeV → reescalado por VEVs ≈ m_H (identificación del
corpus; el reescalado no está implementado aquí).

Excitación mínima en la retícula (ec. D.6):
    E_min(S) = (1/a_lat)·sqrt(2αS + 3β3(φ*² − v3²))·Θ_λ(S − 0.999)
cuyo escalón en S = 1.001 el tratado identifica con la masa del Higgs
m_H = sqrt(2β3)·v3 (β3 = 0.13). En el MCMC el confinamiento no es solo
topológico: es cosmológico.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C

M_PL_GEV = 1.22e19


def E_min(S: float, k: float = M_PL_GEV) -> float:
    """E_min(S) = k · ΔS — ec. (D.5) (independiente de S en la nominal)."""
    return k * C.DELTA_S


def E_min_lattice(S: np.ndarray | float, a_lat: float = 1.0,
                  alpha: float = 0.0, beta3: float = 0.13,
                  phi_star: float | None = None,
                  v3: float = C.V3_GEV,
                  lam: float = C.DELTA_S / 4.0) -> np.ndarray | float:
    """Excitación mínima E1 − E0 = sqrt(V''(φ*, S)) en la retícula (ec. D.6):

        E_min(S) = (1/a_lat)·sqrt(2αS + 3β3(φ*² − v3²))·Θ_λ(S − 0.999)

    con Θ_λ el escalón suave de grosor λ. El tratado no fija λ pero exige
    que «el escalón en S = 1.001 reproduce m_H»: el default λ = ΔS/4 hace
    el escalón completo (>0.9999) en S = 1.001. Con la elección
    φ*² = v3²·(1+2/3) y α = 0, el valor post-escalón es sqrt(2β3)·v3/a_lat
    — la identidad del Higgs de la Prop. 12.1 (misma auditoría de
    circularidad de la Obs. 12.2: β3 es calibrado, no derivado).
    """
    S = np.asarray(S, dtype=float)
    if phi_star is None:
        phi_star = v3 * np.sqrt(1.0 + 2.0 / 3.0)
    arg = 2.0 * alpha * S + 3.0 * beta3 * (phi_star ** 2 - v3 ** 2)
    theta = 0.5 * (1.0 + np.tanh((S - 0.999) / lam))
    out = np.sqrt(np.clip(arg, 0.0, None)) * theta / a_lat
    return float(out) if out.ndim == 0 else out


def glueball_correction(lambda_IR: float, c: float = 0.02) -> float:
    """Δm(λ_IR) ≡ corrección MCMC al espectro glueball.

    Δm → 0 en el límite λ_IR → 0 (recupera QCD pura).
    Forma operativa: Δm = c · λ_IR (lineal a primer orden).
    """
    return c * lambda_IR
