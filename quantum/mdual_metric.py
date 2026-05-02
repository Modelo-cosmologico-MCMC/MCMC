"""Métrica dual relativa (MDR) en el régimen 3+1D.

    ds^2 = -(1 + ζ Φ_ten / M_Pl) dt^2 + a^2(t) dx^2

La fase tensorial modifica g_tt: cuantifica cuánto "espacio" ha infiltrado
el tiempo.

Índice tensional externo:
    S_ext(t) = S_{1.001} + ∫ dt' / τ^ext_Cronos(t')
Índice tensional local (presente estratificado):
    S_local(x, t) = S_ext(t) - ΔS_grav(x, t)
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C


M_PL_GEV = 1.22e19


def g_tt(Phi_ten: float | np.ndarray, zeta: float = C.ZETA_CRONOS,
         M_Pl: float = M_PL_GEV) -> float | np.ndarray:
    """Componente g_tt = -(1 + ζ Φ_ten / M_Pl)."""
    return -(1.0 + zeta * np.asarray(Phi_ten, dtype=float) / M_Pl)


def S_external(t: np.ndarray, tau_cronos: np.ndarray,
               S_birth: float = C.S_SEALS["C4"]) -> np.ndarray:
    """S_ext(t) integrando 1/τ_cronos a partir de t_{1.001}.

    Acepta τ_cronos(t) como un array del mismo shape que t.
    """
    t = np.asarray(t, dtype=float)
    tau = np.asarray(tau_cronos, dtype=float)
    integrand = 1.0 / tau
    dt = np.gradient(t)
    integral = np.cumsum(integrand * dt)
    return S_birth + integral


def S_local(S_ext: np.ndarray, dS_grav: np.ndarray) -> np.ndarray:
    """S_local(x, t) = S_ext(t) - ΔS_grav(x, t)."""
    return S_ext - dS_grav
