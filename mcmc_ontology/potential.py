"""Potencial tensional escalonado V(Phi_Ad; S).

Realiza los axiomas 1-4 (unidad dual, imperfección, tensión, Camino) del
Tratado de Fundamentos (v35, §1.2) en la forma escalonada del Unificado.

Lagrangiano del Campo de Adrián:

    L = 1/2 (∂Phi)^2 - V(Phi, S) + κ Phi tr(F F)

Potencial:

    V(Phi, S) = V0 + alpha S Phi^2
              + sum_{n=1..4} (beta_n / 4) (Phi^2 - v_n^2)^2
                              · 1/2 [1 + tanh((S - S_n)/lambda)]

Condiciones de matching C^1 (Ec. 310, v32):  beta_n = 2 alpha S_n / v_n^2.
Curvatura total en el vacío Phi* = v_n (Ec. P3, v32):

    V''_total(v_n) = 8 beta_n v_n^2 + d_n (m_P(S_n) v_n sqrt(d_n))^2

Δm_eff (curvatura efectiva normalizada):
    Δm_eff,n = sqrt(V''_total) / (m_P(S_n) · K_norm)
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .S_map import m_P


_SEAL_ORDER = ["C1", "C2", "C3", "C4"]
DIM_AT_SEAL = {"C1": 1, "C2": 2, "C3": 3, "C4": 3}  # dimensiones espaciales


def alpha_from_matching(seal: str = "C3") -> float:
    """alpha tal que beta_n = 2 alpha S_n / v_n^2 fija beta del sello dado.

    Por convención usamos C3 como anclaje, con beta_3 = 0.13 (λ_H del
    empalme C¹). La derivación de beta_3 independiente de m_H es el frente
    abierto nº 7 del Tratado de Fundamentos (Obs. 12.2): mientras no exista,
    beta_3 es un valor calibrado, no una constante derivada.
    """
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    return C.BETA[seal] * vn ** 2 / (2.0 * Sn)


def beta_match(seal: str, alpha: float | None = None) -> float:
    """β_n = 2 α S_n / v_n^2 (Ec. 310, v32)."""
    if alpha is None:
        alpha = alpha_from_matching("C3")
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    return 2.0 * alpha * Sn / vn ** 2


def chi_inf(n: int) -> float:
    """Fracción de sellado latente χ∞_n = 1 - S_{n-1}/S_n  (Ec. 312, v32)."""
    if n == 1:
        return 0.0
    Sn  = C.S_SEALS[_SEAL_ORDER[n - 1]]
    Snm = C.S_SEALS[_SEAL_ORDER[n - 2]]
    return 1.0 - Snm / Sn


def step(S: np.ndarray | float, S_n: float, lam: float = C.LAMBDA_ONT) -> np.ndarray | float:
    """Función escalón suave 1/2 [1 + tanh((S - S_n)/lambda)]."""
    return 0.5 * (1.0 + np.tanh((np.asarray(S, dtype=float) - S_n) / lam))


def V(Phi: np.ndarray | float, S: float, alpha: float | None = None,
      lam: float = C.LAMBDA_ONT, V0: float = 0.0) -> np.ndarray | float:
    """Potencial V(Phi; S) — Ec. principal del Campo de Adrián."""
    if alpha is None:
        alpha = alpha_from_matching("C3")
    Phi2 = np.asarray(Phi, dtype=float) ** 2
    out = V0 + alpha * S * Phi2
    for seal in _SEAL_ORDER:
        Sn = C.S_SEALS[seal]
        vn = C.V_GEV[seal]
        bn = C.BETA[seal]
        out = out + (bn / 4.0) * (Phi2 - vn ** 2) ** 2 * step(S, Sn, lam)
    return out


def V_pp_quartic(seal: str) -> float:
    """Curvatura aportada por el término cuártico en el mínimo: 8 β_n v_n^2."""
    return 8.0 * C.BETA[seal] * C.V_GEV[seal] ** 2


def V_pp_kinetic(seal: str) -> float:
    """Curvatura aportada por I_dD: d_n (m_P(S_n) v_n sqrt(d_n))^2."""
    d_n = DIM_AT_SEAL[seal]
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    mp = m_P(Sn)
    return d_n * (mp * vn * np.sqrt(d_n)) ** 2


def V_pp_total(seal: str) -> float:
    """V''_total(v_n) = V''_quartic + V''_kinetic (Ec. P3, v32)."""
    return V_pp_quartic(seal) + V_pp_kinetic(seal)


def delta_m_eff(seal: str, K_norm: float = C.K_NORM) -> float:
    """Δm_eff,n = sqrt(V''_total) / (m_P(S_n) · K_norm).

    Por convención K_norm se calibra desde C4 (donde β_4=10^7 domina I_dD).
    """
    Sn = C.S_SEALS[seal]
    return float(np.sqrt(V_pp_total(seal)) / (m_P(Sn) * K_norm))
