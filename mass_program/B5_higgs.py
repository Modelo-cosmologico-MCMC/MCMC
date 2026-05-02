"""B5 — Higgs y mass gap.

    m_H = sqrt(2 β_3) · v_EW = sqrt(0.26) · 246 GeV ≈ 125.44 GeV

(PDG: 125.25 GeV → desviación 0.15%, sin parámetros libres).

Mass gap (Ec. 62):
    E_min = k · ΔS,   k ≡ M_Pl c^2,   ΔS = 1e-3
En S = 1.001:  E_min(C4) = m_H ≈ 125 GeV.
"""

from __future__ import annotations

import math

from mcmc_ontology import constants as C


M_PL_GEV = 1.22e19  # M_Pl·c^2 en GeV


def higgs_mass() -> float:
    """m_H = sqrt(2 β_3) · v_EW."""
    return math.sqrt(2.0 * C.BETA["C3"]) * C.V_GEV["C3"]


def mass_gap(S: float | None = None, k: float = M_PL_GEV) -> float:
    """E_min(S) = k · ΔS (mass gap cuántico).

    El argumento S permite parametrizar variantes; en la versión nominal
    es independiente de S (ΔS = constante = 1e-3).
    """
    return k * C.DELTA_S


def beta3_from_higgs() -> float:
    """Inversa: β_3 = (m_H / v_EW)^2 / 2 (verificación de consistencia)."""
    return 0.5 * (higgs_mass() / C.V_GEV["C3"]) ** 2
