"""B5 — Higgs y mass gap.

    m_H = sqrt(2·β3) · v3 = sqrt(2·0.13) · 246 GeV ≃ 125.3 GeV
    (PDG: 125.25 ± 0.17 GeV; acuerdo 0.04%)

AUDITORÍA DE CIRCULARIDAD (Tratado de Fundamentos v35, Obs. 12.2):
esta expresión es exactamente la identidad del Modelo Estándar
m_H = sqrt(2·λ_H)·v con λ_H = β3. El acuerdo numérico NO constituye,
por sí solo, una predicción: cualquier β3 ajustado a m_H reproduce m_H.
El resultado tendrá contenido predictivo si y solo si β3 se deriva de
las condiciones de empalme del potencial sin usar m_H como entrada.
Esa derivación independiente es el frente abierto nº 7 del tratado.

Nota numérica: con β3 = 0.13 el cómputo exacto da sqrt(0.26)·246 =
125.436 GeV; el tratado publica 125.3 GeV (Prop. 12.1, redondeo propio
del tratado). Este módulo usa los valores del tratado.

Mass gap (Tratado Unificado v32, Ec. 62):
    E_min = k · ΔS,   k ≡ M_Pl c^2,   ΔS = 1e-3
En S = 1.001:  E_min(C4) = m_H ≈ 125 GeV.
"""

from __future__ import annotations

import math

from mcmc_ontology import constants as C


M_PL_GEV = 1.22e19  # M_Pl·c^2 en GeV


def higgs_mass() -> float:
    """m_H = sqrt(2 β_3) · v3 — identidad del SM con λ_H = β3 (Obs. 12.2).

    v3 = 246 GeV sellada por V3+1D (v35 F.4).
    """
    return math.sqrt(2.0 * C.BETA["C3"]) * C.V3_GEV


def mass_gap(S: float | None = None, k: float = M_PL_GEV) -> float:
    """E_min(S) = k · ΔS (mass gap cuántico).

    El argumento S permite parametrizar variantes; en la versión nominal
    es independiente de S (ΔS = constante = 1e-3).
    """
    return k * C.DELTA_S


def beta3_calibrated_from_higgs() -> float:
    """Calibración inversa: β3 = (m_H / v3)² / 2.

    Es la operación que la Obs. 12.2 identifica como circular: parte de la
    masa medida del Higgs. Se conserva como utilidad de calibración, no
    como verificación ni como derivación independiente (frente abierto nº 7).
    """
    return 0.5 * (higgs_mass() / C.V3_GEV) ** 2
