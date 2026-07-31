"""P4 — Cascada SO(10), Yukawa GUT (Ec. 844-845, Tratado Unificado v32).

Matriz Yukawa GUT y_ij^(0), activada en S = 0.999. Columnas = sellos
de emergencia (familias Clifford). Filas = tipo fermiónico.

Relaciones Georgi-Jarlskog (escala GUT): m_b ≈ m_τ, m_s ≈ m_μ/3, m_d ≈ m_e/3.

El sector neutrino se obtiene por seesaw tensional (v35, Prop. 12.5):
    m_ν ~ m_D² / M_R
El neutrino es el modo cuya supervivencia a los sellos es máxima; por eso
es el más ligero. Las escalas v2/v3 usadas abajo son la asignación v32
(bloque LEGACY_V32 de constants.py).
"""

from __future__ import annotations

from mcmc_ontology import constants as C


def yukawa_gut(ftype: str, family: str) -> float:
    """Yukawa GUT y_ij^(0) por tipo y familia (F1, F2, F3)."""
    return C.YUKAWA_GUT[ftype][family]


def yukawa_dominant(seal: str, ftype: str) -> float:
    """Yukawa dominante en el sello dado, por tipo (post-emergente)."""
    family_seal = {"C1": "F1", "C2": "F2", "C3": "F3", "C4": "F3"}
    return yukawa_gut(ftype, family_seal[seal])


def neutrino_mass(family: str) -> float:
    """Masa de neutrino por seesaw tensional: m_ν = y_ν · v3² / v2 [GeV].

    Prescripción de orden: Yukawa del leptón cargado de la misma familia
    multiplicado por el factor seesaw v3²/v2 = 246²/1e16 ≈ 6.05e-12 GeV
    (Prop. 12.5, m_ν ~ m_D²/M_R con M_R en la escala v2 de la asignación
    v32). Produce m_ν(F1) ≈ 4.4e-5 eV, m_ν(F2) ≈ 2.6e-6 eV,
    m_ν(F3) ≈ 1.3e-8 eV (Σm_ν ≈ 4.6e-5 eV, muy por debajo de la cota
    observacional de 0.12 eV).
    """
    y_l = C.YUKAWA_GUT["lepton"][family]
    return y_l * C.V_GEV["C3"] ** 2 / C.V_GEV["C2"]
