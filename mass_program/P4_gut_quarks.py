"""P4 — Cascada SO(10), Yukawa GUT (Ec. 844-845).

Matriz Yukawa GUT y_ij^(0), activada en S = 0.999. Columnas = sellos
de emergencia (familias Clifford). Filas = tipo fermiónico.

Relaciones Georgi-Jarlskog (escala GUT): m_b ≈ m_τ, m_s ≈ m_μ/3, m_d ≈ m_e/3.

El sector neutrino se obtiene por seesaw tensional:
    m_νi = y_νi · v3^2 / v2
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
    """Masa de neutrino por seesaw tensional: m_ν = y_ν · v3^2 / v2 [GeV].

    Como prescripción de orden, usamos los Yukawa de leptón cargado de la
    misma familia escalados por (v3/v2)^2 ~ 6e-28 para producir la jerarquía.
    """
    y_l = C.YUKAWA_GUT["lepton"][family]
    return y_l * C.V_GEV["C3"] ** 2 / C.V_GEV["C2"]
