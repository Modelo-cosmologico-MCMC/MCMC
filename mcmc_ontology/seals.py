"""Sellos ontológicos C0..C4 — eventos de colapso entrópico.

Realiza el axioma 5 (colapsos y sellos) del Tratado de Fundamentos
(v35, §1.2).

NOTA DE VERSIÓN: la asociación sello → (gauge, familia fermiónica) de este
módulo es la presentación del Tratado Unificado (v32). En la v35 (Tabla
F.1) S₀,₀₉₉ y S₀,₉₉₉ son los sellos de c y de c² — no emergencias de
familias — y la supervivencia por sello la codifican los pesos c_in del
Funcional del Camino (Def. 12.3). Ver README, «Cronología tensional».

Funciones:
  · seal_info(name): dict con S, V, beta, dimensión, gauge, familia (v32).
  · matching_betas(): verifica las condiciones C^1 (Ec. 310, v32).
  · chi_inf_table(): fracción de sellado latente χ∞_n (Ec. 312, v32).
"""

from __future__ import annotations

from . import constants as C
from .potential import alpha_from_matching, beta_match, chi_inf

# Asignaciones v32 (ver nota de versión en el docstring del módulo):
SEAL_GAUGE = {
    "C0": "—",
    "C1": "—",
    "V1D": "U(1)",
    "C2": "SO(10) → GPS",
    "V2D": "U(1)×SU(2)",
    "C3": "GSM",
    "V3D": "U(1)×SU(2)×SU(3)",
    "C4": "SM completo",
}

SEAL_FAMILY = {"C1": "F1", "C2": "F2", "C3": "F3", "C4": "—"}
SEAL_DIM    = {"C0": 0, "C1": 0, "V1D": 1, "C2": 1, "V2D": 2,
               "C3": 2, "V3D": 3, "C4": 4}  # 4 = 3+1 D Lorentziano


def seal_info(name: str) -> dict:
    """Información completa de un sello."""
    if name not in C.S_SEALS:
        raise KeyError(f"Sello desconocido: {name}")
    return {
        "name":   name,
        "S":      C.S_SEALS[name],
        "v_GeV":  C.V_GEV.get(name),
        "beta":   C.BETA.get(name),
        "dim":    SEAL_DIM.get(name),
        "gauge":  SEAL_GAUGE.get(name, "—"),
        "family": SEAL_FAMILY.get(name, "—"),
    }


def matching_betas() -> dict:
    """β_n derivados de la condición de matching C^1 (Ec. 310, v32)."""
    alpha = alpha_from_matching("C3")
    return {seal: beta_match(seal, alpha) for seal in ("C1", "C2", "C3", "C4")}


def chi_inf_table() -> dict:
    """Fracción de sellado latente para n=1..4 (Ec. 312, v32)."""
    return {f"chi_inf_{n}": chi_inf(n) for n in (1, 2, 3, 4)}
