"""Sellos ontológicos C0..C4 — eventos de colapso entrópico.

Cada sello marca:
  · una transición dimensional (Vn-1 D → Vn D),
  · un cambio en el grupo gauge,
  · la emergencia de una familia fermiónica F_n.

Funciones:
  · seal_info(name): dict con S, V, beta, dimensión, gauge, familia.
  · matching_betas(): verifica las condiciones C^1 (Ec. 310).
  · chi_inf_table(): fracción de sellado latente χ∞_n (Ec. 312).
"""

from __future__ import annotations

from . import constants as C
from .potential import beta_match, chi_inf, alpha_from_matching


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
    """β_n derivados de la condición de matching C^1 (Ec. 310)."""
    alpha = alpha_from_matching("C3")
    return {seal: beta_match(seal, alpha) for seal in ("C1", "C2", "C3", "C4")}


def chi_inf_table() -> dict:
    """Fracción de sellado latente para n=1..4 (Ec. 312)."""
    return {f"chi_inf_{n}": chi_inf(n) for n in (1, 2, 3, 4)}
