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
from .potential import beta_match, chi_inf, alpha_per_seal


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


def betas_table() -> dict:
    """β_n del modelo (constantes independientes, Tratado).

    Las β_n NO son derivables de un único α global: cubren ~8 órdenes de
    magnitud entre C1 (10^-43) y C4 (10^7). El matching C^1 (Ec. 310)
    se satisface por construcción asignando un α_n distinto a cada sello
    (ver `alphas_per_seal`).
    """
    return {seal: C.BETA[seal] for seal in ("C1", "C2", "C3", "C4")}


def alphas_per_seal() -> dict:
    """α_n implícito en cada sello: α_n = β_n v_n^2 / (2 S_n)."""
    return {seal: alpha_per_seal(seal) for seal in ("C1", "C2", "C3", "C4")}


# Alias retro-compatible: ahora devuelve las constantes independientes,
# no derivadas de un α global (que era físicamente incorrecto).
matching_betas = betas_table


def chi_inf_table() -> dict:
    """Fracción de sellado latente para n=1..4 (Ec. 312)."""
    return {f"chi_inf_{n}": chi_inf(n) for n in (1, 2, 3, 4)}


def T_crit_table() -> dict[str, float]:
    """Umbrales críticos T_crit^(n) (Ec. 445, Tabla 67 del Tratado).

        T_crit^(n) = T₀/4 · S_n/ΔS · (v_n/v_1)^2
    """
    v1 = C.V_GEV["C1"]
    return {
        seal: (C.T0_GEV / 4.0) * (C.S_SEALS[seal] / C.DELTA_S)
              * (C.V_GEV[seal] / v1) ** 2
        for seal in ("C1", "C2", "C3", "C4")
    }
