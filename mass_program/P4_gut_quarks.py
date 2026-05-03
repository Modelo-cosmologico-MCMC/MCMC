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

    NOTA: Existe una formulación alternativa en el Tratado (Ec. 490-492)
    que usa c_{τ2}^(ν) ≈ 1.0 vía GUT y produce m_ν_τ ~ 6 meV. Ambas
    formulaciones son compatibles con la cota Planck+DESI Σm_ν < 0.12 eV.
    """
    y_l = C.YUKAWA_GUT["lepton"][family]
    return y_l * C.V_GEV["C3"] ** 2 / C.V_GEV["C2"]


# --- Factor de color ξ_c para quarks en C4 (Ecs. 487-488 del Tratado) ---

def alpha_s_at_v4(alpha_s_QCD: float = 0.30) -> float:
    """α_s evaluada en la escala v_4 (≈ 0.20 GeV ~ Λ_QCD)."""
    return alpha_s_QCD


def xi_color() -> float:
    """ξ_c = (4/3) · α_s(v_4) · (v_4 / v_3)   (Ec. 487).

    Numéricamente: (4/3) × 0.3 × 0.2/246 ≈ 3.25×10⁻⁴.
    """
    alpha_s = alpha_s_at_v4()
    return (4.0 / 3.0) * alpha_s * (C.V_GEV["C4"] / C.V_GEV["C3"])


def color_factor_at_seal(seal: str, ftype: str) -> float:
    """Factor (1 + ξ_c · δ_{n4}) sólo para quarks en C4   (Ec. 488).

    Para leptones, neutrinos o sellos n ≠ C4 devuelve 1 (no aplica).
    """
    if seal != "C4":
        return 1.0
    if ftype not in {"up", "down"}:
        return 1.0
    return 1.0 + xi_color()
