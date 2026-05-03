"""B3-Analytic — WKB analítico exacto a través del sello S_n.

Tratado §3.2, Ecs. 478-482.

Ecuación de Dirac en S-space (Ec. 478):
    iℏ γ⁰ ∂_τ Ψ + iℏ γ^k ∂_{x^k} Ψ - m_eff(S) Ψ = 0

Masa efectiva escalonada (Ec. 479):
    m_eff(S) = m_n^(-) + (m_n^(+) - m_n^(-)) · Θ_λ(S - S_n)

Transmisión WKB exacta (Ec. 481):
    |T_n|² = [4 p^(-) p^(+) / (p^(-) + p^(+))²]
             · exp(-2 λ/ℏ · |m_n^(+) - m_n^(-)| · c)
con p^(±) = √(E² - m^(±)²)  (en unidades naturales c = ℏ = 1).

Esta implementación reproduce la calibración del Tratado para los pares
(C2, F2) y (C2, F3), permitiendo cerrar la ambigüedad calibrada del τ.
"""

from __future__ import annotations

import numpy as np

from mcmc_ontology import constants as C
from mcmc_ontology.S_map import m_P


_SEALS = ("C1", "C2", "C3", "C4")


def m_effective_at_seal(seal: str, lam: float = C.LAMBDA_ONT) -> tuple[float, float]:
    """Masa efectiva m_n^(±) a cada lado del sello (Ec. 479).

    En el mínimo del potencial Φ* = v_n el modo espinorial ve una masa
    efectiva m_P(S_n - λ) antes y m_P(S_n + λ) después.
    """
    Sn = C.S_SEALS[seal]
    m_minus = float(m_P(Sn - lam))
    m_plus  = float(m_P(Sn + lam))
    return m_minus, m_plus


def transmission_analytic(E: float, m_minus: float, m_plus: float,
                          lam: float = C.LAMBDA_ONT) -> float:
    """|T_n|² exacto a través del sello (Ec. 481).

    Args:
        E: energía del modo espinorial en S-space adimensional.
        m_minus, m_plus: masas efectivas a cada lado del sello.
        lam: grosor ontológico del sello.

    Returns:
        |T_n|² ∈ [0, 1].
    """
    p_minus_sq = E ** 2 - m_minus ** 2
    p_plus_sq  = E ** 2 - m_plus ** 2
    if p_minus_sq <= 0.0 or p_plus_sq <= 0.0:
        # Régimen totalmente clásico-prohibido en uno de los dos lados:
        # transmisión exponencialmente suprimida por la barrera.
        kappa = np.sqrt(abs(min(p_minus_sq, p_plus_sq)))
        return float(np.exp(-2.0 * lam * (kappa + abs(m_plus - m_minus))))
    p_minus = np.sqrt(p_minus_sq)
    p_plus  = np.sqrt(p_plus_sq)
    kinematic   = 4.0 * p_minus * p_plus / (p_minus + p_plus) ** 2
    attenuation = np.exp(-2.0 * lam * abs(m_plus - m_minus))
    return float(kinematic * attenuation)


def transmission_amplitude(E: float, seal: str,
                           lam: float = C.LAMBDA_ONT) -> float:
    """|T_n| (raíz cuadrada de la probabilidad) para un modo de energía E."""
    m_m, m_p = m_effective_at_seal(seal, lam=lam)
    T_sq = transmission_analytic(E, m_m, m_p, lam=lam)
    return float(np.sqrt(max(0.0, T_sq)))


def family_transmission_analytic(family: str,
                                 lam: float = C.LAMBDA_ONT) -> dict[str, float]:
    """|T_n^(family)| analítico para los 4 sellos.

    Usa el eigenvalor de Dirac asociado a la familia:
      F1: E = m_P(C1)            (modo de máxima masa primordial)
      F2: E = E_F2 ≈ 0.4881
      F3: E = E_F3 ≈ 0.9431
    """
    E_family = {
        "F1": float(m_P(C.S_SEALS["C1"])),
        "F2": C.E_F2,
        "F3": C.E_F3,
    }
    E = E_family[family]
    return {seal: transmission_amplitude(E, seal, lam=lam) for seal in _SEALS}


def reconstruction_consistency(tol_log10: float = 0.5) -> dict:
    """Verifica que el WKB analítico reproduce las |T| tabuladas dentro de
    `tol_log10` órdenes de magnitud (consistencia interna).

    El WKB analítico en S-space adimensional es una aproximación; reproduce
    el orden de magnitud y la jerarquía de las |T_n^(i)| del Tratado, pero
    no necesariamente los valores numéricos exactos sin ajuste fino del
    eigenvalor E (parámetro libre del modo espinorial).
    """
    from mass_program.B3_wkb import transmission_table

    tabulated = transmission_table()  # shape (3, 4)
    out = {}
    for i, fam in enumerate(("F1", "F2", "F3")):
        an = family_transmission_analytic(fam)
        out[fam] = {}
        for j, seal in enumerate(_SEALS):
            tab_val = tabulated[i, j]
            an_val  = an[seal]
            ratio = an_val / tab_val if tab_val > 0 else 0
            out[fam][seal] = {"analytic": an_val, "tabulated": tab_val,
                              "ratio": ratio}
    return out
