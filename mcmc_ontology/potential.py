"""Potencial tensional escalonado V(Phi_Ad; S).

Lagrangiano del Campo de Adrián:

    L = 1/2 (∂Phi)^2 - V(Phi, S) + κ Phi tr(F F)

Potencial:

    V(Phi, S) = V0 + alpha S Phi^2
              + sum_{n=1..4} (beta_n / 4) (Phi^2 - v_n^2)^2
                              · 1/2 [1 + tanh((S - S_n)/lambda)]

Condiciones de matching C^1 (Ec. 310):  beta_n = 2 alpha S_n / v_n^2.
Curvatura total en el vacío Phi* = v_n (Ec. P3):

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


def alpha_per_seal(seal: str) -> float:
    """α implícito del sello: α_n = β_n v_n^2 / (2 S_n)  (inversa de Ec. 310).

    Las β_n son CONSTANTES INDEPENDIENTES del modelo (cubren ~8 órdenes
    de magnitud entre C1 y C4); cada sello fija su propio α_n. El
    "matching C^1" (Ec. 310) establece consistencia interna *por sello*,
    no a través de un único α global.
    """
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    return C.BETA[seal] * vn ** 2 / (2.0 * Sn)


def alpha_from_matching(seal: str = "C3") -> float:
    """Alias histórico: α implícito en `seal` (por defecto C3, escala EW)."""
    return alpha_per_seal(seal)


def beta_match(seal: str, alpha: float | None = None) -> float:
    """Devuelve la β_n del modelo (Ec. 310 satisfecha por construcción).

    Si se pasa `alpha` explícito, devuelve el β derivado por la Ec. 310 con
    ese α; en caso contrario, devuelve la β CONSTANTE del modelo (idéntica
    a `C.BETA[seal]`). Las β_n no son derivables de un único α global.
    """
    if alpha is None:
        return C.BETA[seal]
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    return 2.0 * alpha * Sn / vn ** 2


def chi_inf(n: int) -> float:
    """Fracción de sellado latente χ∞_n = 1 - S_{n-1}/S_n  (Ec. 312)."""
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
    """V''_total(v_n) = V''_quartic + V''_kinetic (Ec. P3)."""
    return V_pp_quartic(seal) + V_pp_kinetic(seal)


def delta_m_eff_raw(seal: str, K_norm: float = C.K_NORM) -> float:
    """Δm_eff numérico crudo: sqrt(V''_total) / (m_P · K_norm) en unidades GeV.

    ATENCIÓN: K_norm se calibra DESDE C4 (donde β_4·v_4^2 domina sobre I_dD).
    Aplicado directamente en C1 (escala Planck, v=1.22e19 GeV) o C2 (escala
    GUT, v=10^16 GeV), I_dD da valores enormes que NO son físicos en
    S-space adimensional. La fórmula sólo coincide con el valor calibrado
    en C4. Para los demás sellos usar `delta_m_eff` (calibrado).
    """
    Sn = C.S_SEALS[seal]
    return float(np.sqrt(V_pp_total(seal)) / (m_P(Sn) * K_norm))


def delta_m_eff(seal: str) -> float:
    """Δm_eff,n CALIBRADO (Tratado, Tabla P3).

    Devuelve los valores canónicos del modelo (Tabla P3), que son los que
    entran en el cálculo final de las transmisiones |T_n^(i)| y en la
    fórmula maestra de masas. La forma cruda numérica (`delta_m_eff_raw`)
    sólo coincide en C4 (ancla de K_norm).
    """
    return C.DELTA_M_EFF_CAL[seal]


# ============================================================
# Tramo pre-geométrico n=0 (Ecs. 20-26 del Tratado)
# ============================================================


def beta0(alpha: float | None = None) -> float:
    """β₀ = 2 α_S · S_{0.001} / v₀²    (Ec. 24).

    Si `alpha` es None se usa α_S = α_per_seal('C1') por consistencia
    con la cascada (cualquier α_S coherente con un sello del modelo).
    """
    if alpha is None:
        alpha = alpha_per_seal("C1")
    return 2.0 * alpha * C.S_PRE / C.V0_NORM ** 2


def gamma0(alpha: float | None = None) -> float:
    """γ₀ = -α_S · S_{0.001} · v₀²    (Ec. 25)."""
    if alpha is None:
        alpha = alpha_per_seal("C1")
    return -alpha * C.S_PRE * C.V0_NORM ** 2


def Mp_initial() -> float:
    """M_p^(0) = ½(1 + v₀)    (Ec. 23)."""
    return 0.5 * (1.0 + C.V0_NORM)


def Ep_initial() -> float:
    """E_p^(0) = ½(1 - v₀)    (Ec. 23)."""
    return 0.5 * (1.0 - C.V0_NORM)


def V_pre(Phi, S: float, alpha: float | None = None,
          lam_pre: float = C.LAMBDA_PRE):
    """Potencial pre-geométrico en V₀D (Ec. 20).

        V_pre(Φ; S) = [β₀(Φ² - v₀²)² - γ₀ Φ] · Θ_{λ_pre}(S - S_{0.001})

    Activado por una función escalón suave en torno a S_{0.001}.
    """
    Phi = np.asarray(Phi, dtype=float)
    b0 = beta0(alpha)
    g0 = gamma0(alpha)
    base = b0 * (Phi ** 2 - C.V0_NORM ** 2) ** 2 - g0 * Phi
    return base * step(S, C.S_PRE, lam_pre)


def V_total(Phi: np.ndarray | float, S: float, alpha: float | None = None,
            lam: float = C.LAMBDA_ONT, lam_pre: float = C.LAMBDA_PRE,
            V0: float = 0.0) -> np.ndarray | float:
    """V_total = α_S·S·Φ² + V_pre(Φ;S) + V_geo(Φ;S)  (Ec. principal generalizada)."""
    return V(Phi, S, alpha=alpha, lam=lam, V0=V0) + V_pre(Phi, S, alpha=alpha, lam_pre=lam_pre)


def k_pre(S: float, alpha: float | None = None,
          Phi_star: float | None = None) -> float:
    """Tasa de colapso k_pre(S) = ∂_S V_pre(Φ*; S) / V_pre(Φ*; S)   (Ec. 449).

    Para S < S₁ y Φ* ≈ v₀ se aproxima al valor de saturación λ_pre
    (Ec. 450).
    """
    if Phi_star is None:
        Phi_star = C.V0_NORM
    eps = 1e-6
    Vp = float(V_pre(Phi_star, S, alpha=alpha))
    Vp_dS = float(V_pre(Phi_star, S + eps, alpha=alpha))
    if abs(Vp) < 1e-30:
        return float(C.LAMBDA_PRE)
    return float((Vp_dS - Vp) / eps / Vp)


def lambda_pre() -> float:
    """λ_pre — tasa asintótica del colapso pre-geométrico (Ec. 450).

    λ_pre ∈ [1e-5, 5e-4]; valor central tabulado en `constants.LAMBDA_PRE`.
    """
    return C.LAMBDA_PRE
