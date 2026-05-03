"""B0 — Imperfección primordial δ₀, tensión T₀ y umbrales T_crit^(n).

Equaciones del Tratado (Apéndice K):

    δ₀ ≡ v₀                                       (Ec. 22)
    M_p^(0) = ½(1 + v₀)                           (Ec. 23)
    E_p^(0) = ½(1 - v₀)                           (Ec. 23)
    T₀ = M_p · c² · δ₀² = E_p · δ₀²              (Ec. 440)
    T₀ ≈ E_p × 10⁻⁴ ≈ 1.22×10¹⁵ GeV               (Ec. 441)

    T_crit^(n) = T₀/4 · S_n/ΔS · (v_n / v_1)^2   (Ec. 445)

    T(S) ≃ T₀ · exp(-λ_pre · S/ΔS)               (Ec. 450)
    λ_pre ∈ [1e-5, 5e-4]
"""

from __future__ import annotations

import math

from mcmc_ontology import constants as C


_SEAL_ORDER = ("C1", "C2", "C3", "C4")


def delta_0() -> float:
    """δ₀ ≡ ε (imperfección primordial)."""
    return C.EPSILON_0


def v0_norm() -> float:
    """v₀ adimensional ≡ δ₀ (Ec. 22)."""
    return C.V0_NORM


def v0_GeV() -> float:
    """v₀ = δ₀ · v₁ [GeV] (anclaje a la escala Planck)."""
    return C.V0_GEV


def Mp0() -> float:
    """M_p^(0) = ½(1 + v₀) (Ec. 23)."""
    return 0.5 * (1.0 + C.V0_NORM)


def Ep0() -> float:
    """E_p^(0) = ½(1 - v₀) (Ec. 23)."""
    return 0.5 * (1.0 - C.V0_NORM)


def T0_GeV() -> float:
    """Tensión primordial T₀ = M_Pl · δ₀² ≈ 1.76×10¹⁵ GeV (Ecs. 440-441).

    Equivalentemente T₀ = E_p × δ₀² (la masa primordial en V₀D coincide
    con la energía planckiana al inicio del ciclo).
    """
    return C.T0_GEV


def T_crit(seal: str) -> float:
    """Umbral crítico T_crit^(n) (Ec. 445):

        T_crit^(n) = T₀/4 · S_n / ΔS · (v_n / v_1)^2
    """
    Sn = C.S_SEALS[seal]
    vn = C.V_GEV[seal]
    v1 = C.V_GEV["C1"]
    return (C.T0_GEV / 4.0) * (Sn / C.DELTA_S) * (vn / v1) ** 2


def T_crit_table() -> dict[str, float]:
    """Tabla 67 del Tratado: T_crit^(n) para C1..C4 [GeV]."""
    return {seal: T_crit(seal) for seal in _SEAL_ORDER}


def tension_decay(S: float, lam_pre: float | None = None) -> float:
    """T(S) ≃ T₀ · exp(-λ_pre · S / ΔS)   (Ec. 450).

    Decaimiento exponencial de la tensión primordial en el tramo
    pre-geométrico.
    """
    if lam_pre is None:
        lam_pre = C.LAMBDA_PRE
    return C.T0_GEV * math.exp(-lam_pre * S / C.DELTA_S)


def lambda_pre() -> float:
    """λ_pre tabulado (valor central). Banda: [1e-5, 5e-4]."""
    return C.LAMBDA_PRE


def lambda_pre_band() -> tuple[float, float]:
    """Banda admisible para λ_pre."""
    return (1e-5, 5e-4)


# --- Balance energético pre-geométrico (Ec. 451-453) ---

def sigma0() -> float:
    """Tensión superficial ontológica σ₀ = T₀ / V₀D^(eq).

    En unidades adimensionales V₀D^(eq) ≡ S_{0.009}, así que σ₀ ≈ T₀/S₁.
    """
    return C.T0_GEV / C.S_SEALS["C1"]


def epsilon_exp(deltaV0D: float) -> float:
    """ε_exp^(n) = σ₀ · ΔV₀D^(n) — canal de expansión V₀D (Ec. 451)."""
    return sigma0() * deltaV0D
