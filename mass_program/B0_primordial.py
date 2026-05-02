"""B0 — Imperfección primordial δ₀ ≡ ε.

Define la asimetría primordial que da origen a la jerarquía de escalas y a
la energía oscura dinámica Λ_rel(z). En el ajuste global bayesiano:

    δ₀ = ε = 0.012 ± 0.003

El VEV del nivel cero v₀ es proporcional al primero:
    v₀ = δ₀ · v₁
"""

from mcmc_ontology import constants as C


def delta_0() -> float:
    """δ₀ ≡ ε (imperfección primordial)."""
    return C.EPSILON_0


def v0() -> float:
    """v₀ = δ₀ · v₁ [GeV]."""
    return C.EPSILON_0 * C.V_GEV["C1"]
