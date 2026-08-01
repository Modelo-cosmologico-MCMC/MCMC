"""Gea, el Sello de Newton y Atlas — el sector gravitatorio (v35, Cap. 9).

Sello de Newton (Prop. 9.3, ec. 9.3): las dos constantes efectivas son

    G_N = G_B/ξ,     G_cosmo = 2·G_B/(3·λK − 1)

y la relatividad general se recupera EXACTAMENTE en el punto fijo
(λK, ξ) → (1, 1) — verificación de signo de §13.5.

Identificación de Atlas (Def. 9.4, ec. 9.4): para λK ≠ 1 la foliación
propaga un modo escalar adicional — el MCMC lo identifica con Φ_ten.
Condición de propagación sana:

    c_s² ∝ (λK − 1)/(3·λK − 1) > 0   ⟹   λK > 1 (o λK < 1/3);
    la aproximación al sello debe ser por arriba, λK ↓ 1⁺.

ESTATUTO: la estabilidad del modo de Atlas (λK ↓ 1⁺, con el Término de
Cronos, Prop. 9.5) está clasificada CONDICIONAL en §13.4; y la relación
del residuo con la observación es la CONJETURA 9.6 (ec. 9.5):

    G_cosmo/G_N − 1 ≃ −(3/2)·εK ≈ −1.8%   (con εK ≡ λK(S_act) − 1 ≈ ε)

conectada al frente abierto nº 6 (contraste de los Residuos). Este
módulo expone ambas condiciones; no las resuelve.
"""

from __future__ import annotations

from mcmc_ontology import constants as C

STATUS_ATLAS = "condicional (§13.4: estabilidad del modo, λK ↓ 1⁺)"
STATUS_RESIDUOS = "conjetura (9.6; frente abierto nº 6)"


def G_newton(G_B: float = 1.0, xi: float = 1.0) -> float:
    """G_N = G_B/ξ (ec. 9.3) — la constante de laboratorio."""
    return G_B / xi


def G_cosmo(G_B: float = 1.0, lambda_K: float = 1.0) -> float:
    """G_cosmo = 2·G_B/(3·λK − 1) (ec. 9.3) — la constante cosmológica."""
    denom = 3.0 * lambda_K - 1.0
    if denom == 0.0:
        raise ValueError("λK = 1/3: G_cosmo diverge (borde de la rama)")
    return 2.0 * G_B / denom


def newton_seal_ratio(lambda_K: float, xi: float, G_B: float = 1.0) -> float:
    """G_cosmo/G_N = 2ξ/(3λK − 1); vale 1 exactamente en el sello (1,1)."""
    return G_cosmo(G_B, lambda_K) / G_newton(G_B, xi)


def cs2_atlas(lambda_K: float) -> float:
    """c_s² del modo de Atlas, salvo factor positivo: (λK−1)/(3λK−1)
    (ec. 9.4). Sano si > 0: λK > 1 o λK < 1/3; el sello se aproxima
    por arriba (λK ↓ 1⁺)."""
    denom = 3.0 * lambda_K - 1.0
    if denom == 0.0:
        raise ValueError("λK = 1/3: borde de la rama")
    return (lambda_K - 1.0) / denom


def atlas_healthy(lambda_K: float) -> bool:
    """Propagación sana del modo de Atlas: c_s² > 0 (ec. 9.4)."""
    return cs2_atlas(lambda_K) > 0.0


def residues_prediction(eps_K: float = C.EPSILON_0) -> float:
    """Conjetura 9.6 (ec. 9.5): G_cosmo/G_N − 1 ≃ −(3/2)·εK.

    Con εK = ε = 0.012 del ajuste: ≈ −1.8%. ES CONJETURA (no resultado
    demostrado): su contraste con BBN/CMB frente a Cavendish es el
    frente abierto nº 6, uno de los tres donde el modelo «se la juega».
    """
    return -1.5 * eps_K
