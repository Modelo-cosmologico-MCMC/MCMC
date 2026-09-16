"""Frente 3 ↔ frente 6 — PPN de marco preferido (α₁, α₂) del sector Atlas
como límite khronométrico (c_ω → ∞) de la teoría de Einstein-aether
(E4_Atlas). Requiere sympy (extra `[derivation]`); la suite no lo importa.

Los parámetros PPN de Einstein-aether (Foster & Jacobson 2006) son

    α₁ = −8(c₃² + c₁c₄)/(2c₁ − c₁² + c₃²),
    α₂ = α₁/2 − (c₁ + 2c₃ − c₄)(2c₁ + 3c₂ + c₃ + c₄)/(c₁₂₃(2 − c₁₄)),

y la clase khronométrica es el límite c_ω ≡ c₁ − c₃ → ∞ con
c₁₄ = α, c₁₃ = β, c₂ = λ fijos (Jacobson 2010; Blas–Pujolàs–Sibiryakov):

    α₁ = 4(α − 2β)/(β − 1),
    α₂ = (α − 2β)(αβ + 2αλ + α − β² − 3βλ − 3β − λ)/((α − 2)(β − 1)(β + λ)).

MAPEO con el sector del tratado (ADM, gauge unitario; #18/#20): α_a = α,
λ_K = 1 + λ y ξ = 1 ⇔ β = 0 (c_T² = ξ en el nuestro, 1/(1 − β) en el
covariante). Con β = 0 se verifican c_s² y G_cosmo/G_N contra las formas
cerradas de cosmology/mu_eta_atlas.py, y

    α₁ = −4α_a,   α₂ = α_a(2α_aλ + α_a − λ)/(λ(2 − α_a)) = −α_a/2 + O(α_a²).

Las cotas de campo débil (LLR sobre α₁, giro solar sobre α₂) acotan
entonces α_a directamente; la más restrictiva gobierna.
"""

from __future__ import annotations

import sympy as sp

a, b, lam, cw = sp.symbols("alpha beta lambda c_omega", positive=True)


def aether_ppn() -> dict:
    """α₁, α₂ de Einstein-aether (FJ06) en los c_i y parametrizados por
    (α, β, λ, c_ω) con c₁ = (β + c_ω)/2, c₃ = (β − c_ω)/2, c₄ = α − c₁,
    c₂ = λ."""
    c1 = (b + cw) / 2
    c3 = (b - cw) / 2
    c4 = a - c1
    c2 = lam
    c14 = c1 + c4
    c123 = c1 + c2 + c3
    alpha1 = -8 * (c3 ** 2 + c1 * c4) / (2 * c1 - c1 ** 2 + c3 ** 2)
    alpha2 = alpha1 / 2 - (c1 + 2 * c3 - c4) * (2 * c1 + 3 * c2 + c3 + c4) / (
        c123 * (2 - c14))
    return {"alpha1": alpha1, "alpha2": alpha2,
            "c14_is_alpha": bool(sp.simplify(c14 - a) == 0),
            "c13_is_beta": bool(sp.simplify(c1 + c3 - b) == 0)}


def khronometric_limit() -> dict:
    """Límite c_ω → ∞: formas cerradas en (α, β, λ) y su especialización a
    β = 0 (ξ = 1)."""
    ae = aether_ppn()
    a1 = sp.factor(sp.limit(ae["alpha1"], cw, sp.oo))
    a2 = sp.factor(sp.limit(ae["alpha2"], cw, sp.oo))
    a1_closed = 4 * (a - 2 * b) / (b - 1)
    a2_closed = (a - 2 * b) * (a * b + 2 * a * lam + a - b ** 2 - 3 * b * lam - 3 * b - lam) / (
        (a - 2) * (b - 1) * (b + lam))
    a1_b0 = sp.simplify(a1.subs(b, 0))
    a2_b0 = sp.factor(a2.subs(b, 0))
    a2_b0_closed = a * (2 * a * lam + a - lam) / (lam * (2 - a))
    return {"alpha1": a1, "alpha2": a2,
            "alpha1_matches_closed": bool(sp.simplify(a1 - a1_closed) == 0),
            "alpha2_matches_closed": bool(sp.simplify(a2 - a2_closed) == 0),
            "alpha1_beta0": a1_b0, "alpha2_beta0": a2_b0,
            "alpha1_beta0_is_minus_4alpha": bool(sp.simplify(a1_b0 + 4 * a) == 0),
            "alpha2_beta0_matches_closed": bool(sp.simplify(a2_b0 - a2_b0_closed) == 0),
            "alpha2_beta0_leading": str(sp.series(a2_b0, a, 0, 2).removeO()),
            "alpha2_beta0_leading_is_minus_alpha_half": bool(
                sp.simplify(sp.series(a2_b0, a, 0, 2).removeO() + a / 2) == 0)}


def mapping_checks() -> dict:
    """Identidades del mapeo ADM ↔ covariante en β = 0 (ξ = 1): c_s² y
    G_cosmo/G_N del tratado (#18) frente a las formas khronométricas
    c_s² = (2−α)(β+λ)/(α(1−β)(2+β+3λ)) y G_cosmo/G_N = (2−α)/(2+β+3λ)."""
    lamK, al_a = sp.symbols("lambda_K alpha_a", positive=True)
    cs2_treatise = (2 - al_a) * (lamK - 1) / (al_a * (3 * lamK - 1))          # ξ = 1
    g_treatise = (2 - al_a) / (3 * lamK - 1)                                   # ξ = 1
    sub = {al_a: a, lamK: 1 + lam}
    cs2_khron_b0 = (2 - a) * lam / (a * (2 + 3 * lam))
    g_khron_b0 = (2 - a) / (2 + 3 * lam)
    cT2_khron = 1 / (1 - b)
    return {"cs2_matches_at_beta0": bool(sp.simplify(cs2_treatise.subs(sub) - cs2_khron_b0) == 0),
            "Gcosmo_matches_at_beta0": bool(sp.simplify(g_treatise.subs(sub) - g_khron_b0) == 0),
            "cT2_xi1_is_beta0": bool(sp.simplify(cT2_khron.subs(b, 0) - 1) == 0),
            "cs2_khron": str(cs2_khron_b0), "g_khron": str(g_khron_b0)}


def alpha_a_bound(bound_alpha1: float, bound_alpha2: float, lam_value: float) -> dict:
    """Cotas sobre α_a (β = 0): de |α₁| = 4α_a ≤ b₁ ⟹ α_a ≤ b₁/4; de
    |α₂(α_a, λ)| ≤ b₂ resolviendo exactamente en α_a (raíz positiva más
    pequeña; la forma es monótona en 0 < α_a ≪ 1)."""
    from_a1 = bound_alpha1 / 4.0
    a2_expr = a * (2 * a * lam + a - lam) / (lam * (2 - a))
    eq = sp.Eq(-a2_expr.subs(lam, lam_value), bound_alpha2)      # α₂ < 0 para α_a pequeño
    sols = [float(s) for s in sp.solve(eq, a) if s.is_real and s > 0]
    from_a2 = min(sols) if sols else float("nan")
    return {"alpha_a_max_from_alpha1": from_a1,
            "alpha_a_max_from_alpha2": from_a2,
            "alpha_a_max": min(from_a1, from_a2),
            "governing": "alpha2" if from_a2 < from_a1 else "alpha1",
            "leading_order_alpha2_bound": 2.0 * bound_alpha2}


def run_all() -> dict:
    ae = aether_ppn()
    kl = khronometric_limit()
    mc = mapping_checks()
    out = {"aether_parametrization": {"c14_is_alpha": ae["c14_is_alpha"],
                                      "c13_is_beta": ae["c13_is_beta"]},
           "khronometric_limit": {k: (str(v) if isinstance(v, sp.Basic) else v)
                                  for k, v in kl.items()},
           "mapping": mc}
    flags = [ae["c14_is_alpha"], ae["c13_is_beta"], kl["alpha1_matches_closed"],
             kl["alpha2_matches_closed"], kl["alpha1_beta0_is_minus_4alpha"],
             kl["alpha2_beta0_matches_closed"], kl["alpha2_beta0_leading_is_minus_alpha_half"],
             mc["cs2_matches_at_beta0"], mc["Gcosmo_matches_at_beta0"], mc["cT2_xi1_is_beta0"]]
    out["identities_pass"] = bool(all(flags))
    out["n_identities"] = len(flags)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
