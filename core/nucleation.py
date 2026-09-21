"""La salida de S₀ por nucleación: el bounce de Coleman sobre el Potencial
Basal y la tasa Γ₀(δ₀) (v35, Prop. 3.5; frente pendiente nombrado en el
repaso del 16-sep, §1.1: «el eslabón inicial del reloj S y hoy no está»).

LO QUE ES. El instantón O(d) del campo radial sobre el corte θ = θ_nuc
del Plano Dual (la salida ocurre hacia el polo de masa, Prop. 3.5),
con el paisaje COMPLETO (inclinación −η·χ incluida): la ecuación del
bounce

    φ'' + (d − 1)/r · φ' = V'(φ),   φ'(0) = 0,   φ(r → ∞) = φ_fv

resuelta por disparo (overshoot/undershoot) sobre φ(0) ∈ (barrera,
vacío verdadero), y la acción euclidiana

    B = Ω_{d−1} ∫₀^∞ r^{d−1} [ ½φ'² + V(φ) − V_fv ] dr,

con Γ₀ = A·e^{−B}. El PREFACTOR A es dimensional y no está en el
tratado: se publica Γ₀/A = e^{−B} y el exponente B; la ley de escala
B(δ₀) se MIDE ajustando sobre un barrido. Argumento de escala con (3.2):
φ ~ δ₀^{1/2}, V ~ δ₀³ ⟹ L ~ φ/√V ~ δ₀^{−1} y B ~ φ²L^{d−2} ~ δ₀^{3−d}:
para d = 4, B ∝ δ₀^{−1} → ∞ y Γ₀(0) = 0 (inercia eterna del perfecto,
Prop. 3.5) SIN prefactor; para d = 3, B → constante y Γ₀(0) = 0 solo si
A(δ₀) → 0 — el módulo publica ambos y declara cuál cumple el axioma.
La inclinación rompe la ley pura (corrección O(δ₀^{1/2}), como en T₀).

LO QUE NO ES. No fija la dimensión d del instantón (el tramo
pre-geométrico no tiene espacio-tiempo: d es una convención declarada,
publicada para d = 3 y d = 4); no fija el prefactor; no fija la
normalización de la acción (rigidez G = 1 y ħ = 1 declaradas). Es el
eslabón que el reloj S consume para arrancar en el centro del bounce
φ(0) con σ = 0 — sustituyendo la convención «punto de escape V = V_fv»
por el estado que el instantón entrega (thick wall: V(φ(0)) < V_fv y
la descarga arranca con f₀ > 0, la «entropía de nucleación», que se
publica).

Estatuto: cálculo (E8) con convenciones declaradas; ninguna
preinscripción (no decide nada); expectativa declarada (E13): Γ₀(δ₀)
creciente en δ₀ y nula en δ₀ = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .basal import B_BAR, C0_DEFAULT, E_BAR, M_BAR, scaled_params
from .s_clock import radial_landscape

STATUS = ("bounce de Coleman sobre el corte radial del Basal completo (E8): dimensión d del "
          "instantón, prefactor A y normalización de la acción DECLARADOS; B(δ₀) medida; "
          "Γ₀(0) = 0 se cumple sin prefactor para d = 4 (B ∝ δ₀^{−1}) y exige A(δ₀) → 0 para d = 3")

OMEGA = {2: 2.0 * np.pi, 3: 4.0 * np.pi, 4: 2.0 * np.pi ** 2}   # Ω_{d−1}


@dataclass(frozen=True)
class BounceResult:
    delta0: float
    d: int
    theta: float
    phi0: float                 # centro del bounce φ(0) (= ρ en el corte)
    phi_fv: float
    phi_barrier: float
    phi_tv: float
    phi_esc_energy: float       # punto de escape a energía cero (V = V_fv)
    B: float                    # acción euclidiana del bounce
    Gamma_over_A: float         # e^{−B}
    V0_minus_Vfv_over_T0: float  # (V(φ0) − V_fv)/T₀ < 0: descarga ya realizada al emerger
    f0: float                   # entropía de nucleación −(V(φ0) − V_fv)/T₀
    thin_wall_B: float | None   # estimación de pared delgada (referencia; NO aplicable si ε ≫ barrera)
    thin_wall_applicable: bool  # ε = T₀ ≪ altura de barrera
    barrier_over_T0: float
    n_shots: int
    converged: bool


def _potential_1d(delta0: float, theta: float, m_bar: float, b_bar: float, e_bar: float, C0: float):
    """V(ρ) y V'(ρ) sobre el corte θ (η incluida)."""
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    tilt = p["eta"] * (np.cos(theta) - np.sin(theta)) / np.sqrt(2.0)

    def V(r):
        return 0.5 * p["M0_sq"] * r ** 2 - 0.25 * p["B"] * r ** 4 + (C0 / 6.0) * r ** 6 - tilt * r

    def dV(r):
        return p["M0_sq"] * r - p["B"] * r ** 3 + C0 * r ** 5 - tilt
    return V, dV


def bounce(delta0: float, d: int = 4, theta: float = 0.0, m_bar: float = M_BAR, b_bar: float = B_BAR,
           e_bar: float = E_BAR, C0: float = C0_DEFAULT, n_shots: int = 80, r_max_factor: float = 60.0) -> BounceResult:
    """Bounce O(d) por disparo sobre φ(0). Overshoot: φ cruza φ_fv con
    velocidad (φ(0) demasiado cerca del vacío verdadero); undershoot: φ'
    cambia de signo antes de llegar (demasiado cerca de la barrera)."""
    if d not in OMEGA:
        raise ValueError("d ∈ {2, 3, 4}")
    land = radial_landscape(delta0, theta, m_bar, b_bar, e_bar, C0)
    if not land["metastable"] or land["rho_esc"] is None:
        raise ValueError(f"sin falso vacío metastable en δ0 = {delta0:g}: no hay bounce")
    V, dV = _potential_1d(delta0, theta, m_bar, b_bar, e_bar, C0)
    phi_fv, phi_bar, phi_tv = land["rho_fv"], land["rho_barrier"], land["rho_tv"]
    T0 = land["T0_full"]
    # escala de longitud natural: L ~ 1/√|V''| en el vacío verdadero
    k2 = abs(dV(phi_tv * (1 + 1e-6)) - dV(phi_tv * (1 - 1e-6))) / (2e-6 * phi_tv)
    L = 1.0 / np.sqrt(max(k2, 1e-300))
    r_max = r_max_factor * L

    def rhs(r, y):
        phi, dphi = y
        fric = 0.0 if r == 0.0 else (d - 1) / r * dphi
        return [dphi, dV(phi) - fric]

    def shoot(phi0):
        # arranque en r₀ ≪ L con la serie φ ≈ φ0 + V'(φ0) r²/(2d)
        r0 = 1e-4 * L
        y0 = [phi0 + dV(phi0) * r0 ** 2 / (2.0 * d), dV(phi0) * r0 / d]
        cross = lambda r, y: y[0] - phi_fv          # noqa: E731  overshoot: cruza el falso vacío
        cross.terminal, cross.direction = True, -1
        turn = lambda r, y: y[1]                    # noqa: E731  undershoot: φ' vuelve a cero
        turn.terminal, turn.direction = True, 1
        sol = solve_ivp(rhs, (r0, r_max), y0, events=(cross, turn), rtol=1e-10, atol=1e-13 * phi_tv,
                        max_step=0.05 * L, dense_output=True)
        if sol.t_events[0].size:
            return "over", sol
        if sol.t_events[1].size:
            return "under", sol
        return "end", sol   # llegó a r_max sin cruzar ni girar: casi solución

    lo, hi = phi_bar * (1 + 1e-9), phi_tv * (1 - 1e-9)
    best, n = None, 0
    for n in range(1, n_shots + 1):
        mid = 0.5 * (lo + hi)
        kind, sol = shoot(mid)
        best = (mid, sol)
        if kind == "over":
            hi = mid
        elif kind == "under":
            lo = mid
        else:
            break
        if hi - lo < 1e-13 * phi_tv:
            break
    phi0, sol = best
    # acción: integrar hasta donde φ se pega al falso vacío (o r_max)
    r = sol.t
    phi, dphi = sol.y
    dens = 0.5 * dphi ** 2 + (V(phi) - V(phi_fv))
    B = OMEGA[d] * np.trapezoid(r ** (d - 1) * dens, r)
    # pared delgada (referencia): σ_w = ∫√(2(V − V_fv)) dφ entre φ_fv y el punto de escape;
    # B_tw = Ω_{d−1}·[σ_w^d / ε^{d−1}]·(d−1)^{d−1}/d  con ε = T₀ (Coleman 1977)
    phis = np.linspace(phi_fv, land["rho_esc"], 2001)
    sig_w = np.trapezoid(np.sqrt(np.clip(2.0 * (V(phis) - V(phi_fv)), 0.0, None)), phis)
    B_tw = OMEGA[d] * sig_w ** d / T0 ** (d - 1) * (d - 1) ** (d - 1) / d if T0 > 0 else None
    converged = abs(phi[-1] - phi_fv) < 1e-3 * (phi_tv - phi_fv) or (hi - lo) < 1e-10 * phi_tv
    return BounceResult(delta0=delta0, d=d, theta=theta, phi0=float(phi0), phi_fv=float(phi_fv),
                        phi_barrier=float(phi_bar), phi_tv=float(phi_tv), phi_esc_energy=float(land["rho_esc"]),
                        B=float(B), Gamma_over_A=float(np.exp(-B)) if B < 700 else 0.0,
                        V0_minus_Vfv_over_T0=float((V(phi0) - V(phi_fv)) / T0), f0=float(-(V(phi0) - V(phi_fv)) / T0),
                        thin_wall_B=None if B_tw is None else float(B_tw),
                        thin_wall_applicable=bool(land["barrier_height"] > 5.0 * T0),
                        barrier_over_T0=float(land["barrier_height"] / T0), n_shots=n, converged=bool(converged))


# ------------------------------------------------ escape de Kramers (Def. 4.4)
def kramers_escape(delta0: float, D_ent: float, theta: float = 0.0, G: float = 1.0, m_bar: float = M_BAR,
                   b_bar: float = B_BAR, e_bar: float = E_BAR, C0: float = C0_DEFAULT) -> dict:
    """La OTRA nucleación, la propia de la dinámica del tratado: el Flujo
    del Camino es disipativo y de primer orden (Axioma 4) y su
    completación estocástica (Def. 4.4) añade difusión entrópica. El
    escape del falso vacío es entonces un problema de Kramers
    sobreamortiguado sobre el corte radial:

        Γ_K = √(V''(φ_fv)·|V''(φ_b)|)/(2π·G) · exp(−ΔV_b/D_ent),

    con D_ent la intensidad de la difusión (DECLARADA: el diccionario
    t ↔ σ de la Def. 4.4 no la fija). Con el escalado (3.2) el prefactor
    va como δ₀² y ΔV_b ∝ δ₀³ → 0: Γ_K(0) = 0 por el PREFACTOR (el paisaje
    plano de δ₀ = 0 no tiene curvatura que fije un ritmo), no por la
    exponencial — el mecanismo opuesto al bounce O(4), donde B → ∞.
    Se publican ambos; decidir cuál es el del tratado es del diccionario."""
    land = radial_landscape(delta0, theta, m_bar, b_bar, e_bar, C0)
    if not land["metastable"]:
        raise ValueError("sin falso vacío metastable: no hay escape que calcular")
    V, dV = _potential_1d(delta0, theta, m_bar, b_bar, e_bar, C0)
    h = 1e-6 * land["rho_tv"]
    Vpp = lambda r: (dV(r + h) - dV(r - h)) / (2.0 * h)  # noqa: E731
    w_fv, w_b = Vpp(land["rho_fv"]), Vpp(land["rho_barrier"])
    dVb = land["barrier_height"]
    pref = np.sqrt(max(w_fv, 0.0) * max(-w_b, 0.0)) / (2.0 * np.pi * G)
    return {"delta0": delta0, "D_ent": D_ent, "barrier_height": float(dVb), "V_pp_fv": float(w_fv),
            "V_pp_barrier": float(w_b), "prefactor": float(pref), "exponent": float(dVb / D_ent),
            "Gamma_K": float(pref * np.exp(-dVb / D_ent)),
            "status": "declarado: D_ent no fijada por el corpus; Γ_K(0) = 0 por el prefactor ∝ δ₀²"}


def kramers_scaling(delta0_grid, D_ent: float, **kw) -> dict:
    rows = [kramers_escape(x, D_ent, **kw) for x in delta0_grid]
    x = np.log(np.asarray(delta0_grid, dtype=float))
    p_pref = float(np.polyfit(x, np.log([r["prefactor"] for r in rows]), 1)[0])
    p_bar = float(np.polyfit(x, np.log([r["barrier_height"] for r in rows]), 1)[0])
    return {"D_ent": D_ent, "rows": rows, "prefactor_exponent_measured": p_pref, "prefactor_exponent_argument": 2.0,
            "barrier_exponent_measured": p_bar, "barrier_exponent_argument": 3.0,
            "Gamma_K_vanishes_at_delta0_zero": bool(p_pref > 0.0)}


def scaling_law(delta0_grid, d: int = 4, **kw) -> dict:
    """B(δ₀) sobre un barrido y ajuste log-log B ∝ δ₀^p; argumento de
    escala: p = 3 − d (sin inclinación). Publica p medido y su desviación."""
    rows = [bounce(x, d=d, **kw) for x in delta0_grid]
    x = np.log(np.asarray(delta0_grid, dtype=float))
    y = np.log(np.array([r.B for r in rows]))
    p, c = np.polyfit(x, y, 1)
    return {"d": d, "rows": [r.__dict__ for r in rows], "exponent_measured": float(p),
            "exponent_scaling_argument": float(3 - d), "intercept": float(c),
            "Gamma0_vanishes_at_delta0_zero_without_prefactor": bool(p < 0.0),
            "note": "la inclinación (∝ δ₀^{7/2}) desplaza el exponente medido respecto de 3 − d en O(δ₀^{1/2})"}


def delta0_for_B(B_target: float, d: int = 4, lo: float = 1e-3, hi: float = 0.1, **kw) -> float:
    """El δ₀ que da una acción B dada (p. ej. B ≈ 1: nucleación rápida)."""
    f = lambda x: bounce(x, d=d, **kw).B - B_target  # noqa: E731
    return float(brentq(f, lo, hi, xtol=1e-6))
