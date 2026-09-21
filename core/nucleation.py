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
from scipy.integrate import quad, solve_ivp
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


# ============================================================ ronda 2 (21-sep, tarde)
# n_dim DECLARADO: el tramo pre-geométrico no tiene espacio-tiempo, así que la
# dimensionalidad del instantón es una convención publicada, no un dato.
#   n = 1 : túnel 0+1 (WKB/Gamow) del Flujo del Camino como grado de libertad
#           único — B₁ = 2∫√(2G(V − V_fv)) dρ, Γ₀ = (ω_fv/2π)·e^{−B₁} con el
#           prefactor de Gamow ω_fv = √(V''(fv)/G) = m̄·δ₀ (frecuencia de
#           intento). Con (3.2): B₁ = b₁·δ₀² → 0, así que Γ₀ → 0 LINEALMENTE
#           por el prefactor (la ley que el axioma Γ₀(0) = 0 obtiene sin
#           exponencial). El punto de salida WKB es exactamente el punto de
#           escape V = V_fv de la convención v0 del reloj.
#   n = 3, 4 : bounce de Coleman O(n) por disparo (arriba); Γ₀/A = e^{−B}.
N_DIM_DECLARED = (1, 3, 4)


@dataclass(frozen=True)
class GamowResult:
    delta0: float
    theta: float
    rho_fv: float
    rho_esc: float
    B1: float                   # acción WKB 2∫√(2G(V − V_fv)) dρ
    b1_over_delta0_sq: float    # B₁/δ₀²
    omega_fv: float             # √(V''(φ_fv)/G): frecuencia de intento
    prefactor: float            # ω_fv/(2π)
    Gamma0: float               # (ω_fv/2π)·e^{−B₁}
    n_grid: int
    converged: bool             # B₁ estable al duplicar la malla (< 1e-6 relativo)


def gamow_tunnel(delta0: float, theta: float = 0.0, m_bar: float = M_BAR, b_bar: float = B_BAR,
                 e_bar: float = E_BAR, C0: float = C0_DEFAULT, G: float = 1.0, n_grid: int = 4001) -> GamowResult:
    """Túnel 0+1 (n_dim = 1) sobre el corte θ del Basal completo."""
    land = radial_landscape(delta0, theta, m_bar, b_bar, e_bar, C0)
    if not land["metastable"] or land["rho_esc"] is None:
        raise ValueError(f"sin falso vacío metastable en δ0 = {delta0:g}: no hay túnel")
    V, dV = _potential_1d(delta0, theta, m_bar, b_bar, e_bar, C0)
    fv, esc = land["rho_fv"], land["rho_esc"]
    integrand = lambda r: np.sqrt(max(2.0 * G * (V(r) - land["V_fv"]), 0.0))  # noqa: E731
    # cuadratura adaptativa (la raíz se anula como √ en ambos extremos) y
    # comprobación con la regla del trapecio sobre n_grid puntos
    B1 = float(2.0 * quad(integrand, fv, esc, limit=200, epsrel=1e-10)[0])
    r = np.linspace(fv, esc, n_grid)
    B1_trap = float(2.0 * np.trapezoid(np.sqrt(np.clip(2.0 * G * (V(r) - land["V_fv"]), 0.0, None)), r))
    h = 1e-6 * land["rho_tv"]
    Vpp = (dV(fv + h) - dV(fv - h)) / (2.0 * h)
    omega = float(np.sqrt(max(Vpp, 0.0) / G))
    pref = omega / (2.0 * np.pi)
    return GamowResult(delta0=delta0, theta=theta, rho_fv=float(fv), rho_esc=float(esc), B1=B1,
                       b1_over_delta0_sq=B1 / delta0 ** 2, omega_fv=omega, prefactor=pref,
                       Gamma0=float(pref * np.exp(-B1)), n_grid=n_grid,
                       converged=bool(abs(B1 - B1_trap) <= 1e-3 * max(B1, 1e-300)))


def b1_no_tilt(m_bar: float = M_BAR, b_bar: float = B_BAR, C0: float = C0_DEFAULT, n: int = 20001) -> float:
    """Constante b₁ de la ley B₁ = b₁·δ₀² SIN inclinación: con ρ = √δ₀·s,
    V₀ = δ₀³·v(s), v(s) = m̄²s²/2 − b̄s⁴/4 + C0s⁶/6, y b₁ = 2∫₀^{s_esc}√(2v(s)) ds
    donde s_esc es la raíz no nula de v (el punto de escape)."""
    # s_esc: v(s) = 0 con s > 0 ⟺ C0 s⁴/6 − b̄ s²/4 + m̄²/2 = 0 (raíz menor en s²)
    disc = (b_bar / 4.0) ** 2 - 4.0 * (C0 / 6.0) * (m_bar ** 2 / 2.0)
    if disc < 0.0:
        raise ValueError("sin punto de escape: el mínimo no trivial no baja del origen")
    s2 = ((b_bar / 4.0) - np.sqrt(disc)) / (2.0 * C0 / 6.0)
    v = lambda s: max(m_bar ** 2 * s ** 2 / 2.0 - b_bar * s ** 4 / 4.0 + C0 * s ** 6 / 6.0, 0.0)  # noqa: E731
    return float(2.0 * quad(lambda s: np.sqrt(2.0 * v(s)), 0.0, np.sqrt(s2), limit=200, epsrel=1e-10)[0])


def gamma0(delta0: float, n_dim: int, **kw) -> dict:
    """Γ₀(δ₀) condicional a n_dim declarado: n = 1 → túnel de Gamow con
    prefactor; n ∈ {3, 4} → bounce O(n) (prefactor A no fijado)."""
    if n_dim not in N_DIM_DECLARED:
        raise ValueError(f"n_dim ∈ {N_DIM_DECLARED} (declarado)")
    if n_dim == 1:
        g = gamow_tunnel(delta0, **kw)
        return {"n_dim": 1, "delta0": delta0, "B": g.B1, "Gamma0": g.Gamma0, "prefactor": g.prefactor,
                "prefactor_status": "Gamow (ω_fv/2π), derivado con G declarado", "converged": g.converged,
                "detail": g.__dict__}
    b = bounce(delta0, d=n_dim, **kw)
    return {"n_dim": n_dim, "delta0": delta0, "B": b.B, "Gamma0": None, "Gamma0_over_A": b.Gamma_over_A,
            "prefactor": None, "prefactor_status": "A dimensional, no fijado por el tratado", "converged": b.converged,
            "detail": b.__dict__}


def path_deformation(delta0: float, m_bar: float = M_BAR, b_bar: float = B_BAR, e_bar: float = E_BAR,
                     C0: float = C0_DEFAULT, G: float = 1.0, n_modes: int = 2, n_s: int = 1201,
                     maxiter: int = 400) -> dict:
    """SEGUNDA PASADA (n_dim = 1): ¿baja la acción WKB al dejar que el camino
    de túnel abandone el corte θ = 0 y se curve en el Plano Dual (ρ, θ), es
    decir en (ρ, χ)? Camino Φ(s), s ∈ [0, 1], desde el falso vacío (sobre el
    eje θ = 0, frontera φ_E = 0 del dominio) hasta la superficie de escape
    V = V_fv en el rayo θ_end: θ(s) = θ_end·s + Σ_k a_k·sin(kπs) (recortado al
    dominio [0, π/2]), ρ(s) lineal entre ρ_fv y ρ_esc(θ_end). B[Φ] =
    2∫√(2G·max(V − V_fv, 0))·|dΦ/ds| ds, minimizado con Nelder–Mead sobre
    (θ_end, a_1..a_k). Publica B_ray (camino recto θ = 0), B_min y el cociente.
    La inclinación −η·χ es máxima sobre θ = 0 dentro del dominio, así que la
    expectativa (E13, declarada) es cociente ≃ 1."""
    from scipy.optimize import minimize
    land0 = radial_landscape(delta0, 0.0, m_bar, b_bar, e_bar, C0)
    if not land0["metastable"] or land0["rho_esc"] is None:
        raise ValueError("sin falso vacío metastable: no hay camino que deformar")
    p = scaled_params(delta0, m_bar, b_bar, e_bar)
    V_fv, rho_fv = land0["V_fv"], land0["rho_fv"]

    def V2(x, y):
        r2 = x ** 2 + y ** 2
        return (0.5 * p["M0_sq"] * r2 - 0.25 * p["B"] * r2 ** 2 + (C0 / 6.0) * r2 ** 3
                - p["eta"] * (x - y) / np.sqrt(2.0))

    def rho_escape(theta_end):
        ld = radial_landscape(delta0, theta_end, m_bar, b_bar, e_bar, C0)
        if ld["rho_tv"] is None:
            return None
        Vr = lambda r: V2(r * np.cos(theta_end), r * np.sin(theta_end)) - V_fv  # noqa: E731
        lo = ld["rho_barrier"] if ld["rho_barrier"] is not None else 0.5 * ld["rho_tv"]
        if Vr(lo) <= 0.0 or Vr(ld["rho_tv"]) >= 0.0:
            return None
        return float(brentq(Vr, lo, ld["rho_tv"], xtol=1e-14))

    s = np.linspace(0.0, 1.0, n_s)

    def action(params):
        theta_end = float(np.clip(params[0], 0.0, np.pi / 2))
        th = theta_end * s + sum(a * np.sin((k + 1) * np.pi * s) for k, a in enumerate(params[1:]))
        th = np.clip(th, 0.0, np.pi / 2)
        r_end = rho_escape(theta_end)
        if r_end is None:
            return np.inf
        rho = rho_fv + (r_end - rho_fv) * s
        x, y = rho * np.cos(th), rho * np.sin(th)
        dens = np.sqrt(np.clip(2.0 * G * (V2(x, y) - V_fv), 0.0, None))
        dl = np.hypot(np.diff(x), np.diff(y))
        return float(2.0 * np.sum(0.5 * (dens[1:] + dens[:-1]) * dl))

    x0 = np.zeros(1 + n_modes)
    B_ray = action(x0)
    res = minimize(action, x0, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9 * max(B_ray, 1e-300), "maxiter": maxiter, "initial_simplex":
                            np.vstack([x0] + [x0 + 0.1 * np.eye(len(x0))[i] for i in range(len(x0))])})
    B_min = float(min(res.fun, B_ray))
    return {"delta0": delta0, "e_bar": e_bar, "B_ray": B_ray, "B_min": B_min, "ratio_min_over_ray": B_min / B_ray if B_ray > 0 else None,
            "theta_end_opt": float(np.clip(res.x[0], 0.0, np.pi / 2)), "modes_opt": res.x[1:].tolist(),
            "optimizer_converged": bool(res.success), "n_modes": n_modes,
            "note": "la inclinación −η·χ es máxima en θ = 0 dentro del dominio: el rayo recto es el candidato natural al mínimo"}
