"""Frente 3 — cola O(e²) del canal Atlas CON el sector de velocidades
(E3_Atlas): escalera exacta del modo creciente sobre el sistema lineal
completo y arnés numérico con condiciones iniciales adiabáticas.

Este módulo NO es ruta de producción (requiere sympy, extra opcional
`[derivation]`); la suite no lo importa. Comparte la acción cuadrática
promediada con validation/atlas_derivation.py (etapa 1) y los mismos
campos (ψ, b, φ, j0, jx, l1) en gauge unitario, materia = polvo.

MÉTODO (escalera). En materia dominante con a = t^{2/3} (κ = m = 1,
ξ = 1 — GW170817) todos los coeficientes del sistema lineal son
potencias de t, así que el modo creciente admite una serie exacta

    X_i(t) = t^g · Σ_n c_{i,n} t^{−n/3},   e² = (aH/ck)² = (4/9k²) t^{−2/3},

con g = 2p/3 y p el índice de crecimiento QS, p(p+½) = (3/2)(3λ_K−1)/(2−α_a).
El orden dominante reproduce µ_sub = 1/(1−α_a/2) y η₀ = 1 (control contra
las formas cerradas QS); el orden n = 2 da los coeficientes VERDADEROS de
η − 1 y de µ_loc − 1 en e², sin truncación cuasi-estática (∂_t y
velocidades incluidas). Las torres admiten cabeceras super-dominantes
(n < 0, p. ej. l1) y la normalización es c_{j0,0} = 1 (la dirección libre
de renormalización del modo no cambia los cocientes).

RESIDUOS DEL POLO (λ_K → 1). Los coeficientes tienen la estructura
coef(λ_K) = P/(λ_K − 1) + Q + O(λ_K − 1). El polo es FÍSICO (horizonte de
sonido del khronon: el parámetro pequeño es aH/(c_s k)) y sus residuos se
extraen por extrapolación de Richardson en h = λ_K − 1:

    P_η = 3α_a/(2 − α_a),     P_µ = −P_η · p(2p−1)/3   (p en λ_K = 1).

Equivalentemente η − 1 = (3/2)(aH/(c_s k))²[1 + O(λ_K−1)] y
µ_loc − 1 = −[p(2p−1)/2](aH/(c_s k))²[1 + O(λ_K−1)]. La cola QS truncada
(cosmology.mu_eta_atlas.eta_tail_coefficient) sobreestima el residuo en
×2(2−α_a)/(3α_a) y no depende de α_a: no es física.

ARNÉS ADIABÁTICO (E3b). Las torres evaluadas en t₀ dan condiciones
iniciales del modo creciente puro (sin excitar los modos oscilatorios del
khronon, que contaminan η con un suelo de ruido que no escala como e²);
el sistema completo se integra y η(e) − 1, µ_loc(e) − 1 se ajustan a
s·e² sobre la trayectoria: la pendiente s se contrasta con la escalera.
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

from validation.atlas_derivation import (
    J0b,
    a,
    alph,
    b,
    build_numeric_rhs,
    ellb,
    j0,
    jx,
    kap,
    l1,
    lamK,
    m,
    phi,
    psi,
    t,
    xi,
)

FIELDS = [psi, b, phi, j0, jx, l1]
NAMES = ["ps", "bb", "ph", "j0", "jx", "l1"]
K = sp.Symbol("k", positive=True)


def growth_exponent(lamv, alv):
    """p (índice de crecimiento en materia dominante, ξ = 1) y g = 2p/3,
    exactos: p(p+½) = (3/2)(3λ−1)/(2−α)."""
    Rg = sp.nsimplify(3 * lamv - 1) / sp.nsimplify(2 - alv)
    p = (-1 + sp.sqrt(1 + 24 * Rg)) / 4
    return p, 2 * p / 3


def linear_equations_matter_era(S: dict, lamv, alv):
    """EL de L2 para los seis campos con el fondo de polvo sustituido,
    ξ = 1, κ = m = 1, a = t^{2/3}, J̄⁰ = (4/3)(3λ−1) (Friedmann del Sello con
    a³H² = 4/9). λ y α racionales exactos."""
    lamv, alv = sp.nsimplify(lamv), sp.nsimplify(alv)
    J0c = sp.Rational(4, 3) * (3 * lamv - 1)
    L2n = S["L2"]
    for old, new in [(sp.Derivative(ellb, t, 2), 0), (sp.Derivative(ellb, t), -m),
                     (sp.Derivative(J0b, t, 2), 0), (sp.Derivative(J0b, t), 0),
                     (J0b, J0c), (kap, 1), (m, 1), (xi, 1), (lamK, lamv),
                     (alph, alv)]:
        L2n = L2n.subs(old, new)
    L2n = L2n.doit().subs(a, t ** sp.Rational(2, 3)).doit()
    Es = [sp.expand((sp.diff(L2n, f) - sp.diff(sp.diff(L2n, f.diff(t)), t)).doit())
          for f in FIELDS]
    return Es, J0c


def ladder_point(S: dict, lamv, alv, nlow: int = -4, nhigh: int = 16,
                 prof: int = 12) -> dict:
    """Escalera exacta en un punto (λ, α) racional, ξ = 1.

    Devuelve los coeficientes de la torre (para ICs adiabáticas), el orden
    dominante (mu0, eta0), los coeficientes de e² de η − 1 (coef_eta) y de
    µ_loc − 1 (coef_mu_relloc), y las comprobaciones estructurales de
    cabecera. Los símbolos libres de la cola profunda (no determinados por
    el corte `prof`) se fijan a 0: no afectan a n ≤ 4 si prof ≥ 12."""
    lamv, alv = sp.nsimplify(lamv), sp.nsimplify(alv)
    Es, J0c = linear_equations_matter_era(S, lamv, alv)
    p, g = growth_exponent(lamv, alv)

    C = {}
    ans = {}
    for f, nm in zip(FIELDS, NAMES):
        tower = 0
        for n in range(nlow, nhigh + 1):
            C[nm, n] = sp.Symbol(f"c_{nm}_{n}")
            tower += C[nm, n] * t ** sp.Rational(-n, 3)
        ans[f] = t ** g * tower

    Tv = sp.Symbol("T", positive=True)
    by_pow: dict = {}
    for jdx, e in enumerate(Es):
        ee = e
        for f in FIELDS:
            ee = ee.subs(f, ans[f])
        ee = sp.expand(sp.powsimp(sp.expand(ee.doit() / t ** g), force=True))
        ee = sp.expand(ee.subs(t, Tv ** 3))
        for term in sp.Add.make_args(ee):
            rest, q = term.as_coeff_exponent(Tv)
            by_pow.setdefault((jdx, sp.Integer(q)), []).append(rest)
    syseqs = []
    for (jdx, q), terms in by_pow.items():
        expr = sp.expand(sp.Add(*terms))
        if expr != 0:
            syseqs.append(((jdx, int(q)), expr))
    top = max(q for (_, q), _ in syseqs)
    sysuse = [e for (key, e) in syseqs if key[1] >= top - prof]
    unks = sorted({s for e in sysuse for s in e.free_symbols
                   if str(s).startswith("c_")}, key=str)
    sol = sp.linsolve([C["j0", 0] - 1] + sysuse, unks)
    if sol is sp.EmptySet or len(sol) == 0:
        raise RuntimeError("escalera inconsistente: g no es raíz del sistema")
    vec = list(sol)[0]
    free = {s for v in vec for s in getattr(v, "free_symbols", set())
            if str(s).startswith("c_")}
    solmap = {}
    for kk, vv in zip(unks, vec):
        solmap[kk] = vv.subs(dict.fromkeys(free, 0)) if hasattr(vv, "subs") else vv

    def cval(nm, n):
        s = C[nm, n]
        if s in free:
            return sp.S(0)
        return sp.simplify(solmap.get(s, sp.S(0)))

    checks = {
        "j0_superdominant_null": all(cval("j0", n) == 0 for n in range(nlow, 0)),
        "psi_starts_at_n2": cval("ps", 0) == 0 and cval("ps", 1) == 0,
        "phi_starts_at_n2": cval("ph", 0) == 0 and cval("ph", 1) == 0,
        "psi_odd_null": cval("ps", 3) == 0,
        "phi_odd_null": cval("ph", 3) == 0,
    }
    ps2, ps4 = cval("ps", 2), cval("ps", 4)
    ph2, ph4 = cval("ph", 2), cval("ph", 4)
    if ps2 == 0:
        raise RuntimeError("ψ sin cabecera en n = 2")
    # η = φ/ψ = (ph2 + ph4 T⁻²)/(ps2 + ps4 T⁻²); T⁻² = t^{−2/3} = (9k²/4) e²
    eta0 = sp.simplify(ph2 / ps2)
    eta2_T = sp.simplify(ph4 / ps2 - ph2 * ps4 / ps2 ** 2)
    coef_eta = sp.simplify(eta2_T * 9 * K ** 2 / 4)
    # µ: (k²/a²)ψ = −4πG_N µ ρ δ, G_N = G_B = 1/(16π), ρ = J0c/t², δ = j0 + 3φ
    d0 = sp.simplify(cval("j0", 0) + 3 * cval("ph", 0))
    d2 = sp.simplify(cval("j0", 2) + 3 * cval("ph", 2))
    mu0 = sp.simplify(-4 * K ** 2 * ps2 / (J0c * d0))
    mu2_T = sp.simplify(-4 * K ** 2 / J0c * (ps4 / d0 - ps2 * d2 / d0 ** 2))
    coef_mu = sp.simplify(mu2_T * 9 * K ** 2 / 4)
    coef_mu_relloc = sp.simplify(coef_mu * (1 - alv / 2))
    tower = {nm: {n: cval(nm, n) for n in range(nlow, nhigh + 1)} for nm in NAMES}
    return {"lambda_K": lamv, "alpha_a": alv, "p": p, "g": g, "J0c": J0c,
            "mu0": mu0, "eta0": eta0, "coef_eta": coef_eta,
            "coef_mu_relloc": coef_mu_relloc,
            "mu_sub_expected": 1 / (1 - alv / 2), "checks": checks,
            "n_equations": len(sysuse), "n_unknowns": len(unks),
            "n_free_deep": len(free), "tower": tower}


def _f(x) -> float:
    return float(sp.N(x, 30))


def ladder_summary(r: dict) -> dict:
    """Números serializables de un punto de la escalera."""
    return {"lambda_K": _f(r["lambda_K"]), "alpha_a": _f(r["alpha_a"]),
            "p": _f(r["p"]), "mu0": _f(r["mu0"]), "eta0": _f(r["eta0"]),
            "mu0_equals_1_over_1_minus_alpha_half": bool(
                sp.simplify(r["mu0"] - r["mu_sub_expected"]) == 0),
            "eta0_is_1": bool(sp.simplify(r["eta0"] - 1) == 0),
            "coef_eta_e2": _f(r["coef_eta"]),
            "coef_mu_local_e2": _f(r["coef_mu_relloc"]),
            "checks": {kk: bool(v) for kk, v in r["checks"].items()},
            "n_equations": r["n_equations"], "n_unknowns": r["n_unknowns"],
            "n_free_deep": r["n_free_deep"]}


H_STEPS = (sp.Rational(1, 200), sp.Rational(1, 100), sp.Rational(1, 50),
           sp.Rational(1, 25))


def pole_residues(S: dict, alv, h_steps=H_STEPS, **kw) -> dict:
    """Residuos P_η, P_µ del polo 1/(λ_K−1) por extrapolación polinómica:
    con f(h) = h·coef(1+h) = P + Q h + R h² + S h³ + O(h⁴), el ajuste
    cúbico exacto sobre cuatro pasos h da P = f(0) con error O(h⁴) y Q =
    f'(0). (Pasos racionales moderados: la aritmética exacta con h ≲ 1e-3
    es intratable en la escalera.) Compara con las formas cerradas
    P_η = 3α/(2−α), P_µ = −P_η p(2p−1)/3 (p en λ_K = 1)."""
    alv = sp.nsimplify(alv)
    hs = [sp.nsimplify(h) for h in h_steps]
    runs = [ladder_point(S, 1 + hh, alv, **kw) for hh in hs]
    hf = np.array([_f(hh) for hh in hs])
    out = {}
    for key in ("coef_eta", "coef_mu_relloc"):
        fv = np.array([_f(hh * r[key]) for hh, r in zip(hs, runs)])
        c = np.polyfit(hf, fv, len(hs) - 1)          # exacto por 4 puntos
        out[key] = {"P": float(c[-1]), "Q": float(c[-2]),
                    "f_values": [float(v) for v in fv]}
    p1, _ = growth_exponent(1, alv)
    P_eta_closed = 3 * alv / (2 - alv)
    P_mu_closed = -P_eta_closed * p1 * (2 * p1 - 1) / 3
    return {"alpha_a": _f(alv), "h_steps": [float(h) for h in hf],
            "coef_eta_at_steps": [_f(r["coef_eta"]) for r in runs],
            "coef_mu_local_at_steps": [_f(r["coef_mu_relloc"]) for r in runs],
            "P_eta": out["coef_eta"]["P"], "Q_eta": out["coef_eta"]["Q"],
            "P_mu_local": out["coef_mu_relloc"]["P"],
            "Q_mu_local": out["coef_mu_relloc"]["Q"],
            "P_eta_closed": _f(P_eta_closed), "P_mu_local_closed": _f(P_mu_closed),
            "P_eta_rel_err": abs(out["coef_eta"]["P"] / _f(P_eta_closed) - 1.0),
            "P_mu_rel_err": abs(out["coef_mu_relloc"]["P"] / _f(P_mu_closed) - 1.0),
            "ratio_P_mu_over_P_eta": out["coef_mu_relloc"]["P"] / out["coef_eta"]["P"],
            "ratio_closed": _f(-p1 * (2 * p1 - 1) / 3),
            "p_at_lamK_1": _f(p1),
            "qs_over_physical_residue": _f(2 * (2 - alv) / (3 * alv))}


# --------------------------------------------------------------------
# Arnés numérico con condiciones iniciales adiabáticas (E3b)
# --------------------------------------------------------------------

def adiabatic_initial_state(r: dict, kk: float, t0: float, nmax: int = 8):
    """Estado (φ, φ', j0, l1) del modo creciente puro en t₀ desde las
    torres de la escalera (truncadas en n ≤ nmax, donde el corte no las
    contamina). La torre lleva la dependencia en k de cada campo, así que
    se evalúa con k = kk."""
    g = r["g"]
    tw = r["tower"]

    def field(nm, tt, deriv=False):
        val = sp.S(0)
        for n, c in tw[nm].items():
            if n > nmax or c == 0:
                continue
            ex = g - sp.Rational(n, 3)
            term = c.subs(K, kk) * (ex * tt ** (ex - 1) if deriv else tt ** ex)
            val += term
        return float(sp.N(val, 30))

    tt = sp.nsimplify(t0)
    return [field("ph", tt), field("ph", tt, deriv=True), field("j0", tt),
            field("l1", tt)]


def run_adiabatic(S: dict, r: dict, k_values, a_start: float = 0.1,
                  e_fit_max: float = 0.02, rtol: float = 1e-10,
                  atol: float = 1e-18, n_eval: int = 600) -> dict:
    """Integra el sistema completo (ξ = 1) desde ICs adiabáticas y ajusta
    η − 1 = s_η e² y µ_loc − 1 = s_µ e² (mínimos cuadrados por el origen)
    sobre los puntos con e ≤ e_fit_max, junto con el exponente log-log de
    |η − 1| frente a e (debe ser 2). Devuelve por k y agrupado."""
    from scipy.integrate import solve_ivp

    lam = _f(r["lambda_K"])
    al = _f(r["alpha_a"])
    F, J0cv = build_numeric_rhs(S, lam, 1.0, al)
    GNn = 1.0 / (16 * np.pi)

    def rhs(tt, y, kk):
        ph, pd, j_, l_ = y
        return [pd, F["ph"](tt, kk, ph, pd, j_, l_),
                F["j0"](tt, kk, ph, pd, j_, l_),
                F["l1"](tt, kk, ph, pd, j_, l_)]

    t0 = a_start ** 1.5
    rows = []
    E2, DETA, DMU = [], [], []
    for kk in k_values:
        y0 = adiabatic_initial_state(r, kk, t0)
        t_eval = np.geomspace(t0, 1.0, n_eval)
        so = solve_ivp(rhs, (t0, 1.0), y0, args=(kk,), method="Radau",
                       rtol=rtol, atol=atol, t_eval=t_eval)
        tt = so.t
        ph, pd, j_, l_ = so.y
        a_ = tt ** (2 / 3)
        dN = j_ + 3 * ph
        psN = np.array([F["psi"](tt[i], kk, ph[i], pd[i], j_[i], l_[i])
                        for i in range(len(tt))])
        rho = J0cv / tt ** 2
        muN = (kk ** 2 / a_ ** 2) * psN / (-4 * np.pi * GNn * rho * dN)
        mu_loc = muN * (1 - al / 2)
        etaN = ph / psN
        e = (2 / 3) * tt ** (-1 / 3) / kk
        w = e <= e_fit_max
        e2 = e[w] ** 2
        de, dm = etaN[w] - 1.0, mu_loc[w] - 1.0
        s_eta = float(np.sum(e2 * de) / np.sum(e2 * e2))
        s_mu = float(np.sum(e2 * dm) / np.sum(e2 * e2))
        # ajuste con término e⁴ (E4b): X − 1 = s·e² + q·e⁴, mínimos cuadrados
        A4 = np.vstack([e2, e2 ** 2]).T
        (s_eta4, q_eta), *_ = np.linalg.lstsq(A4, de, rcond=None)
        (s_mu4, q_mu), *_ = np.linalg.lstsq(A4, dm, rcond=None)
        ok = de > 0
        expo = (float(np.polyfit(np.log(e[w][ok]), np.log(de[ok]), 1)[0])
                if ok.sum() > 10 else float("nan"))
        lnd, lna = np.log(np.abs(dN[w])), np.log(a_[w])
        p_loc = np.gradient(lnd, lna)
        rows.append({"k_over_H0": float(kk), "e_range_fit": [float(e[w].min()),
                                                              float(e[w].max())],
                     "n_points_fit": int(w.sum()),
                     "slope_eta_e2": s_eta, "slope_mu_local_e2": s_mu,
                     "slope_eta_e2_quartic": float(s_eta4), "coef_eta_e4": float(q_eta),
                     "slope_mu_local_e2_quartic": float(s_mu4),
                     "coef_mu_local_e4": float(q_mu),
                     "loglog_exponent_eta": expo,
                     "eta_minus_one_at_e_max": float(de[np.argmax(e2)]),
                     "p_num_median": float(np.median(p_loc)),
                     "p_spread_std": float(np.std(p_loc)),
                     "solver_success": bool(so.success)})
        E2.append(e2); DETA.append(de); DMU.append(dm)
    E2 = np.concatenate(E2); DETA = np.concatenate(DETA); DMU = np.concatenate(DMU)
    pooled = {"slope_eta_e2": float(np.sum(E2 * DETA) / np.sum(E2 * E2)),
              "slope_mu_local_e2": float(np.sum(E2 * DMU) / np.sum(E2 * E2)),
              "n_points": int(len(E2))}
    return {"rows": rows, "pooled": pooled, "a_start": a_start,
            "e_fit_max": e_fit_max, "p_qs": _f(r["p"])}


def run_qs_ic_control(S: dict, lam: float, al: float, k_values,
                      a_start: float = 1e-3, window_a: float = 0.3,
                      e_fit_max: float = 0.02, rtol: float = 1e-9,
                      atol: float = 1e-15) -> list[dict]:
    """CONTROL (no gobierna el desenlace): la misma integración con las
    ICs QS-consistentes del arnés E2 de #18 (excitan los modos oscilatorios
    del khronon). Publica la pendiente y el exponente log-log de |η − 1|
    frente a e en la ventana tardía, para documentar el suelo de ruido
    que no escala como e²."""
    from scipy.integrate import solve_ivp

    from validation.atlas_derivation import eta_qs_num, mu_qs_num

    F, J0cv = build_numeric_rhs(S, lam, 1.0, al)

    def rhs(tt, y, kk):
        ph, pd, j_, l_ = y
        return [pd, F["ph"](tt, kk, ph, pd, j_, l_),
                F["j0"](tt, kk, ph, pd, j_, l_),
                F["l1"](tt, kk, ph, pd, j_, l_)]

    t0 = a_start ** 1.5
    rows = []
    for kk in k_values:
        d0 = 1e-3
        e0 = (2 / 3) * t0 ** (-1 / 3) / kk
        ps0 = (-1.5 * ((2 / 3) * t0 ** (-1 / 3)) ** 2 * mu_qs_num(e0, lam, 1.0, al)
               * d0 / kk ** 2 * t0 ** (4 / 3))
        ph0 = eta_qs_num(e0, lam, 1.0, al) * ps0
        y0 = [ph0, 0.0, d0 - 3 * ph0, 0.0]
        t_eval = np.geomspace(t0, 1.0, 600)
        so = solve_ivp(rhs, (t0, 1.0), y0, args=(kk,), method="Radau",
                       rtol=rtol, atol=atol, t_eval=t_eval)
        tt = so.t
        ph, pd, j_, l_ = so.y
        a_ = tt ** (2 / 3)
        psN = np.array([F["psi"](tt[i], kk, ph[i], pd[i], j_[i], l_[i])
                        for i in range(len(tt))])
        etaN = ph / psN
        e = (2 / 3) * tt ** (-1 / 3) / kk
        w = (a_ >= window_a) & (e <= e_fit_max)
        de = etaN[w] - 1.0
        e2 = e[w] ** 2
        ok = de > 0
        expo = (float(np.polyfit(np.log(e[w][ok]), np.log(de[ok]), 1)[0])
                if ok.sum() > 10 else float("nan"))
        rows.append({"k_over_H0": float(kk),
                     "slope_eta_e2": float(np.sum(e2 * de) / np.sum(e2 * e2)),
                     "eta_minus_one_median": float(np.median(de)),
                     "eta_minus_one_min": float(de.min()),
                     "eta_minus_one_max": float(de.max()),
                     "loglog_exponent_eta": expo,
                     "fraction_negative": float(np.mean(de < 0)),
                     "solver_success": bool(so.success)})
    return rows


# --------------------------------------------------------------------
# Invariantes de gauge (E5): potenciales de Bardeen y Δ comóvil
# --------------------------------------------------------------------
# Los seis campos de la escalera son los de GAUGE UNITARIO (foliación del
# khronon). Bajo la reparametrización temporal t → t + T, con T = ε T(t)
# cos(kx), sobre el ansatz N = 1 + εψ cos, N_x = ∂_x(εb cos),
# q_ij = a²(1 − 2εφ cos)δ_ij y polvo con ℓ̄̇ = −m (δg_μν → δg_μν − L_ξ ḡ_μν):
#
#     ψ → ψ − Ṫ,   b → b + T,   φ → φ + HT,   δ → δ + 3HT,   l1 → l1 + mT.
#
# El gauge newtoniano (b = 0) se alcanza con T = −b; los invariantes son
#
#     Ψ_N = ψ + ḃ,   Φ_N = φ − H b,   Δ = δ − 3H l1/m     (m = 1).
#
# En la torre t^g Σ c_n t^{−n/3} con H = 2/(3t), tanto ḃ como H b desplazan
# el índice n → n + 3: como b arranca en n = 1, la corrección de gauge cae
# exactamente en n = 4, el MISMO orden e² de las colas de E3/E4. Los
# coeficientes de e² de η y µ_loc publicados en E3/E4 son, por tanto, los
# de los potenciales de gauge unitario, no observables.

GAUGE_TRANSFORMATION = {"psi": "ψ − Ṫ", "b": "b + T", "phi": "φ + HT",
                        "delta": "δ + 3HT", "l1": "l1 + mT",
                        "invariants": {"Psi_N": "ψ + ḃ", "Phi_N": "φ − Hb",
                                       "Delta": "δ − 3H l1 (m = 1)"}}


def gauge_invariant_towers(r: dict, nmax: int = 8) -> dict:
    """Torres de Ψ_N, Φ_N y Δ (coeficientes exactos, dependientes de k) a
    partir de la torre de gauge unitario de `ladder_point`. Con a = t^{2/3},
    H = 2/(3t): ḃ → (g − (n−3)/3)·c_{b,n−3}, H b → (2/3)·c_{b,n−3},
    3H l1 → 2·c_{l1,n−3}."""
    g = r["g"]
    tw = r["tower"]

    def c(nm, n):
        return tw[nm].get(n, sp.S(0))

    Psi = {n: sp.simplify(c("ps", n) + (g - sp.Rational(n - 3, 3)) * c("bb", n - 3))
           for n in range(0, nmax + 1)}
    Phi = {n: sp.simplify(c("ph", n) - sp.Rational(2, 3) * c("bb", n - 3))
           for n in range(0, nmax + 1)}
    Del = {n: sp.simplify(c("j0", n) + 3 * c("ph", n) - 2 * c("l1", n - 3))
           for n in range(0, nmax + 1)}
    return {"Psi_N": Psi, "Phi_N": Phi, "Delta": Del}


def gauge_invariant_coefficients(r: dict, n_identity_max: int = 4) -> dict:
    """η_N = Φ_N/Ψ_N y µ_Δ (respecto de la G local) del modo creciente, en
    la expansión T⁻² = t^{−2/3} = (9k²/4)e²: orden dominante y coeficiente
    de e². Comprueba además la identidad de torre Φ_N,n = Ψ_N,n (ausencia
    de estrés anisótropo lineal ⟹ η_N ≡ 1) para n ≤ n_identity_max — con
    prof = 12 la torre es fiable hasta n = 4; para n = 6, 8 hace falta
    prof ≥ 16 en `ladder_point`.

    µ_Δ: (k²/a²)Ψ_N = −4πG_N µ_Δ ρ Δ con G_N = G_local = G_B/(1 − α_a/2),
    κ = 16πG_B = 1, ρ = J0c/t². (La normalización respecto de G_B difiere
    en el factor (1 − α_a/2); ambas coinciden al orden dominante en α_a.)"""
    alv = r["alpha_a"]
    J0c = r["J0c"]
    gi = gauge_invariant_towers(r, nmax=max(4, n_identity_max))
    Psi, Phi, Del = gi["Psi_N"], gi["Phi_N"], gi["Delta"]
    if Psi[2] == 0:
        raise RuntimeError("Ψ_N sin cabecera en n = 2")
    T2 = 9 * K ** 2 / 4
    etaN0 = sp.simplify(Phi[2] / Psi[2])
    coef_etaN = sp.simplify((Phi[4] / Psi[2] - Phi[2] * Psi[4] / Psi[2] ** 2) * T2)
    muD0 = sp.simplify(-4 * K ** 2 * Psi[2] / (J0c * Del[0]) * (1 - alv / 2))
    coef_muD = sp.simplify(-4 * K ** 2 / J0c * (Psi[4] / Del[0] - Psi[2] * Del[2] / Del[0] ** 2)
                           * T2 * (1 - alv / 2))
    identity = {n: bool(sp.simplify(Phi[n] - Psi[n]) == 0) for n in range(0, n_identity_max + 1)}
    tw = r["tower"]
    b_head = min((n for n, v in tw["bb"].items() if v != 0), default=None)
    l1_head = min((n for n, v in tw["l1"].items() if v != 0), default=None)
    lam = r["lambda_K"]
    cs2 = (2 - alv) * (lam - 1) / (alv * (3 * lam - 1))
    closed = -alv / cs2                       # −α_a (aH/(c_s k))², orden dominante en α_a
    return {"lambda_K": _f(lam), "alpha_a": _f(alv), "cs2": _f(cs2),
            "etaN0": etaN0, "coef_etaN_e2": coef_etaN,
            "etaN0_is_1": bool(sp.simplify(etaN0 - 1) == 0),
            "coef_etaN_e2_is_zero": bool(coef_etaN == 0),
            "muD0": muD0, "coef_muD_e2": coef_muD,
            "muD0_is_1": bool(sp.simplify(muD0 - 1) == 0),
            "coef_muD_e2_closed_leading": _f(closed),
            "ratio_muD_to_closed": _f(coef_muD) / _f(closed),
            "phi_equals_psi_tower": identity,
            "b_head_n": b_head, "l1_head_n": l1_head,
            "unitary_coef_eta_e2": _f(r["coef_eta"]),
            "unitary_coef_mu_local_e2": _f(r["coef_mu_relloc"])}


def gauge_invariant_summary(gi: dict) -> dict:
    """Números serializables de `gauge_invariant_coefficients`."""
    out = dict(gi)
    for key in ("etaN0", "coef_etaN_e2", "muD0", "coef_muD_e2"):
        out[key] = _f(gi[key])
    out["phi_equals_psi_tower"] = {str(n): v for n, v in gi["phi_equals_psi_tower"].items()}
    return out


def build_gauge_invariant_rhs(S: dict, lam: float, al: float):
    """Como `build_numeric_rhs` (ξ = 1) pero devuelve además b y ḃ sobre la
    trayectoria, para formar Ψ_N = ψ + ḃ y Φ_N = φ − Hb. ḃ se obtiene por
    derivada total simbólica de la solución algebraica de b, sustituyendo
    φ̈, j̇0, l̇1 (y l̈1 si aparece) por las ecuaciones de movimiento."""
    L2 = S["L2"]
    J0cv = 4.0 / 3.0 * (3 * lam - 1)
    subs = [(sp.Derivative(ellb, t, 2), 0), (sp.Derivative(ellb, t), -m),
            (sp.Derivative(J0b, t, 2), 0), (sp.Derivative(J0b, t), 0),
            (J0b, J0cv), (kap, 1), (m, 1), (lamK, lam), (xi, 1.0), (alph, al)]
    L2n = L2
    for o, n in subs:
        L2n = L2n.subs(o, n)
    L2n = L2n.doit()
    auxeqs = [sp.expand(sp.diff(L2n, f)) for f in (psi, b, jx)]
    Msys, rvec = sp.linear_eq_to_matrix(auxeqs, [psi, b, jx])
    solvec = Msys.LUsolve(rvec)
    aux = {psi: solvec[0], b: solvec[1], jx: solvec[2]}
    L2red = L2n.subs(aux)
    Es = [(sp.diff(L2red, f) - sp.diff(sp.diff(L2red, f.diff(t)), t)).doit()
          for f in (phi, j0, l1)]
    phidd, j0d, l1d, phid, l1dd = sp.symbols("phidd j0d l1d phid l1dd")
    rep = [(sp.Derivative(phi, t, 2), phidd), (sp.Derivative(j0, t), j0d),
           (sp.Derivative(l1, t, 2), l1dd), (sp.Derivative(l1, t), l1d),
           (sp.Derivative(phi, t), phid)]
    Es = [e.subs(rep) for e in Es]
    M2, r2 = sp.linear_eq_to_matrix(Es, [phidd, j0d, l1d])
    sol2 = M2.LUsolve(r2)
    # ecuaciones de movimiento como expresiones en (t, φ, φ̇, j0, l1)
    back = {phid: sp.Derivative(phi, t)}
    eom = {sp.Derivative(phi, t, 2): sol2[0].subs(back),
           sp.Derivative(j0, t): sol2[1].subs(back),
           sp.Derivative(l1, t): sol2[2].subs(back)}
    l1dd_expr = sp.diff(eom[sp.Derivative(l1, t)], t)
    for dv in (sp.Derivative(phi, t, 2), sp.Derivative(j0, t), sp.Derivative(l1, t)):
        l1dd_expr = l1dd_expr.subs(dv, eom[dv])
    bdot = sp.diff(aux[b], t)
    for dv, ex in [(sp.Derivative(l1, t, 2), l1dd_expr), (sp.Derivative(phi, t, 2), eom[sp.Derivative(phi, t, 2)]),
                   (sp.Derivative(j0, t), eom[sp.Derivative(j0, t)]),
                   (sp.Derivative(l1, t), eom[sp.Derivative(l1, t)])]:
        bdot = bdot.subs(dv, ex)
    bg = [(sp.Derivative(a, t, 2), -sp.Rational(2, 9) * t ** sp.Rational(-4, 3)),
          (sp.Derivative(a, t), sp.Rational(2, 3) * t ** sp.Rational(-1, 3)),
          (a, t ** sp.Rational(2, 3))]
    PHI, J0S, L1S = sp.symbols("PHI J0S L1S")
    base = [(phi, PHI), (j0, J0S), (l1, L1S)]
    F = {}
    for nm, expr in [("ph", sol2[0]), ("j0", sol2[1]), ("l1", sol2[2]),
                     ("psi", aux[psi].subs(rep)), ("b", aux[b].subs(rep)),
                     ("bdot", bdot.subs(rep))]:
        e = expr.subs(bg).subs(base)
        free = {str(s) for s in e.free_symbols} - {"t", "k", "PHI", "phid", "J0S", "L1S"}
        if free:
            raise RuntimeError(f"símbolos no resueltos en {nm}: {free}")
        F[nm] = sp.lambdify((t, K, PHI, phid, J0S, L1S), e, "numpy")
    return F, J0cv


def _fit_e2_e4(e2, dx):
    A4 = np.vstack([e2, e2 ** 2]).T
    (s4, q4), *_ = np.linalg.lstsq(A4, dx, rcond=None)
    return float(s4), float(q4)


def run_adiabatic_gauge_invariant(S: dict, r: dict, k_values, a_start: float = 0.1,
                                  a_fit_min: float = 0.0, e_fit_max: float = 0.02,
                                  rtol: float = 1e-10, atol: float = 1e-18,
                                  n_eval: int = 600) -> dict:
    """Arnés E5b: integra el sistema completo (ξ = 1) desde ICs adiabáticas
    y ajusta X − 1 = s·e² + q·e⁴ para X ∈ {η_N, µ_Δ} (invariantes de gauge)
    y, como control, para los de gauge unitario {η, µ_loc}. La ventana de
    ajuste es a ≥ a_fit_min y e ≤ e_fit_max (a_fit_min > 0 descarta el
    transitorio de las ICs truncadas en n ≤ 8)."""
    from scipy.integrate import solve_ivp

    lam = _f(r["lambda_K"])
    al = _f(r["alpha_a"])
    F, J0cv = build_gauge_invariant_rhs(S, lam, al)
    GNn = 1.0 / (16 * np.pi)

    def rhs(tt, y, kk):
        ph, pd, j_, l_ = y
        return [pd, F["ph"](tt, kk, ph, pd, j_, l_),
                F["j0"](tt, kk, ph, pd, j_, l_),
                F["l1"](tt, kk, ph, pd, j_, l_)]

    t0 = a_start ** 1.5
    rows = []
    for kk in k_values:
        y0 = adiabatic_initial_state(r, kk, t0)
        t_eval = np.geomspace(t0, 1.0, n_eval)
        so = solve_ivp(rhs, (t0, 1.0), y0, args=(kk,), method="Radau",
                       rtol=rtol, atol=atol, t_eval=t_eval)
        tt = so.t
        ph, pd, j_, l_ = so.y
        a_ = tt ** (2 / 3)
        H_ = 2.0 / (3.0 * tt)
        ev = lambda nm: np.array([F[nm](tt[i], kk, ph[i], pd[i], j_[i], l_[i])  # noqa: E731
                                  for i in range(len(tt))])
        psN, bN, bdN = ev("psi"), ev("b"), ev("bdot")
        Psi_N = psN + bdN
        Phi_N = ph - H_ * bN
        dN = j_ + 3 * ph
        Delta = dN - 3 * H_ * l_
        rho = J0cv / tt ** 2
        eta_u = ph / psN
        mu_u = (kk ** 2 / a_ ** 2) * psN / (-4 * np.pi * GNn * rho * dN) * (1 - al / 2)
        eta_N = Phi_N / Psi_N
        mu_D = (kk ** 2 / a_ ** 2) * Psi_N / (-4 * np.pi * GNn * rho * Delta) * (1 - al / 2)
        e = (2 / 3) * tt ** (-1 / 3) / kk
        w = (e <= e_fit_max) & (a_ >= a_fit_min)
        e2 = e[w] ** 2
        s_etaN, q_etaN = _fit_e2_e4(e2, eta_N[w] - 1.0)
        s_muD, q_muD = _fit_e2_e4(e2, mu_D[w] - 1.0)
        s_eta_u, q_eta_u = _fit_e2_e4(e2, eta_u[w] - 1.0)
        s_mu_u, q_mu_u = _fit_e2_e4(e2, mu_u[w] - 1.0)
        lnd, lna = np.log(np.abs(Delta[w])), np.log(a_[w])
        p_loc = np.gradient(lnd, lna)
        rows.append({"k_over_H0": float(kk),
                     "e_range_fit": [float(e[w].min()), float(e[w].max())],
                     "a_range_fit": [float(a_[w].min()), float(a_[w].max())],
                     "n_points_fit": int(w.sum()),
                     "slope_etaN_e2": s_etaN, "coef_etaN_e4": q_etaN,
                     "slope_muD_e2": s_muD, "coef_muD_e4": q_muD,
                     "etaN_minus_one_max_abs": float(np.max(np.abs(eta_N[w] - 1.0))),
                     "muD_minus_one_at_e_max": float((mu_D[w] - 1.0)[np.argmax(e2)]),
                     "control_unitary": {"slope_eta_e2": s_eta_u, "coef_eta_e4": q_eta_u,
                                         "slope_mu_local_e2": s_mu_u, "coef_mu_local_e4": q_mu_u},
                     "p_num_median": float(np.median(p_loc)),
                     "p_spread_std": float(np.std(p_loc)),
                     "solver_success": bool(so.success)})
    return {"rows": rows, "a_start": a_start, "a_fit_min": a_fit_min,
            "e_fit_max": e_fit_max, "p_qs": _f(r["p"])}


if __name__ == "__main__":     # desarrollo: un punto cronometrado
    import pickle
    import sys
    T0 = time.time()
    S = pickle.load(open(sys.argv[1], "rb"))
    r = ladder_point(S, sp.Rational(105, 100), sp.Rational(3, 10))
    print(f"{time.time()-T0:6.1f}s", ladder_summary(r))
