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


if __name__ == "__main__":     # desarrollo: un punto cronometrado
    import pickle
    import sys
    T0 = time.time()
    S = pickle.load(open(sys.argv[1], "rb"))
    r = ladder_point(S, sp.Rational(105, 100), sp.Rational(3, 10))
    print(f"{time.time()-T0:6.1f}s", ladder_summary(r))
