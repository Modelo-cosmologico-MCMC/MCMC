"""Frente 3 — canal Atlas de µ/η RE-DERIVADO desde la Acción de Gea (9.1)
+ Término de Cronos (9.4), y arnés numérico del sistema lineal completo.

Este módulo NO es ruta de producción: re-deriva simbólicamente (sympy)
los resultados que cosmology/mu_eta_atlas.py fija como formas cerradas,
para que el candado E1 compare fórmula fijada contra derivación
independiente (patrón #14), y ejecuta la integración numérica sin
aproximación cuasi-estática (E2). Requiere sympy (extra opcional
`[derivation]`); la suite no lo importa.

TEORÍA (tratado v35, gauge unitario = la foliación entrópica es física,
c = 1; materia = polvo por la acción de Sorkin–Schutz reducida):

    L_g = (1/κ) N√q [ K_ij K^ij − λ_K K² + ξ R³[q] + α_a a_i a^i ],
    κ = 16πG_B,  a_i = ∂_i ln N
    L_m = −m √(−g_μν J^μ J^ν) − J^μ ∂_μ ℓ

Perturbaciones escalares (un modo de Fourier, onda real cos(kx),
promedio espacial):
    N = 1 + ε ψ(t) cos(kx);  N_x = ∂_x[ε b(t) cos(kx)];
    q_ij = a²(1 − 2ε φ(t) cos(kx)) δ_ij;
    J⁰ = J̄⁰(t)(1 + ε j0(t) cos(kx));  J^x = ε jx(t) sin(kx);
    ℓ = ℓ̄(t) + ε l1(t) cos(kx).
Contraste del polvo: δ = j0 + 3φ (comprobado simbólicamente).

Etapas: (1) acción cuadrática promediada; (2) fondo → G_cosmo del Sello
(9.3); (3) límite QS sub-horizonte → µ_Atlas(e), η_Atlas(e), e = aH/k;
(4) G estática (laboratorio); (5) khronon en Minkowski → coef. cinético
y c_s²; (6) sector tensorial → c_T²; (7) integración numérica completa.

Método de (7): eliminación algebraica de los campos no dinámicos
(ψ, b, jx), EL de (φ, j0, l1), fondo a = t^{2/3} (κ = m = 1, J̄⁰ tal
que H² = κmJ̄⁰/(3(3λ_K−1)a³)), ICs QS-consistentes en a_start = 1e-3 y
RELAJACIÓN: los modos decrecientes caen como potencias de a, así que en
la ventana tardía sobrevive el modo creciente puro; la pureza se mide
por la constancia del índice p = d ln|δ|/d ln a (p_QS analítico:
p(p+½) = (3/2)(3λ_K−1)/(2ξ−α_a)). µ_num y η_num se miden en esa
ventana. El D/D₀ del arnés no es físico (amplitud inicial repartida
entre modos) y no se publica.
"""

from __future__ import annotations

import time

import numpy as np
import sympy as sp

# --------------------------------------------------------------------
# Símbolos compartidos
# --------------------------------------------------------------------
t, x, k, eps = sp.symbols("t x k epsilon", positive=True)
lamK, xi, alph = sp.symbols("lambda_K xi alpha_a", positive=True)
GB, m = sp.symbols("G_B m", positive=True)
kap = sp.Symbol("kappa", positive=True)          # κ = 16πG_B
Hs = sp.Symbol("H", positive=True)               # Hubble instantáneo
J0c = sp.Symbol("J0c", positive=True)            # J̄⁰ constante
eps_h = sp.Symbol("epsilon_H", positive=True)    # e = aH/k
delta = sp.Symbol("delta")

a = sp.Function("a", positive=True)(t)
psi = sp.Function("psi")(t)
b = sp.Function("b")(t)
phi = sp.Function("phi")(t)
j0 = sp.Function("j0")(t)
jx = sp.Function("jx")(t)
l1 = sp.Function("l1")(t)
J0b = sp.Function("J0b", positive=True)(t)
ellb = sp.Function("ellb")(t)

_COORDS = [x, sp.Symbol("y"), sp.Symbol("z")]


def _d3(f, i):
    return sp.diff(f, _COORDS[i]) if i == 0 else sp.S(0)


def _christoffel(q, qinv):
    G = [[[sp.S(0)] * 3 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for kk in range(3):
                e = sp.S(0)
                for ll in range(3):
                    e += qinv[i, ll] * (_d3(q[ll, j], kk) + _d3(q[ll, kk], j)
                                        - _d3(q[j, kk], ll))
                G[i][j][kk] = sp.together(e / 2)
    return G


def _ricci_scalar(q, qinv, G):
    Ric = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            e = sp.S(0)
            for kk in range(3):
                e += _d3(G[kk][i][j], kk) - _d3(G[kk][i][kk], j)
                for ll in range(3):
                    e += (G[kk][kk][ll] * G[ll][i][j]
                          - G[kk][j][ll] * G[ll][i][kk])
            Ric[i, j] = e
    return sum(qinv[i, j] * Ric[i, j] for i in range(3) for j in range(3))


def _gravity_lagrangian(N, Nvec, q):
    """N√q [K_ijK^ij − λK² + ξR³ + α a_ia^i]/κ para lapse, shift (índice
    abajo) y métrica espacial dados."""
    qinv = q.inv()
    sqrtq = sp.sqrt(q.det())
    G = _christoffel(q, qinv)
    R3 = _ricci_scalar(q, qinv, G)

    def DN(i, j):
        e = _d3(Nvec[j], i)
        for kk in range(3):
            e -= G[kk][i][j] * Nvec[kk]
        return e

    Kt = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            Kt[i, j] = (sp.diff(q[i, j], t) - DN(i, j) - DN(j, i)) / (2 * N)
    Kmix = qinv * Kt
    trK = sp.trace(Kmix)
    KK = sum((Kmix * Kmix)[i, i] for i in range(3))
    ai2 = sum(qinv[i, j] * _d3(sp.log(N), i) * _d3(sp.log(N), j)
              for i in range(3) for j in range(3))
    return N * sqrtq * (KK - lamK * trK ** 2 + xi * R3 + alph * ai2) / kap, qinv


def _xavg(expr):
    """Promedio espacial sobre un período del modo."""
    expr = expr.rewrite(sp.cos).expand(trig=True)
    return sp.integrate(expr, (x, 0, 2 * sp.pi / k)) * k / (2 * sp.pi)


def build_quadratic_action() -> dict:
    """Etapa 1: L0, L1, L2 promediadas (fondo, lineal, cuadrática)."""
    c_ = sp.cos(k * x)
    s_ = sp.sin(k * x)
    N = 1 + eps * psi * c_
    Nx = sp.diff(eps * b * c_, x)
    Nvec = [Nx, sp.S(0), sp.S(0)]
    qd = a ** 2 * (1 - 2 * eps * phi * c_)
    q = sp.diag(qd, qd, qd)
    L_grav, qinv = _gravity_lagrangian(N, Nvec, q)

    Nup = [sum(qinv[i, j] * Nvec[j] for j in range(3)) for i in range(3)]
    g = sp.zeros(4, 4)
    g[0, 0] = -(N ** 2 - sum(Nvec[i] * Nup[i] for i in range(3)))
    for i in range(3):
        g[0, i + 1] = Nvec[i]
        g[i + 1, 0] = Nvec[i]
        for j in range(3):
            g[i + 1, j + 1] = q[i, j]
    J = [J0b * (1 + eps * j0 * c_), eps * jx * s_, sp.S(0), sp.S(0)]
    JJ = sum(g[mu, nu] * J[mu] * J[nu] for mu in range(4) for nu in range(4))
    ell = ellb + eps * l1 * c_
    dell = [sp.diff(ell, t), sp.diff(ell, x), sp.S(0), sp.S(0)]
    L_mat = -m * sp.sqrt(-JJ) - sum(J[mu] * dell[mu] for mu in range(4))

    L = L_grav + L_mat
    Lser = sp.series(L, eps, 0, 3).removeO().expand()
    return {"L0": _xavg(Lser.coeff(eps, 0)),
            "L1": _xavg(Lser.coeff(eps, 1)),
            "L2": _xavg(Lser.coeff(eps, 2))}


def _EL(Lag, f):
    return sp.expand(sp.diff(Lag, f) - sp.diff(sp.diff(Lag, f.diff(t)), t))


def stage_background(S: dict) -> dict:
    """Etapa 2: EL del fondo; la primera integral de la ecuación de a con
    polvo debe ser H² = κmJ̄⁰/(3(3λ_K−1)a³), i.e. G_cosmo = 2G_B/(3λ_K−1)
    (Sello de Newton 9.3)."""
    L0 = S["L0"]
    bg_a = _EL(L0, a)
    bg_J0 = _EL(L0, J0b)
    bg_ell = _EL(L0, ellb)
    # Con ℓ̄' = −m (de bg_J0), J̄⁰ = J0c, la ecuación de a (segundo orden)
    # debe anularse en la solución de materia dominante con la constante
    # de Friedmann del Sello: a' = aH, a'' = −aH²/2, H² = κmJ0c/(3(3λ−1)a³).
    Hsq = kap * m * J0c / (3 * (3 * lamK - 1) * a ** 3)
    e = bg_a.subs({sp.Derivative(ellb, t): -m, J0b: J0c})
    e = e.subs(sp.Derivative(a, t, 2), -a * Hsq / 2).subs(
        sp.Derivative(a, t), a * sp.sqrt(Hsq))
    sello_ok = sp.simplify(e) == 0
    return {"bg_a": str(bg_a), "bg_J0": str(bg_J0), "bg_ell": str(bg_ell),
            "G_cosmo_over_GB": "2/(3*lambda_K - 1)",
            "sello_newton_reproduced": bool(sello_ok)}


def _linear_equations_on_background(S: dict) -> dict:
    """EL de L2 para los seis campos, con el fondo de polvo sustituido."""
    L2 = S["L2"]
    fields = {"psi": psi, "b": b, "phi": phi, "j0": j0, "jx": jx, "l1": l1}
    eqs = {nm: _EL(L2, f) for nm, f in fields.items()}
    sub_bg = [(sp.Derivative(ellb, t, 2), 0), (sp.Derivative(ellb, t), -m),
              (sp.Derivative(J0b, t, 2), 0), (sp.Derivative(J0b, t), 0),
              (J0b, J0c),
              (sp.Derivative(a, t, 2), -a * Hs ** 2 / sp.S(2)),
              (sp.Derivative(a, t), a * Hs)]
    out = {}
    for nm, e in eqs.items():
        e = e.subs(sub_bg[:5]).doit()
        for old, new in sub_bg[5:]:
            e = e.subs(old, new)
        out[nm] = sp.expand(e)
    return out


def stage_quasi_static(S: dict) -> dict:
    """Etapa 3: contraste δ = j0 + 3φ; límite QS (sin ∂_t de los
    potenciales, jx despreciada); µ y η exactas en e = aH/k con la
    normalización G_N ≡ G_B/ξ y Poisson GR (k²/a²)ψ = −4πG_N ρ δ."""
    eqs = _linear_equations_on_background(S)
    # contraste de densidad del polvo
    Nl = 1 + eps * psi * sp.cos(k * x)
    detq = (a ** 2 * (1 - 2 * eps * phi * sp.cos(k * x))) ** 3
    Jt = J0b * (1 + eps * j0 * sp.cos(k * x))
    nfull = Nl * Jt / (Nl * sp.sqrt(detq))
    d_expr = sp.series(nfull / (J0b / a ** 3), eps, 0, 2).removeO().coeff(eps, 1)
    d_expr = sp.simplify(d_expr.subs(sp.cos(k * x), 1))
    delta_ok = sp.simplify(d_expr - (j0 + 3 * phi)) == 0

    fried = sp.Eq(kap * m * J0c, 3 * (3 * lamK - 1) * a ** 3 * Hs ** 2)
    solJ = sp.solve(fried, J0c)[0]
    qs_kill = []
    for f in (psi, b, phi):
        qs_kill += [(sp.Derivative(f, t, 2), 0), (sp.Derivative(f, t), 0)]
    qs_kill += [(sp.Derivative(jx, t), 0), (jx, 0), (sp.Derivative(l1, t, 2), 0)]
    grav = {}
    for nm in ("psi", "b", "phi"):
        e = eqs[nm]
        for old, new in qs_kill:
            e = e.subs(old, new)
        e = e.subs(j0, delta - 3 * phi).subs(J0c, solJ)
        grav[nm] = sp.expand(sp.simplify(e))
    sol = sp.solve([grav["psi"], grav["b"], grav["phi"]], [psi, phi, b],
                   dict=True)
    if len(sol) != 1:
        raise RuntimeError(f"sistema QS con {len(sol)} soluciones")
    s = sol[0]
    rhob = m * J0c / a ** 3
    GN = GB / xi
    poisson_rhs = -4 * sp.pi * GN * rhob * delta
    mu = sp.simplify((k ** 2 / a ** 2 * s[psi] / poisson_rhs)
                     .subs(kap, 16 * sp.pi * GB))
    eta = sp.simplify(s[phi] / s[psi])
    fried_sub = [(J0c, 3 * (3 * lamK - 1) * a ** 3 * Hs ** 2 / (kap * m)),
                 (kap, 16 * sp.pi * GB)]
    mu_e = sp.simplify(mu.subs(fried_sub).subs(Hs, eps_h * k / a))
    eta_e = sp.simplify(eta.subs(fried_sub).subs(Hs, eps_h * k / a))
    mu_sub = sp.simplify(sp.limit(mu_e, eps_h, 0))
    eta_sub = sp.simplify(sp.limit(eta_e, eps_h, 0))
    # formas cerradas fijadas en cosmology/mu_eta_atlas.py
    mu_closed = 2 * xi / ((2 * xi - alph) + 3 * (3 * lamK - 1) * eps_h ** 2)
    eta_closed = -(9 * eps_h ** 2 * lamK ** 2 - 9 * eps_h ** 2 * lamK * xi
                   - 6 * eps_h ** 2 * lamK + 3 * eps_h ** 2 * xi + eps_h ** 2
                   + 2 * lamK * xi - 2 * xi) / (
        xi * (9 * eps_h ** 2 * lamK - 3 * eps_h ** 2 - 2 * lamK + 2))
    gr = {lamK: 1, xi: 1, alph: 0}
    return {
        "delta_is_j0_plus_3phi": bool(delta_ok),
        "mu_qs": str(sp.factor(mu_e)), "eta_qs": str(sp.factor(eta_e)),
        "mu_subhorizon": str(mu_sub), "eta_subhorizon": str(eta_sub),
        "mu_matches_closed_form": bool(sp.simplify(mu_e - mu_closed) == 0),
        "eta_matches_closed_form": bool(sp.simplify(eta_e - eta_closed) == 0),
        "mu_sub_is_1_over_1_minus_alpha_over_2xi": bool(
            sp.simplify(mu_sub - 1 / (1 - alph / (2 * xi))) == 0),
        "eta_sub_is_1": bool(sp.simplify(eta_sub - 1) == 0),
        "gr_limit_mu_eta_1": [bool(sp.simplify(mu_sub.subs(gr) - 1) == 0),
                              bool(sp.simplify(eta_sub.subs(gr) - 1) == 0)],
        "eta_qs_at_lamK_1_finite_e": str(sp.simplify(eta_e.subs(gr))),
        "_mu_e": mu_e, "_eta_e": eta_e, "_grav": grav, "_eqs": eqs,
    }


def stage_static_G(qs: dict) -> dict:
    """Etapa 4: la G efectiva de crecimiento (e→0) es G_B/(ξ − α_a/2); la
    G estática de laboratorio (Cavendish) debe coincidir: cancelación.

    La G estática se calcula de forma INDEPENDIENTE del límite e→0: las
    ecuaciones lineales en Minkowski (a = 1, H = 0) con una fuente de
    polvo ESTÁTICA de densidad ρ̄ = mJ̄⁰ libre (sin sustituir Friedmann —
    en un fondo en expansión ρ̄ ∝ H² y H → 0 apagaría la fuente), todas
    las derivadas temporales nulas y jx = 0: Poisson modificada
    k²ψ = −4πG_local ρ̄ δ."""
    mu_sub = sp.simplify(sp.limit(qs["_mu_e"], eps_h, 0))
    G_growth = sp.simplify(mu_sub * GB / xi)
    eqs = qs["_eqs"]            # fondo sustituido, Friedmann NO sustituida
    static_kill = [(Hs, 0), (a, 1)]
    for f in (psi, b, phi, l1):
        static_kill += [(sp.Derivative(f, t, 2), 0), (sp.Derivative(f, t), 0)]
    static_kill += [(sp.Derivative(jx, t), 0), (jx, 0), (j0, delta - 3 * phi)]
    stat = {}
    for nm in ("psi", "b", "phi"):
        e = eqs[nm]
        for old, new in static_kill:
            e = e.subs(old, new)
        stat[nm] = sp.expand(e.doit())
    sol0 = sp.solve([stat["psi"], stat["b"], stat["phi"]], [psi, phi, b],
                    dict=True)
    if len(sol0) != 1:
        raise RuntimeError(f"sistema estático con {len(sol0)} soluciones")
    rhob = m * J0c                       # a = 1
    mu_static_full = sp.simplify((k ** 2 * sol0[0][psi]
                                  / (-4 * sp.pi * (GB / xi) * rhob * delta))
                                 .subs(kap, 16 * sp.pi * GB))
    # Un fondo Minkowski con ρ̄ ≠ 0 uniforme NO es solución (Jeans
    # swindle): deja un término residual ∝ κρ̄, independiente de k, en el
    # denominador. La G de laboratorio es el límite de fondo vacío,
    # ρ̄ → 0 a δρ = ρ̄δ fijo (equivalente a k → ∞); ambos se registran.
    mu_static = sp.simplify(sp.limit(mu_static_full, J0c, 0))
    mu_static_kinf = sp.simplify(sp.limit(mu_static_full, k, sp.oo))
    jeans_residual = sp.simplify(1 / mu_static_full - 1 / mu_static)
    G_local = sp.simplify(mu_static * GB / xi)
    return {"G_growth_over_GB": str(sp.simplify(G_growth / GB)),
            "G_local_over_GB": str(sp.simplify(G_local / GB)),
            "G_local_limits_agree": bool(
                sp.simplify(mu_static - mu_static_kinf) == 0),
            "jeans_residual_in_1_over_mu": str(jeans_residual),
            "G_growth_equals_G_local": bool(
                sp.simplify(G_growth - G_local) == 0),
            "G_growth_is_GB_over_xi_minus_alpha_half": bool(
                sp.simplify(G_growth - GB / (xi - alph / 2)) == 0),
            "G_cosmo_over_G_local": str(sp.simplify(
                (2 * GB / (3 * lamK - 1)) / G_local))}


def stage_khronon(S: dict) -> dict:
    """Etapa 5: L2 en Minkowski sin materia; eliminados ψ y b (no
    dinámicos) queda A φ'² − B φ² ⟹ c_s² = B/(A k²), no-fantasma ⟺ A>0."""
    L2 = S["L2"]
    mink = [(j0, 0), (jx, 0), (l1, 0), (sp.Derivative(l1, t), 0),
            (sp.Derivative(J0b, t), 0), (J0b, 0),
            (sp.Derivative(ellb, t), -m),
            (sp.Derivative(a, t, 2), 0), (sp.Derivative(a, t), 0), (a, 1)]
    L2m = L2
    for old, new in mink:
        L2m = L2m.subs(old, new)
    L2m = sp.expand(L2m.doit())
    ELpsi = _EL(L2m, psi)
    ELb = _EL(L2m, b)
    solc = sp.solve([ELpsi, ELb], [psi, b], dict=True)[0]
    L2red = sp.expand(sp.together(
        sp.expand(L2m.subs({psi: solc[psi], b: solc[b]}).doit())))
    # coeficientes como polinomio en (φ', φ): la forma debe ser A φ'² − B φ²
    phid = sp.Symbol("phid_sym")
    phs = sp.Symbol("phi_sym")
    poly = sp.Poly(sp.expand(L2red.subs(phi.diff(t), phid).subs(phi, phs)),
                   phid, phs)
    A = sp.simplify(poly.coeff_monomial(phid ** 2))
    B = sp.simplify(-poly.coeff_monomial(phs ** 2))
    other = [str(mn) for mn in poly.monoms()
             if mn not in ((2, 0), (0, 2))]
    cs2 = sp.simplify(B / (A * k ** 2))
    cs2_closed = xi * (2 * xi - alph) * (lamK - 1) / (alph * (3 * lamK - 1))
    return {"kinetic_coefficient": str(sp.factor(A)),
            "cs2": str(sp.factor(cs2)),
            "cs2_matches_closed_form": bool(sp.simplify(cs2 - cs2_closed) == 0),
            "kinetic_sign_factor": str(sp.factor(
                sp.simplify(A * kap * (lamK - 1) / (3 * lamK - 1)))),
            "unexpected_monomials": other}


def stage_tensor() -> dict:
    """Etapa 6: modo tensorial TT (k ∥ x, polarización +):
    q_ij = a²(δ_ij + ε h(t) cos(kx) e_ij), e = diag(0, 1, −1); N = 1,
    N_i = 0, sin materia ⟹ L2 ∝ ḣ² − ξ k² h²/a² ⟹ c_T² = ξ."""
    h = sp.Function("h")(t)
    c_ = sp.cos(k * x)
    q = sp.diag(a ** 2, a ** 2 * (1 + eps * h * c_), a ** 2 * (1 - eps * h * c_))
    L_grav, _ = _gravity_lagrangian(sp.S(1), [sp.S(0)] * 3, q)
    Lser = sp.series(L_grav, eps, 0, 3).removeO().expand()
    L2 = sp.expand(_xavg(Lser.coeff(eps, 2)))
    hd = sp.Symbol("hd_sym")
    hs = sp.Symbol("h_sym")
    poly = sp.Poly(sp.expand(L2.subs(h.diff(t), hd).subs(h, hs)), hd, hs)
    A = sp.simplify(poly.coeff_monomial(hd ** 2))
    B_full = sp.simplify(-poly.coeff_monomial(hs ** 2))
    # B_full = (ξ k²/a²)·A + términos ∝ H² del fondo (masa efectiva de
    # Hubble, no propagación): la velocidad es el coeficiente de k².
    B_k2 = sp.simplify(sp.diff(B_full, k) / (2 * k))
    cT2 = sp.simplify(B_k2 / A * a ** 2)
    # comprobación cruzada: en Minkowski (a' = 0) B_full/A·a²/k² = ξ
    mink = [(sp.Derivative(a, t, 2), 0), (sp.Derivative(a, t), 0)]
    cT2_mink = sp.simplify((B_full / (A * k ** 2) * a ** 2).subs(mink))
    return {"kinetic_coefficient": str(sp.factor(A)),
            "cT2": str(sp.simplify(cT2)),
            "cT2_minkowski": str(cT2_mink),
            "hubble_mass_terms": str(sp.factor(sp.simplify(
                B_full - B_k2 * k ** 2))),
            "cT2_is_xi": bool(sp.simplify(cT2 - xi) == 0
                              and sp.simplify(cT2_mink - xi) == 0)}


# --------------------------------------------------------------------
# Etapa 7: integración numérica del sistema lineal completo
# --------------------------------------------------------------------

def growth_index_qs(lam: float, xi_: float, al: float) -> float:
    """p_QS: δ ∝ a^p en materia dominante con G_growth/G_cosmo =
    (3λ−1)/(2ξ−α): p(p+½) = (3/2)(3λ−1)/(2ξ−α)."""
    g = (3 * lam - 1) / (2 * xi_ - al)
    return (-0.5 + np.sqrt(0.25 + 6.0 * g)) / 2.0


def mu_qs_num(e, lam, xi_, al):
    return 2 * xi_ / (2 * xi_ - al + 3 * (3 * lam - 1) * e ** 2)


def eta_qs_num(e, lam, xi_, al):
    e2 = e * e
    return -(9 * e2 * lam ** 2 - 9 * e2 * lam * xi_ - 6 * e2 * lam + 3 * e2 * xi_
             + e2 + 2 * lam * xi_ - 2 * xi_) / (
        xi_ * (9 * e2 * lam - 3 * e2 - 2 * lam + 2))


def build_numeric_rhs(S: dict, lam: float, xi_: float, al: float):
    """Elimina (ψ, b, jx) algebraicamente, obtiene EL de (φ, j0, l1),
    resuelve las derivadas altas y lambdifica sobre a = t^{2/3}."""
    L2 = S["L2"]
    J0cv = 4.0 / 3.0 * (3 * lam - 1)
    subs = [(sp.Derivative(ellb, t, 2), 0), (sp.Derivative(ellb, t), -m),
            (sp.Derivative(J0b, t, 2), 0), (sp.Derivative(J0b, t), 0),
            (J0b, J0cv), (kap, 1), (m, 1), (lamK, lam), (xi, xi_), (alph, al)]
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
    phidd, j0d, l1d, phid = sp.symbols("phidd j0d l1d phid")
    rep = [(sp.Derivative(phi, t, 2), phidd), (sp.Derivative(j0, t), j0d),
           (sp.Derivative(l1, t, 2), sp.Symbol("l1dd")),
           (sp.Derivative(l1, t), l1d), (sp.Derivative(phi, t), phid)]
    Es = [e.subs(rep) for e in Es]
    M2, r2 = sp.linear_eq_to_matrix(Es, [phidd, j0d, l1d])
    sol2 = M2.LUsolve(r2)
    bg = [(sp.Derivative(a, t, 2), -sp.Rational(2, 9) * t ** sp.Rational(-4, 3)),
          (sp.Derivative(a, t), sp.Rational(2, 3) * t ** sp.Rational(-1, 3)),
          (a, t ** sp.Rational(2, 3))]
    PHI, J0S, L1S = sp.symbols("PHI J0S L1S")
    base = [(phi, PHI), (j0, J0S), (l1, L1S)]
    F = {}
    for nm, expr in [("ph", sol2[0]), ("j0", sol2[1]), ("l1", sol2[2]),
                     ("psi", aux[psi].subs(rep))]:
        e = expr.subs(bg).subs(base)
        F[nm] = sp.lambdify((t, k, PHI, phid, J0S, L1S), e, "numpy")
    return F, J0cv


def run_numeric(S: dict, lam: float, xi_: float, al: float, k_values,
                a_start: float = 1e-3, window=(0.3, 1.0),
                rtol: float = 1e-9, atol: float = 1e-15) -> list[dict]:
    """Integra el sistema completo desde a_start con ICs QS-consistentes
    y mide, en la ventana tardía donde el modo creciente domina, la
    mediana de µ_num/µ_QS(e), de η_num, y el índice p con su dispersión
    (pureza de modo)."""
    from scipy.integrate import solve_ivp

    F, J0cv = build_numeric_rhs(S, lam, xi_, al)
    GNn = 1.0 / (16 * np.pi) / xi_          # G_N ≡ G_B/ξ, κ = 1

    def rhs(tt, y, kk):
        ph, pd, j_, l_ = y
        return [pd, F["ph"](tt, kk, ph, pd, j_, l_),
                F["j0"](tt, kk, ph, pd, j_, l_),
                F["l1"](tt, kk, ph, pd, j_, l_)]

    p_qs = growth_index_qs(lam, xi_, al)
    out = []
    t0 = a_start ** 1.5
    for kk in k_values:
        d0 = 1e-3
        e0 = (2 / 3) * t0 ** (-1 / 3) / kk
        ps0 = (-1.5 * ((2 / 3) * t0 ** (-1 / 3)) ** 2 * mu_qs_num(e0, lam, xi_, al)
               * d0 / kk ** 2 * t0 ** (4 / 3))
        ph0 = eta_qs_num(e0, lam, xi_, al) * ps0
        y0 = [ph0, 0.0, d0 - 3 * ph0, 0.0]
        t_eval = np.geomspace(t0, 1.0, 400)
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
        etaN = ph / psN
        eH = (2 / 3) * tt ** (-1 / 3) / kk
        w = (a_ >= window[0]) & (a_ <= window[1])
        lnd = np.log(np.abs(dN[w]))
        lna = np.log(a_[w])
        p_loc = np.gradient(lnd, lna)
        out.append({
            "k_over_H0": float(kk),
            "e_end": float(eH[-1]),
            "mu_num_median": float(np.median(muN[w])),
            "mu_qs_at_window": float(np.median(mu_qs_num(eH[w], lam, xi_, al))),
            "mu_ratio_minus_one": float(np.median(muN[w])
                                        / np.median(mu_qs_num(eH[w], lam, xi_, al))
                                        - 1.0),
            "eta_num_median": float(np.median(etaN[w])),
            "eta_num_minus_one": float(np.median(etaN[w]) - 1.0),
            "eta_qs_tail_at_window": float(np.median(eta_qs_num(eH[w], lam, xi_, al))
                                           - 1.0),
            "p_num": float(np.median(p_loc)),
            "p_qs": float(p_qs),
            "p_rel_err": float(np.median(p_loc) / p_qs - 1.0),
            "p_spread_std": float(np.std(p_loc)),
            "delta_sign_flips_in_window": int(np.sum(np.diff(np.sign(dN[w])) != 0)),
            "solver_success": bool(so.success),
        })
    return out


def run_all_symbolic(verbose: bool = True) -> dict:
    """Etapas 1–6 con cronometraje; devuelve resultados serializables y
    la acción cuadrática (para el arnés)."""
    T0 = time.time()

    def tick(msg):
        if verbose:
            print(f"{time.time() - T0:7.1f}s {msg}", flush=True)

    S = build_quadratic_action(); tick("etapa 1: acción cuadrática")
    bg = stage_background(S); tick("etapa 2: fondo / Sello de Newton")
    qs = stage_quasi_static(S); tick("etapa 3: límite QS µ, η")
    st = stage_static_G(qs); tick("etapa 4: G estática")
    kh = stage_khronon(S); tick("etapa 5: khronon c_s², cinético")
    tn = stage_tensor(); tick("etapa 6: tensores c_T²")
    qs_public = {kk: v for kk, v in qs.items() if not kk.startswith("_")}
    return {"S": S, "background": bg, "quasi_static": qs_public,
            "static_G": st, "khronon": kh, "tensor": tn,
            "seconds": round(time.time() - T0, 1)}
