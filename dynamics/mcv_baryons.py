"""La MCV (c) con bariones generales — núcleo isotermo con la ECV, mapa inverso (ρ₀, σ₀) → (M200, c, r₁) que cubre
todas las ramas, empalme directo envoltura → ramas, la rejilla vectorizada de núcleos y la edad desde la concentración
(bloques P0 del autor del 26 y del 27-oct-2026; física general, sin datos, sin letra).

Es la parte del brazo MCV (c) de PR-S que no toca ningún catálogo: sirve igual para SPARC que para cualquier disco o
esferoide con bariones tabulados (TabulatedBaryons) o de Plummer (PlummerBaryons). El ajuste a SPARC, que no usa
maquinaria de optimización (rejilla congelada, como exige el candado anti-ajuste del 5E sobre dynamics/*sparc*.py), está
en dynamics/mcv_sparc.py.

MODELO (documento del 26-oct). ∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E); dentro de r₁, ρ_M = ρ₀·exp[−(Φ − Φ₀)/σ₀²] con los bariones y
la ECV en Φ; r₁ por Γ_x(r₁)·t_edad = 1 con Γ_x = ρ_M·(σ_x/m)·⟨v_rel⟩ y ⟨v_rel⟩ = 4σ₀/√π; envoltura NFW empalmada en r₁
con ρ y M continuos (Kaplinghat, Tulin y Yu 2016). Para unos bariones dados, el núcleo no depende de σ_x/m ni de t_edad:
solo dónde cae r₁ (donde ρ_M baja a ρ₁ = 1/[(σ_x/m)·⟨v_rel⟩·t_edad]). Cada (ρ₀, σ₀) da como mucho una envoltura; una
envoltura puede venir de varios (ρ₀, σ₀), sus ramas.

σ_x/m ES UN PARÁMETRO EFECTIVO (enmienda 1 del 27-oct): mide la tasa de autointercambio Γ_x de la energía sellada, no la
sección eficaz de una partícula nueva.

COSMOLOGÍA FIJA Y DECLARADA (punto 9 del bloque del 28-oct). PLANCK18 de dynamics/ecv_border.py (h = 0.674,
Ω_m = 0.315, Ω_Λ = 0.685, plana, sin radiación), ρ_E = Ω_Λ·ρ_crit,0 constante y H0_DEFAULT = 67.4 de
dynamics/mcv_equilibrium.py. No entra ninguna ley de intercambio Q (#65) ni ninguna ley de H: con la conversión de la
evaluación del 28-oct (la ley constante, ξ_0 = 0.06), R₀ en z = 1 cambiaría ×0.986 y t_edad entre +0.15 % y +0.74 %, frente al ±29 % de la
c–M y al 30–55 % que la edad mueve el núcleo. t_edad se convierte con el t(a) de este fondo.

Unidades: kpc, M☉, km/s, Gyr; σ_x/m en cm²/g; densidades en M☉/kpc³.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from dynamics.ecv_border import PLANCK18, rho_crit, rho_ecv
from dynamics.mcv_equilibrium import (
    CM2_PER_G_TO_KPC2_PER_MSUN,
    G_KPC,
    GYR_PER_KPC_KMS,
    H0_DEFAULT,
    _bilinear_common_zero,
    mean_relative_speed,
    nfw_density,
    nfw_halo,
    nfw_mass,
    plummer_density,
    plummer_mass,
    r1_from_rate,
)

C_KMS = 299792.458
# Cosmología fija y declarada (punto 9 del bloque del 28-oct): la de dynamics/ecv_border.py, sin ley de Q ni de H.
H0 = H0_DEFAULT                             # 67.4 km/s/Mpc (dynamics/mcv_equilibrium.py)
OMEGA_M, OMEGA_L = PLANCK18["Omega_m"], PLANCK18["Omega_L"]
RHO_CRIT = rho_crit(0.0)                    # M☉/kpc³
RHO_E = rho_ecv()                           # ρ_E = Ω_Λ·ρ_crit,0, constante
T_AGE_FIXED = 10.0                          # Gyr: la convención del documento del 26-oct


# ------------------------------------------------------------------ funciones NFW auxiliares
def nfw_m(x):
    """m(x) = ln(1 + x) − x/(1 + x), con la serie por debajo de x = 0.01 (la resta pierde dígitos)."""
    x = np.asarray(x, dtype=float)
    xs = np.where(x < 1e-2, x, 0.0)
    series = xs ** 2 * (0.5 - xs * (2.0 / 3.0 - xs * (0.75 - xs * (0.8 - xs * (5.0 / 6.0 - xs * (6.0 / 7.0 - xs * 0.875))))))
    with np.errstate(invalid="ignore"):
        full = np.log1p(x) - x / (1.0 + x)
    return np.where(x < 1e-2, series, full)


def _mean_to_local(x):
    """⟨ρ⟩/ρ de una NFW en x = r/r_s: 3m(x)(1 + x)²/x², creciente desde 3/2."""
    x = np.asarray(x, dtype=float)
    return 3.0 * nfw_m(x) * (1.0 + x) ** 2 / x ** 2


def _bisect_log(f, target, lo, hi, n_iter=64):
    """Raíz de f(x) = target con f creciente, por bisección en ln x (vectorizada); NaN fuera de [f(lo), f(hi)]."""
    target = np.asarray(target, dtype=float)
    a = np.full(target.shape, math.log(lo))
    b = np.full(target.shape, math.log(hi))
    ok = (target > f(np.exp(a))) & (target < f(np.exp(b)))
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        up = f(np.exp(m)) < target
        a, b = np.where(up, m, a), np.where(up, b, m)
    return np.where(ok, np.exp(0.5 * (a + b)), np.nan)


def envelope_from_match(r1, rho1, M1, rho_crit: float = RHO_CRIT) -> dict:
    """NFW que empalma en r₁ con ρ(r₁) = ρ₁ y M(<r₁) = M₁ (arrays): x₁ = r₁/r_s, r_s, ρ_s, c, r200 y M200 (M200 a
    200·ρ_crit). NaN donde no hay solución (⟨ρ⟩/ρ ≤ 3/2, o c fuera de [1e-3, 1e4])."""
    r1, rho1, M1 = (np.asarray(v, dtype=float) for v in (r1, rho1, M1))
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 3.0 * M1 / (4.0 * math.pi * r1 ** 3 * rho1)
    x1 = _bisect_log(_mean_to_local, q, 1e-8, 1e6)
    r_s = r1 / x1
    rho_s = rho1 * x1 * (1.0 + x1) ** 2
    c = _bisect_log(lambda cc: cc ** 3 / nfw_m(cc), rho_s / (200.0 / 3.0 * rho_crit), 1e-3, 1e4)
    r200 = c * r_s
    return {"x1": x1, "r_s": r_s, "rho_s": rho_s, "c": c, "r200": r200,
            "M200": 4.0 / 3.0 * math.pi * 200.0 * rho_crit * r200 ** 3}



# ------------------------------------------------------------------ bariones
class TabulatedBaryons:
    """M_b(r) = r·V_bar²(r)/G desde los puntos (R, V_bar² ≥ 0): lineal en r entre puntos, ∝ r³ dentro del primero y
    constante más allá del último."""

    def __init__(self, R_kpc, v2_kms2):
        self.R = np.asarray(R_kpc, dtype=float)
        self.v2 = np.asarray(v2_kms2, dtype=float)
        if self.R.ndim != 1 or self.R.size < 2 or np.any(self.R <= 0.0) or np.any(np.diff(self.R) <= 0.0):
            raise ValueError("radios de los bariones no positivos o no crecientes")
        if np.any(~np.isfinite(self.v2)) or np.any(self.v2 < 0.0):
            raise ValueError("V_bar² negativa o no finita: se pone a cero y se cuenta antes (baryon_v2)")
        self.M_first = self.R[0] * self.v2[0] / G_KPC
        self.M_last = self.R[-1] * self.v2[-1] / G_KPC

    def mass(self, r):
        r = np.asarray(r, dtype=float)
        M = r * np.interp(r, self.R, self.v2) / G_KPC
        M = np.where(r < self.R[0], self.M_first * (r / self.R[0]) ** 3, M)
        return np.where(r > self.R[-1], self.M_last, M)

    def rho_center(self) -> float:
        return 3.0 * self.M_first / (4.0 * math.pi * self.R[0] ** 3)


class PlummerBaryons:
    """Bariones de Plummer con la interfaz de TabulatedBaryons (para comparar con dynamics/mcv_equilibrium)."""

    def __init__(self, M: float, a: float):
        self.M, self.a = M, a

    def mass(self, r):
        return plummer_mass(r, self.M, self.a)

    def rho_center(self) -> float:
        return float(plummer_density(0.0, self.M, self.a))


def _baryon_mass(baryons, r):
    return 0.0 if baryons is None else baryons.mass(r)


def _baryon_rho0(baryons) -> float:
    return 0.0 if baryons is None else baryons.rho_center()



# ------------------------------------------------------------------ autointercambio
def sigma_over_m_power(sigma100: float, a: float):
    """σ_x/m(v) = σ₁₀₀·(⟨v_rel⟩/100 km/s)^(−a) en cm²/g; a = 0 es el caso constante."""
    if a == 0.0:
        return float(sigma100)
    return lambda v: sigma100 * (np.asarray(v, dtype=float) / 100.0) ** (-a)


def _som_value(sigma_over_m, v_rel):
    return sigma_over_m(v_rel) if callable(sigma_over_m) else sigma_over_m * np.ones_like(np.asarray(v_rel, dtype=float))


def rho1_of(sigma0, sigma_over_m, t_age):
    """ρ₁ = 1/[(σ_x/m)·⟨v_rel⟩·t_edad]: la densidad a la que Γ_x·t_edad = 1 (M☉/kpc³), para σ₀ (y t_edad) en arrays."""
    v = mean_relative_speed(1.0) * np.asarray(sigma0, dtype=float)
    k = _som_value(sigma_over_m, v) * CM2_PER_G_TO_KPC2_PER_MSUN * v / GYR_PER_KPC_KMS * np.asarray(t_age, dtype=float)
    return 1.0 / k


# ------------------------------------------------------------------ núcleo isotermo con bariones generales (exacto)
def _core_rhs(rr, phi, M, rho0, sigma0, baryons, rho_E):
    M_tot = M + _baryon_mass(baryons, rr) - 8.0 / 3.0 * math.pi * rho_E * rr ** 3
    return G_KPC * M_tot / (rr * sigma0 ** 2), 4.0 * math.pi * rr ** 3 * rho0 * np.exp(-phi)


def _core_start(r, rho0, sigma0, baryons, rho_E):
    phi = 2.0 * math.pi * G_KPC / (3.0 * sigma0 ** 2) * (rho0 + _baryon_rho0(baryons) - 2.0 * rho_E) * r ** 2
    return phi, 4.0 / 3.0 * math.pi * rho0 * r ** 3


def core_profile(rho0: float, sigma0: float, r, baryons=None, rho_E: float = RHO_E) -> dict:
    """Núcleo isotermo ρ_M = ρ₀·exp(−φ) con φ = (Φ − Φ₀)/σ₀²: MCV, bariones (TabulatedBaryons o PlummerBaryons) y
    ECV en Φ. Devuelve ρ_M, M_M(<r) y Φ − Φ₀ en los radios pedidos (crecientes)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    r_start = min(1e-6 * float(r[0]) if r[0] > 0 else 1e-9, 1e-9 + 1e-6 * float(r[-1]))

    def f(s, y):
        rr = math.exp(s)
        dphi, dM = _core_rhs(rr, y[0], y[1], rho0, sigma0, baryons, rho_E)
        return [float(dphi), float(dM)]
    sol = solve_ivp(f, (math.log(r_start), math.log(float(r[-1]))), list(_core_start(r_start, rho0, sigma0, baryons, rho_E)),
                    method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True)
    y = sol.sol(np.log(np.maximum(r, r_start)))
    return {"r": r, "rho_M": rho0 * np.exp(-y[0]), "M_M": y[1], "dPhi": sigma0 ** 2 * y[0]}


def inverse_map(rho0: float, sigma0: float, sigma_over_m, t_age: float = T_AGE_FIXED, baryons=None,
                rho_E: float = RHO_E, rho_crit: float = RHO_CRIT, r_max: float = 3e3) -> dict | None:
    """Mapa inverso exacto (ρ₀, σ₀) → (M200, c, r₁): se integra el núcleo hasta que ρ_M = ρ₁ y se empalma la NFW.
    None si ρ₀ ≤ ρ₁, si ρ_M no baja a ρ₁ antes de r_max o si no hay NFW que empalme."""
    rho1 = float(rho1_of(sigma0, sigma_over_m, t_age))
    if rho0 <= rho1:
        return None
    target = math.log(rho0 / rho1)
    r_start = 1e-9

    def f(s, y):
        rr = math.exp(s)
        dphi, dM = _core_rhs(rr, y[0], y[1], rho0, sigma0, baryons, rho_E)
        return [float(dphi), float(dM)]

    def hit(s, y):
        return y[0] - target
    hit.terminal, hit.direction = True, 1.0
    sol = solve_ivp(f, (math.log(r_start), math.log(r_max)), list(_core_start(r_start, rho0, sigma0, baryons, rho_E)),
                    method="LSODA", rtol=1e-11, atol=1e-13, events=hit)
    if not sol.t_events[0].size:
        return None
    r1 = math.exp(float(sol.t_events[0][0]))
    M1 = float(sol.y_events[0][0][1])
    env = {k: float(v) for k, v in envelope_from_match(r1, rho1, M1, rho_crit).items()}
    if not math.isfinite(env["c"]):
        return None
    return {"rho0": rho0, "sigma0": sigma0, "rho1": rho1, "r1": r1, "M1": M1, "u": rho0 / rho1, **env}


# ------------------------------------------------------------------ empalme directo con bariones generales (todas las ramas)
def _match_residuals_grid(sig, u, halo, sigma_over_m, t_age, baryons, rho_E, n_steps=900, s_min=-12.0):
    """f1 = ln[ρ_M(r₁)/ρ_env(r₁)] y f2 = ln[M_M(r₁)/M_env(r₁)] en la rejilla (σ₀, u = ρ₀/ρ_env(r₁)): el mismo RK4 que
    dynamics/mcv_equilibrium._residuals_grid, con bariones generales."""
    r1 = np.array([r1_from_rate(s, sigma_over_m, t_age, halo["rho_s"], halo["r_s"]) for s in sig])[:, None]
    rho1 = nfw_density(r1, halo["rho_s"], halo["r_s"])
    S, U = sig[:, None] * np.ones_like(u)[None, :], np.ones_like(sig)[:, None] * u[None, :]
    rho0 = U * rho1
    h = -s_min / n_steps
    r_lo = r1 * math.exp(s_min)
    phi = 2.0 * math.pi * G_KPC / (3.0 * S ** 2) * (rho0 + _baryon_rho0(baryons) - 2.0 * rho_E) * r_lo ** 2
    M = 4.0 / 3.0 * math.pi * rho0 * r_lo ** 3

    def F(s, phi, M):
        rr = r1 * math.exp(s)
        M_tot = M - 8.0 / 3.0 * math.pi * rho_E * rr ** 3 + _baryon_mass(baryons, rr)
        return G_KPC * M_tot / (rr * S ** 2), 4.0 * math.pi * rr ** 3 * rho0 * np.exp(-phi)
    s = s_min
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            k1 = F(s, phi, M)
            k2 = F(s + h / 2, phi + h / 2 * k1[0], M + h / 2 * k1[1])
            k3 = F(s + h / 2, phi + h / 2 * k2[0], M + h / 2 * k2[1])
            k4 = F(s + h, phi + h * k3[0], M + h * k3[1])
            phi = phi + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            M = M + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            s += h
        f1 = np.log(U) - phi
        f2 = np.log(M / nfw_mass(r1, halo["rho_s"], halo["r_s"]))
    return f1, f2


def _match_residuals_exact(x, halo, sigma_over_m, t_age, baryons, rho_E):
    sigma0, u = math.exp(x[0]), math.exp(x[1])
    r1 = r1_from_rate(sigma0, sigma_over_m, t_age, halo["rho_s"], halo["r_s"])
    rho1 = float(nfw_density(r1, halo["rho_s"], halo["r_s"]))
    prof = core_profile(u * rho1, sigma0, [r1], baryons, rho_E)
    return [math.log(prof["rho_M"][0] / rho1), math.log(prof["M_M"][0] / float(nfw_mass(r1, halo["rho_s"], halo["r_s"])))]


def half_density_radius(rho0: float, sigma0: float, baryons=None, rho_E: float = RHO_E, r_max: float = 1e3) -> float:
    """r_c: radio en que ρ_M = ρ₀/2 en el núcleo isotermo (la «media densidad» de la lectura r_c(M200))."""
    def h(lr):
        return float(core_profile(rho0, sigma0, [math.exp(lr)], baryons, rho_E)["rho_M"][0]) / rho0 - 0.5
    r0 = sigma0 / math.sqrt(4.0 * math.pi * G_KPC * rho0)
    return math.exp(brentq(h, math.log(1e-4 * r0), math.log(min(r_max, 1e4 * r0)), xtol=1e-12))


def match_branches(M200: float, c: float, sigma_over_m, t_age: float = T_AGE_FIXED, baryons=None, rho_E: float = RHO_E,
                   H0_kms_mpc: float = H0, sigma0_range=(0.5, 300.0), u_range=(1.0 + 1e-6, 1e10), n_sigma: int = 72,
                   n_u: int = 140, prefilter: bool = False) -> dict:
    """Empalme directo envoltura → ramas con bariones generales: la búsqueda de dynamics/mcv_equilibrium.jeans_match
    (rejilla, candidatas, refinado) con TabulatedBaryons o PlummerBaryons. Ramas ordenadas por ρ₀; la primera es
    «con núcleo» y las demás «densas»."""
    halo = nfw_halo(M200, c, H0_kms_mpc)
    sig = np.geomspace(*sigma0_range, n_sigma)
    u = np.geomspace(*u_range, n_u)
    f1, f2 = _match_residuals_grid(sig, u, halo, sigma_over_m, t_age, baryons, rho_E)
    cands = []
    for i in range(n_sigma - 1):
        for j in range(n_u - 1):
            a, b = f1[i:i + 2, j:j + 2], f2[i:i + 2, j:j + 2]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if a.min() < 0.0 < a.max() and b.min() < 0.0 < b.max() and (not prefilter or _bilinear_common_zero(a, b)):
                cands.append((math.log(math.sqrt(sig[i] * sig[i + 1])), math.log(math.sqrt(u[j] * u[j + 1]))))
    sols = []
    for x0 in cands:
        try:
            r_ = root(_match_residuals_exact, x0, args=(halo, sigma_over_m, t_age, baryons, rho_E), method="hybr",
                      options={"xtol": 1e-12})
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        if not r_.success or max(abs(v) for v in r_.fun) > 1e-8:
            continue
        sigma0, u_ = math.exp(r_.x[0]), math.exp(r_.x[1])
        if any(abs(sigma0 / s["sigma0"] - 1.0) < 1e-6 and abs(u_ / s["u"] - 1.0) < 1e-6 for s in sols):
            continue
        r1 = r1_from_rate(sigma0, sigma_over_m, t_age, halo["rho_s"], halo["r_s"])
        rho1 = float(nfw_density(r1, halo["rho_s"], halo["r_s"]))
        sols.append({"sigma0": sigma0, "u": u_, "rho0": u_ * rho1, "r1": r1, "rho1": rho1,
                     "M1": float(nfw_mass(r1, halo["rho_s"], halo["r_s"])),
                     "r_c": half_density_radius(u_ * rho1, sigma0, baryons, rho_E)})
    sols.sort(key=lambda s: s["rho0"])
    for k, s in enumerate(sols):
        s["branch"] = "con núcleo" if k == 0 else "densa"
    return {"halo": halo, "t_age": t_age, "rho_E": rho_E, "branches": sols}


# ------------------------------------------------------------------ núcleos de toda una rejilla (vectorizado)
def radial_nodes(r_lo: float = 1e-5, r_hi: float = 1e3, h: float = 0.04) -> np.ndarray:
    """Nodos en s = ln r, uniformes de paso h; M_M se lee entre nodos por Hermite cúbico (CoreTable.mass_at)."""
    return np.arange(math.log(r_lo), math.log(r_hi) + 0.5 * h, h)


def integrate_cores(rho0, sigma0, s_nodes, Mb_nodes, Mb_mid, rho_b0, rho_E: float) -> tuple[np.ndarray, np.ndarray]:
    """RK4 sobre los nodos s = ln r para todos los (ρ₀, σ₀) a la vez (arrays planos): φ y M_M en cada nodo, con la
    serie central en el primer nodo. La masa bariónica va en los nodos y en los puntos medios, común (K,) o por punto
    (n, K), con su densidad central ρ_b0 (escalar o por punto). Devuelve dos arrays (n_puntos, n_nodos)."""
    rho0 = np.asarray(rho0, dtype=float)
    s2 = np.asarray(sigma0, dtype=float) ** 2
    r = np.exp(s_nodes)
    n, K = rho0.size, s_nodes.size
    phi = np.empty((n, K))
    M = np.empty((n, K))
    phi[:, 0] = 2.0 * math.pi * G_KPC / (3.0 * s2) * (rho0 + np.asarray(rho_b0, float) - 2.0 * rho_E) * r[0] ** 2
    M[:, 0] = 4.0 / 3.0 * math.pi * rho0 * r[0] ** 3
    mid = np.exp(0.5 * (s_nodes[:-1] + s_nodes[1:]))
    Mb_n = np.broadcast_to(np.asarray(Mb_nodes, float), (n, K)) if np.ndim(Mb_nodes) == 2 else np.asarray(Mb_nodes, float)
    Mb_m = np.broadcast_to(np.asarray(Mb_mid, float), (n, K - 1)) if np.ndim(Mb_mid) == 2 else np.asarray(Mb_mid, float)
    per_point = np.ndim(Mb_nodes) == 2
    E_n = 8.0 / 3.0 * math.pi * rho_E * r ** 3
    E_m = 8.0 / 3.0 * math.pi * rho_E * mid ** 3

    def F(rr, Mb, E, p, m):
        return G_KPC * (m + Mb - E) / (rr * s2), 4.0 * math.pi * rr ** 3 * rho0 * np.exp(-p)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        for k in range(K - 1):
            h = s_nodes[k + 1] - s_nodes[k]
            p, m = phi[:, k], M[:, k]
            b0, bm, b1 = (Mb_n[:, k], Mb_m[:, k], Mb_n[:, k + 1]) if per_point else (Mb_n[k], Mb_m[k], Mb_n[k + 1])
            k1 = F(r[k], b0, E_n[k], p, m)
            k2 = F(mid[k], bm, E_m[k], p + h / 2 * k1[0], m + h / 2 * k1[1])
            k3 = F(mid[k], bm, E_m[k], p + h / 2 * k2[0], m + h / 2 * k2[1])
            k4 = F(r[k + 1], b1, E_n[k + 1], p + h * k3[0], m + h * k3[1])
            phi[:, k + 1] = p + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            M[:, k + 1] = m + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return phi, M


@dataclass
class CoreTable:
    """Núcleos integrados de una rejilla para unos bariones (comunes o por punto): lo que hace falta para colocar r₁
    con cualquier ρ₁ y para leer M_M en cualquier radio."""
    rho0: np.ndarray
    sigma0: np.ndarray
    s_nodes: np.ndarray
    phi: np.ndarray
    M: np.ndarray
    Mb_nodes: np.ndarray                    # (n, K)
    rho_E: float
    phi_mono: np.ndarray = field(init=False)

    def __post_init__(self):
        self.phi_mono = np.maximum.accumulate(np.where(np.isfinite(self.phi), self.phi, np.inf), axis=1)

    def derivatives(self, idx, k):
        """dφ/ds y dM/ds exactas del sistema en los nodos k de los puntos idx."""
        rr = np.exp(self.s_nodes[k])
        E = 8.0 / 3.0 * math.pi * self.rho_E * rr ** 3
        dphi = G_KPC * (self.M[idx, k] + self.Mb_nodes[idx, k] - E) / (rr * self.sigma0[idx] ** 2)
        with np.errstate(under="ignore"):
            dM = 4.0 * math.pi * rr ** 3 * self.rho0[idx] * np.exp(-self.phi[idx, k])
        return dphi, dM

    def mass_at(self, s_query) -> np.ndarray:
        """M_M en s = ln r (forma (n_datos,) común o (n, n_datos) por punto) por Hermite cúbico entre nodos."""
        n, K = self.M.shape
        sq = np.broadcast_to(np.asarray(s_query, float), (n, np.shape(s_query)[-1]))
        k = np.clip(np.searchsorted(self.s_nodes, sq, side="right") - 1, 0, K - 2)
        idx = np.arange(n)[:, None]
        h = self.s_nodes[k + 1] - self.s_nodes[k]
        t = (sq - self.s_nodes[k]) / h
        _, d0 = self.derivatives(idx, k)
        _, d1 = self.derivatives(idx, k + 1)
        return _hermite(self.M[idx, k], self.M[idx, k + 1], d0, d1, h, t)


def baryon_nodes(baryons, s_nodes):
    """Masa bariónica en los nodos y en los puntos medios de una rejilla s = ln r, y la densidad central."""
    r = np.exp(s_nodes)
    mid = np.exp(0.5 * (s_nodes[:-1] + s_nodes[1:]))
    K = s_nodes.size
    return (np.asarray(_baryon_mass(baryons, r), float) * np.ones(K),
            np.asarray(_baryon_mass(baryons, mid), float) * np.ones(K - 1), _baryon_rho0(baryons))


def core_table(rho0, sigma0, baryons, rho_E: float, s_nodes) -> CoreTable:
    """Tabla de núcleos con unos bariones comunes a todos los puntos."""
    Mb_n, Mb_m, rb0 = baryon_nodes(baryons, s_nodes)
    phi, M = integrate_cores(rho0, sigma0, s_nodes, Mb_n, Mb_m, rb0, rho_E)
    rho0 = np.asarray(rho0, float)
    return CoreTable(rho0, np.asarray(sigma0, float), s_nodes, phi, M, np.broadcast_to(Mb_n, (rho0.size, Mb_n.size)), rho_E)


def core_table_per_point(rho0, sigma0, baryons_list, rho_E: float, s_nodes) -> CoreTable:
    """Tabla de núcleos con bariones propios de cada punto (lista de objetos, uno por punto)."""
    parts = [baryon_nodes(b, s_nodes) for b in baryons_list]
    Mb_n = np.stack([p[0] for p in parts])
    Mb_m = np.stack([p[1] for p in parts])
    rb0 = np.array([p[2] for p in parts])
    phi, M = integrate_cores(rho0, sigma0, s_nodes, Mb_n, Mb_m, rb0, rho_E)
    return CoreTable(np.asarray(rho0, float), np.asarray(sigma0, float), s_nodes, phi, M, Mb_n, rho_E)


def _hermite(y0, y1, d0, d1, h, t):
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * h * d0 + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * h * d1


def _hermite_dt(y0, y1, d0, d1, h, t):
    t2 = t * t
    return (6 * t2 - 6 * t) * y0 + (3 * t2 - 4 * t + 1) * h * d0 + (-6 * t2 + 6 * t) * y1 + (3 * t2 - 2 * t) * h * d1


def place_r1(tab: CoreTable, rho1) -> dict:
    """r₁ donde ρ_M = ρ₁ (φ = ln ρ₀/ρ₁) y M_M(r₁) para cada punto de la tabla: bisección vectorizada sobre los nodos y
    Hermite cúbico dentro del intervalo (con las derivadas exactas del sistema). NaN donde ρ₀ ≤ ρ₁ o ρ_M no llega a ρ₁."""
    n, K = tab.phi.shape
    T = np.log(tab.rho0 / np.asarray(rho1, dtype=float))
    idx = np.arange(n)
    ok = np.isfinite(T) & (T > tab.phi[:, 0]) & (T < tab.phi_mono[:, -1])
    lo, hi = np.zeros(n, dtype=int), np.full(n, K - 1)
    for _ in range(int(math.ceil(math.log2(K))) + 1):
        mid = (lo + hi) // 2
        below = tab.phi_mono[idx, mid] < T
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    lo = np.clip(lo, 0, K - 2)
    k1 = lo + 1
    s0, s1 = tab.s_nodes[lo], tab.s_nodes[k1]
    h = s1 - s0
    p0, p1 = tab.phi[idx, lo], tab.phi[idx, k1]
    m0, m1 = tab.M[idx, lo], tab.M[idx, k1]
    dp0, dm0 = tab.derivatives(idx, lo)
    dp1, dm1 = tab.derivatives(idx, k1)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        t = np.clip((T - p0) / (p1 - p0), 0.0, 1.0)
        for _ in range(6):
            f = _hermite(p0, p1, dp0, dp1, h, t) - T
            d = _hermite_dt(p0, p1, dp0, dp1, h, t)
            t = np.clip(t - np.where(np.abs(d) > 0.0, f / d, 0.0), 0.0, 1.0)
        s1_ = s0 + t * h
        M1 = _hermite(m0, m1, dm0, dm1, h, t)
    r1 = np.exp(s1_)
    return {"r1": np.where(ok, r1, np.nan), "M1": np.where(ok, M1, np.nan), "T": T, "ok": ok}


def map_table(tab: CoreTable, sigma_over_m, t_age, rho_crit: float = RHO_CRIT) -> dict:
    """Mapa inverso de toda la tabla para un σ_x/m(v) y un t_edad (escalar o por punto): r₁, M₁ y la envoltura."""
    rho1 = rho1_of(tab.sigma0, sigma_over_m, t_age)
    pl = place_r1(tab, rho1)
    env = envelope_from_match(pl["r1"], rho1, pl["M1"], rho_crit)
    return {"rho1": rho1, **pl, **env}


# ------------------------------------------------------------------ edad desde la concentración (enmienda 2)
def cosmic_time_gyr(a, H0_kms_mpc: float = H0, omega_m: float = OMEGA_M, omega_l: float = OMEGA_L):
    """t(a) del fondo declarado (plano, materia y ECV, sin radiación): t = 2/(3H₀√Ω_Λ)·asinh(√(Ω_Λ/Ω_m)·a^{3/2}) (Gyr),
    la integral exacta de da/(a·H) con dynamics.ecv_border.E_of_z."""
    a = np.asarray(a, dtype=float)
    hubble_time = 977.792221 / H0_kms_mpc
    return 2.0 / (3.0 * math.sqrt(omega_l)) * np.arcsinh(math.sqrt(omega_l / omega_m) * a ** 1.5) * hubble_time


T_UNIVERSE = float(cosmic_time_gyr(1.0))


def formation_age_gyr(c, c1: float):
    """t_edad desde la concentración con la forma de Wechsler et al. (2002), c = c₁·a₀/a_c (a₀ = 1): la edad desde la
    época de formación a_c = c₁/c, t_edad = t(1) − t(a_c). NaN si c ≤ c₁ (a_c ≥ 1: el halo no tiene edad que medir).
    c₁ es decisión del autor (decisión 2 del 27-oct); Wechsler et al. dan c₁ ≈ 4.1 para c_vir."""
    c = np.asarray(c, dtype=float)
    a_c = c1 / c
    with np.errstate(invalid="ignore"):
        return np.where(a_c < 1.0, T_UNIVERSE - cosmic_time_gyr(np.minimum(a_c, 1.0)), np.nan)


def concentration_age_maps(tab: CoreTable, sigma100s, a: float, c1: float, t_lo: float = 0.05, n_scan: int = 144,
                           n_iter: int = 10, max_roots: int = 3, rho_crit: float = RHO_CRIT) -> list[list[dict]]:
    """Mapa inverso con t_edad = formation_age_gyr(c) de la propia envoltura (punto fijo t = t_W(c(t))), para varios
    σ₁₀₀ con el mismo a. El mapa solo depende de σ₁₀₀·t_edad (y de a), así que c se tabula una vez en P = σ₁₀₀·t
    (n_scan puntos en ln P) y cada σ₁₀₀ busca los cambios de signo de g(t) = t_W(c) − t entre puntos válidos; cada
    raíz se refina con regula falsi (Illinois) y el mapa exacto, y se comprueba (|g| < 1e-6·t).

    Puede haber varias raíces: cada una es una envoltura autoconsistente distinta para el mismo núcleo, así que se
    devuelven todas (hasta max_roots, por orden de t), cada una como un mapa completo con NaN donde falta; el ajuste
    toma el mejor χ² entre ellas. Las raíces donde g cruza de + a − son las estables bajo t → t_W(c(t))."""
    som1 = sigma_over_m_power(1.0, a)
    s100 = np.asarray(sigma100s, dtype=float)
    P = np.geomspace(s100.min() * t_lo, s100.max() * T_UNIVERSE, n_scan)
    c_tab = np.stack([map_table(tab, som1, p, rho_crit)["c"] for p in P])            # (n_P, n)
    n = tab.rho0.size
    cols = np.arange(n)
    result = []

    def g_at(s, t):
        return formation_age_gyr(map_table(tab, som1, s * t, rho_crit)["c"], c1) - t

    def edge(s, t_valid, t_invalid, has):
        """Borde de validez entre un punto válido y uno inválido del barrido, por bisección en ln t (lado válido)."""
        v, w = np.log(t_valid), np.log(t_invalid)
        for _ in range(12):
            m_ = 0.5 * (v + w)
            fin = np.isfinite(g_at(s, np.exp(m_)))
            v, w = np.where(has & fin, m_, v), np.where(has & ~fin, m_, w)
        return np.exp(v), g_at(s, np.exp(v))

    for s in s100:
        sel = (P / s >= t_lo * (1.0 - 1e-12)) & (P / s <= T_UNIVERSE * (1.0 + 1e-12))
        ts = P[sel] / s
        g = formation_age_gyr(c_tab[sel], c1) - ts[:, None]
        valid = np.isfinite(g)
        brackets = [[np.full(n, np.nan) for _ in range(4)] for _ in range(max_roots)]
        count = np.zeros(n, dtype=int)

        def add(t_a, t_b, g_a, g_b, count=count):
            new = np.isfinite(g_a) & np.isfinite(g_b) & (np.sign(g_a) != np.sign(g_b)) & (count < max_roots)
            for slot in range(max_roots):
                here = new & (count == slot)
                br = brackets[slot]
                br[0], br[1] = np.where(here, t_a, br[0]), np.where(here, t_b, br[1])
                br[2], br[3] = np.where(here, g_a, br[2]), np.where(here, g_b, br[3])
            count += new
        # Bordes de validez exteriores (donde empieza y donde acaba el empalme con t_W definido): junto al borde
        # inferior c → c₁ y t_W → 0, así que g < 0 y suele haber una raíz entre el borde y el primer punto válido.
        any_valid = valid.any(axis=0)
        f = np.where(any_valid, np.argmax(valid, axis=0), 0)
        last = np.where(any_valid, valid.shape[0] - 1 - np.argmax(valid[::-1], axis=0), 0)
        has_lo = any_valid & (f > 0)
        t_e, g_e = edge(s, ts[f], ts[np.maximum(f - 1, 0)], has_lo)
        add(np.where(has_lo, t_e, np.nan), ts[f], g_e, g[f, cols])
        for k in range(ts.size - 1):
            add(np.full(n, ts[k]), np.full(n, ts[k + 1]), g[k], g[k + 1])
        has_hi = any_valid & (last < ts.size - 1)
        t_e, g_e = edge(s, ts[last], ts[np.minimum(last + 1, ts.size - 1)], has_hi)
        add(ts[last], np.where(has_hi, t_e, np.nan), g[last, cols], g_e)
        maps = []
        for slot in range(max_roots):
            lo, hi, g_lo, g_hi = brackets[slot]
            ok = np.isfinite(lo)
            if not ok.any():
                break
            maps.append(_refine_fixed_point(tab, som1, s, c1, ok, lo, hi, g_lo, g_hi, n_iter, rho_crit))
        result.append(maps)
    return result


def _refine_fixed_point(tab, som1, s, c1, ok, lo, hi, g_lo, g_hi, n_iter, rho_crit):
    n = tab.rho0.size
    a_ = np.where(ok, np.log(np.where(ok, lo, 1.0)), 0.0)
    b_ = np.where(ok, np.log(np.where(ok, hi, 1.0)), 0.0)
    ga, gb = np.where(ok, g_lo, 1.0), np.where(ok, g_hi, -1.0)
    side = np.zeros(n)
    for _ in range(n_iter):
        with np.errstate(invalid="ignore", divide="ignore"):
            m_ = (a_ * gb - b_ * ga) / (gb - ga)
        m_ = np.where(np.isfinite(m_) & (m_ > np.minimum(a_, b_)) & (m_ < np.maximum(a_, b_)), m_, 0.5 * (a_ + b_))
        mp = map_table(tab, som1, s * np.exp(m_), rho_crit)
        gm = formation_age_gyr(mp["c"], c1) - np.exp(m_)
        bad = ~np.isfinite(gm)
        m_ = np.where(bad, 0.5 * (a_ + b_), m_)
        if bad.any():
            mp2 = map_table(tab, som1, s * np.exp(m_), rho_crit)
            gm = np.where(bad, formation_age_gyr(mp2["c"], c1) - np.exp(m_), gm)
        gm = np.where(np.isfinite(gm), gm, ga)
        left = np.sign(gm) == np.sign(ga)
        a_, ga = np.where(left, m_, a_), np.where(left, gm, ga)
        b_, gb = np.where(left, b_, m_), np.where(left, gb, gm)
        gb = np.where(left & (side == 1), gb / 2.0, gb)
        ga = np.where(~left & (side == -1), ga / 2.0, ga)
        side = np.where(left, 1, -1)
    t_star = np.exp(np.where(np.abs(ga) < np.abs(gb), a_, b_))
    out = map_table(tab, som1, s * np.where(ok, t_star, T_AGE_FIXED), rho_crit)
    resid = formation_age_gyr(out["c"], c1) - t_star
    ok = ok & np.isfinite(resid) & (np.abs(resid) < 1e-6 * t_star)
    for k in ("r1", "M1", "x1", "r_s", "rho_s", "c", "r200", "M200"):
        out[k] = np.where(ok, out[k], np.nan)
    out["t_age"] = np.where(ok, t_star, np.nan)
    return out


# ------------------------------------------------------------------ curva de rotación del modelo
def model_v2_at_data(tab: CoreTable, mp: dict, R_data, v2_bar_data) -> np.ndarray:
    """V_mod² en los radios de los datos (n_puntos, n_datos): bariones + MCV (núcleo dentro de r₁, NFW fuera) + el
    término de la ECV, −(8π/3)Gρ_E·R². R_data y v2_bar_data, comunes (n_datos,) o por punto (n, n_datos)."""
    n = tab.rho0.size
    R = np.broadcast_to(np.asarray(R_data, dtype=float), (n, np.shape(R_data)[-1]))
    M_core = tab.mass_at(np.log(R))
    with np.errstate(invalid="ignore"):
        M_nfw = 4.0 * math.pi * mp["rho_s"][:, None] * mp["r_s"][:, None] ** 3 * nfw_m(R / mp["r_s"][:, None])
        M_dm = np.where(R < mp["r1"][:, None], M_core, M_nfw)
    return np.asarray(v2_bar_data, dtype=float) + G_KPC * M_dm / R - 8.0 / 3.0 * math.pi * G_KPC * RHO_E * R ** 2


def ecv_force_fraction(r, M_enclosed, rho_E: float = RHO_E):
    """Peso de la ECV frente a la atracción, (8π/3)ρ_E·r³/M(<r) (comprobación C1)."""
    return 8.0 / 3.0 * math.pi * rho_E * np.asarray(r, dtype=float) ** 3 / np.asarray(M_enclosed, dtype=float)


# ------------------------------------------------------------------ preimágenes de una envoltura
def preimage_cells(f1, f2):
    """Celdas de una rejilla (ρ₀, σ₀) donde ln M200 − ln M200* y ln c − ln c* cambian de signo y el filtro bilineal
    admite un cero común (índices (i, j) de la esquina inferior)."""
    cells = []
    for i in range(f1.shape[0] - 1):
        for j in range(f1.shape[1] - 1):
            a, b = f1[i:i + 2, j:j + 2], f2[i:i + 2, j:j + 2]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if a.min() < 0.0 < a.max() and b.min() < 0.0 < b.max() and _bilinear_common_zero(a, b):
                cells.append((i, j))
    return cells


def refine_preimage(M200: float, c: float, x0, sigma_over_m, t_age: float, baryons=None, rho_E: float = RHO_E,
                    tol: float = 1e-7) -> dict | None:
    """El (ρ₀, σ₀) cuya envoltura es (M200, c), desde x0 = (ln ρ₀, ln σ₀), con el mapa inverso exacto; None si no
    converge."""
    def resid(x):
        m = inverse_map(math.exp(x[0]), math.exp(x[1]), sigma_over_m, t_age, baryons, rho_E)
        return [1e3, 1e3] if m is None else [math.log(m["M200"] / M200), math.log(m["c"] / c)]
    try:
        r_ = root(resid, x0, method="hybr", options={"xtol": 1e-10})
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if not r_.success or max(abs(v) for v in r_.fun) > tol:
        return None
    return {"rho0": math.exp(r_.x[0]), "sigma0": math.exp(r_.x[1])}


def burkert_half_density_x() -> float:
    """x = r/r_c de media densidad de un Burkert, raíz real de (1 + x)(1 + x²) = 2."""
    return brentq(lambda x: (1.0 + x) * (1.0 + x * x) - 2.0, 0.1, 1.0, xtol=1e-14)
