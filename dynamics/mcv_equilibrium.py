"""MCV (c): la materia del vacío capturada en los pozos, en equilibrio por autointercambio — Jeans isotermo empalmado con
una envoltura NFW (decisiones del autor del 14, 18, 21 y 23-oct-2026; CONDICIONAL, sin datos).

DECISIONES DEL AUTOR, no derivadas aquí. La MCV es la energía sellada atrapada en los pozos: la misma sustancia que la
ECV, con masa ínfima pero no nula, y existe antes que los agujeros negros. Su concentración la frena el equilibrio por dos
canales, el autointercambio (MCV con MCV) y el intercambio con la ECV: decisión (c), confirmada el 23-oct. La fuerza
local +c²∇ε_c(ρ_m) de la Cor. 11.3(c) deja de ser la MCV; sus filas quedan como resultados sobre la ley local. σ_x/m es
global: se calibra una vez y se prueba fuera de la muestra (su derivación desde el sellado sigue abierta).

ECUACIONES (bloque P0 del 23-oct):

  1. ∇²Φ = 4πG(ρ_b + ρ_M − 2ρ_E): la ECV (w = −1) entra con ρ + 3p = −2ρ_E.
  2. Γ_x = ρ_M·(σ_x/m)·⟨v_rel⟩, ⟨v_rel⟩ = 4σ/√π; el radio r₁ cumple Γ_x(r₁)·t_edad = 1.
  3. Si r < r₁: ρ_M = ρ_M0·exp[−(Φ − Φ₀)/σ₀²], con los bariones dentro de Φ.
  4. Empalme en r₁ con la envoltura: ρ_M y M(<r) continuos — el modelo de Jeans isotermo de Kaplinghat, Tulin y Yu
     (2016, PRL 116, 041302). La regla de r₁ usa la densidad de la envoltura en r₁ (que el empalme hace igual a la del
     núcleo) y la dispersión σ₀ del núcleo.
  5. Junto a un agujero negro: ρ_M ∝ r^{−(3+a)/4} si σ_x/m ∝ v^{−a} (Shapiro y Paschalidis 2014, PRD 89, 023506).
  6. Colapso gravotérmico: t_c ≈ (150/β)/[(σ_x/m)·ρ_s·r_s·√(4πGρ_s)], β entre 0.45 y 0.75 (Essig et al. 2019,
     PRL 123, 121102).

EL EMPALME TIENE VARIAS RAMAS. Para una envoltura y un σ_x/m dados, las condiciones (ρ y M continuos en r₁, con r₁ fijado
por la tasa) pueden tener una solución, varias o ninguna: la rama de menor ρ₀ es la «con núcleo» y las de mayor ρ₀ las
«densas» (hacia el colapso). `jeans_match` devuelve TODAS las ramas que encuentra en su rejilla, ordenadas por ρ₀; elegir
una es decisión de quien preinscribe.

Unidades: kpc, M☉, km/s y Gyr; σ_x/m en cm²/g (número, o función de ⟨v_rel⟩ en km/s); densidades en M☉/kpc³.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

G_KPC = 4.30091e-6                       # kpc·(km/s)²/M☉
GYR_PER_KPC_KMS = 0.977792               # 1 kpc/(km/s) en Gyr
MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
CM2_PER_G_TO_KPC2_PER_MSUN = MSUN_G / KPC_CM ** 2
H0_DEFAULT = 67.4                        # km/s/Mpc: el de las cifras de referencia del autor (M200 a 200·ρ_crit)
T_AGE_DEFAULT = 10.0                     # Gyr
BETA_COLLAPSE = (0.45, 0.75)             # intervalo de β de Essig et al. (2019)


# ------------------------------------------------------------------ envoltura NFW y bariones de Plummer
def nfw_halo(M200: float, c: float, H0: float = H0_DEFAULT) -> dict:
    """Parámetros NFW de una envoltura (M200 a 200·ρ_crit): r200 y r_s en kpc, ρ_s en M☉/kpc³."""
    rho_crit = 3.0 * (H0 / 1000.0) ** 2 / (8.0 * math.pi * G_KPC)
    r200 = (3.0 * M200 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    r_s = r200 / c
    rho_s = M200 / (4.0 * math.pi * r_s ** 3 * (math.log(1.0 + c) - c / (1.0 + c)))
    return {"M200": M200, "c": c, "H0": H0, "rho_crit": rho_crit, "r200": r200, "r_s": r_s, "rho_s": rho_s}


def nfw_density(r, rho_s: float, r_s: float):
    x = np.asarray(r, dtype=float) / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def nfw_mass(r, rho_s: float, r_s: float):
    x = np.asarray(r, dtype=float) / r_s
    return 4.0 * math.pi * rho_s * r_s ** 3 * (np.log1p(x) - x / (1.0 + x))


def plummer_density(r, M_b: float, a: float):
    return 3.0 * M_b / (4.0 * math.pi * a ** 3) * (1.0 + (np.asarray(r, dtype=float) / a) ** 2) ** -2.5


def plummer_mass(r, M_b: float, a: float):
    r = np.asarray(r, dtype=float)
    return M_b * r ** 3 / (r ** 2 + a ** 2) ** 1.5


# ------------------------------------------------------------------ autointercambio
def mean_relative_speed(sigma_kms: float) -> float:
    """⟨v_rel⟩ = 4σ/√π para una maxwelliana de dispersión 1D σ."""
    return 4.0 * sigma_kms / math.sqrt(math.pi)


def _sigma_over_m(sigma_over_m, v_rel: float) -> float:
    """σ_x/m en cm²/g: un número o una función de ⟨v_rel⟩ (km/s)."""
    return float(sigma_over_m(v_rel)) if callable(sigma_over_m) else float(sigma_over_m)


def interaction_rate(rho, sigma_over_m, sigma_kms: float):
    """Γ_x = ρ·(σ_x/m)·⟨v_rel⟩ en 1/Gyr (ρ en M☉/kpc³)."""
    v = mean_relative_speed(sigma_kms)
    return np.asarray(rho) * _sigma_over_m(sigma_over_m, v) * CM2_PER_G_TO_KPC2_PER_MSUN * v / GYR_PER_KPC_KMS


def r1_from_rate(sigma0: float, sigma_over_m, t_age: float, rho_s: float, r_s: float) -> float:
    """r₁ con Γ_x(r₁)·t_edad = 1, usando la densidad de la envoltura en r₁ y la dispersión σ₀ del núcleo."""
    k = float(interaction_rate(1.0, sigma_over_m, sigma0)) * t_age          # Γ·t por unidad de densidad
    return math.exp(brentq(lambda lr: math.log(float(nfw_density(math.exp(lr), rho_s, r_s)) * k),
                           math.log(1e-9 * r_s), math.log(1e6 * r_s), xtol=1e-14))


# ------------------------------------------------------------------ núcleo isotermo con bariones
def _rhs(r, phi, M, rho0, sigma0, plummer, rho_E):
    """dφ/dr y dM/dr con φ = (Φ − Φ₀)/σ₀² (adimensional) y M la masa de MCV dentro de r."""
    M_tot = M - 2.0 * rho_E * 4.0 / 3.0 * math.pi * r ** 3
    if plummer is not None:
        M_tot = M_tot + plummer_mass(r, plummer["M"], plummer["a"])
    return G_KPC * M_tot / (r ** 2 * sigma0 ** 2), 4.0 * math.pi * r ** 2 * rho0 * np.exp(-phi)


def _central_start(r, rho0, sigma0, plummer, rho_E):
    rho_b0 = 0.0 if plummer is None else float(plummer_density(0.0, plummer["M"], plummer["a"]))
    phi = 2.0 * math.pi * G_KPC / (3.0 * sigma0 ** 2) * (rho0 + rho_b0 - 2.0 * rho_E) * r ** 2
    return phi, 4.0 / 3.0 * math.pi * rho0 * r ** 3


def isothermal_profile(rho0: float, sigma0: float, r, plummer: dict | None = None, rho_E: float = 0.0) -> dict:
    """Núcleo isotermo de MCV: ρ_M = ρ₀·exp[−(Φ − Φ₀)/σ₀²] con Φ de la MCV, los bariones de Plummer ({"M", "a"}) y la
    ECV (−2ρ_E). Devuelve ρ_M(r), M_M(<r) y (Φ − Φ₀)(r) en los radios pedidos (crecientes)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    r_start = min(1e-6 * float(r[0]) if r[0] > 0 else 1e-9, 1e-9 + 1e-6 * float(r[-1]))
    y0 = _central_start(r_start, rho0, sigma0, plummer, rho_E)

    def f(s, y):
        rr = math.exp(s)
        dphi, dM = _rhs(rr, y[0], y[1], rho0, sigma0, plummer, rho_E)
        return [rr * dphi, rr * dM]
    sol = solve_ivp(f, (math.log(r_start), math.log(float(r[-1]))), list(y0), method="LSODA", rtol=1e-10, atol=1e-12,
                    dense_output=True)
    y = sol.sol(np.log(np.maximum(r, r_start)))
    return {"r": r, "rho_M": rho0 * np.exp(-y[0]), "M_M": y[1], "dPhi": sigma0 ** 2 * y[0]}


def core_radius(rho0: float, sigma0: float, plummer: dict | None = None, rho_E: float = 0.0, r_max: float = 1e3) -> float:
    """r_c: radio en que ρ_M = ρ₀/2 en el núcleo isotermo."""
    def h(lr):
        return float(isothermal_profile(rho0, sigma0, [math.exp(lr)], plummer, rho_E)["rho_M"][0]) / rho0 - 0.5
    r0 = sigma0 / math.sqrt(4.0 * math.pi * G_KPC * rho0)
    return math.exp(brentq(h, math.log(1e-4 * r0), math.log(min(r_max, 1e4 * r0)), xtol=1e-12))


# ------------------------------------------------------------------ empalme de Jeans (todas las ramas)
def _residuals_grid(sig, u, halo, sigma_over_m, t_age, plummer, rho_E, n_steps=900, s_min=-12.0):
    """f1 = ln[ρ_M(r₁)/ρ_env(r₁)] y f2 = ln[M_M(r₁)/M_env(r₁)] sobre la rejilla (σ₀, u = ρ₀/ρ_env(r₁)), por RK4
    vectorizado en s = ln(r/r₁) ∈ [s_min, 0]."""
    r1 = np.array([r1_from_rate(s, sigma_over_m, t_age, halo["rho_s"], halo["r_s"]) for s in sig])[:, None]
    rho1 = nfw_density(r1, halo["rho_s"], halo["r_s"])
    S, U = sig[:, None] * np.ones_like(u)[None, :], np.ones_like(sig)[:, None] * u[None, :]
    rho0 = U * rho1
    h = -s_min / n_steps
    r_lo = r1 * math.exp(s_min)
    rho_b0 = 0.0 if plummer is None else float(plummer_density(0.0, plummer["M"], plummer["a"]))
    phi = 2.0 * math.pi * G_KPC / (3.0 * S ** 2) * (rho0 + rho_b0 - 2.0 * rho_E) * r_lo ** 2
    M = 4.0 / 3.0 * math.pi * rho0 * r_lo ** 3

    def F(s, phi, M):
        rr = r1 * math.exp(s)
        M_tot = M - 8.0 / 3.0 * math.pi * rho_E * rr ** 3
        if plummer is not None:
            M_tot = M_tot + plummer_mass(rr, plummer["M"], plummer["a"])
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


def _residuals_exact(x, halo, sigma_over_m, t_age, plummer, rho_E):
    sigma0, u = math.exp(x[0]), math.exp(x[1])
    r1 = r1_from_rate(sigma0, sigma_over_m, t_age, halo["rho_s"], halo["r_s"])
    rho1 = float(nfw_density(r1, halo["rho_s"], halo["r_s"]))
    prof = isothermal_profile(u * rho1, sigma0, [r1], plummer, rho_E)
    return [math.log(prof["rho_M"][0] / rho1), math.log(prof["M_M"][0] / float(nfw_mass(r1, halo["rho_s"], halo["r_s"])))]


def jeans_match(M200: float, c: float, sigma_over_m, t_age: float = T_AGE_DEFAULT, H0: float = H0_DEFAULT,
                plummer: dict | None = None, rho_E: float = 0.0, sigma0_range=(0.5, 300.0), u_range=(1.0 + 1e-6, 1e10),
                n_sigma: int = 72, n_u: int = 140) -> dict:
    """Empalme de Jeans isotermo con la envoltura NFW (M200, c): TODAS las soluciones (σ₀, ρ₀) en la rejilla, refinadas.

    Cada rama trae σ₀ (km/s), ρ₀ (M☉/kpc³), r₁ (kpc), r_c (kpc, ρ = ρ₀/2), la densidad y la masa en r₁ y su clase:
    'con núcleo' la de menor ρ₀, 'densa' las demás. Sin solución en la rejilla, la lista va vacía (se declara)."""
    halo = nfw_halo(M200, c, H0)
    sig = np.geomspace(*sigma0_range, n_sigma)
    u = np.geomspace(*u_range, n_u)
    f1, f2 = _residuals_grid(sig, u, halo, sigma_over_m, t_age, plummer, rho_E)
    cands = []
    for i in range(n_sigma - 1):
        for j in range(n_u - 1):
            a, b = f1[i:i + 2, j:j + 2], f2[i:i + 2, j:j + 2]
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                continue
            if a.min() < 0.0 < a.max() and b.min() < 0.0 < b.max():
                cands.append((math.log(math.sqrt(sig[i] * sig[i + 1])), math.log(math.sqrt(u[j] * u[j + 1]))))
    sols = []
    for x0 in cands:
        try:
            r_ = root(_residuals_exact, x0, args=(halo, sigma_over_m, t_age, plummer, rho_E), method="hybr",
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
        rho0 = u_ * rho1
        sols.append({"sigma0": sigma0, "u": u_, "rho0": rho0, "r1": r1, "rho1": rho1,
                     "M1": float(nfw_mass(r1, halo["rho_s"], halo["r_s"])),
                     "r_c": core_radius(rho0, sigma0, plummer, rho_E), "residuals": [float(v) for v in r_.fun]})
    sols.sort(key=lambda s: s["rho0"])
    for k, s in enumerate(sols):
        s["branch"] = "con núcleo" if k == 0 else "densa"
    return {"halo": halo, "sigma_over_m": sigma_over_m if not callable(sigma_over_m) else "función de ⟨v_rel⟩",
            "t_age": t_age, "plummer": plummer, "rho_E": rho_E, "branches": sols,
            "grid": {"sigma0_range": list(sigma0_range), "u_range": list(u_range), "n_sigma": n_sigma, "n_u": n_u}}


def dm_profile(match: dict, branch: dict, r) -> dict:
    """Perfil completo de la MCV para una rama: núcleo isotermo dentro de r₁ y envoltura NFW fuera (ρ y M continuos)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    halo = match["halo"]
    inner = r <= branch["r1"]
    rho, M = np.empty_like(r), np.empty_like(r)
    if inner.any():
        prof = isothermal_profile(branch["rho0"], branch["sigma0"], r[inner], match["plummer"], match["rho_E"])
        rho[inner], M[inner] = prof["rho_M"], prof["M_M"]
    rho[~inner] = nfw_density(r[~inner], halo["rho_s"], halo["r_s"])
    M[~inner] = nfw_mass(r[~inner], halo["rho_s"], halo["r_s"])
    return {"r": r, "rho_M": rho, "M_M": M}


# ------------------------------------------------------------------ colapso gravotérmico y cúspide del agujero negro
def gravothermal_collapse_time(sigma_over_m: float, rho_s: float, r_s: float, beta: float = 0.75) -> float:
    """t_c ≈ (150/β)/[(σ_x/m)·ρ_s·r_s·√(4πGρ_s)] en Gyr (Essig et al. 2019; β entre 0.45 y 0.75)."""
    t = (150.0 / beta) / (sigma_over_m * CM2_PER_G_TO_KPC2_PER_MSUN * rho_s * r_s * math.sqrt(4.0 * math.pi * G_KPC * rho_s))
    return t * GYR_PER_KPC_KMS


def bh_cusp_slope(a: float) -> float:
    """Pendiente de la cúspide de MCV junto a un agujero negro: ρ_M ∝ r^{−(3+a)/4} si σ_x/m ∝ v^{−a}."""
    return (3.0 + a) / 4.0


def bh_influence_radius(M_bh: float, sigma0: float) -> float:
    """r_h = G·M_BH/σ₀², radio de influencia del agujero negro sobre el núcleo isotermo (kpc)."""
    return G_KPC * M_bh / sigma0 ** 2


def bh_cusp_density(r, M_bh: float, sigma0: float, rho0: float, a: float = 0.0):
    """ρ_M(r) = ρ₀·(r/r_h)^{−(3+a)/4} dentro de r_h y ρ₀ fuera (forma declarada: la cúspide de Shapiro y Paschalidis
    empalmada con la densidad central del núcleo isotermo en r_h)."""
    r = np.asarray(r, dtype=float)
    r_h = bh_influence_radius(M_bh, sigma0)
    return np.where(r < r_h, rho0 * (r / r_h) ** -bh_cusp_slope(a), rho0)
