"""Ventana de validez de la Ley de Cronos débil en halos NFW y el cierre
ÚNICO de la amplitud A ≡ α₀⁻¹/ρ_c^(3/2) (ronda de registro, 17-sep-2026).

HALLAZGO DE CONSISTENCIA (cálculo del autor en la sesión de verificación
del 17-sep, reproducido aquí de forma independiente): los parámetros
(α₀⁻¹, ρ_c) de la Def. 11.1 entran en TODA la dinámica solo por la
amplitud A (ε_c = A·ρ^{3/2}; dynamics/weak_field.py), y el repositorio
llevaba dos cierres incompatibles:

  galáctico   A_Sculptor = 6.201589e-7 (M☉/pc³)^{-3/2}   (5E, congelada
              en results/2026-08-16_front5e_sparc; problema inverso en
              Sculptor: dynamics/sculptor_transfer.frozen_config)
  cosmológico ρ_c = 200·ρ̄_m con α₀⁻¹ = 1e-6              (cosmology/
              mu_eta_cronos.RHO_C_OVER_MEAN; elección de la nota de
              teoría del 13-sep, no derivada)

Difieren en ~6.6e7 en A. La condición de la que nace la cota (11.5) es la
SUBDOMINANCIA punto a punto c²ε_c ≲ |Φ_N| (lectura (P), Cor. 11.3c). El
cociente de dominancia D(r) ≡ c²ε_c(r)/|Φ_N(r)| decide:

  - cierre cosmológico dentro de un halo NFW de 1e11 M☉ (c = 10): D ≫ 1
    en todo el halo (5.4 en r₂₀₀, ~1e4 en r_s): EXCLUIDO dinámicamente.
    En la parte externa la exclusión es por subdominancia y no por el
    régimen débil (en r₂₀₀ ε_c ~ 5e-7 ≪ 1); en la interna se rompe
    también el régimen débil (ε_c = 0.09 en 2.3 kpc, 2.2 en 0.38 kpc,
    17 en 0.1 kpc) — matiz que el cálculo añade a la nota del autor;
  - cierre galáctico: en el halo de 1e11 M☉, D_Φ < 1 en todo r ≥ 0.1 kpc
    (0.77 en 0.1 kpc, ~4e-3 en 2.3 kpc) y A_Sculptor ≈ A_max (1.3), la
    amplitud máxima subdominante EN POTENCIAL con suavizado 0.1 kpc. En
    un halo de 1e10 M☉ (c = 13, suavizado 0.05 kpc) esa subdominancia se
    rompe por debajo de ~0.15 kpc (A_max = 0.2·A_Sculptor).

PERO la subdominancia en potencial no es subdominancia en fuerza. El
cociente dinámico D_F = |c² dε_c/dr|/g_N vale, en una cúspide NFW,
(3/2)|dlnρ/dlnr|·|Φ_N| r/(G M(<r)) veces D_Φ — un factor 20–600 porque
|Φ_N| es finito en el centro mientras G M(<r)/r → 0. Con A_Sculptor en el
halo de 1e11 M☉: D_F = 0.73 en 1 kpc, 7.4 en 0.4 kpc y 230 en 0.1 kpc;
D_F = 1 en r ≈ 0.9 kpc. El término +c²∇ε_c DOMINA la gravedad dentro de
~0.9 kpc. Esto no invalida la exclusión del cierre cosmológico (peor aún
en fuerza) ni el régimen débil (ε_c ≤ 3e-7), pero sí la frase «A_Sculptor
está donde la lectura (P) dice»: lo está en potencial, no en fuerza.
Coherente con su origen: A_Sculptor se calibró (5A/5B) para que Cronos
sustituya a la materia oscura en Sculptor (F_Cronos ≈ 7·g_bariones), así
que sumada a un halo CDM domina su centro. El Nivel A mide qué hace esa
fuerza dentro de un halo en equilibrio (expectativa: contracción del
interior, posiblemente progresiva); la puerta de régimen del Nivel A es
la del campo débil (ε_c < 1e-3, N > 0), y D_Φ, D_F se publican como
diagnóstico, no como puerta.

Consecuencias que el registro recoge: la amplitud del modelo operativo
es UNA y es A_Sculptor; el sector µ/η de Cronos (E1, results/2026-09-13_
mu_eta_cronos) se computó con el cierre excluido — con A_Sculptor,
ε̄_c(hoy) ~ 5e-18 y µ − 1 ~ 1e-12 en k = 0.2 h/Mpc: la cola k² muere por
consistencia interna, no solo por falta de sensibilidad.

Esto es un CÁLCULO DE CONSISTENCIA (E8), no un experimento con
desenlaces: no hay umbrales que congelar; el artefacto publica los
números. Unidades: radios en kpc, densidades en M☉/pc³, potenciales en
(km/s)² (G = 4.30091e-3 pc·(km/s)²/M☉).
"""

from __future__ import annotations

import numpy as np

from cronos.cronos_v3 import ALPHA0_INV_MAX
from dynamics.sculptor_transfer import frozen_config
from dynamics.weak_field import C_KMS, G_PC, cronos_amplitude, epsilon_c_of_rho

RHO_CRIT_H2_MSUN_MPC3 = 2.775e11       # ρ_crit/h² [M☉/Mpc³]
A_SCULPTOR = frozen_config().A_sculptor  # (M☉/pc³)^{-3/2}, congelada en 5E


def rho_mean_matter_today(H0: float, Omega_m: float) -> float:
    """ρ̄_m(z = 0) en M☉/pc³ para (H0 [km/s/Mpc], Ω_m)."""
    h = H0 / 100.0
    return Omega_m * RHO_CRIT_H2_MSUN_MPC3 * h ** 2 / 1e18


def rho_crit_today(H0: float) -> float:
    """ρ_crit(z = 0) en M☉/pc³."""
    return RHO_CRIT_H2_MSUN_MPC3 * (H0 / 100.0) ** 2 / 1e18


def cosmological_closure_amplitude(alpha0_inv: float, rho_mean0: float,
                                   f_over_mean: float = 200.0) -> float:
    """A del cierre cosmológico «comóvil» ρ_c = f·ρ̄_m (hoy)."""
    return cronos_amplitude(alpha0_inv, f_over_mean * rho_mean0)


def galactic_closure_rho_c(A: float, alpha0_inv: float = ALPHA0_INV_MAX) -> float:
    """ρ_c equivalente [M☉/pc³] del cierre galáctico a α₀⁻¹ dado:
    A = α₀⁻¹/ρ_c^{3/2} ⟹ ρ_c = (α₀⁻¹/A)^{2/3}."""
    if A <= 0.0:
        raise ValueError("A debe ser > 0")
    return (alpha0_inv / A) ** (2.0 / 3.0)


def galactic_closure_f_over_mean(A: float, rho_mean0: float,
                                 alpha0_inv: float = ALPHA0_INV_MAX) -> float:
    """f = ρ_c/ρ̄_m(0) del cierre galáctico: el valor que hace que
    cosmology.mu_eta_cronos con rho_c_mode='physical' reproduzca A."""
    return galactic_closure_rho_c(A, alpha0_inv) / rho_mean0


def nfw_halo(M200: float, c: float, H0: float, r_min_kpc: float,
             n: int = 600) -> dict:
    """Perfil NFW en equilibrio: r [kpc], ρ [M☉/pc³], |Φ_N| [(km/s)²],
    M(<r) [M☉], r_s y r₂₀₀ [kpc]. Φ_N = −4πGρ_s r_s² ln(1+x)/x."""
    rho_crit = rho_crit_today(H0)                      # M☉/pc³
    r200_pc = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    rs_pc = r200_pc / c
    fc = np.log(1.0 + c) - c / (1.0 + c)
    rho_s = rho_crit * (200.0 / 3.0) * c ** 3 / fc
    r_kpc = np.geomspace(r_min_kpc, r200_pc / 1e3, n)
    x = r_kpc * 1e3 / rs_pc
    rho = rho_s / (x * (1.0 + x) ** 2)
    M_enc = 4.0 * np.pi * rho_s * rs_pc ** 3 * (np.log(1.0 + x) - x / (1.0 + x))
    abs_phi = 4.0 * np.pi * G_PC * rho_s * rs_pc ** 2 * np.log(1.0 + x) / x
    return {"r_kpc": r_kpc, "rho": rho, "abs_phi": abs_phi, "M_enc": M_enc,
            "r_s_kpc": rs_pc / 1e3, "r200_kpc": r200_pc / 1e3, "rho_s": rho_s,
            "v_circ_rs_kms": float(np.sqrt(G_PC * M_enc[np.argmin(abs(x - 1.0))] / rs_pc))}


def dominance_ratio(halo: dict, A: float) -> np.ndarray:
    """D_Φ(r) = c²ε_c(r)/|Φ_N(r)| con ε_c = A·ρ^{3/2} — la condición LITERAL
    de Cor. 11.3c («c²ε_c ≲ |Φ_N|»), la que mide dynamics.weak_field."""
    return C_KMS ** 2 * epsilon_c_of_rho(halo["rho"], A) / halo["abs_phi"]


def force_dominance_ratio(halo: dict, A: float) -> np.ndarray:
    """D_F(r) = |c² dε_c/dr| / g_N(r), g_N = G M(<r)/r² — el cociente de
    FUERZAS, que es lo que «dominar sobre la gravedad» significa en la
    dinámica. En una cúspide NFW, D_F/D_Φ = (3/2)|dlnρ/dlnr|·|Φ_N| r/(G M(<r))
    ≫ 1 (|Φ_N| es finito en r → 0 mientras G M/r → 0): con A_Sculptor en
    un halo de 1e11 M☉, D_Φ = 0.77 en 0.1 kpc pero D_F ≈ 230, y D_F = 1
    en r ≈ 0.9 kpc. Hallazgo de la sesión de código (17-sep) al pilotar
    el Nivel A; corrige la lectura «A_Sculptor está donde (P) dice»."""
    r_pc = halo["r_kpc"] * 1e3
    eps = epsilon_c_of_rho(halo["rho"], A)
    deps_dr = np.gradient(eps, r_pc)                       # 1/pc
    g_N = G_PC * halo["M_enc"] / r_pc ** 2                # (km/s)²/pc
    return C_KMS ** 2 * np.abs(deps_dr) / g_N


def radius_where_force_ratio_is_one(halo: dict, A: float):
    """Mayor radio [kpc] con D_F ≥ 1; None si nunca."""
    DF = force_dominance_ratio(halo, A)
    idx = np.where(DF >= 1.0)[0]
    if idx.size == 0:
        return None
    i = int(idx[-1])
    if i == len(DF) - 1:
        return float(halo["r_kpc"][-1])
    r0, r1 = np.log(halo["r_kpc"][i]), np.log(halo["r_kpc"][i + 1])
    d0, d1 = np.log(DF[i]), np.log(DF[i + 1])
    return float(np.exp(r0 - d0 * (r1 - r0) / (d1 - d0)))


def max_subdominant_amplitude(halo: dict) -> float:
    """A_max: la mayor A con D(r) ≤ 1 en toda la malla (r ≥ r_min)."""
    return float(np.min(halo["abs_phi"] / (C_KMS ** 2 * halo["rho"] ** 1.5)))


def _at(halo: dict, arr: np.ndarray, r_kpc: float) -> float:
    return float(arr[np.argmin(np.abs(halo["r_kpc"] - r_kpc))])


def radius_where_dominance_is_one(halo: dict, A: float):
    """Mayor radio [kpc] con D ≥ 1 (D decrece hacia fuera); None si D < 1
    en toda la malla. Interpolación log-log entre nodos vecinos."""
    D = dominance_ratio(halo, A)
    idx = np.where(D >= 1.0)[0]
    if idx.size == 0:
        return None
    i = int(idx[-1])
    if i == len(D) - 1:
        return float(halo["r_kpc"][-1])
    r0, r1 = np.log(halo["r_kpc"][i]), np.log(halo["r_kpc"][i + 1])
    d0, d1 = np.log(D[i]), np.log(D[i + 1])
    return float(np.exp(r0 + (0.0 - d0) * (r1 - r0) / (d1 - d0)))


def validity_report(H0: float, Omega_m: float, alpha0_inv: float = ALPHA0_INV_MAX,
                    f_cosmo: float = 200.0, A_gal: float = A_SCULPTOR,
                    halos=((1e11, 10.0, 0.10), (1e10, 13.0, 0.05))) -> dict:
    """Números serializables del hallazgo: los dos cierres, su cociente,
    y D(r) en radios de referencia por halo."""
    rho_mean0 = rho_mean_matter_today(H0, Omega_m)
    A_cosmo = cosmological_closure_amplitude(alpha0_inv, rho_mean0, f_cosmo)
    rho_c_gal = galactic_closure_rho_c(A_gal, alpha0_inv)
    out = {"inputs": {"H0": H0, "Omega_m": Omega_m, "alpha0_inv": alpha0_inv,
                      "f_cosmo_over_mean": f_cosmo, "A_galactic": A_gal,
                      "A_units": "(M_sol/pc^3)^(-3/2); epsilon_c = A * rho^(3/2)"},
           "rho_mean0_msun_pc3": rho_mean0,
           "A_cosmological": A_cosmo,
           "A_cosmo_over_A_galactic": A_cosmo / A_gal,
           "rho_c_galactic_equiv_msun_pc3": rho_c_gal,
           "rho_c_galactic_over_mean": rho_c_gal / rho_mean0,
           "epsilon_c_background_today": {
               "galactic": float(epsilon_c_of_rho(rho_mean0, A_gal)),
               "cosmological": float(epsilon_c_of_rho(rho_mean0, A_cosmo))},
           "halos": []}
    for M200, c, soft in halos:
        h = nfw_halo(M200, c, H0, soft)
        A_max = max_subdominant_amplitude(h)
        row = {"M200": M200, "c": c, "soft_kpc": soft, "r_s_kpc": h["r_s_kpc"],
               "r200_kpc": h["r200_kpc"], "v_circ_rs_kms": h["v_circ_rs_kms"],
               "A_max_subdominant": A_max, "A_max_over_A_galactic": A_max / A_gal,
               "M_within_2p3kpc": _at(h, h["M_enc"], 2.3),
               "M_within_0p4kpc": _at(h, h["M_enc"], 0.4),
               "closures": {}}
        for name, A in (("galactic", A_gal), ("A_max", A_max), ("cosmological", A_cosmo)):
            D = dominance_ratio(h, A)
            DF = force_dominance_ratio(h, A)
            eps = epsilon_c_of_rho(h["rho"], A)
            row["closures"][name] = {
                "A": A, "D_soft": float(D[0]), "D_0p38kpc": _at(h, D, 0.38),
                "D_2p3kpc": _at(h, D, 2.3), "D_rs": _at(h, D, h["r_s_kpc"]),
                "D_r200": float(D[-1]), "D_max": float(D.max()),
                "r_D_equals_one_kpc": radius_where_dominance_is_one(h, A),
                "DF_soft": float(DF[0]), "DF_0p4kpc": _at(h, DF, 0.4),
                "DF_1kpc": _at(h, DF, 1.0), "DF_2p3kpc": _at(h, DF, 2.3),
                "DF_rs": _at(h, DF, h["r_s_kpc"]),
                "r_DF_equals_one_kpc": radius_where_force_ratio_is_one(h, A),
                "force_subdominant_everywhere": bool(DF.max() <= 1.0 + 1e-9),
                "eps_c_max": float(eps.max()), "eps_c_r200": float(eps[-1]),
                "eps_c_2p3kpc": _at(h, eps, 2.3), "eps_c_0p38kpc": _at(h, eps, 0.38),
                "subdominant_everywhere": bool(D.max() <= 1.0 + 1e-9),
                "weak_everywhere": bool(eps.max() < 1e-3),
                "weak_at_r200": bool(eps[-1] < 1e-3)}
        out["halos"].append(row)
    return out
