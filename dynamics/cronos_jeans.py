"""El criterio de Cronos–Jeans para la Ley de Cronos débil (Def. 11.1 +
Cor. 11.3c): derivación del autor (sesión de verificación del
21-sep-2026), reproducida en el repositorio (E8).

DERIVACIÓN. Medio colisionless isotermo con dispersión 1D σ y potencial
efectivo Ψ = Φ_N − c²ε_c(ρ) (Prop. 11.2). La respuesta de Boltzmann a
una perturbación estática es δρ = −ρ·δΨ/σ²; la parte de Cronos es
autoconsistente, δΨ_C = −c²ε_c'(ρ)·δρ, luego δρ = +ρc²ε_c'(ρ)δρ/σ²
admite solución no trivial cuando ρc²ε_c'/σ² ≥ 1. Con ε_c = A·ρ^{3/2},
ρε_c' = (3/2)ε_c:

    q(ρ, σ) ≡ (3/2)·c²·ε_c(ρ) / σ_1D²  ≥ 1   ⟹   inestable a TODA k.

En el límite fluido ω² = (σ² − (3/2)c²ε_c)k² − 4πGρ: la gravedad es
infrarroja (tasa independiente de k) y la fuerza de Cronos, LOCAL, es
ultravioleta (tasa ∝ k·√(q − 1)·σ). No hay longitud de Jeans propia: por
encima del umbral la escala más pequeña resuelta crece más deprisa.
Consecuencia para el N-cuerpos: con q > 1 refinar la resolución del
campo AUMENTA la tasa — no existe solución convergida del estado inicial.

TRES APLICACIONES (las tablas de la nota del autor):
  (1) el halo NFW del Nivel A (1e11 M☉, c = 10, Jeans isótropo) con
      A_Sculptor: q(r), r_CJ (q = 1) y la masa dentro;
  (2) la vecindad solar (McKee, Parravano & Hollenbach 2015): fuerza
      vertical de Cronos frente a K_z de losa y límite de Oort efectivo;
  (3) Sculptor con el montaje CONGELADO del 5E (dynamics.sculptor_transfer):
      q(r) y el perfil σ_los(R) que la misma amplitud predice.

Estatuto: derivación y cálculos de consistencia (E8) con expectativas
declaradas (E13) y sin datos ingeridos; nada es resultado observacional
hasta que se preinscriba y se ejecute con bytes oficiales. Ningún
umbral ni A_Sculptor se toca.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from dynamics.dsph_data import SCULPTOR, plummer_scale_from_Rhalf, stellar_mass
from dynamics.jeans import sigma_los_sq, sigma_r_sq_grid
from dynamics.sculptor_transfer import UPSILON_SCULPTOR_FIDUCIAL, frozen_config
from dynamics.weak_field import (
    C_KMS,
    G_PC,
    g_eff_plummer,
    plummer_density,
    plummer_g_newton,
)

G_KPC = 4.30091e-6      # kpc·(km/s)²/M☉
A_SCULPTOR = frozen_config().A_sculptor

STATUS = ("derivación (E8): criterio local de inestabilidad ultravioleta de la ley "
          "ε_c = A·ρ^(3/2); aplicaciones sin datos ingeridos; expectativas declaradas (E13)")


# ---------------------------------------------------------------- criterio
def q_criterion(rho_msun_pc3, sigma2_kms2, A: float = A_SCULPTOR):
    """q = (3/2)·c²·A·ρ^{3/2} / σ_1D²  (ρ en M☉/pc³, σ² en (km/s)²)."""
    return 1.5 * C_KMS ** 2 * A * np.asarray(rho_msun_pc3, dtype=float) ** 1.5 / np.asarray(sigma2_kms2, dtype=float)


def fluid_dispersion_omega2(k, rho_msun_pc3, sigma2_kms2, A: float = A_SCULPTOR,
                            with_gravity: bool = True):
    """ω² = (σ² − (3/2)c²ε_c)·k² − 4πGρ  [k en 1/pc; ω en km/s/pc]."""
    eps = A * float(rho_msun_pc3) ** 1.5
    grav = 4.0 * np.pi * G_PC * float(rho_msun_pc3) if with_gravity else 0.0
    return (float(sigma2_kms2) - 1.5 * C_KMS ** 2 * eps) * np.asarray(k, dtype=float) ** 2 - grav


def growth_rate_over_k(q: float, sigma_kms: float = 1.0) -> float:
    """Tasa de crecimiento por unidad de k en el límite fluido sin
    gravedad: γ/k = σ·√(q − 1) para q > 1; 0 si q ≤ 1."""
    return float(sigma_kms * np.sqrt(max(q - 1.0, 0.0)))


# ------------------------------------------------------------ (1) halo NFW
def nfw_jeans_table(M200: float = 1e11, c: float = 10.0, H0: float = 67.86705532886631,
                    A: float = A_SCULPTOR, radii_kpc=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.5, 2.3, 5.0),
                    amplitude_factors=(0.1, 0.05), v_well_stop_kms: float = 1000.0) -> dict:
    """q(r) en un NFW isótropo (Jeans) con la amplitud única; r_CJ y la
    masa dentro; r_CJ para fracciones de A; y a qué densidad equivale la
    regla de parada v_well del Nivel A."""
    rho_crit = 3.0 * (H0 / 1000.0) ** 2 / (8.0 * np.pi * G_KPC)          # M☉/kpc³
    r200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    rs = r200 / c
    fc = np.log(1.0 + c) - c / (1.0 + c)
    rho_s = M200 / (4.0 * np.pi * rs ** 3 * fc)

    def rho(r):                                   # M☉/kpc³
        return rho_s / ((r / rs) * (1.0 + r / rs) ** 2)

    def M(r):
        return 4.0 * np.pi * rho_s * rs ** 3 * (np.log(1.0 + r / rs) - (r / rs) / (1.0 + r / rs))

    def sig2(r):                                  # Jeans isótropo, (km/s)²
        val, _ = quad(lambda x: rho(x) * G_KPC * M(x) / x ** 2, r, 50.0 * r200, limit=200)
        return val / rho(r)

    rows = []
    for r in radii_kpc:
        rp = rho(r) * 1e-9
        eps = A * rp ** 1.5
        s2 = sig2(r)
        dlnrho = -(1.0 + 3.0 * r / rs) / (1.0 + r / rs)
        DF = C_KMS ** 2 * 1.5 * eps * abs(dlnrho) / r / (G_KPC * M(r) / r ** 2)
        rows.append({"r_kpc": r, "rho_msun_pc3": rp, "eps_c": eps, "cronos_term_kms2": 1.5 * C_KMS ** 2 * eps,
                     "sigma_r2_kms2": s2, "v_c2_kms2": G_KPC * M(r) / r, "q": 1.5 * C_KMS ** 2 * eps / s2, "D_F": DF})

    def f_q1(r, fac=1.0):
        return 1.5 * C_KMS ** 2 * fac * A * (rho(r) * 1e-9) ** 1.5 - sig2(r)
    r_CJ = brentq(f_q1, 0.2, 5.0)
    r_CJ_frac = {}
    for fac in amplitude_factors:
        try:
            r_CJ_frac[str(fac)] = float(brentq(lambda r: f_q1(r, fac), 0.02, 5.0))
        except ValueError:
            r_CJ_frac[str(fac)] = None
    eps_stop = (v_well_stop_kms / C_KMS) ** 2 / 2.0
    rho_stop = (eps_stop / A) ** (2.0 / 3.0)
    return {"M200": M200, "c": c, "H0": H0, "A": A, "r_s_kpc": rs, "r200_kpc": r200, "rows": rows,
            "r_CJ_kpc": float(r_CJ), "M_within_r_CJ": float(M(r_CJ)), "r_CJ_for_A_fraction": r_CJ_frac,
            "stop_rule": {"v_well_kms": v_well_stop_kms, "eps_c": eps_stop, "rho_msun_pc3": rho_stop,
                          "r_kpc_holding_M_within_r_CJ": float((3.0 * M(r_CJ) / (4.0 * np.pi * rho_stop * 1e9)) ** (1.0 / 3.0))}}


# --------------------------------------------------- (2) vecindad solar
def solar_neighbourhood_table(A: float = A_SCULPTOR, sigma_z_kms: float = 20.0,
                              z_pc=(0, 50, 100, 200, 300, 500, 800, 1100)) -> dict:
    """Modelo local de losa (McKee, Parravano & Hollenbach 2015):
    estrellas 0.043 M☉/pc³ (sech², h ≈ 300 pc), gas 0.041 (h ≈ 125 pc),
    materia oscura 0.013 (constante). Fuerza vertical de Cronos
    g_C = −c²dε_c/dz frente a K_z = 2πGΣ(<|z|); límite de Oort efectivo
    (1/4πG)·dg_C/dz|₀; cotas sobre A por g_C ≤ 0.1·K_z en 300 pc y por
    estabilidad (q < 1) en el plano."""
    def rho(z):
        return 0.043 / np.cosh(z / 600.0) ** 2 + 0.041 / np.cosh(z / 250.0) ** 2 + 0.013
    z = np.linspace(0.0, 1500.0, 15001)
    r = rho(z)
    eps = A * r ** 1.5
    gC = -C_KMS ** 2 * np.gradient(eps, z)                # (km/s)²/pc, hacia el plano
    Sigma = 2.0 * np.cumsum(r) * (z[1] - z[0])
    Kz = 2.0 * np.pi * G_PC * Sigma
    rows = []
    for zz in z_pc:
        i = int(np.searchsorted(z, zz))
        rows.append({"z_pc": zz, "rho_msun_pc3": float(r[i]), "c2_eps_c_kms2": float(C_KMS ** 2 * eps[i]),
                     "v_well_kms": float(np.sqrt(2.0 * C_KMS ** 2 * eps[i])),
                     "g_C_kms2_per_kpc": float(gC[i] * 1e3), "K_z_slab_kms2_per_kpc": float(Kz[i] * 1e3),
                     "g_C_over_K_z": float(gC[i] / max(Kz[i], 1e-30)),
                     "q": float(q_criterion(r[i], sigma_z_kms ** 2, A))})
    rho2 = -(2.0 * 0.043 / 600.0 ** 2 + 2.0 * 0.041 / 250.0 ** 2)      # ρ''(0)
    dgdz = -1.5 * C_KMS ** 2 * A * np.sqrt(r[0]) * rho2
    oort_cronos = dgdz / (4.0 * np.pi * G_PC)
    oort_stars_only = (-1.5 * C_KMS ** 2 * A * np.sqrt(r[0]) * (-2.0 * 0.043 / 600.0 ** 2)) / (4.0 * np.pi * G_PC)
    i300 = int(np.searchsorted(z, 300.0))
    A_bound_Kz = 0.1 * Kz[i300] / gC[i300] * A
    A_bound_stab = sigma_z_kms ** 2 / (1.5 * C_KMS ** 2 * r[0] ** 1.5)
    return {"A": A, "sigma_z_kms": sigma_z_kms, "Sigma_1p1kpc_model": float(np.interp(1100.0, z, Sigma)),
            "rows": rows, "oort_limit_cronos_msun_pc3": float(oort_cronos),
            "oort_limit_cronos_stars_only": float(oort_stars_only),
            "oort_observed_msun_pc3": {"value": 0.10, "err": 0.01, "source": "Holmberg & Flynn 2000; análisis Gaia 0.08–0.10 (referencia, no ingerida)"},
            "A_bound_gC_le_0p1_Kz_300pc": float(A_bound_Kz), "A_bound_over_A_sculptor": float(A_bound_Kz / A),
            "A_bound_stability_plane": float(A_bound_stab), "A_bound_stability_over_A_sculptor": float(A_bound_stab / A)}


# ------------------------------------------------------------ (3) Sculptor
def sculptor_profile_table(A: float | None = None, beta: float = 0.0,
                           radii_pc=(5, 50, 100, 150, 200, 260, 400, 600, 1000),
                           R_pc=(10, 30, 60, 100, 150, 200, 260, 350, 500, 700, 1000)) -> dict:
    """Con el montaje CONGELADO del 5E (Plummer, Υ fiducial, β = 0): q(r)
    y el perfil σ_los(R) newtoniano y newtoniano + Cronos a la amplitud
    única. El promedio pesado por luminosidad reproduce σ_obs por
    construcción (A_Sculptor se obtiene invirtiendo ese número); la
    FORMA del perfil es la predicción."""
    cfg = frozen_config()
    A = cfg.A_sculptor if A is None else A
    d = SCULPTOR
    a0 = plummer_scale_from_Rhalf(d["R_half_pc"])
    M = stellar_mass(d["L_V_Lsun"], UPSILON_SCULPTOR_FIDUCIAL)
    r = np.geomspace(0.05, 120.0 * a0, 800)
    nu = plummer_density(r, M, a0)
    gN, gE = plummer_g_newton(r, M, a0), g_eff_plummer(r, M, a0, A)
    s2N, s2E = sigma_r_sq_grid(r, nu, gN, beta=beta), sigma_r_sq_grid(r, nu, gE, beta=beta)
    eps = A * nu ** 1.5
    rows = []
    for rr in radii_pc:
        i = int(np.searchsorted(r, rr))
        rows.append({"r_pc": rr, "rho_star_msun_pc3": float(nu[i]), "cronos_term_kms2": float(1.5 * C_KMS ** 2 * eps[i]),
                     "sigma_r2_newton": float(s2N[i]), "sigma_r2_newton_cronos": float(s2E[i]),
                     "q": float(1.5 * C_KMS ** 2 * eps[i] / s2E[i]), "g_C_over_g_N": float((gE[i] - gN[i]) / gN[i])})
    R = np.asarray(R_pc, dtype=float)
    sN = np.sqrt(sigma_los_sq(R, r, nu, gN, beta=beta, u_max=120.0 * a0))
    sE = np.sqrt(sigma_los_sq(R, r, nu, gE, beta=beta, u_max=120.0 * a0))
    ok = (nu > 0.0) & (s2E > 0.0)
    q_arr = np.where(ok, 1.5 * C_KMS ** 2 * eps / np.where(ok, s2E, 1.0), 0.0)
    unstable = r[q_arr >= 1.0]
    return {"A": A, "beta": beta, "M_star": M, "upsilon": UPSILON_SCULPTOR_FIDUCIAL, "a0_pc": a0,
            "sigma_obs_kms": d["sigma_los_kms"], "rows": rows,
            "profile": [{"R_pc": float(x), "sigma_los_newton": float(a), "sigma_los_newton_cronos": float(b)}
                        for x, a, b in zip(R, sN, sE)],
            "r_unstable_max_pc": float(unstable.max()) if unstable.size else 0.0,
            "observed_reference": "≈ 9–10 km/s, plano hasta ≳ 1 kpc (Walker et al. 2007, 2009; Battaglia et al. 2008) — referencia, no ingerida",
            "shape_prediction": "pico central ≈ 2× lo observado y exterior newtoniano ÷3–4 (β = 0): la forma es la predicción congelada de jeans-dsph"}


def full_report(A: float = A_SCULPTOR) -> dict:
    return {"status": STATUS, "A_sculptor": A, "criterion": "q = (3/2)c²ε_c/σ_1D² ≥ 1 ⟹ inestable a toda k (tasa ∝ k)",
            "nfw_nivelA": nfw_jeans_table(A=A), "solar_neighbourhood": solar_neighbourhood_table(A=A),
            "sculptor_frozen_5E": sculptor_profile_table(A=A)}
