"""La MCV (c) en Sculptor (MCV-b, bloque P0 del autor del 23-oct): σ_los(R) de las estrellas de Plummer en el potencial
de la MCV en equilibrio y de las propias estrellas, frente al perfil binado de PR-W.

Modelo. La MCV es el Jeans isotermo con autointercambio empalmado con una envoltura NFW (M200, c) de molestia
(dynamics/mcv_equilibrium.py, con los bariones de Plummer dentro de Φ y ρ_E = 0). Las estrellas son el trazador de
Plummer del montaje congelado del 5E (a = R_half = 260 pc, L_V = 2.6e6 L☉) con M⋆ = Υ⋆·L_V, Υ⋆ ∈ {1, 2, 3}. σ_los(R)
sale de Jeans con β constante (el brazo de la ley local, −0.5, 0, +0.3) y la proyección de dynamics/jeans.py.

Banco. Para cada punto de la rejilla congelada (σ_x/m, Υ⋆, log10 M200, k) con c = c200_DM14(M200)·10^(0.11·k), todas
las ramas del empalme dentro del dominio de la rejilla del empalme y con r₁ < r200; se calcula ANTES de los bytes y su
sha256 queda en la preinscripción. Con los bytes solo se proyecta σ_los en los R de los bins y se cuenta χ².

Letras (preinscripción; «cabe» = p(χ²_mín, ν = N_bins − 2) ≥ p_min, σ_x/m fijo en cada valor de la rejilla):
A cabe con σ_x/m ≤ 10, envoltura en las previas y rama con núcleo; B no A, pero cabe con algún σ_x/m ≤ 50 (rama densa,
envoltura fuera de las previas o 10 < σ_x/m ≤ 50); C no cabe con ningún σ_x/m ≤ 50 en la caja; INDETERMINADO, puertas.

Estatuto: instrumento (E8), sin datos; ninguna letra es afirmación sobre el tratado (E13).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from dynamics.jeans import sigma_los_sq
from dynamics.mcv_equilibrium import (
    BETA_COLLAPSE,
    dm_profile,
    gravothermal_collapse_time,
    jeans_match,
    nfw_halo,
    plummer_mass,
)
from dynamics.weak_field import G_PC

_EDGE_TOL = 1e-9


def c200_dm14(M200: float, rel: dict) -> float:
    """Mediana de c200 a z = 0 de Dutton y Macciò 2014 (log10 c = a + b·log10[M200/(10¹² h⁻¹ M☉)]), extrapolada por
    debajo de su rango simulado (declarado en la preinscripción)."""
    return 10.0 ** (rel["a"] + rel["b"] * math.log10(M200 * rel["h"] / 1e12))


def envelope(logM: float, k: float, rel: dict) -> tuple[float, float]:
    M200 = 10.0 ** logM
    return M200, c200_dm14(M200, rel) * 10.0 ** (k * rel["scatter_dex"])


def envelope_grid(spec: dict) -> list[tuple[float, float]]:
    """Rejilla congelada de la envoltura: fina dentro de las previas, gruesa en la caja (unión, sin duplicados)."""
    pts = set()
    for g in (spec["grid_prior"], spec["grid_box"]):
        lm = np.round(np.arange(g["logM"][0], g["logM"][1] + 1e-9, g["dlogM"]), 6)
        ks = np.round(np.arange(-g["k_abs_max"], g["k_abs_max"] + 1e-9, g["dk"]), 6)
        pts.update((float(a), float(b)) for a in lm for b in ks)
    return sorted(pts)


def in_prior(logM: float, k: float, spec: dict) -> bool:
    p = spec["prior"]
    return p["logM"][0] - _EDGE_TOL <= logM <= p["logM"][1] + _EDGE_TOL and abs(k) <= p["k_abs_max"] + _EDGE_TOL


def on_box_edge(logM: float, k: float, spec: dict) -> bool:
    g = spec["grid_box"]
    return min(abs(logM - g["logM"][0]), abs(logM - g["logM"][1])) < _EDGE_TOL or abs(abs(k) - g["k_abs_max"]) < _EDGE_TOL


def bank_point(sigma_over_m: float, upsilon: float, logM: float, k: float, spec: dict, prefilter: bool = True) -> dict:
    """Todas las ramas del empalme en un punto de la rejilla, dentro del dominio (σ₀ y u en la rejilla del empalme) y con
    r₁ < r200; las descartadas se cuentan."""
    M200, c = envelope(logM, k, spec["c_relation"])
    pl = {"M": upsilon * spec["L_V_Lsun"], "a": spec["plummer_a_kpc"]}
    d = spec["domain"]
    m = jeans_match(M200, c, sigma_over_m, t_age=spec["t_age_gyr"], H0=spec["H0"], plummer=pl,
                    sigma0_range=tuple(d["sigma0_range"]), u_range=(1.0 + 1e-6, d["u_max"]), prefilter=prefilter)
    keep, out_domain, beyond = [], 0, 0
    for b in m["branches"]:
        if b["u"] > d["u_max"] or not d["sigma0_range"][0] <= b["sigma0"] <= d["sigma0_range"][1]:
            out_domain += 1
        elif b["r1"] >= m["halo"]["r200"]:
            beyond += 1
        else:
            keep.append([b["sigma0"], b["u"], b["rho0"], b["r1"], b["branch"]])
    return {"som": sigma_over_m, "ups": upsilon, "logM": logM, "k": k, "M200": M200, "c": c, "branches": keep,
            "dropped_outside_domain": out_domain, "dropped_r1_beyond_r200": beyond}


def sigma_los_model(point: dict, branch: list, betas, R_pc, spec: dict) -> dict:
    """σ_los(R) [km/s] de las estrellas de Plummer en el potencial MCV + estrellas, para cada β pedido."""
    a_pc = 1000.0 * spec["plummer_a_kpc"]
    M_star = point["ups"] * spec["L_V_Lsun"]
    match = {"halo": nfw_halo(point["M200"], point["c"], spec["H0"]), "plummer": {"M": M_star, "a": spec["plummer_a_kpc"]},
             "rho_E": 0.0}
    br = {"sigma0": branch[0], "rho0": branch[2], "r1": branch[3]}
    r = np.geomspace(0.05, spec["jeans"]["r_max_over_a"] * a_pc, spec["jeans"]["n_r"])
    M = dm_profile(match, br, r / 1000.0)["M_M"] + plummer_mass(r / 1000.0, M_star, spec["plummer_a_kpc"])
    g = G_PC * M / r ** 2
    nu = (1.0 + (r / a_pc) ** 2) ** -2.5
    R = np.asarray(R_pc, dtype=float)
    return {float(b): np.sqrt(sigma_los_sq(R, r, nu, g, beta=float(b), u_max=spec["jeans"]["r_max_over_a"] * a_pc)) for b in betas}


def chi2_rows(bank: list[dict], bins: list[dict], spec: dict) -> list[dict]:
    """χ² de cada (punto, rama, β) frente a los bins (σ_obs, σ_err en R_bin)."""
    R = [b["R_pc"] for b in bins]
    sig = np.array([b["sigma_kms"] for b in bins])
    err = np.array([b["sigma_err_kms"] for b in bins])
    rows = []
    for p in bank:
        for br in p["branches"]:
            for beta, s in sigma_los_model(p, br, spec["beta_arm"], R, spec).items():
                rows.append({"som": p["som"], "ups": p["ups"], "logM": p["logM"], "k": p["k"], "branch": br[4], "beta": beta,
                             "chi2": float(np.sum(((s - sig) / err) ** 2)), "in_prior": in_prior(p["logM"], p["k"], spec)})
    return rows


def letters(rows: list[dict], n_bins: int, spec: dict) -> dict:
    """Letra de la preinscripción sobre las filas de χ² (puertas de datos y de reproducción, fuera de aquí)."""
    nu = n_bins - spec["fit"]["n_fitted"]
    if nu < spec["fit"]["nu_min"]:
        return {"letter": "INDETERMINADO", "reason": f"ν = N_bins − {spec['fit']['n_fitted']} = {nu} < {spec['fit']['nu_min']}"}
    p_min = spec["fit"]["p_min"]
    per = {}
    for som in spec["sigma_over_m_grid"]:
        sel = [r for r in rows if r["som"] == som]
        a_rows = [r for r in sel if r["branch"] == "con núcleo" and r["in_prior"]]
        best_a = min(a_rows, key=lambda r: r["chi2"]) if a_rows else None
        best_any = min(sel, key=lambda r: r["chi2"]) if sel else None
        per[str(som)] = {"best_A": best_a, "p_A": float(chi2_distribution.sf(best_a["chi2"], nu)) if best_a else 0.0,
                         "best_any": best_any, "p_any": float(chi2_distribution.sf(best_any["chi2"], nu)) if best_any else 0.0}
    fits_a = [float(s) for s, v in per.items() if float(s) <= spec["sigma_over_m_A_max"] and v["p_A"] >= p_min]
    fits_any = [float(s) for s, v in per.items() if float(s) <= spec["sigma_over_m_C_max"] and v["p_any"] >= p_min]
    out = {"nu": nu, "per_sigma_over_m": per, "fits_A_sigma_over_m": fits_a, "fits_any_sigma_over_m": fits_any}
    if fits_a:
        return {**out, "letter": "A", "reason": f"cabe con rama con núcleo y envoltura en las previas para σ_x/m ∈ {fits_a} cm²/g"}
    if fits_any:
        return {**out, "letter": "B", "reason": f"solo cabe fuera del régimen de A (σ_x/m ∈ {fits_any} cm²/g)"}
    cands = [v["best_any"] for s, v in per.items() if float(s) <= spec["sigma_over_m_C_max"] and v["best_any"]]
    best = min(cands, key=lambda r: r["chi2"]) if cands else None
    if best is not None and on_box_edge(best["logM"], best["k"], spec):
        return {**out, "letter": "INDETERMINADO", "reason": "el mejor ajuste sin restricciones cae en el borde de la caja: "
                                                         "la caja no basta para afirmar C"}
    return {**out, "letter": "C", "reason": f"no cabe con ningún σ_x/m ≤ {spec['sigma_over_m_C_max']} cm²/g en la caja"}


def gravothermal_reference(row: dict, spec: dict) -> dict:
    """t_c gravotérmico (β = 0.45 y 0.75) de la envoltura de un ajuste, frente a t_edad: sin letra."""
    M200, c = envelope(row["logM"], row["k"], spec["c_relation"])
    h = nfw_halo(M200, c, spec["H0"])
    return {"t_c_gyr": {str(b): gravothermal_collapse_time(row["som"], h["rho_s"], h["r_s"], beta=b) for b in BETA_COLLAPSE},
            "t_age_gyr": spec["t_age_gyr"]}
