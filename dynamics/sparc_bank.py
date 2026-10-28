"""PR-S: el banco de mecanismos sobre SPARC con el brazo MCV (c) — brazos, molestias idénticas, reparto A/B, calibración
en A, letras sobre B frente a ΛCDM-2, puertas (cualificación E8-Q, tamaño de B, empalmes, borde de la rejilla, bytes) y
lecturas sin letra (bloques P0 del autor del 26 y del 27-oct-2026; sin datos, sin letra, sin congelar).

BRAZOS (documento del 26-oct, «El banco, con la MCV (c)»):

  ΛCDM-1    NFW libre: (M200, c) por galaxia, log M200 ∈ [9, 13.5] y log c ∈ [0, 2] planas            referencia
  ΛCDM-2    NFW con la c–M de Dutton y Macciò (2014, 0.11 dex): (M200, c) con esa previa              referencia
  MOND      g_obs = g_bar/(1 − e^{−√(g_bar/a₀)}) con a₀ = 1.2e-10 m/s² fijo                         referencia
  MCMC-C    bariones + Cronos local con A_S (el M1 congelado del 5E, disco exponencial y ζ = 0.15)   ley local
  MCMC-J62  Burkert con r_c = r⋆·(M200/1e11 M☉)^(1/3) (J.6.2, ec. 767), M200 por galaxia, r⋆ en A    control
  MCV (c)   Jeans isotermo con bariones y envoltura NFW (dynamics/mcv_sparc.py), σ_x/m(v) en A      condicional

y, solo para la cualificación E8-Q, Burkert libre (ρ_B, r_c) por galaxia. Molestias idénticas en todos: Υ_d y Υ_b
lognormales (0.5 y 0.7, 0.1 dex), D e i gaussianas con los errores del catálogo, i perfilada. Los brazos con halo y los
del MCMC llevan el término de la ECV, −(8π/3)Gρ_E·R² (en ΛCDM es el mismo término de Λ); MOND, no.

LETRAS sobre la mitad B, de la MCV (c) frente a ΛCDM-2 (mismas previas y mismo número de parámetros por galaxia, así que
ΔBIC = Δχ²_mín): A — mediana de Δχ²(MCV − ΛCDM-2) negativa con IC95 bootstrap entre galaxias que excluye 0 y más de la
mitad con Δχ² < 0; C — mediana positiva con IC95 que excluye 0 y menos de un tercio con Δχ² < 0; B — el resto;
INDETERMINADO — puertas. Orden: puertas, A, C, B. Ninguna letra es una afirmación sobre el tratado (E8).

ALCANCE DE CADA LETRA (enmienda 1 del 27-oct, sección «Qué prueba PR-S y qué no»): `LETTER_SCOPE`.

Estatuto: instrumento (E8), sin datos; ninguna cifra es señal (E13).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from dynamics import mcv_baryons as mb
from dynamics import mcv_sparc as ms
from dynamics.sculptor_transfer import frozen_config
from dynamics.sparc_crossfalsification import predict_cronos_fixed_A

A0_MOND_SI = 1.2e-10                                  # m/s²
KMS2_PER_KPC_SI = 1e6 / 3.0856775814913673e19         # 1 (km/s)²/kpc en m/s²
A0_MOND = A0_MOND_SI / KMS2_PER_KPC_SI                # (km/s)²/kpc
ECV_TERM = 8.0 / 3.0 * math.pi * mb.G_KPC * mb.RHO_E  # V² de la ECV: −ECV_TERM·R²

LETTER_SCOPE = {
    "MCV (c), letra sobre la mitad B": {
        "prueba": "que la MCV en equilibrio por autointercambio, con un σ_x/m(v) único para todas las galaxias, describe "
                  "los discos fuera de la muestra",
        "no_puede_decir": "nada sobre el intercambio con la ECV, el borde R₀, el punto de ruptura ni el origen de la MCV; "
                          "un modelo de materia oscura autointeractuante sin ontología daría la misma letra (Ren et al. "
                          "2019)",
        "A": "el freno por equilibrio con una sola ley de autointercambio basta para los discos",
        "C": "excluye esa ley de autointercambio con la forma declarada, no la MCV como energía atrapada ni el "
             "intercambio con la ECV",
        "lectura_fuerte_u_operativa": "ninguna letra decide entre la lectura fuerte y la operativa del fondo",
    },
    "MCMC-C (5E)": {"prueba": "la ley local ε_c = A_S·ρ^(3/2) con bariones solos y A congelada",
                    "no_puede_decir": "nada sobre la MCV (c); A_S ya está «excluida bajo regla congelada» por Oort–K_z, "
                                      "con bytes no verificados"},
    "Escalón de la RAR": {"prueba": "la forma saturante de ε_c y su densidad ρ*",
                          "no_puede_decir": "con ρ* = 0.01–0.03 la letra esperada es INDETERMINADO por no comprobable; "
                                            "no informa sobre la MCV"},
    "MCMC-J62": {"prueba": "el tratado escrito (Maestro J.6.2), con r⋆ libre",
                 "no_puede_decir": "no es la MCV (c): su pendiente 1/3 no sale del equilibrio"},
    "Lecturas sin letra": {"prueba": "universalidad de σ_x/m, proximidad al colapso (rama densa), escala r_c(M200), cruce "
                                     "con Sculptor, historia física acumulada del halo (memoria), lapse y q_CJ(0) "
                                     "cruzado",
                           "no_puede_decir": "ninguna da letra"},
}

BANK_SPEC = {
    "nfw": {"log_M200": [9.0, 13.5], "dlog_M200": 0.02, "log_c": [0.0, 2.0], "dlog_c": 0.01},
    "burkert": {"log_rho": [5.0, 11.0], "dlog_rho": 0.02, "log_rc": [-1.0, 2.0], "dlog_rc": 0.01},
    "j62": {"log_M200": [9.0, 13.5], "dlog_M200": 0.005, "log_rstar": [-0.5, 1.5], "dlog_rstar": 0.025},
    "mond_free": {"log_a0": [-11.0, -9.0], "dlog_a0": 0.005},
    "sigma100_grid": [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0],
    "a_grid": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    "calibration_delta_2lnL": 5.99,
    "letters": {"n_boot": 10000, "seed": 20261027, "n_B_min": 40, "max_frac_no_match": 0.10},
    "quality_cut": {"Q_max": 2, "inc_min_deg": 30.0, "n_points_min": 5},
    "split_salt": "PR-S",
    "e8q": {"chi2nu_rel_tol": 0.10, "frac_min": 0.90, "a0_ref": 1.13e-10, "a0_ref_err": 0.02e-10, "n_sigma": 2.0},
    "resolution": {"delta_chi2_max": 0.5, "frac_min": 0.95},
}


# ------------------------------------------------------------------ reparto A/B y submuestra de calidad
def quality_sample(galaxies: list, spec: dict = BANK_SPEC) -> list:
    """La submuestra de calidad del 5E: Q ≤ 2, i ≥ 30° y al menos 5 puntos (metadatos del catálogo, nunca residuos)."""
    q = spec["quality_cut"]
    return [g for g in galaxies if (g.quality is None or g.quality <= q["Q_max"]) and g.inc >= q["inc_min_deg"]
            and np.asarray(g.R).size >= q["n_points_min"]]


def split_halves(names: list[str], mode: str = "sha256", salt: str = "PR-S", seed: int | None = None) -> dict:
    """Mitades A y B fijadas antes de los bytes (decisión 3 del 26-oct, pendiente): «sha256» ordena por
    sha256(identificador + salt) y alterna A, B, A, …; «semilla» baraja con una semilla congelada y alterna."""
    if mode == "sha256":
        order = sorted(names, key=lambda n: hashlib.sha256((n + salt).encode("utf-8")).hexdigest())
    elif mode == "semilla":
        if seed is None:
            raise ValueError("el reparto aleatorio exige una semilla congelada")
        order = [names[i] for i in np.random.default_rng(seed).permutation(len(names))]
    else:
        raise ValueError(f"reparto desconocido: {mode!r}")
    return {"A": order[0::2], "B": order[1::2]}


# ------------------------------------------------------------------ modelos de los brazos rápidos (V² total)
def nfw_v2(log_M200, log_c, R):
    """V² de una NFW (M200 a 200·ρ_crit) en R (arrays (n,) y (n, n_datos) o (n_datos,))."""
    M200, c = 10.0 ** np.asarray(log_M200, float), 10.0 ** np.asarray(log_c, float)
    r200 = (3.0 * M200 / (4.0 * math.pi * 200.0 * mb.RHO_CRIT)) ** (1.0 / 3.0)
    r_s = r200 / c
    M = M200[:, None] * mb.nfw_m(R / r_s[:, None]) / mb.nfw_m(c)[:, None]
    return mb.G_KPC * M / R


def burkert_mass(r, rho_B, r_c):
    """M(<r) de un Burkert ρ_B·r_c³/[(r + r_c)(r² + r_c²)]: πρ_B·r_c³·[ln(1 + x²) + 2ln(1 + x) − 2 arctan x]."""
    x = np.asarray(r, float) / r_c
    return math.pi * rho_B * r_c ** 3 * (np.log1p(x * x) + 2.0 * np.log1p(x) - 2.0 * np.arctan(x))


def burkert_v2(log_rho, log_rc, R):
    rho_B, r_c = 10.0 ** np.asarray(log_rho, float)[:, None], 10.0 ** np.asarray(log_rc, float)[:, None]
    return mb.G_KPC * burkert_mass(R, rho_B, r_c) / R


def j62_v2(log_M200, r_star, R):
    """Burkert de J.6.2: r_c = r⋆·(M200/1e11)^(1/3) y la normalización con M(<r200) = M200."""
    M200 = 10.0 ** np.asarray(log_M200, float)
    r200 = (3.0 * M200 / (4.0 * math.pi * 200.0 * mb.RHO_CRIT)) ** (1.0 / 3.0)
    r_c = r_star * (M200 / 1e11) ** (1.0 / 3.0)
    rho_B = M200 / burkert_mass(r200, 1.0, r_c)
    return mb.G_KPC * burkert_mass(R, rho_B[:, None], r_c[:, None]) / R


def mond_v2(v2_bar, R, a0: float = A0_MOND):
    """g_obs = g_bar/(1 − e^{−√(g_bar/a₀)}) (la función de McGaugh, Lelli y Schombert 2016), V² = R·g_obs."""
    g = np.maximum(np.asarray(v2_bar, float), 0.0) / R
    with np.errstate(invalid="ignore", divide="ignore"):
        nu = np.where(g > 0.0, 1.0 / -np.expm1(-np.sqrt(g / a0)), 0.0)
    return R * g * nu


def cronos_v2(gal, cfg: dict, R):
    """V_cronos² del M1 congelado del 5E: disco exponencial con Σ₀ = Υ_d·SB_disk(0), R_d y ζ = 0.15, A = A_S."""
    if gal.SB_disk0 is None or gal.R_d is None:
        raise ValueError(f"{gal.name}: sin R_d o brillo central del disco no hay M1 del 5E")
    ratio = cfg["D"] / gal.D
    return predict_cronos_fixed_A(1e3 * np.asarray(R, float), cfg["ups_disk"] * gal.SB_disk0, 1e3 * gal.R_d * ratio,
                                  frozen_config())


ARMS = {
    "ΛCDM-1": {"params": ("nfw", 2), "prior": "plana", "ecv": True},
    "ΛCDM-2": {"params": ("nfw", 2), "prior": "c–M", "ecv": True},
    "MOND": {"params": (None, 0), "prior": None, "ecv": False},
    "MCMC-C": {"params": (None, 0), "prior": None, "ecv": True},
    "MCMC-J62": {"params": ("j62", 1), "prior": "plana", "ecv": True},
    "Burkert libre (E8-Q)": {"params": ("burkert", 2), "prior": "plana", "ecv": True},
    "MOND a₀ libre": {"params": ("mond_free", 1), "prior": "plana", "ecv": False},
}


def _halo_grid(kind: str, spec: dict = BANK_SPEC):
    if kind == "nfw":
        g = spec["nfw"]
        a = np.arange(g["log_M200"][0], g["log_M200"][1] + 1e-9, g["dlog_M200"])
        b = np.arange(g["log_c"][0], g["log_c"][1] + 1e-9, g["dlog_c"])
    elif kind == "burkert":
        g = spec["burkert"]
        a = np.arange(g["log_rho"][0], g["log_rho"][1] + 1e-9, g["dlog_rho"])
        b = np.arange(g["log_rc"][0], g["log_rc"][1] + 1e-9, g["dlog_rc"])
    elif kind == "j62":
        g = spec["j62"]
        return (np.arange(g["log_M200"][0], g["log_M200"][1] + 1e-9, g["dlog_M200"]),)
    elif kind == "mond_free":
        g = spec["mond_free"]
        return (np.arange(g["log_a0"][0], g["log_a0"][1] + 1e-9, g["dlog_a0"]),)
    else:
        return ()
    A, B = np.meshgrid(a, b, indexing="ij")
    return A.ravel(), B.ravel()


def _halo_steps(kind: str, spec: dict = BANK_SPEC):
    return {"nfw": (spec["nfw"]["dlog_M200"], spec["nfw"]["dlog_c"]),
            "burkert": (spec["burkert"]["dlog_rho"], spec["burkert"]["dlog_rc"]),
            "j62": (spec["j62"]["dlog_M200"],), "mond_free": (spec["mond_free"]["dlog_a0"],)}.get(kind, ())


def arm_v2(arm: str, P: tuple, R, v2_bar, gal, cfg: dict, global_param=None):
    """V_mod² (n, n_datos) de un brazo rápido para los parámetros de halo P (arrays (n,)) y unos bariones."""
    kind = ARMS[arm]["params"][0]
    n = P[0].size if P else 1
    R2 = np.broadcast_to(np.asarray(R, float), (n, np.shape(R)[-1]))
    vb = np.broadcast_to(np.asarray(v2_bar, float), (n, np.shape(R)[-1]))
    if arm == "MOND":
        v2 = mond_v2(vb, R2)
    elif arm == "MOND a₀ libre":
        v2 = mond_v2(vb, R2, (10.0 ** P[0])[:, None] / KMS2_PER_KPC_SI)
    elif arm == "MCMC-C":
        v2 = vb + cronos_v2(gal, cfg, np.asarray(R, float))
    elif kind == "nfw":
        v2 = vb + nfw_v2(P[0], P[1], R2)
    elif kind == "burkert":
        v2 = vb + burkert_v2(P[0], P[1], R2)
    elif kind == "j62":
        v2 = vb + j62_v2(P[0], global_param, R2)
    else:
        raise ValueError(arm)
    return v2 - ECV_TERM * R2 ** 2 if ARMS[arm]["ecv"] else v2


def arm_prior(arm: str, P: tuple) -> np.ndarray:
    if ARMS[arm]["prior"] == "c–M":
        return ms.c_prior_chi2(10.0 ** P[0], 10.0 ** P[1])
    return np.zeros(P[0].size if P else 1)


def _evaluate(gal, arm: str, P: tuple, cfgs: list[dict], global_param, spec_mcv: dict) -> dict:
    """χ² total de todas las combinaciones (configuración de molestias × parámetros de halo)."""
    rows = []
    for cfg in cfgs:
        R, v2_bar, n_neg, _ = ms.galaxy_baryons(gal, cfg)
        v2 = arm_v2(arm, P, R, v2_bar, gal, cfg, global_param)
        inc = spec_mcv["inclination"]
        chi2_data, i_best = ms.profile_inclination(gal.v_obs, gal.e_v, gal.inc, gal.e_inc, v2, inc["k_range"],
                                                   inc["n_grid"], inc["n_newton"])
        total = chi2_data + arm_prior(arm, P) + cfg["prior"]
        j = int(np.argmin(total))
        rows.append({"chi2": float(total[j]), "chi2_data": float(chi2_data[j]), "inc": float(i_best[j]),
                     "params": tuple(float(p[j]) for p in P), "cfg": dict(cfg), "n_negative_vbar2": n_neg})
    return min(rows, key=lambda r: r["chi2"])


def fit_arm(gal, arm: str, global_param=None, spec: dict = BANK_SPEC, spec_mcv: dict = ms.FIT_SPEC,
            zoom: bool | None = None) -> dict:
    """Ajuste de un brazo rápido a una galaxia: rejilla de halo × rejilla de molestias (las mismas que la MCV (c)),
    con i perfilada, y el mismo refinado local que la MCV (c) si zoom."""
    zoom = spec_mcv["zoom"]["enabled"] if zoom is None else zoom
    kind, k_halo = ARMS[arm]["params"]
    P = _halo_grid(kind, spec)
    best = _evaluate(gal, arm, P, ms.nuisance_configs(gal, spec_mcv), global_param, spec_mcv)
    best["stage"] = "rejilla"
    if zoom:
        best = _zoom_arm(gal, arm, best, global_param, spec, spec_mcv)
    best["k_params"] = k_halo + _k_nuisance(gal)
    best["arm"] = arm
    return best


def _k_nuisance(gal) -> int:
    """Molestias ajustadas por galaxia: Υ_d, D, i y Υ_b si hay bulbo."""
    return 3 + (1 if gal.has_bulge else 0)


def _zoom_arm(gal, arm: str, fit: dict, global_param, spec: dict, spec_mcv: dict) -> dict:
    z = spec_mcv["zoom"]
    kind, _ = ARMS[arm]["params"]
    steps = list(_halo_steps(kind, spec))
    d_k = z["dk0"]
    cur = dict(fit)
    for _ in range(z["max_levels"]):
        steps = [s / 2.0 for s in steps]
        d_k /= 2.0
        c0 = cur["cfg"]
        ks = [c0["k_disk"] + d_k * o for o in (-1.0, 0.0, 1.0)]
        kb = [c0["k_bul"] + d_k * o for o in (-1.0, 0.0, 1.0)] if gal.has_bulge else [0.0]
        kD = [c0["k_D"] + d_k * o for o in (-1.0, 0.0, 1.0)]
        cfgs = [ms.nuisance_point(gal, a, b, d, spec_mcv) for a in ks for b in kb for d in kD
                if max(abs(a), abs(b), abs(d)) <= z["k_abs_max"]]
        cfgs = [c for c in cfgs if c is not None]
        if steps:
            axes = [p + s * np.arange(-2.0, 3.0) for p, s in zip(cur["params"], steps)]
            P = tuple(m.ravel() for m in np.meshgrid(*axes, indexing="ij"))
        else:
            P = ()
        cand = _evaluate(gal, arm, P, cfgs, global_param, spec_mcv)
        if cand["chi2"] < cur["chi2"]:
            cur = cand
        if (not steps or max(steps) < z["tol_dex"]) and d_k < z["tol_k"]:
            break
    cur["stage"] = "rejilla + refinado"
    return cur


def bic(chi2_min: float, k: int, n_points: int) -> float:
    return chi2_min + k * math.log(n_points)


# ------------------------------------------------------------------ calibración en A y letras sobre B
def calibrate(chi2_by_setting: dict, delta: float = BANK_SPEC["calibration_delta_2lnL"]) -> dict:
    """Calibración global: el punto de la rejilla que minimiza Σ_galaxias χ²_mín (= maximiza Σ máx ln L) y la región del
    95 % (Δ(2 ln L) = ΔΣχ² ≤ 5.99). chi2_by_setting: {clave del ajuste global: [χ²_mín por galaxia]}."""
    tot = {k: float(np.sum(v)) for k, v in chi2_by_setting.items() if np.all(np.isfinite(v))}
    if not tot:
        return {"best": None, "region": [], "totals": {}}
    best = min(tot, key=tot.get)
    return {"best": best, "region": sorted(k for k, v in tot.items() if v - tot[best] <= delta), "totals": tot}


def on_grid_edge(setting: tuple, spec: dict = BANK_SPEC) -> bool:
    s100, a = setting
    return s100 in (spec["sigma100_grid"][0], spec["sigma100_grid"][-1]) or a in (spec["a_grid"][0], spec["a_grid"][-1])


def bootstrap_median(delta_chi2, n_boot: int, seed: int) -> dict:
    d = np.asarray(delta_chi2, float)
    rng = np.random.default_rng(seed)
    med = np.median(d[rng.integers(0, d.size, size=(n_boot, d.size))], axis=1)
    return {"median": float(np.median(d)), "ci95": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
            "frac_negative": float(np.mean(d < 0.0)), "n": int(d.size)}


def letters_mcv_vs_lcdm2(delta_chi2_B, gates: dict, spec: dict = BANK_SPEC) -> dict:
    """La letra de la MCV (c) sobre la mitad B. gates: {"cualificacion_E8Q": bool, "bytes_con_manifiesto": bool,
    "frac_B_sin_empalme": float, "calibracion_en_borde": bool}. Orden: puertas, A, C, B."""
    L = spec["letters"]
    d = np.asarray(delta_chi2_B, float)
    reasons = []
    if not gates.get("bytes_con_manifiesto", False):
        reasons.append("bytes sin manifiesto")
    if not gates.get("cualificacion_E8Q", False):
        reasons.append("cualificación del ajustador (E8-Q) no superada")
    if d.size < L["n_B_min"]:
        reasons.append(f"menos de {L['n_B_min']} galaxias en B ({d.size})")
    if gates.get("frac_B_sin_empalme", 1.0) > L["max_frac_no_match"]:
        reasons.append(f"más del {100 * L['max_frac_no_match']:.0f} % de B sin ningún empalme válido en la rejilla")
    if gates.get("calibracion_en_borde", True):
        reasons.append("calibración en el borde de la rejilla")
    if reasons:
        return {"letter": "INDETERMINADO", "reasons": reasons}
    s = bootstrap_median(d, L["n_boot"], L["seed"])
    lo, hi = s["ci95"]
    if s["median"] < 0.0 and hi < 0.0 and s["frac_negative"] > 0.5:
        return {"letter": "A", **s}
    if s["median"] > 0.0 and lo > 0.0 and s["frac_negative"] < 1.0 / 3.0:
        return {"letter": "C", **s}
    return {"letter": "B", **s}


# ------------------------------------------------------------------ cualificación E8-Q y resolución
def e8q_gate(our_chi2nu: dict, li2020_chi2nu: dict | None, a0_population: tuple | None,
             spec: dict = BANK_SPEC) -> dict:
    """E8-Q, antes de abrir cualquier brazo MCMC: con las previas de Li+2020, sus ajustes NFW y Burkert se reproducen
    con χ²_ν a un 10 % en al menos el 90 % de las galaxias; y la a₀ poblacional sale a 2σ de 1.13 ± 0.02 (×1e-10 m/s²)
    con el montaje de 2610.05871. Sin las tablas de Li+2020 o sin la a₀ poblacional, falla cerrado.

    our_chi2nu / li2020_chi2nu: {modelo: {galaxia: χ²_ν}}; a0_population: (a₀, σ) en m/s²."""
    q = spec["e8q"]
    if li2020_chi2nu is None or a0_population is None:
        return {"pass": False, "reason": "sin las tablas de Li+2020 o sin la a₀ poblacional: E8-Q falla cerrado"}
    per = {}
    for model in ("NFW", "Burkert"):
        ours, ref = our_chi2nu.get(model, {}), li2020_chi2nu.get(model, {})
        common = [g for g in ref if g in ours]
        ok = [abs(ours[g] / ref[g] - 1.0) <= q["chi2nu_rel_tol"] for g in common]
        per[model] = {"n": len(common), "frac_within": float(np.mean(ok)) if ok else 0.0}
    a0, err = a0_population
    pull = abs(a0 - q["a0_ref"]) / math.hypot(err, q["a0_ref_err"])
    passed = all(v["n"] > 0 and v["frac_within"] >= q["frac_min"] for v in per.values()) and pull <= q["n_sigma"]
    return {"pass": bool(passed), "models": per, "a0_pull_sigma": pull}


def resolution_check(chi2_grid: dict, chi2_doubled: dict, spec: dict = BANK_SPEC) -> dict:
    """La rejilla (ρ₀, σ₀) vale si duplicarla mueve χ²_mín menos de 0.5 en el 95 % de las galaxias (sin letra)."""
    r = spec["resolution"]
    d = np.array([abs(chi2_doubled[g] - chi2_grid[g]) for g in chi2_grid])
    frac = float(np.mean(d < r["delta_chi2_max"])) if d.size else 0.0
    return {"pass": frac >= r["frac_min"], "frac_within": frac, "max_shift": float(d.max()) if d.size else None}


# ------------------------------------------------------------------ lecturas sin letra
def globality(cal_A: dict, cal_B: dict) -> dict:
    """¿Cae la calibración hecha solo con B dentro de la región del 95 % de A?"""
    return {"calibracion_A": cal_A["best"], "calibracion_B": cal_B["best"], "B_dentro_region_A": cal_B["best"] in cal_A["region"]}


def sculptor_cross_reading(setting: tuple, sigma_los_kms: float = 9.2) -> dict:
    """SPARC → Sculptor: σ_x/m(v) calibrado, evaluado en el ⟨v_rel⟩ = 4σ/√π de Sculptor (≈ 21 km/s con σ = 9.2 km/s);
    se compara con el intervalo de σ_x/m con el que la preinscripción de Sculptor da A, el día que las dos tengan bytes."""
    s100, a = setting
    v = ms.mean_relative_speed(sigma_los_kms)
    som = mb.sigma_over_m_power(s100, a)
    return {"v_rel_kms": v, "sigma_over_m_at_sculptor": float(som(v)) if callable(som) else float(som)}


def pairwise_preferences(bics: dict) -> dict:
    """Recuento de preferencias por pares: {(brazo1, brazo2): galaxias con BIC1 < BIC2}. bics: {brazo: {galaxia: BIC}}."""
    arms = sorted(bics)
    out = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            common = [g for g in bics[a] if g in bics[b]]
            out[f"{a} < {b}"] = int(sum(bics[a][g] < bics[b][g] for g in common))
            out[f"{b} < {a}"] = int(sum(bics[b][g] < bics[a][g] for g in common))
    return out
