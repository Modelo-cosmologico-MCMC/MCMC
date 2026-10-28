"""PR-S: la MCV (c) en las curvas de rotación de SPARC — bariones tabulados, núcleo isotermo con la ECV, mapa inverso
(ρ₀, σ₀) → (M200, c, r₁) que cubre todas las ramas y ajuste por galaxia sobre una rejilla congelada, con las lecturas sin
letra de las enmiendas del 27-oct (bloques P0 del autor del 26 y del 27-oct-2026; sin datos, sin letra, sin congelar).

MODELO (documento del 26-oct, «El brazo MCV (c) para SPARC»). Es el Jeans isotermo de dynamics/mcv_equilibrium.py (#55)
llevado a discos, con dos cambios: los bariones salen de las componentes de SPARC y σ_x/m se calibra con media muestra.

  - Bariones. V_bar² = V_gas·|V_gas| + Υ_d·V_disk·|V_disk| + Υ_b·V_bul·|V_bul| (componentes con Υ = 1), reescalada con
    la distancia (radios ∝ D, V_bar² ∝ D). Donde V_bar² < 0 (huecos de gas) se pone a cero y se cuenta. Masa esférica
    equivalente M_b(r) = r·V_bar²(r)/G: V_bar² lineal en r entre puntos, M_b ∝ r³ por dentro del primero y constante
    más allá del último (el Jeans isotermo es esférico; el disco entra como en Kaplinghat, Tulin y Yu y en Ren et al.).
  - MCV. ρ_M = ρ₀·exp[−(Φ − Φ₀)/σ₀²] dentro de r₁, con los bariones y la ECV (−2ρ_E) dentro de Φ; r₁ por Γ_x(r₁)·t_edad
    = 1; envoltura NFW empalmada en r₁ con ρ y M continuos.
  - Mapa inverso. Para cada (ρ₀, σ₀) se integra el núcleo hacia fuera, r₁ es donde ρ_M baja a ρ₁ = 1/[(σ_x/m)·⟨v_rel⟩·
    t_edad] y la NFW que empalma sale de dos raíces: x₁ = r₁/r_s de 3M₁/(4πr₁³ρ₁) = 3m(x)(1 + x)²/x² y c de ρ_s. Cada
    (ρ₀, σ₀) da como mucho una envoltura; una envoltura puede venir de varios (ρ₀, σ₀), que son sus ramas. Recorrer
    (ρ₀, σ₀) cubre todas las ramas a la vez; la rama del mejor ajuste se clasifica buscando las demás preimágenes de su
    envoltura (`classify_branch`).
  - El núcleo no depende de σ_x/m ni de t_edad: solo dónde cae r₁. Por eso se integra una vez por bariones y se reutiliza
    para toda la rejilla global de σ_x/m(v) y para cualquier t_edad.

DÓNDE ESTÁ CADA COSA. La física general (bariones, núcleo, mapa inverso, empalme directo, rejilla vectorizada, edad desde
la concentración) está en dynamics/mcv_baryons.py; aquí, el ajuste a las galaxias y las lecturas. Este fichero cae bajo
el candado anti-ajuste del 5E (tests/test_front5e_lock.py vigila dynamics/*sparc*.py y prohíbe la maquinaria de
optimización): el ajuste es una búsqueda en rejilla, como pide el documento del 26-oct, con un refinado local también en
rejillas, y A_S no se ajusta en ningún brazo.

σ_x/m ES UN PARÁMETRO EFECTIVO (enmienda 1 del 27-oct): mide la tasa de autointercambio Γ_x de la energía sellada, no la
sección eficaz de una partícula nueva («masa ínfima pero no nula, sin partículas nuevas»). Su forma global es
σ_x/m(v) = σ₁₀₀·(⟨v_rel⟩/100 km/s)^(−a), con ⟨v_rel⟩ = 4σ₀/√π de cada galaxia; a = 0 es el caso constante.

LECTURAS SIN LETRA (enmiendas del 27-oct): memoria del halo (t_edad desde la concentración, mcv_baryons), rama
densa y colapso (t_c por galaxia y la variante que solo admite la rama densa con t_c ≤ t_edad), la lapse (ΔN(0) y σ₀),
r_c(M200) frente a sus referencias del corpus y el q₀ cruzado con la ley local. La cúspide del agujero negro queda fuera
y declarada: con las combinaciones de C2, r_h ≤ 43 pc, por debajo de los radios de SPARC.

Estatuto: instrumento (E8), sin datos; ninguna cifra es señal (E13). Unidades: kpc, M☉, km/s, Gyr; σ_x/m en cm²/g;
densidades en M☉/kpc³ salvo donde se indica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynamics.mcv_baryons import (
    C_KMS,
    G_KPC,
    RHO_E,
    T_AGE_FIXED,
    CoreTable,
    TabulatedBaryons,
    baryon_nodes,
    burkert_half_density_x,
    concentration_age_maps,
    core_table,
    integrate_cores,
    map_table,
    model_v2_at_data,
    preimage_cells,
    radial_nodes,
    refine_preimage,
    sigma_over_m_power,
)
from dynamics.mcv_equilibrium import (
    bh_influence_radius,
    gravothermal_collapse_time,
    mean_relative_speed,
)

DM14 = {"a": 0.905, "b": -0.101, "h": 0.674, "scatter_dex": 0.11}  # c200(M200) de Dutton y Macciò (2014) a z = 0
BETAS = (0.75, 0.45)                        # β de Essig et al. (2019) en t_c (enmienda 3)

# Cúspide del agujero negro (enmienda 6): r_h = G·M_BH/σ₀² con las combinaciones de la comprobación C2 del 27-oct.
BH_C2_COMBINATIONS = (((1e5, 1e6), (20.0, 50.0)), ((1e7, 1e8), (100.0, 150.0)))
BH_OUT_OF_SCOPE = ("la cúspide de MCV junto al agujero negro (ec. 5 del bloque) no entra en PR-S: con M_BH de 1e5–1e6 M☉ "
                   "y σ₀ = 20–50 km/s, r_h = 0.2–11 pc; con 1e7–1e8 M☉ y σ₀ = 100–150 km/s, r_h = 2–43 pc; como mucho "
                   "43 pc, por debajo de los radios de SPARC")

# Referencias de la lectura r_c(M200) (enmienda 5). J.6.2 (Maestro, ec. 767): Burkert con r_c = 1.8 kpc·(M200/1e11)^(1/3),
# cuya media densidad cae en 0.979 kpc; U (166): pendiente 0.35; «∼0.4 en SPARC» del Unificado. La Tabla 45 del
# Unificado no está transcrita en el repositorio: la lectura la pide y falla cerrado sin ella.
J62_R_STAR_KPC = 1.8
RC_REFERENCES = {"J.6.2 (M 767), control": {"slope": 1.0 / 3.0, "r_half_at_1e11_kpc": None},
                 "U (166)": {"slope": 0.35, "r_half_at_1e11_kpc": None},
                 "«∼0.4 en SPARC» (U §7)": {"slope": 0.4, "r_half_at_1e11_kpc": None}}
U_TABLE_45 = None                           # pendiente de transcripción del autor
HALO_PROFILE_LABEL = "calibración fiducial sin origen en el corpus"   # cronos/halo_profile.py, fuera de PR-S


# ------------------------------------------------------------------ bariones de SPARC
def baryon_v2(v_gas, v_disk, v_bul, ups_disk: float, ups_bul: float, dist_ratio: float = 1.0):
    """V_bar² con los signos de SPARC, reescalada con D/D_cat; los valores negativos se ponen a cero y se cuentan."""
    v2 = (np.asarray(v_gas, float) * np.abs(v_gas) + ups_disk * np.asarray(v_disk, float) * np.abs(v_disk)
          + ups_bul * np.asarray(v_bul, float) * np.abs(v_bul)) * dist_ratio
    neg = v2 < 0.0
    return np.where(neg, 0.0, v2), int(neg.sum())


# ------------------------------------------------------------------ χ² con la inclinación perfilada y las previas
def profile_inclination(v_obs, e_v, inc_deg: float, e_inc_deg: float, v2_model, k_range: float = 4.0, n_grid: int = 41,
                        n_newton: int = 4):
    """χ²_datos + previa gaussiana en i, minimizado en i: con u = sen i/sen i_cat, χ²_datos = A − 2uB + u²C (A, B, C
    sumas sobre los puntos). Rejilla en i_cat ± k_range·e_i ∩ (0°, 90°] y Newton dentro del intervalo vecino.
    Devuelve (χ²_mín, i_mejor) con la forma de v2_model sin el último eje."""
    v, e = np.asarray(v_obs, float), np.asarray(e_v, float)
    Vm = np.sqrt(np.maximum(np.asarray(v2_model, float), 0.0))
    A = float(np.sum(v * v / (e * e)))
    B = np.sum(Vm * (v / (e * e)), axis=-1)
    C = np.sum(Vm * Vm / (e * e), axis=-1)
    lo, hi = max(inc_deg - k_range * e_inc_deg, 1.0), min(inc_deg + k_range * e_inc_deg, 90.0)
    grid = np.linspace(lo, hi, n_grid)
    s_cat = math.sin(math.radians(inc_deg))

    def total(i_deg):
        u = np.sin(np.radians(i_deg)) / s_cat
        return A - 2.0 * u * B + u * u * C + ((i_deg - inc_deg) / e_inc_deg) ** 2
    vals = np.stack([total(g) for g in grid], axis=-1)
    j = np.argmin(vals, axis=-1)
    i_best = grid[j]
    a_lo, a_hi = grid[np.maximum(j - 1, 0)], grid[np.minimum(j + 1, n_grid - 1)]
    for _ in range(n_newton):
        ir = np.radians(i_best)
        u = np.sin(ir) / s_cat
        du = np.cos(ir) / s_cat * math.pi / 180.0
        d2u = -np.sin(ir) / s_cat * (math.pi / 180.0) ** 2
        g1 = (-2.0 * B + 2.0 * u * C) * du + 2.0 * (i_best - inc_deg) / e_inc_deg ** 2
        g2 = 2.0 * C * du * du + (-2.0 * B + 2.0 * u * C) * d2u + 2.0 / e_inc_deg ** 2
        step = np.where(g2 > 0.0, g1 / np.where(g2 > 0.0, g2, 1.0), 0.0)
        i_best = np.clip(i_best - step, a_lo, a_hi)
    best = np.minimum(total(i_best), np.min(vals, axis=-1))
    i_out = np.where(total(i_best) <= np.min(vals, axis=-1), i_best, grid[j])
    return best, i_out


def c_prior_chi2(M200, c, rel: dict = DM14):
    """((log c − log c_DM14(M200))/0.11)²: la previa c–M de los brazos con halo."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lc_med = rel["a"] + rel["b"] * np.log10(np.asarray(M200, float) * rel["h"] / 1e12)
        return ((np.log10(np.asarray(c, float)) - lc_med) / rel["scatter_dex"]) ** 2


def c200_dm14(M200, rel: dict = DM14):
    return 10.0 ** (rel["a"] + rel["b"] * np.log10(np.asarray(M200, float) * rel["h"] / 1e12))


# ------------------------------------------------------------------ galaxias y molestias
@dataclass
class Galaxy:
    """Una galaxia de SPARC (o sintética, declarada) con las componentes de Rotmod_LTG (Υ = 1) en D_cat e i_cat."""
    name: str
    R: np.ndarray
    v_obs: np.ndarray
    e_v: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bul: np.ndarray
    D: float
    e_D: float
    inc: float
    e_inc: float
    R_d: float | None = None              # kpc, de la tabla maestra (lecturas q₀ y escalón)
    SB_disk0: float | None = None         # L☉/pc², brillo central del disco (tabla maestra)
    quality: int | None = None
    synthetic: bool = False

    @property
    def has_bulge(self) -> bool:
        return bool(np.any(np.asarray(self.v_bul) != 0.0))


FIT_SPEC = {
    "log_rho0": [4.5, 11.5], "dlog_rho0": 0.025,
    "log_sigma0": [0.5, 2.6], "dlog_sigma0": 0.025,
    "upsilon_disk": {"center": 0.5, "dex": 0.1},
    "upsilon_bulge": {"center": 0.7, "dex": 0.1},
    "k_grid": [-2.0, -1.0, 0.0, 1.0, 2.0],      # Υ_d, Υ_b y D en la rejilla, en unidades de su anchura
    "inclination": {"k_range": 4.0, "n_grid": 41, "n_newton": 4},
    "log_M200": [9.0, 13.5],
    "c_relation": DM14,
    "nodes": {"r_lo": 1e-5, "r_hi": 1e3, "h": 0.04},
    "require_r1_inside_r200": True,
    "chunk": 4096,
    # Refinado local (propuesta del ejecutor, pendiente del autor): rejillas locales que se estrechan a la mitad en
    # cada nivel alrededor del mejor punto, en (log ρ₀, log σ₀) y en las molestias, hasta tol.
    "zoom": {"enabled": False, "dk0": 1.0, "tol_dex": 2e-4, "tol_k": 2e-3, "max_levels": 12, "k_abs_max": 5.0},
}


def nuisance_point(gal: Galaxy, k_disk: float, k_bul: float, k_D: float, spec: dict = FIT_SPEC) -> dict | None:
    """Υ_d y Υ_b lognormales (centro, 0.1 dex) y D gaussiana (D_cat, e_D), en unidades k de su anchura; la previa es
    k_d² + k_b² + k_D² (k_b solo si hay bulbo). None si D ≤ 0."""
    ud, ub = spec["upsilon_disk"], spec["upsilon_bulge"]
    D = gal.D + k_D * gal.e_D
    if D <= 0.0:
        return None
    k_bul = k_bul if gal.has_bulge else 0.0
    return {"k_disk": k_disk, "k_bul": k_bul, "k_D": k_D, "ups_disk": ud["center"] * 10.0 ** (k_disk * ud["dex"]),
            "ups_bul": ub["center"] * 10.0 ** (k_bul * ub["dex"]), "D": D, "prior": k_disk ** 2 + k_bul ** 2 + k_D ** 2}


def nuisance_configs(gal: Galaxy, spec: dict = FIT_SPEC) -> list[dict]:
    """La rejilla de molestias del primer paso (k en k_grid para Υ_d, Υ_b si hay bulbo, y D)."""
    kb = spec["k_grid"] if gal.has_bulge else [0.0]
    cfgs = [nuisance_point(gal, a, b, d, spec) for a in spec["k_grid"] for b in kb for d in spec["k_grid"]]
    return [c for c in cfgs if c is not None]


def galaxy_baryons(gal: Galaxy, cfg: dict):
    """Radios, V_bar² (en los radios) y bariones tabulados de una configuración de molestias."""
    ratio = cfg["D"] / gal.D
    R = np.asarray(gal.R, float) * ratio
    v2, n_neg = baryon_v2(gal.v_gas, gal.v_disk, gal.v_bul, cfg["ups_disk"], cfg["ups_bul"], ratio)
    return R, v2, n_neg, TabulatedBaryons(R, v2)


def core_grid(spec: dict = FIT_SPEC, refine: int = 1) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """La rejilla congelada (log ρ₀, log σ₀); refine = 2 la duplica (comprobación de resolución)."""
    lr = np.arange(spec["log_rho0"][0], spec["log_rho0"][1] + 1e-9, spec["dlog_rho0"] / refine)
    ls = np.arange(spec["log_sigma0"][0], spec["log_sigma0"][1] + 1e-9, spec["dlog_sigma0"] / refine)
    R0, S0 = np.meshgrid(10.0 ** lr, 10.0 ** ls, indexing="ij")
    return R0.ravel(), S0.ravel(), (lr.size, ls.size)


def _setting_som(st: dict):
    return sigma_over_m_power(st["sigma100"], st["a"])


def _map_for_setting(tab: CoreTable, st: dict) -> list[dict]:
    """Los mapas de una tabla para un ajuste global: uno con t_edad fijo; las raíces del punto fijo con t_edad desde la
    concentración."""
    if st.get("t_mode", "fijo") == "fijo":
        out = map_table(tab, _setting_som(st), st.get("t_age", T_AGE_FIXED))
        out["t_age"] = np.full(tab.rho0.size, float(st.get("t_age", T_AGE_FIXED)))
        return [out]
    return concentration_age_maps(tab, [st["sigma100"]], st["a"], st["c1"])[0]


def _maps_for_settings(tab: CoreTable, settings: list[dict]) -> list[list[dict]]:
    """Los mapas de una tabla para todos los ajustes globales; los de t_edad desde la concentración se agrupan por
    (a, c₁) para compartir el barrido en σ₁₀₀·t."""
    out: list[list[dict] | None] = [None] * len(settings)
    groups: dict[tuple, list[int]] = {}
    for k, st in enumerate(settings):
        if st.get("t_mode", "fijo") == "fijo":
            out[k] = _map_for_setting(tab, st)
        else:
            groups.setdefault((st["a"], st["c1"]), []).append(k)
    for (a, c1), ks in groups.items():
        for k, maps in zip(ks, concentration_age_maps(tab, [settings[k]["sigma100"] for k in ks], a, c1)):
            out[k] = maps
    return out


def chi2_table(gal: Galaxy, tab: CoreTable, mp: dict, prior, R, v2_bar, spec: dict = FIT_SPEC) -> dict:
    """χ² total (datos con i perfilada + previa c–M + previas de las molestias) de cada punto de la tabla para un mapa
    ya hecho; inf donde el empalme no vale (sin r₁, sin NFW, M200 fuera de la previa o r₁ ≥ r200)."""
    V2 = model_v2_at_data(tab, mp, R, v2_bar)
    inc = spec["inclination"]
    chi2_data, i_best = profile_inclination(gal.v_obs, gal.e_v, gal.inc, gal.e_inc, V2, inc["k_range"], inc["n_grid"],
                                            inc["n_newton"])
    with np.errstate(invalid="ignore", divide="ignore"):
        lM = np.log10(mp["M200"])
    valid = np.isfinite(mp["c"]) & np.isfinite(lM) & (lM >= spec["log_M200"][0]) & (lM <= spec["log_M200"][1])
    if spec.get("require_r1_inside_r200", True):
        valid &= mp["r1"] < mp["r200"]
    valid &= np.isfinite(chi2_data)
    total = chi2_data + c_prior_chi2(mp["M200"], mp["c"], spec["c_relation"]) + prior
    return {"chi2": np.where(valid, total, np.inf), "chi2_data": np.where(valid, chi2_data, np.inf), "inc": i_best,
            "valid": valid}


_ENV_KEYS = ("r1", "M1", "rho1", "x1", "r_s", "rho_s", "c", "r200", "M200", "t_age")


def _record(tab: CoreTable, mp: dict, ct: dict, j: int, cfg: dict, n_neg: int) -> dict:
    return {"chi2": float(ct["chi2"][j]), "chi2_data": float(ct["chi2_data"][j]), "rho0": float(tab.rho0[j]),
            "sigma0": float(tab.sigma0[j]), "inc": float(ct["inc"][j]), "cfg": dict(cfg), "n_negative_vbar2": n_neg,
            **{q: float(mp[q][j]) for q in _ENV_KEYS}}


def fit_mcv(gal: Galaxy, settings: list[dict], spec: dict = FIT_SPEC, refine: int = 1, zoom: bool | None = None) -> list[dict]:
    """Ajuste de la MCV (c) a una galaxia para cada ajuste global (σ₁₀₀, a, modo de t_edad): mínimo de χ² sobre la
    rejilla (ρ₀, σ₀) × molestias, con i perfilada (primer paso, el del documento del 26-oct), y el refinado local si
    zoom (propuesta). Devuelve, por ajuste global, el mejor punto con su envoltura y cuántos puntos de la rejilla
    tienen un empalme válido."""
    zoom = spec["zoom"]["enabled"] if zoom is None else zoom
    rho0_all, sig_all, _ = core_grid(spec, refine)
    nodes = radial_nodes(**spec["nodes"])
    best = [{"chi2": math.inf, "n_valid": 0} for _ in settings]
    for cfg in nuisance_configs(gal, spec):
        R, v2_bar, n_neg, bar = galaxy_baryons(gal, cfg)
        for start in range(0, rho0_all.size, spec["chunk"]):
            sl = slice(start, start + spec["chunk"])
            tab = core_table(rho0_all[sl], sig_all[sl], bar, RHO_E, nodes)
            for k, maps in enumerate(_maps_for_settings(tab, settings)):
                valid_any = np.zeros(tab.rho0.size, dtype=bool)
                for mp in maps:
                    ct = chi2_table(gal, tab, mp, cfg["prior"], R, v2_bar, spec)
                    valid_any |= ct["valid"]
                    j = int(np.argmin(ct["chi2"]))
                    if ct["chi2"][j] < best[k]["chi2"]:
                        best[k] = {**_record(tab, mp, ct, j, cfg, n_neg), "n_valid": best[k]["n_valid"]}
                best[k]["n_valid"] += int(valid_any.sum())
    for k, st in enumerate(settings):
        best[k]["stage"] = "rejilla"
        if zoom and math.isfinite(best[k]["chi2"]):
            best[k] = zoom_refine(gal, best[k], st, spec)
    return best


def zoom_refine(gal: Galaxy, fit: dict, setting: dict, spec: dict = FIT_SPEC) -> dict:
    """Refinado local (propuesta del ejecutor): alrededor del mejor punto, rejillas 5×5 en (log ρ₀, log σ₀) y 3 valores
    por molestia, con los pasos a la mitad en cada nivel, hasta tol. Todo el nivel se integra de una vez (bariones por
    punto). Nunca empeora el χ² de partida."""
    z = spec["zoom"]
    nodes = radial_nodes(**spec["nodes"])
    cur = dict(fit)
    d_r, d_s, d_k = spec["dlog_rho0"], spec["dlog_sigma0"], z["dk0"]
    offs = np.arange(-2.0, 3.0)
    for _ in range(z["max_levels"]):
        d_r, d_s, d_k = d_r / 2.0, d_s / 2.0, d_k / 2.0
        c0 = cur["cfg"]
        ks = [c0["k_disk"] + d_k * o for o in (-1.0, 0.0, 1.0)]
        kb = [c0["k_bul"] + d_k * o for o in (-1.0, 0.0, 1.0)] if gal.has_bulge else [0.0]
        kD = [c0["k_D"] + d_k * o for o in (-1.0, 0.0, 1.0)]
        cfgs = [nuisance_point(gal, a, b, d, spec) for a in ks for b in kb for d in kD
                if max(abs(a), abs(b), abs(d)) <= z["k_abs_max"]]
        cfgs = [c for c in cfgs if c is not None]
        lr0, ls0 = math.log10(cur["rho0"]), math.log10(cur["sigma0"])
        LR, LS = np.meshgrid(lr0 + d_r * offs, ls0 + d_s * offs, indexing="ij")
        lr, ls = LR.ravel(), LS.ravel()
        parts, Rs, V2s, priors, negs = [], [], [], [], []
        for cfg in cfgs:
            R, v2_bar, n_neg, bar = galaxy_baryons(gal, cfg)
            parts.append(baryon_nodes(bar, nodes))
            Rs.append(R)
            V2s.append(v2_bar)
            priors.append(cfg["prior"])
            negs.append(n_neg)
        m = lr.size
        Mb_n = np.repeat(np.stack([p[0] for p in parts]), m, axis=0)
        Mb_m = np.repeat(np.stack([p[1] for p in parts]), m, axis=0)
        rb0 = np.repeat(np.array([p[2] for p in parts]), m)
        rho0 = np.tile(10.0 ** lr, len(cfgs))
        sig0 = np.tile(10.0 ** ls, len(cfgs))
        phi, M = integrate_cores(rho0, sig0, nodes, Mb_n, Mb_m, rb0, RHO_E)
        tab = CoreTable(rho0, sig0, nodes, phi, M, Mb_n, RHO_E)
        for mp in _map_for_setting(tab, setting):
            ct = chi2_table(gal, tab, mp, np.repeat(np.array(priors), m), np.repeat(np.stack(Rs), m, axis=0),
                            np.repeat(np.stack(V2s), m, axis=0), spec)
            j = int(np.argmin(ct["chi2"]))
            if ct["chi2"][j] < cur["chi2"]:
                cur = {**_record(tab, mp, ct, j, cfgs[j // m], negs[j // m]), "n_valid": cur.get("n_valid")}
        if d_r < z["tol_dex"] and d_k < z["tol_k"]:
            break
    cur["stage"] = "rejilla + refinado"
    return cur


# ------------------------------------------------------------------ rama del mejor ajuste (todas las preimágenes)
def branch_fields(gal: Galaxy, cfg: dict, setting: dict, t_age: float, spec: dict = FIT_SPEC, refine: int = 1) -> dict:
    """ln M200 y ln c en toda la rejilla (ρ₀, σ₀) con unas molestias y t_edad fijo: el mapa en el que se buscan las
    preimágenes de una envoltura."""
    _, _, _, bar = galaxy_baryons(gal, cfg)
    nodes = radial_nodes(**spec["nodes"])
    st = {"sigma100": setting["sigma100"], "a": setting["a"], "t_mode": "fijo", "t_age": t_age}
    rho0_all, sig_all, shape = core_grid(spec, refine)
    lnM = np.full(rho0_all.size, np.nan)
    lnc = np.full(rho0_all.size, np.nan)
    for start in range(0, rho0_all.size, spec["chunk"]):
        sl = slice(start, start + spec["chunk"])
        tab = core_table(rho0_all[sl], sig_all[sl], bar, RHO_E, nodes)
        mp = _map_for_setting(tab, st)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            lnM[sl], lnc[sl] = np.log(mp["M200"]), np.log(mp["c"])
    return {"lnM": lnM.reshape(shape), "lnc": lnc.reshape(shape), "lr": np.log(rho0_all.reshape(shape)[:, 0]),
            "ls": np.log(sig_all.reshape(shape)[0, :]), "bar": bar}


def classify_branch(gal: Galaxy, fit: dict, setting: dict, spec: dict = FIT_SPEC, refine: int = 1,
                    tol: float = 1e-3, fields: dict | None = None) -> dict:
    """¿Es el mejor ajuste la rama con núcleo de su envoltura? Se buscan las demás preimágenes de (M200, c) en la
    rejilla (ρ₀, σ₀) con las molestias del ajuste (cambios de signo de ln M200 − ln M200* y ln c − ln c*, con el filtro
    bilineal) y se refinan con el mapa inverso exacto. «con núcleo» si ninguna tiene ρ₀ menor; si no, «densa».

    Con t_edad desde la concentración basta fijar t_edad en el del ajuste: toda preimagen de la misma envoltura tiene
    el mismo c y, por tanto, la misma edad t_W(c). `fields` (de branch_fields, con las mismas molestias y t_edad)
    evita repetir la rejilla."""
    fl = fields if fields is not None else branch_fields(gal, fit["cfg"], setting, fit["t_age"], spec, refine)
    f1 = fl["lnM"] - math.log(fit["M200"])
    f2 = fl["lnc"] - math.log(fit["c"])
    lr, ls, bar = fl["lr"], fl["ls"], fl["bar"]
    som = _setting_som(setting)
    roots = [{"rho0": fit["rho0"], "sigma0": fit["sigma0"], "self": True}]
    for i, j in preimage_cells(f1, f2):
        q = refine_preimage(fit["M200"], fit["c"], [0.5 * (lr[i] + lr[i + 1]), 0.5 * (ls[j] + ls[j + 1])], som,
                            fit["t_age"], bar)
        if q is None:
            continue
        if any(abs(q["rho0"] / p["rho0"] - 1.0) < tol and abs(q["sigma0"] / p["sigma0"] - 1.0) < tol for p in roots):
            continue
        roots.append({**q, "self": False})
    roots.sort(key=lambda q: q["rho0"])
    lower = [q for q in roots if q["rho0"] < fit["rho0"] * (1.0 - tol)]
    return {"branch": "densa" if lower else "con núcleo", "n_preimages": len(roots), "preimages": roots}


def _point_record(gal: Galaxy, cfg: dict, rho0: float, sigma0: float, setting: dict, spec: dict) -> dict:
    """El registro completo (envoltura, χ², i) de un solo punto (ρ₀, σ₀) con unas molestias."""
    R, v2_bar, n_neg, bar = galaxy_baryons(gal, cfg)
    tab = core_table(np.array([rho0]), np.array([sigma0]), bar, RHO_E, radial_nodes(**spec["nodes"]))
    mp = _map_for_setting(tab, setting)[0]
    return _record(tab, mp, chi2_table(gal, tab, mp, cfg["prior"], R, v2_bar, spec), 0, cfg, n_neg)


def restricted_fit(gal: Galaxy, fit: dict, setting: dict, spec: dict = FIT_SPEC, beta: float = 0.75,
                   max_candidates: int = 300) -> dict:
    """Sensibilidad sin letra de la enmienda 3: la rama densa solo se admite si t_c(β) ≤ t_edad. Si el mejor ajuste es
    la rama con núcleo, o densa con t_c ≤ t_edad, no cambia; si no, se recorren los puntos de la rejilla por χ²
    creciente (como mucho max_candidates) hasta el primero admisible, y se refina dentro de lo admisible. Con t_edad
    fijo (el modo con t_edad desde la concentración clasifica el mejor ajuste, sin recorrido)."""
    if setting.get("t_mode", "fijo") != "fijo":
        raise ValueError("la variante restringida se define con t_edad fijo")
    cache: dict[tuple, dict] = {}

    def admissible(rec):
        key = tuple(round(rec["cfg"][k], 9) for k in ("k_disk", "k_bul", "k_D"))
        if key not in cache:
            cache[key] = branch_fields(gal, rec["cfg"], setting, rec["t_age"], spec)
        br = classify_branch(gal, rec, setting, spec, fields=cache[key])["branch"]
        t_c = collapse_reading(rec, setting)["t_c_gyr"][str(beta)]
        return br == "con núcleo" or t_c <= rec["t_age"], br, t_c

    ok, br, t_c = admissible(fit)
    if ok:
        return {**fit, "branch": br, "t_c_beta": t_c, "restricted": "igual que sin restringir"}
    rho0_all, sig_all, _ = core_grid(spec)
    nodes = radial_nodes(**spec["nodes"])
    cands = []
    cfgs = nuisance_configs(gal, spec)
    for ci, cfg in enumerate(cfgs):
        R, v2_bar, _, bar = galaxy_baryons(gal, cfg)
        for start in range(0, rho0_all.size, spec["chunk"]):
            sl = slice(start, start + spec["chunk"])
            tab = core_table(rho0_all[sl], sig_all[sl], bar, RHO_E, nodes)
            mp = _map_for_setting(tab, setting)[0]
            chi2 = chi2_table(gal, tab, mp, cfg["prior"], R, v2_bar, spec)["chi2"]
            for j in np.flatnonzero(np.isfinite(chi2)):
                cands.append((float(chi2[j]), ci, start + int(j)))
    cands.sort()
    for n, (_, ci, j) in enumerate(cands[:max_candidates]):
        rec = _point_record(gal, cfgs[ci], float(rho0_all[j]), float(sig_all[j]), setting, spec)
        ok, br, t_c = admissible(rec)
        if ok:
            z = zoom_refine(gal, rec, setting, spec) if spec["zoom"]["enabled"] else rec
            if z is not rec:
                ok_z, br_z, t_c_z = admissible(z)
                if ok_z:
                    rec, br, t_c = z, br_z, t_c_z
            return {**rec, "branch": br, "t_c_beta": t_c, "restricted": f"candidato {n + 1} por χ²"}
    return {"chi2": math.inf, "restricted": f"sin candidato admisible entre los {min(len(cands), max_candidates)} mejores"}


# ------------------------------------------------------------------ lecturas por galaxia
def collapse_reading(fit: dict, setting: dict) -> dict:
    """t_c (β = 0.75 y 0.45) de la envoltura ajustada con el σ_x/m calibrado en ⟨v_rel⟩ = 4σ₀/√π del ajuste, frente a
    t_edad (enmienda 3)."""
    som = _setting_som(setting)
    v = mean_relative_speed(fit["sigma0"])
    s = float(som(v)) if callable(som) else float(som)
    t_c = {str(b): gravothermal_collapse_time(s, fit["rho_s"], fit["r_s"], beta=b) for b in BETAS}
    return {"sigma_over_m_at_v": s, "v_rel": v, "t_c_gyr": t_c, "t_age_gyr": fit["t_age"],
            "t_c_exceeds_age": {k: v_ > fit["t_age"] for k, v_ in t_c.items()}}


def lapse_depth(fit: dict, gal: Galaxy) -> dict:
    """ΔN(0) = |Φ(0)|/c² del modelo ajustado (MCV + bariones, Φ(∞) = 0; la ECV fuera, declarado) y σ₀ (enmienda 4).
    Φ(0) = Φ(r₁) − [σ₀²φ(r₁) + (4π/3)Gρ_E·r₁²], con Φ(r₁) de la envoltura NFW y de los bariones tabulados."""
    _, _, _, bar = galaxy_baryons(gal, fit["cfg"])
    r1 = fit["r1"]
    phi1 = math.log(fit["rho0"] / fit["rho1"])
    x1 = r1 / fit["r_s"]
    phi_nfw_r1 = -4.0 * math.pi * G_KPC * fit["rho_s"] * fit["r_s"] ** 3 * math.log1p(x1) / r1
    r_out = np.geomspace(r1, max(r1, float(bar.R[-1])) * 1.0001, 2001)
    Mb = bar.mass(r_out)
    integral = np.trapezoid(Mb / r_out ** 2, r_out) if hasattr(np, "trapezoid") else np.trapz(Mb / r_out ** 2, r_out)
    phi_b_r1 = -G_KPC * (float(integral) + float(bar.M_last) / float(r_out[-1]))
    phi0 = phi_nfw_r1 + phi_b_r1 - (fit["sigma0"] ** 2 * phi1 + 4.0 / 3.0 * math.pi * G_KPC * RHO_E * r1 ** 2)
    return {"Delta_N0": abs(phi0) / C_KMS ** 2, "Phi0_kms2": phi0, "sigma0_kms": fit["sigma0"],
            "Delta_N0_nfw_only": 4.0 * math.pi * G_KPC * fit["rho_s"] * fit["r_s"] ** 2 / C_KMS ** 2}


def midplane_density_center(gal: Galaxy, ups_disk: float, zeta: float = 0.15) -> float | None:
    """ρ del plano medio del disco en el centro, Υ_d·Σ_d(0)/(2ζR_d), en M☉/pc³ (convenciones del 5E)."""
    if gal.SB_disk0 is None or gal.R_d is None or gal.R_d <= 0.0:
        return None
    return ups_disk * gal.SB_disk0 / (2.0 * zeta * gal.R_d * 1e3)


def q0_cross_reading(fit: dict, gal: Galaxy, eps_c, deps_c=None, zeta: float = 0.15) -> dict:
    """Lectura cruzada (enmienda 7): q₀ = 1.5·c²·ε_c(ρ)/σ₀² con la forma de ε_c que reciba A (5E o escalón), en la
    densidad del plano medio del centro; y el q general c²·ρ·ε_c'(ρ)/σ₀², que coincide con el anterior para la ley
    débil. Las dos hipótesis no pueden valer a la vez donde q₀ supera 1 (comprobación C3)."""
    rho = midplane_density_center(gal, fit["cfg"]["ups_disk"], zeta)
    if rho is None:
        return {"q0": None, "q_general": None, "rho_mid0": None}
    s2 = fit["sigma0"] ** 2
    q0 = 1.5 * C_KMS ** 2 * float(eps_c(rho)) / s2
    qg = None if deps_c is None else C_KMS ** 2 * rho * float(deps_c(rho)) / s2
    return {"q0": q0, "q_general": qg, "rho_mid0": rho}


def bh_influence_table() -> list[dict]:
    """r_h = G·M_BH/σ₀² (pc) en los extremos de las combinaciones de C2 (enmienda 6)."""
    rows = []
    for masses, sigmas in BH_C2_COMBINATIONS:
        for M in masses:
            for s in sigmas:
                rows.append({"M_bh": M, "sigma0": s, "r_h_pc": 1e3 * bh_influence_radius(M, s)})
    return rows


def rc_m200_reading(r_half_kpc, M200) -> dict:
    """Lectura r_c(M200) (enmienda 5): pendiente y valor en 1e11 M☉ del ajuste log r_c = a + b·log(M200/1e11) frente a
    J.6.2 (control, r_½ = 0.979 kpc·(M200/1e11)^(1/3)), la pendiente 0.35 de U (166) y «∼0.4 en SPARC». La Tabla 45 del
    Unificado falla cerrado mientras no esté transcrita."""
    lr, lm = np.log10(np.asarray(r_half_kpc, float)), np.log10(np.asarray(M200, float) / 1e11)
    ok = np.isfinite(lr) & np.isfinite(lm)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "slope": None, "r_half_at_1e11_kpc": None}
    b, a = np.polyfit(lm[ok], lr[ok], 1)
    refs = {k: dict(v) for k, v in RC_REFERENCES.items()}
    refs["J.6.2 (M 767), control"]["r_half_at_1e11_kpc"] = burkert_half_density_radius(J62_R_STAR_KPC)
    return {"n": int(ok.sum()), "slope": float(b), "r_half_at_1e11_kpc": float(10.0 ** a), "references": refs,
            "U_table_45": "pendiente de transcripción del autor" if U_TABLE_45 is None else U_TABLE_45}


def burkert_half_density_radius(r_c: float) -> float:
    """Radio de media densidad de un Burkert: (1 + x)(1 + x²) = 2, x ≈ 0.5437."""
    return r_c * burkert_half_density_x()
