"""Diccionario, ecuación 1 (orden del autor del 22-sep, §10): la forma
SATURANTE de ε_c(ρ) y sus tres ligaduras.

La Def. 6.4 (N = r/r̄ ≥ 0) hace ontológica la saturación de la energía de
Cronos: la forma mínima compatible con la ley débil ε_c = A·ρ^{3/2} a
baja densidad (Def. 11.1) y con un techo ε_max a alta densidad (§11.4 la
admite como regularización) es

    ε_c(ρ) = ε_max · ρ^{3/2} / (ρ^{3/2} + ρ*^{3/2}),        (E1)

con dos números: ε_max (el techo, adimensional) y ρ* (la densidad de
saturación, M☉/pc³). A ≡ ε_max/ρ*^{3/2} es la amplitud de la ley débil
que la forma reproduce para ρ ≪ ρ*. La fuerza local (Cor. 11.3) es
+c²∇ε_c = c²·ε_c'(ρ)∇ρ con

    ε_c'(ρ) = (3/2)·ε_max·ρ^{1/2}·ρ*^{3/2} / (ρ^{3/2} + ρ*^{3/2})²,

que se ANULA a alta densidad: la fuerza vive en ρ ≈ ρ*. Tres ligaduras
independientes, ya derivadas en el programa, acotan (ε_max, ρ*) SIN
ajustar nada:

  (i)  Oort–K_z (§3.26): Δρ_eff(0) = −c²·ε_c'(ρ₀)·ρ''(0)/(4πG) debe caber
       en el margen ρ_dyn − ρ_bar (+2σ) de la vecindad solar (ρ₀ ≈ 0.097
       M☉/pc³, losa declarada). Es una cota superior a ε_c'(ρ₀).
  (ii) estabilidad (criterio de Cronos–Jeans, §3.21 generalizado): en
       cada sistema virializado q ≡ c²·ρ·ε_c'(ρ)/σ² < 1 (para la ley
       débil ρ·ε_c' = (3/2)ε_c y q recupera (3/2)c²ε_c/σ²). Tabla
       declarada de sistemas (plano solar, Sculptor, halo NFW interior).
  (iii) el perfil σ_los(R) de Sculptor (§3.27): la FORMA del perfil bajo
       (E1); fallo cerrado hasta la ingesta de walker2009.

Y una condición de CALIBRACIÓN, no ligadura: el 5E calibró A_Sculptor
por la fuerza en Sculptor; que (E1) reproduzca esa fuerza exige
ε_c'(ρ_S) = (3/2)·A_Sculptor·ρ_S^{1/2} en la densidad estelar central de
Sculptor ρ_S (Plummer congelado, Υ⋆ del escaneo declarado). Como
Sculptor (ρ_S ≈ 0.04–0.1) y el plano solar (ρ₀ ≈ 0.1) tienen densidades
comparables, la pregunta que la forma responde es si hay algún ρ* para
el que ε_c' cae lo bastante entre ρ_S y ρ₀: ese es el primer resultado
del diccionario que puede falsarse, y se publica sea cual sea.

PREDICCIÓN (falsable en SPARC cuando haya bytes): con saturación la
fuerza extra g_C = c²·ε_c'(ρ)|∇ρ| solo actúa donde ρ ≈ ρ*, así que en la
relación aceleración radial (RAR) g_obs(g_bar) aparece como un ESCALÓN
localizado en la densidad de saturación, no como una desviación
continua. Se publica para un disco exponencial declarado.

Estatuto: cálculo de consistencia (E8) sobre la forma declarada;
ningún parámetro se ajusta a datos; A_Sculptor no se toca.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0e, i1e, k0e, k1e

from dynamics.cronos_jeans import nfw_jeans_table
from dynamics.local_kz import SLAB_DECLARED, load_bounds, room
from dynamics.sculptor_transfer import SCULPTOR, frozen_config
from dynamics.weak_field import C_KMS, G_PC, plummer_density

A_SCULPTOR = frozen_config().A_sculptor
UPSILON_SCAN = SCULPTOR["upsilon_scan"]                       # (1.0, 2.0, 3.0), declarado en el 5E
R_HALF_PC = SCULPTOR["R_half_pc"]
L_V = SCULPTOR["L_V_Lsun"]
PLUMMER_A_OVER_RHALF = 1.0 / np.sqrt(2.0 ** (2.0 / 3.0) - 1.0)      # a = R_half/√(2^{2/3} − 1) (Plummer proyectado)

FORM = "ε_c(ρ) = ε_max·ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}); A_débil ≡ ε_max/ρ*^{3/2}"


# ------------------------------------------------------------------ la forma
def eps_c(rho, eps_max: float, rho_star: float):
    x = np.clip(np.asarray(rho, dtype=float), 0.0, None) ** 1.5
    return eps_max * x / (x + rho_star ** 1.5)


def deps_c_drho(rho, eps_max: float, rho_star: float):
    r = np.clip(np.asarray(rho, dtype=float), 0.0, None)
    s = rho_star ** 1.5
    return 1.5 * eps_max * np.sqrt(r) * s / (r ** 1.5 + s) ** 2


def weak_law_amplitude(eps_max: float, rho_star: float) -> float:
    """A de la ley débil que (E1) reproduce para ρ ≪ ρ*."""
    return eps_max / rho_star ** 1.5


def q_local(rho, sigma2_kms2, eps_max: float, rho_star: float):
    """q = c²·ρ·ε_c'(ρ)/σ² (criterio de Cronos–Jeans para una ley general)."""
    r = np.asarray(rho, dtype=float)
    return C_KMS ** 2 * r * deps_c_drho(r, eps_max, rho_star) / np.asarray(sigma2_kms2, dtype=float)


# ----------------------------------------------------------- las ligaduras
def slab_rho0_and_rho2(slab: dict = SLAB_DECLARED) -> tuple[float, float]:
    rho0 = slab["rho_star_0"] + slab["rho_gas_0"] + slab["rho_dm"]
    rho2 = -(2.0 * slab["rho_star_0"] / slab["h_star_pc"] ** 2 + 2.0 * slab["rho_gas_0"] / slab["h_gas_pc"] ** 2)
    return float(rho0), float(rho2)


def oort_bound_on_deps(n_sigma: float = 2.0) -> dict:
    """Cota (i): ε_c'(ρ₀) ≤ D_max con Δρ_eff(0) = −c²ε_c'(ρ₀)ρ''(0)/(4πG) ≤
    margen + n_σ·σ_margen (todo el margen para Cronos: la cota más laxa).
    Consume local_kz_bounds (falla cerrado si no está AVAILABLE)."""
    b = load_bounds()
    rm = room(b)
    rho0, rho2 = slab_rho0_and_rho2()
    allowed = rm["rho_room_0"] + n_sigma * rm["rho_room_0_err"]
    D_max = allowed * 4.0 * np.pi * G_PC / (-C_KMS ** 2 * rho2)
    return {"rho0_msun_pc3": rho0, "rho2_msun_pc5": rho2, "room": rm, "n_sigma": n_sigma,
            "deps_max_at_rho0": float(D_max),
            "power_law_equivalent_A_max": float(D_max / (1.5 * np.sqrt(rho0))),
            "dataset_caveat": b["provenance_caveat"]}


def sculptor_central_density(upsilon: float) -> float:
    """ρ_S: densidad estelar central del Plummer congelado de Sculptor
    (M⋆ = Υ⋆·L_V, a = R_half/√(2^{2/3} − 1)), en M☉/pc³."""
    a = R_HALF_PC * PLUMMER_A_OVER_RHALF
    return float(plummer_density(0.0, upsilon * L_V, a))


def stability_systems(upsilon_sculptor: float = 2.0) -> list[dict]:
    """Tabla DECLARADA de sistemas virializados (ρ en M☉/pc³, σ_1D² en
    (km/s)²) donde q < 1 debe cumplirse: plano solar (losa declarada,
    σ_z = 20), Sculptor (centro del Plummer congelado, σ_los observada),
    halo NFW de la ronda (filas de Jeans interiores, σ_r² newtoniana)."""
    rho0, _ = slab_rho0_and_rho2()
    rows = [{"system": "plano solar (losa MPH2015)", "rho": rho0, "sigma2": 20.0 ** 2},
            {"system": f"Sculptor centro (Plummer, Υ⋆ = {upsilon_sculptor})", "rho": sculptor_central_density(upsilon_sculptor),
             "sigma2": SCULPTOR["sigma_los_kms"] ** 2}]
    for r in nfw_jeans_table()["rows"]:
        if r["r_kpc"] in (0.05, 0.4, 1.0, 5.0):
            rows.append({"system": f"halo NFW 1e11 (r = {r['r_kpc']} kpc)", "rho": r["rho_msun_pc3"], "sigma2": r["sigma_r2_kms2"]})
    return rows


def sculptor_force_calibration_eps_max(rho_star, upsilon: float) -> np.ndarray:
    """ε_max(ρ*) para el que (E1) reproduce la FUERZA del 5E en Sculptor:
    ε_c'(ρ_S) = (3/2)·A_Sculptor·ρ_S^{1/2} ⟹ ε_max = A_S·(ρ_S^{3/2} + ρ*^{3/2})²/ρ*^{3/2}."""
    rho_s = sculptor_central_density(upsilon)
    s = np.asarray(rho_star, dtype=float) ** 1.5
    return A_SCULPTOR * (rho_s ** 1.5 + s) ** 2 / s


def constraint_map(log_eps_max=(-14.0, -2.0, 241), log_rho_star=(-5.0, 2.0, 141), upsilon_sculptor: float = 2.0) -> dict:
    """Región permitida en (ε_max, ρ*) por (i) y (ii); (iii) fallo cerrado.
    Publica también la curva de calibración de Sculptor y si corta la región."""
    oort = oort_bound_on_deps()
    systems = stability_systems(upsilon_sculptor)
    le = np.linspace(*log_eps_max)
    lr = np.linspace(*log_rho_star)
    E, R = np.meshgrid(10.0 ** le, 10.0 ** lr, indexing="ij")
    ok_oort = deps_c_drho(oort["rho0_msun_pc3"], E, R) <= oort["deps_max_at_rho0"]
    ok_stab = np.ones_like(ok_oort, dtype=bool)
    worst = {}
    for s in systems:
        q = q_local(s["rho"], s["sigma2"], E, R)
        ok_stab &= q < 1.0
        worst[s["system"]] = float(np.mean(q < 1.0))
    allowed = ok_oort & ok_stab
    # calibración de Sculptor: para cada ρ*, el ε_max que reproduce la fuerza; ¿cumple (i) y (ii)?
    cal = {}
    for ups in UPSILON_SCAN:
        # la tabla de estabilidad se reconstruye con el MISMO Υ⋆ que la calibración
        sys_u = stability_systems(ups)
        e_cal = sculptor_force_calibration_eps_max(10.0 ** lr, ups)
        ok_o = deps_c_drho(oort["rho0_msun_pc3"], e_cal, 10.0 ** lr) <= oort["deps_max_at_rho0"]
        ok_s, ok_s_no_self = np.ones_like(ok_o, dtype=bool), np.ones_like(ok_o, dtype=bool)
        for s in sys_u:
            q = q_local(s["rho"], s["sigma2"], e_cal, 10.0 ** lr) < 1.0
            ok_s &= q
            if not s["system"].startswith("Sculptor"):
                ok_s_no_self &= q
        ratio = deps_c_drho(oort["rho0_msun_pc3"], e_cal, 10.0 ** lr) / oort["deps_max_at_rho0"]
        rho_s = sculptor_central_density(ups)
        # q de Sculptor bajo su propia calibración: fijado por la fuerza, INDEPENDIENTE de la forma
        q_self = float(C_KMS ** 2 * rho_s * 1.5 * A_SCULPTOR * np.sqrt(rho_s) / SCULPTOR["sigma_los_kms"] ** 2)
        cal[str(ups)] = {"rho_S_msun_pc3": rho_s, "log10_rho_star": lr.tolist(), "log10_eps_max": np.log10(e_cal).tolist(),
                         "oort_ok": ok_o.tolist(), "stability_ok": ok_s.tolist(), "stability_ok_without_sculptor_self": ok_s_no_self.tolist(),
                         "both_ok_any": bool(np.any(ok_o & ok_s)), "n_rho_star_both_ok": int(np.sum(ok_o & ok_s)),
                         "oort_and_others_ok_any": bool(np.any(ok_o & ok_s_no_self)), "n_rho_star_oort_and_others_ok": int(np.sum(ok_o & ok_s_no_self)),
                         "q_sculptor_self_form_independent": q_self,
                         "min_oort_excess_ratio": float(np.min(ratio)), "rho_star_at_min_ratio": float(10.0 ** lr[int(np.argmin(ratio))])}
    return {"form": FORM, "grid": {"log10_eps_max": le.tolist(), "log10_rho_star": lr.tolist()},
            "allowed_fraction": float(np.mean(allowed)), "allowed_by_oort_fraction": float(np.mean(ok_oort)),
            "allowed_by_stability_fraction": float(np.mean(ok_stab)), "stability_pass_fraction_by_system": worst,
            "allowed_mask": allowed.astype(int).tolist(), "oort": oort, "systems": systems,
            "sculptor_calibration": cal, "sculptor_profile_constraint": "no evaluable: walker2009 DATA_UNAVAILABLE (fallo cerrado)",
            "form_independent_note": ("calibrar la FUERZA en el centro de Sculptor fija ρ_S·ε_c'(ρ_S) = (3/2)A_S·ρ_S^{3/2} y por tanto "
                                      "q_S = c²ρ_Sε_c'(ρ_S)/σ² sea cual sea la forma ε_c(ρ): la (in)estabilidad de Sculptor bajo su "
                                      "propia calibración no depende del diccionario, solo de A_S, ρ_S(Υ⋆) y σ")}


# -------------------------------------------------------- predicción RAR
def exponential_disc_g_bar(R_kpc, Sigma0_msun_pc2: float, h_kpc: float):
    """Aceleración radial newtoniana en el plano de un disco exponencial
    infinitamente delgado (Freeman 1970): g = 2πGΣ₀·(R/h)·[I₀K₀ − I₁K₁](R/2h).
    Unidades: (km/s)²/kpc con G en kpc·(km/s)²/M☉ y Σ₀ en M☉/kpc²."""
    R = np.asarray(R_kpc, dtype=float)
    y = R / (2.0 * h_kpc)
    G_kpc = G_PC * 1e-3
    Sigma0 = Sigma0_msun_pc2 * 1e6
    return 2.0 * np.pi * G_kpc * Sigma0 * (R / h_kpc) * (i0e(y) * k0e(y) - i1e(y) * k1e(y))


def rar_step_prediction(eps_max: float, rho_star: float, Sigma0_msun_pc2: float = 300.0, h_kpc: float = 3.0,
                        h_z_kpc: float = 0.3, R_kpc=None) -> dict:
    """g_obs = g_bar + g_C sobre un disco exponencial declarado, con
    ρ(R, 0) = Σ(R)/(2h_z) y g_C = c²·ε_c'(ρ)·|∂ρ/∂R| = c²·ε_c'(ρ)·ρ/h (radial,
    hacia el centro). El escalón aparece donde ρ(R) cruza ρ*."""
    R = np.geomspace(0.2, 30.0, 200) if R_kpc is None else np.asarray(R_kpc, dtype=float)
    Sigma = Sigma0_msun_pc2 * np.exp(-R / h_kpc)                       # M☉/pc²
    rho = Sigma / (2.0 * h_z_kpc * 1e3)                                 # M☉/pc³
    g_bar = exponential_disc_g_bar(R, Sigma0_msun_pc2, h_kpc)
    g_C = C_KMS ** 2 * deps_c_drho(rho, eps_max, rho_star) * rho / h_kpc   # (km/s)²/kpc
    R_star = float(h_kpc * np.log(Sigma0_msun_pc2 / (2.0 * h_z_kpc * 1e3 * rho_star))) if Sigma0_msun_pc2 > 2.0 * h_z_kpc * 1e3 * rho_star else None
    return {"eps_max": eps_max, "rho_star": rho_star, "disc": {"Sigma0_msun_pc2": Sigma0_msun_pc2, "h_kpc": h_kpc, "h_z_kpc": h_z_kpc},
            "R_kpc": R.tolist(), "rho_msun_pc3": rho.tolist(), "g_bar": g_bar.tolist(), "g_C": g_C.tolist(),
            "g_obs": (g_bar + g_C).tolist(), "R_where_rho_equals_rho_star_kpc": R_star,
            "max_boost": float(np.max(g_C / g_bar)), "R_of_max_boost_kpc": float(R[int(np.argmax(g_C / g_bar))])}
