"""PR-S de principio a fin sobre SPARC: submuestra, reparto, ajustes de todos los brazos, calibración en A, letra de la
MCV (c) sobre B con sus puertas y las lecturas sin letra de los documentos del 26 y del 27-oct (bloques P0 del autor;
sin datos, sin letra, sin congelar).

Este módulo solo orquesta: no ajusta A_S en ningún brazo y no usa maquinaria de optimización (cae bajo el candado
anti-ajuste del 5E, tests/test_front5e_lock.py). Con bytes, el único acceso al catálogo es
dynamics.sparc_data.load_sparc_catalogue, que falla cerrado sin el dataset AVAILABLE.

Lo que el autor tiene que decidir antes de congelar (cuatro decisiones del 27-oct, seis del 26-oct y los hallazgos del
ejecutor) está en PENDING_DECISIONS y EXECUTOR_FINDINGS; la congelación se niega mientras quede alguna pendiente
(scripts/run_pr_s.py freeze).
"""

from __future__ import annotations

import math
from multiprocessing import Pool

import numpy as np

from dynamics import mcv_baryons as mb
from dynamics import mcv_sparc as ms
from dynamics import sparc_bank as sb

PENDING_DECISIONS = [
    {"id": "27oct-1", "decision": "Enmiendas 1–4 (σ_x/m efectivo y alcance de las letras; memoria del halo; rama densa "
     "y colapso; lapse)", "recomendacion": "las cuatro, antes de congelar",
     "alternativa": "dejarlas para otra ronda, con el riesgo de leer una A sin su alcance",
     "en_el_codigo": "las cuatro están implementadas; ninguna toca la letra"},
    {"id": "27oct-2", "decision": "Relación de época de formación para t_edad", "recomendacion":
     "c = c₁·a₀/a_c de Wechsler et al. (2002) con c₁ congelado antes de los bytes (Wechsler: c₁ ≈ 4.1 para c_vir)",
     "alternativa": "tres valores fijos (6, 10 y 13 Gyr) como sensibilidad, sin relación",
     "en_el_codigo": "los dos modos; falta el valor de c₁"},
    {"id": "27oct-3", "decision": "Rama densa", "recomendacion": "lectura de coherencia con t_c y la variante "
     "restringida (solo t_c ≤ t_edad) como sensibilidad, sin letra", "alternativa": "publicar solo la fracción",
     "en_el_codigo": "las dos (mcv_sparc.restricted_fit)"},
    {"id": "27oct-4", "decision": "Ley local en la lapse de la MCV", "recomendacion": "mantener la separación del 18-oct "
     "con la lectura cruzada de q_CJ(0)", "alternativa": "brazo combinado sin letra con la forma saturante de ρ* = 0.01–0.03",
     "en_el_codigo": "la lectura cruzada; el brazo combinado no (ver el hallazgo H3, que cambia su premisa)"},
    {"id": "26oct-1", "decision": "J.6.2 en el banco", "recomendacion": "control MCMC-J62 con r⋆ libre en A; el factor de "
     "época, señalado como incoherencia del Maestro", "alternativa": "retirarlo y comparar solo la pendiente",
     "en_el_codigo": "el brazo MCMC-J62 con r⋆ en rejilla"},
    {"id": "26oct-2", "decision": "Forma de σ_x/m(v)", "recomendacion": "ley de potencias (σ₁₀₀, a) con la constante "
     "como sub-brazo anidado", "alternativa": "solo constante", "en_el_codigo": "la rejilla (σ₁₀₀, a) completa; a = 0 "
     "es la constante"},
    {"id": "26oct-3", "decision": "Reparto A/B", "recomendacion": "por sha256 del identificador",
     "alternativa": "aleatorio con semilla congelada", "en_el_codigo": "los dos (sparc_bank.split_halves)"},
    {"id": "26oct-4", "decision": "Letras del escalón", "recomendacion": "lectura aditiva con ventanas; la literal, aparte",
     "alternativa": "solo la literal", "en_el_codigo": "las dos (dynamics/rar_step_letters.py)"},
    {"id": "26oct-5", "decision": "Previas del halo", "recomendacion": "c–M de Dutton y Macciò (0.11 dex), las de Sculptor",
     "alternativa": "la c–M de 2610.05871, si difiere", "en_el_codigo": "DM14; la otra exige transcribirla"},
    {"id": "26oct-6", "decision": "Lo que pide #65 (afueras, E3, datos del fondo)", "recomendacion": "afueras = capa del "
     "borde del código; E3 fuera hasta tener vacíos; datos del fondo, del autor", "alternativa": "—",
     "en_el_codigo": "no es de PR-S"},
]

EXECUTOR_FINDINGS = [
    {"id": "H1", "hallazgo": "Con la rejilla sola, χ²_mín queda cuantizado: el paso de las molestias (Υ y D en ±2σ) y "
     "el de (ρ₀, σ₀) mueven χ²_mín más de 0.5 en galaxias con errores pequeños.",
     "opciones": "rejilla sola (documento), o rejilla y refinado local en rejillas que se estrechan (propuesta, "
     "FIT_SPEC['zoom'])", "efecto": "la comprobación de resolución de scripts/run_pr_s.py checks da las cifras"},
    {"id": "H2", "hallazgo": "En las letras del escalón, δ_pred = log(1 + g_C/g_obs) del documento queda por debajo del "
     "escalón que deja la propia ley aditiva cuando g_C ~ g_obs (ρ* = 0.001): la A sería inalcanzable.",
     "opciones": "«documento», o «autoconsistente», log[g_obs/(g_obs − g_C)]", "efecto": "solo ρ* = 0.001"},
    {"id": "H3", "hallazgo": "Para la forma saturante, 1.5·c²·ε_c/σ₀² (fórmula de C3 y de la enmienda 7) no es el q que "
     "decide la estabilidad de Cronos–Jeans, c²·ρ·ε_c'(ρ)/σ₀²; coinciden solo en la ley débil. Con ρ ≫ ρ* el segundo es "
     "cientos de veces menor: con ρ* = 0.001, q < 1 en vez de 0.5–9.3.",
     "opciones": "publicar los dos (hecho)", "efecto": "cambia la premisa de la alternativa de la decisión 27oct-4"},
    {"id": "H4", "hallazgo": "El punto fijo t_edad = t_W(c(t_edad)) tiene dos raíces por núcleo: una junto al borde "
     "(c ≈ c₁, t_W → 0, inestable) y la estable; las dos son envolturas autoconsistentes.",
     "opciones": "todas, y el ajuste toma el mejor χ² (hecho), o solo la estable", "efecto": "lectura de memoria"},
    {"id": "H5", "hallazgo": "«ECV por debajo de 1e-4 de la atracción dentro de 10·R_d» depende del tamaño del disco "
     "(1.7e-4 con R_d = 2 kpc en 1e11 M☉); la forma de C1 (≤ 6e-5 hasta 0.1·r200) se cumple.",
     "opciones": "test con C1 (hecho)", "efecto": "ninguno sobre las letras"},
    {"id": "H6", "hallazgo": "El candado anti-ajuste del 5E vigila dynamics/*sparc*.py y prohíbe la maquinaria de "
     "optimización: la física general del brazo (raíces del empalme) va en dynamics/mcv_baryons.py y el ajuste a SPARC "
     "es en rejilla.", "opciones": "mantener el candado tal cual (hecho), o ampliarlo a mano", "efecto": "ninguno"},
    {"id": "H7", "hallazgo": "Precisiones del ejecutor que el documento no fija: χ² = datos + previas (−2 ln de las "
     "previas); ΛCDM-1 con log c ∈ [0, 2]; Υ_b solo con bulbo; r₁ < r200; rangos de (ρ₀, σ₀); paso de los nodos "
     "(0.04 en ln r); χ²_ν de E8-Q = χ²_datos/(N − k_halo); la a₀ poblacional con la RAR de un parámetro.",
     "opciones": "aceptarlas o fijar otras", "efecto": "todas entran en la preinscripción"},
    {"id": "H8", "hallazgo": "Faltan entradas del corpus y de la literatura: la Tabla 45 del Unificado, el valor de c₁, "
     "las previas exactas de Li+2020 y el montaje de 2610.05871 (arxiv.org no es accesible desde la sesión).",
     "opciones": "que el autor las transcriba", "efecto": "E8-Q y la lectura r_c(M200) fallan cerrado sin ellas"},
    {"id": "H9", "hallazgo": "Con discos sintéticos de tres tamaños (0.5, 1 y 2 veces R_ref) y h_z = 0.15·R_d, el escalón "
     "previsto S_pred vale 0.30 dex con ρ* = 0.001 y 0.024–0.032 dex con ρ* = 0.01–0.03, entre 2 y 8 veces lo del documento "
     "del 26-oct (0.10–0.14 y 0.004–0.007, con los discos de la predicción congelada): g_C ∝ 1/R_d crece en los discos "
     "compactos. Con σ_S ≈ 0.005 dex, el escalón con ρ* = 0.01–0.03 podría ser comprobable. Con ρ* = 0.001, la ley "
     "predice g_C ≥ g_obs en la mitad de esos discos.", "opciones": "revisar la expectativa E13 del escalón con el "
     "reparto de R_d de SPARC antes de congelar, o dejarla como está", "efecto": "solo la expectativa E13; las letras "
     "no cambian"},
]

# Las cinco decisiones del documento del 28-oct: ninguna bloquea PR-S. Lo que hace el código con cada una.
DECISIONS_28OCT = [
    {"id": "28oct-1", "decision": "Lectura de la energía sellada", "recomendacion": "(a) operativa para los ajustes y (b) "
     "como frente de derivación de C(S), sin decidir la ley de Cronos antes de Walker", "en_PR_S": "nada: PR-S usa (a)"},
    {"id": "28oct-2", "decision": "Ley constante de la evaluación", "recomendacion": "brazo de control en el catálogo de "
     "#65, como ξ_0, fuera de PR-S", "en_PR_S": "nada (fuera de PR-S y no ahora)"},
    {"id": "28oct-3", "decision": "S_local en PR-S", "recomendacion": "lectura operativa con t_edad(c) y ΔN(0), sin letra",
     "en_PR_S": "texto «S_local operativa» (punto 10), sin parámetros nuevos"},
    {"id": "28oct-4", "decision": "Respaldo de la rama", "recomendacion": "sí, con las cinco condiciones",
     "en_PR_S": "hecho: mcv/pr-s-sparc-instrument-wip, commit «borrador sin congelar»"},
    {"id": "28oct-5", "decision": "Nombres", "recomendacion": "ξ_0 para la ley constante, q_dec para la deceleración y "
     "q_CJ(0) en PR-S", "en_PR_S": "q_CJ(0) en el código y en el texto; ε solo con subíndice (punto 12)"},
]

FAST_ARMS = ("ΛCDM-1", "ΛCDM-2", "MOND", "MCMC-C", "Burkert libre (E8-Q)", "MOND a₀ libre")


def settings_grid(spec_bank: dict, t_mode: str = "fijo", t_age: float = 10.0, c1: float | None = None) -> list[dict]:
    """La rejilla global (σ₁₀₀, a) de σ_x/m(v) con un modo de t_edad."""
    extra = {"t_mode": "fijo", "t_age": t_age} if t_mode == "fijo" else {"t_mode": "concentración", "c1": c1}
    return [{"sigma100": s, "a": a, **extra} for s in spec_bank["sigma100_grid"] for a in spec_bank["a_grid"]]


def setting_key(st: dict) -> tuple:
    return st["sigma100"], st["a"]


def default_plan(c1: float | None = None, fixed_ages=(6.0, 13.0), zoom: bool = False) -> dict:
    """El plan del día de bytes: la rejilla principal con t_edad = 10 Gyr y las de la memoria del halo."""
    modes = {"t10": settings_grid(sb.BANK_SPEC)}
    if c1 is not None:
        modes["concentración"] = settings_grid(sb.BANK_SPEC, "concentración", c1=c1)
    for t in fixed_ages:
        modes[f"t{t:g}"] = settings_grid(sb.BANK_SPEC, "fijo", t_age=t)
    rs = sb.BANK_SPEC["j62"]
    return {"spec_mcv": {**ms.FIT_SPEC, "zoom": {**ms.FIT_SPEC["zoom"], "enabled": zoom}}, "spec_bank": sb.BANK_SPEC,
            "modes": modes, "j62_rstars": list(10.0 ** np.arange(rs["log_rstar"][0], rs["log_rstar"][1] + 1e-9,
                                                                     rs["dlog_rstar"]))}


def fit_galaxy(args) -> dict:
    """Todos los brazos para una galaxia: los rápidos, J62 en cada r⋆ y la MCV (c) en todos los modos a la vez (los
    núcleos se integran una sola vez para todos los ajustes globales)."""
    gal, plan = args
    spec_mcv, spec_bank = plan["spec_mcv"], plan["spec_bank"]
    out = {"name": gal.name, "n": int(np.asarray(gal.R).size), "has_bulge": gal.has_bulge}
    for arm in FAST_ARMS:
        if arm == "MCMC-C" and (gal.R_d is None or gal.SB_disk0 is None):
            continue
        out[arm] = sb.fit_arm(gal, arm, spec=spec_bank, spec_mcv=spec_mcv)
    out["MCMC-J62"] = {r: sb.fit_arm(gal, "MCMC-J62", r, spec=spec_bank, spec_mcv=spec_mcv) for r in plan["j62_rstars"]}
    flat = [(mode, st) for mode, sts in plan["modes"].items() for st in sts]
    fits = ms.fit_mcv(gal, [st for _, st in flat], spec_mcv)
    out["MCV"] = {mode: {} for mode in plan["modes"]}
    for (mode, st), f in zip(flat, fits):
        out["MCV"][mode][setting_key(st)] = f
    return out


def fit_all(galaxies: list, plan: dict, processes: int = 1) -> dict:
    args = [(g, plan) for g in galaxies]
    if processes > 1:
        with Pool(processes) as pool:
            res = pool.map(fit_galaxy, args)
    else:
        res = [fit_galaxy(a) for a in args]
    return {r["name"]: r for r in res}


def _calibrate_mode(fits: dict, names: list[str], mode: str) -> dict:
    keys = list(next(iter(fits.values()))["MCV"][mode])
    return sb.calibrate({k: [fits[n]["MCV"][mode][k]["chi2"] for n in names] for k in keys})


def population_a0(galaxies: list, a0_grid_si, spec_mcv: dict = ms.FIT_SPEC) -> dict:
    """a₀ poblacional (para E8-Q): Σ_galaxias χ²_mín(a₀) con las molestias perfiladas en cada a₀; mínimo y 1σ por
    Δχ² = 1 (con la RAR de un parámetro; el montaje exacto de 2610.05871 está pendiente, hallazgo H8)."""
    a0 = np.asarray(a0_grid_si, float)
    total = np.zeros(a0.size)
    inc = spec_mcv["inclination"]
    for gal in galaxies:
        best = np.full(a0.size, np.inf)
        for cfg in ms.nuisance_configs(gal, spec_mcv):
            R, v2_bar, _, _ = ms.galaxy_baryons(gal, cfg)
            v2 = sb.mond_v2(np.broadcast_to(v2_bar, (a0.size, R.size)), np.broadcast_to(R, (a0.size, R.size)),
                            (a0 / sb.KMS2_PER_KPC_SI)[:, None])
            chi2, _ = ms.profile_inclination(gal.v_obs, gal.e_v, gal.inc, gal.e_inc, v2, inc["k_range"], inc["n_grid"],
                                             inc["n_newton"])
            best = np.minimum(best, chi2 + cfg["prior"])
        total += best
    j = int(np.argmin(total))
    inside = a0[total - total[j] <= 1.0]
    return {"a0": float(a0[j]), "sigma": float(0.5 * (inside.max() - inside.min())), "on_edge": j in (0, a0.size - 1)}


def readings(galaxies: dict, fits: dict, halves: dict, cal: dict, plan: dict, local_law_A: dict | None = None,
             restricted: bool = True) -> dict:
    """Las lecturas sin letra con la calibración de A (enmiendas 2–7 del 27-oct y lecturas del 26-oct)."""
    best = cal["best"]
    st = next(s for s in plan["modes"]["t10"] if setting_key(s) == best)
    spec = plan["spec_mcv"]
    rows = []
    for name, f in fits.items():
        gal, fit = galaxies[name], f["MCV"]["t10"][best]
        if not math.isfinite(fit["chi2"]):
            rows.append({"name": name, "sin_empalme": True})
            continue
        cb = ms.classify_branch(gal, fit, st, spec)
        col = ms.collapse_reading(fit, st)
        lap = ms.lapse_depth(fit, gal)
        r_half = (mb.half_density_radius(fit["rho0"], fit["sigma0"], ms.galaxy_baryons(gal, fit["cfg"])[3])
                  if cb["branch"] == "con núcleo" else math.nan)
        row = {"name": name, "half": "A" if name in halves["A"] else "B", "branch": cb["branch"],
               "n_preimages": cb["n_preimages"], "t_c_gyr": col["t_c_gyr"], "t_age_gyr": fit["t_age"],
               "densa_con_t_c_mayor_que_edad": cb["branch"] == "densa" and col["t_c_exceeds_age"]["0.75"],
               "Delta_N0": lap["Delta_N0"], "sigma0": fit["sigma0"], "M200": fit["M200"], "c": fit["c"],
               "c_fuera_de_2sigma": bool(ms.c_prior_chi2(fit["M200"], fit["c"]) > 4.0), "r_half_kpc": r_half,
               "SB_disk0": gal.SB_disk0}
        if restricted:
            rf = ms.restricted_fit(gal, fit, st, spec)
            row["restringido"] = {"chi2": rf["chi2"], "nota": rf["restricted"]}
        if local_law_A:
            row["q_CJ0_cruzado"] = {form: ms.q_cj_cross_reading(fit, gal, *fns) for form, fns in local_law_A.items()}
        rows.append(row)
    ok = [r for r in rows if not r.get("sin_empalme")]
    sb_med = float(np.median([r["SB_disk0"] for r in ok if r["SB_disk0"] is not None])) if ok else math.nan

    def frac(sel, key):
        return float(np.mean([r[key] for r in sel])) if sel else None
    hsb = [r for r in ok if r["SB_disk0"] is not None and r["SB_disk0"] > sb_med]
    lsb = [r for r in ok if r["SB_disk0"] is not None and r["SB_disk0"] <= sb_med]
    out = {"por_galaxia": rows,
           "ramas": {"fraccion_densa": float(np.mean([r["branch"] == "densa" for r in ok])) if ok else None,
                     "fraccion_densa_incoherente": frac(ok, "densa_con_t_c_mayor_que_edad"),
                     "c_fuera_de_2sigma": {"todas": frac(ok, "c_fuera_de_2sigma"), "HSB": frac(hsb, "c_fuera_de_2sigma"),
                                           "LSB": frac(lsb, "c_fuera_de_2sigma")}},
           "r_c_M200": ms.rc_m200_reading([r["r_half_kpc"] for r in ok], [r["M200"] for r in ok]),
           "sparc_sculptor": sb.sculptor_cross_reading(best),
           "S_local_operativa": ms.S_LOCAL_TEXT, "gap": ms.GAP_OUT_OF_SCOPE,
           "cuspide_agujero_negro": ms.BH_OUT_OF_SCOPE}
    return out


def memory_reading(fits: dict, halves: dict, cal_t10: dict) -> dict:
    """Enmienda 2: recalibrar en A con cada modo de t_edad y probar en B; Δχ² en B frente a t_edad = 10 Gyr (cada uno
    con su calibración) y el cambio de la calibración."""
    out = {}
    modes = [m for m in next(iter(fits.values()))["MCV"] if m != "t10"]
    for mode in modes:
        cal = _calibrate_mode(fits, halves["A"], mode)
        if cal["best"] is None:
            out[mode] = {"calibracion": None}
            continue
        d = [fits[n]["MCV"][mode][cal["best"]]["chi2"] - fits[n]["MCV"]["t10"][cal_t10["best"]]["chi2"] for n in halves["B"]]
        out[mode] = {"calibracion": cal["best"], "calibracion_t10": cal_t10["best"],
                     "delta_chi2_B": {"mediana": float(np.median(d)), "suma": float(np.sum(d)),
                                      "fraccion_negativa": float(np.mean(np.array(d) < 0.0))}}
    return out


def bank_readings(fits: dict, names: list[str], cal_j62: dict, best: tuple) -> dict:
    """Reglas comunes del banco: BIC con k completo, preferencias por pares y ΔBIC frente a ΛCDM-2 (sin letra)."""
    bics = {}
    for n in names:
        f = fits[n]
        k_nuis = 3 + (1 if f["has_bulge"] else 0)
        cand = {arm: (f[arm]["chi2"], f[arm]["k_params"]) for arm in FAST_ARMS if arm in f}
        cand["MCMC-J62"] = (f["MCMC-J62"][cal_j62["best"]]["chi2"], 1 + k_nuis)
        cand["MCV (c)"] = (f["MCV"]["t10"][best]["chi2"], 2 + k_nuis)
        for arm, (c2, k) in cand.items():
            bics.setdefault(arm, {})[n] = sb.bic(c2, k, f["n"])
    ref = bics["ΛCDM-2"]
    dist = {arm: [v[n] - ref[n] for n in v if n in ref] for arm, v in bics.items() if arm != "ΛCDM-2"}
    return {"pares": sb.pairwise_preferences(bics),
            "delta_BIC_frente_a_LCDM2": {a: {"mediana": float(np.median(d)), "p16": float(np.percentile(d, 16)),
                                            "p84": float(np.percentile(d, 84))} for a, d in dist.items() if d}}


def analysis(galaxies: list, plan: dict, split_mode: str = "sha256", split_seed: int | None = None,
             li2020: dict | None = None, a0_population: tuple | None = None, bytes_ok: bool = False,
             processes: int = 1, local_law_A: dict | None = None, restricted: bool = True) -> dict:
    """El análisis completo: submuestra de calidad, reparto, ajustes, calibración en A (MCV (c) y r⋆ de J62), letra
    sobre B frente a ΛCDM-2 con sus puertas y las lecturas sin letra."""
    sample = sb.quality_sample(galaxies, plan["spec_bank"])
    by_name = {g.name: g for g in sample}
    halves = sb.split_halves(sorted(by_name), split_mode, plan["spec_bank"]["split_salt"], split_seed)
    fits = fit_all(sample, plan, processes)
    cal_A = _calibrate_mode(fits, halves["A"], "t10")
    cal_B = _calibrate_mode(fits, halves["B"], "t10")
    cal_j62 = sb.calibrate({r: [fits[n]["MCMC-J62"][r]["chi2"] for n in halves["A"]] for r in plan["j62_rstars"]})
    out = {"n_muestra": len(sample), "mitades": halves, "calibracion_A": cal_A, "calibracion_J62_A": cal_j62}
    if cal_A["best"] is None:
        out["letra"] = {"letter": "INDETERMINADO", "reasons": ["sin calibración posible en A"]}
        return out
    best = cal_A["best"]
    d = [fits[n]["MCV"]["t10"][best]["chi2"] - fits[n]["ΛCDM-2"]["chi2"] for n in halves["B"]]
    no_match = float(np.mean([fits[n]["MCV"]["t10"][best]["n_valid"] == 0 for n in halves["B"]])) if halves["B"] else 1.0
    chi2nu = {"NFW": {n: fits[n]["ΛCDM-1"]["chi2_data"] / max(fits[n]["n"] - 2, 1) for n in fits},
              "Burkert": {n: fits[n]["Burkert libre (E8-Q)"]["chi2_data"] / max(fits[n]["n"] - 2, 1) for n in fits}}
    e8q = sb.e8q_gate(chi2nu, li2020, a0_population, plan["spec_bank"])
    gates = {"bytes_con_manifiesto": bytes_ok, "cualificacion_E8Q": e8q["pass"], "frac_B_sin_empalme": no_match,
             "calibracion_en_borde": sb.on_grid_edge(best, plan["spec_bank"])}
    out.update({"letra": sb.letters_mcv_vs_lcdm2(d, gates, plan["spec_bank"]), "puertas": gates, "E8Q": e8q,
                "globalidad": sb.globality(cal_A, cal_B), "alcance": sb.LETTER_SCOPE,
                "memoria_del_halo": memory_reading(fits, halves, cal_A),
                "banco": bank_readings(fits, sorted(fits), cal_j62, best)})
    out["lecturas"] = readings(by_name, fits, halves, cal_A, plan, local_law_A, restricted)
    out["J62_B_delta_chi2_frente_a_LCDM2"] = [fits[n]["MCMC-J62"][cal_j62["best"]]["chi2"] - fits[n]["ΛCDM-2"]["chi2"]
                                             for n in halves["B"]] if cal_j62["best"] is not None else None
    return out
