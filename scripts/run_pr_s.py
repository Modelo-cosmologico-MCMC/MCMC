#!/usr/bin/env python
"""PR-S (bloques P0 del autor del 26 y del 27-oct-2026): SPARC alineado con el registro, letras del escalón de la RAR y
banco de mecanismos con el brazo MCV (c), con las enmiendas del 27-oct. Sin bytes, sin letra y sin congelar.

    python scripts/run_pr_s.py checks [procesos]   # comprobaciones sin datos → results/2026-10-27_pr_s/checks.{json,md}
    python scripts/run_pr_s.py draft               # BORRADOR de la preinscripción (no congelado) → preregistration_draft.*
    python scripts/run_pr_s.py freeze              # se niega mientras quede pendiente una decisión del autor
    python scripts/run_pr_s.py analyze             # FALLA CERRADO sin preinscripción congelada y sin sparc AVAILABLE

Nada de esto toca A_Sculptor, ρ*, q_S, la RAR congelada (9566a2ac0457) ni las preinscripciones del 5E, de Sculptor o
del escalón. Las galaxias de las comprobaciones son sintéticas y declaradas; ninguna cifra es señal (E13).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics import mcv_baryons as mb  # noqa: E402
from dynamics import mcv_sparc as ms  # noqa: E402
from dynamics import rar_step_letters as rs  # noqa: E402
from dynamics import sparc_bank as sb  # noqa: E402
from dynamics import sparc_data as sd  # noqa: E402
from dynamics import sparc_pr_s as pr  # noqa: E402
from dynamics.ecv_border import r0_over_r200  # noqa: E402
from dynamics.epsilon_c_saturating import (  # noqa: E402
    deps_c_drho,
    eps_c,
    exponential_disc_g_bar,
)
from dynamics.mcv_equilibrium import (  # noqa: E402
    gravothermal_collapse_time,
    nfw_halo,
    nfw_mass,
)
from dynamics.sculptor_transfer import frozen_config  # noqa: E402
from mcmc_ontology.data_registry import manifest_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-27_pr_s"
CHECKS = OUTDIR / "checks.json"
DRAFT = OUTDIR / "preregistration_draft.json"
PREREG = OUTDIR / "preregistration.json"
DECISIONS = OUTDIR / "decisiones_autor.json"
DOC_26 = "https://claude.ai/artifact/MtXFLxRgx59VxS8ek3nbiX"
DOC_27 = "https://claude.ai/artifact/BaZVZaDTmxw5P96GcQXR4z"
DOC_28 = "https://claude.ai/artifact/Bmm1d5tMgQjEateRk64Qwb"
FROZEN_UNTOUCHED = {
    "5E (ley local con A_S)": ROOT / "results" / "2026-08-16_front5e_sparc" / "preregistration.json",
    "Sculptor con la MCV (c)": ROOT / "results" / "2026-10-23_mcv_sculptor" / "preregistration.json",
    "escalón de la RAR (predicción)": ROOT / "results" / "2026-09-22_dictionary_eps_c" / "rar_step_prediction.json",
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _refuse_if_bytes() -> None:
    raw = ROOT / "data" / "raw" / "sparc"
    present = [p.name for p in raw.glob("*")] if raw.exists() else []
    if manifest_path("sparc").exists() or present:
        raise SystemExit(f"FALLO CERRADO: hay manifest o bytes de SPARC ({present}); PR-S se congela antes de los bytes")


# ------------------------------------------------------------------ galaxias sintéticas declaradas
def moster13(M200: float) -> float:
    """M⋆ de Moster, Naab y White (2013) a z = 0 (relación de las galaxias sintéticas del 26-oct)."""
    x = M200 / 10 ** 11.590
    return 2 * 0.0351 * M200 / (x ** -1.376 + x ** 0.608)


GAS_FRACTION = {10.5: 1.0, 11.0: 0.5, 11.5: 0.25, 12.0: 0.1}           # M_gas/M⋆ declarada por masa del halo
SIZES = {"compacto": 0.5, "normal": 1.0, "extendido": 2.0}              # R_d en unidades de R_ref(M⋆)


def r_ref(M_star: float) -> float:
    """R_ref = 2.5 kpc·(M⋆/1e10)^0.3 (declarado)."""
    return 2.5 * (M_star / 1e10) ** 0.3


def synthetic_galaxy(logM: float, size: str, som: float | None = 1.0, n: int = 20, noise: float = 0.0, seed: int = 0,
                     r_max_over_rd: float = 6.0):
    """Galaxia SINTÉTICA declarada: halo con la c–M mediana, M⋆ de Moster et al. (2013), gas en la proporción declarada
    en un disco de 2R_d, R_d = R_ref·{0.5, 1, 2}; con som la MCV (c) en su rama con núcleo (t_edad = 10 Gyr), sin som
    la envoltura NFW sola. Errores del 3 % (mínimo 2 km/s), ruido gaussiano opcional."""
    M200 = 10 ** logM
    c = float(ms.c200_dm14(M200))
    Ms = moster13(M200)
    Rd = SIZES[size] * r_ref(Ms)
    R = np.geomspace(0.3 * Rd, r_max_over_rd * Rd, n)
    v2s = exponential_disc_g_bar(R, Ms / (2 * math.pi * Rd ** 2) / 1e6, Rd) * R
    v2g = exponential_disc_g_bar(R, GAS_FRACTION[logM] * Ms / (2 * math.pi * (2 * Rd) ** 2) / 1e6, 2 * Rd) * R
    bar = mb.TabulatedBaryons(R, v2s + v2g)
    h = nfw_halo(M200, c, mb.H0)
    truth = {"M200": M200, "c": c, "R_d": Rd, "M_star": Ms}
    M_dm = nfw_mass(R, h["rho_s"], h["r_s"])
    if som is not None:
        brs = mb.match_branches(M200, c, som, baryons=bar, prefilter=True)["branches"]
        if not brs:
            return None, truth
        br = brs[0]
        M_dm = np.where(R < br["r1"], mb.core_profile(br["rho0"], br["sigma0"], R, bar)["M_M"], M_dm)
        truth.update({"rho0": br["rho0"], "sigma0": br["sigma0"], "r_half": br["r_c"]})
    v = np.sqrt(v2s + v2g + mb.G_KPC * M_dm / R - sb.ECV_TERM * R ** 2)
    e = np.maximum(2.0, 0.03 * v)
    if noise:
        v = v + np.random.default_rng(seed).normal(0.0, noise * e)
    gal = ms.Galaxy(f"SINT-{logM}-{size}", R, v, e, np.sqrt(v2g), np.sqrt(v2s / 0.5), np.zeros_like(R), 10.0, 1.0, 60.0,
                    4.0, R_d=Rd, SB_disk0=Ms / 0.5 / (2 * math.pi * (Rd * 1e3) ** 2), quality=1, synthetic=True)
    return gal, truth


# ------------------------------------------------------------------ comprobaciones sin datos
def check_document_numbers() -> dict:
    """Las cifras de los tests sin datos de los dos documentos (26-oct y enmienda 8 del 27-oct)."""
    M, c = 1e11, float(ms.c200_dm14(1e11))

    def rh(som, t=10.0):
        return mb.match_branches(M, c, som, t, prefilter=True)["branches"][0]["r_c"]
    Rg = np.geomspace(0.1, 40.0, 60)
    disc = mb.TabulatedBaryons(Rg, exponential_disc_g_bar(Rg, 3e9 / (8.0 * math.pi) / 1e6, 2.0) * Rg)
    br = mb.match_branches(M, c, 1.0, baryons=disc, prefilter=True)["branches"][0]
    inv = mb.inverse_map(br["rho0"], br["sigma0"], 1.0, 10.0, disc)
    r_j62 = ms.burkert_half_density_radius(ms.J62_R_STAR_KPC)
    lo, hi = math.log(0.1), math.log(0.5)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rh(math.exp(mid)) < r_j62 else (lo, mid)
    return {"r_half_sin_bariones_kpc": {str(s): rh(s) for s in (0.3, 1.0, 3.0, 10.0)},
            "r_half_por_edad_kpc": {f"σ_x/m={s}": {str(t): rh(s, t) for t in (5.0, 10.0, 12.5)} for s in (1.0, 3.0)},
            "control_producto": {"(1, 5 Gyr)": rh(1.0, 5.0), "(0.5, 10 Gyr)": rh(0.5, 10.0)},
            "J62": {"r_half_burkert_kpc": r_j62, "sigma_over_m_cm2_g": math.exp(0.5 * (lo + hi))},
            "mapa_inverso_con_disco": {"dM200_rel": abs(inv["M200"] / M - 1.0), "dc_rel": abs(inv["c"] / c - 1.0),
                                       "tolerancia_documento": [2e-4, 1e-4]}}


def check_ecv_and_bh() -> dict:
    """C1 (peso de la ECV y R₀), H5 (dentro de 10·R_d según el tamaño del disco) y C2 (r_h del agujero negro)."""
    c1 = {}
    for lm in (10.0, 11.0, 12.0):
        h = nfw_halo(10 ** lm, float(ms.c200_dm14(10 ** lm)), mb.H0)
        r = np.linspace(1e-3, 0.1 * h["r200"], 400)
        c1[str(lm)] = {"max_hasta_0.1r200": float(np.max(mb.ecv_force_fraction(r, nfw_mass(r, h["rho_s"], h["r_s"])))),
                       "en_r200_medio": float(mb.ecv_force_fraction(0.5 * h["r200"], nfw_mass(0.5 * h["r200"], h["rho_s"], h["r_s"]))),
                       "R0_kpc": r0_over_r200(0.0) * h["r200"]}
    h5 = {}
    for lm in (10.5, 11.0, 11.5, 12.0):
        for size in SIZES:
            gal, tr = synthetic_galaxy(lm, size, som=None, r_max_over_rd=10.0)
            R = np.linspace(2.0, 10.0 * tr["R_d"], 200)
            h = nfw_halo(tr["M200"], tr["c"], mb.H0)
            _, _, _, bar = ms.galaxy_baryons(gal, ms.nuisance_point(gal, 0.0, 0.0, 0.0))
            f = mb.ecv_force_fraction(R, nfw_mass(R, h["rho_s"], h["r_s"]) + bar.mass(R))
            h5[f"{lm}-{size}"] = {"R_d_kpc": tr["R_d"], "max_2kpc_a_10Rd": float(np.max(f)), "min": float(np.min(f))}
    return {"C1": c1, "H5_10Rd": h5, "C2": ms.bh_influence_table()}


def check_q_cj() -> dict:
    """C3 y el hallazgo H3: q_CJ(0) = 1.5·c²ε_c/σ₀² y c²·ρ·ε_c'/σ₀² en el centro (densidad del plano medio, ζ = 0.15),
    con el σ₀ de la rama con núcleo (σ_x/m = 1, t_edad = 10 Gyr), para las leyes locales con letra posible."""
    A = frozen_config().A_sculptor
    forms = {"A_S": (lambda r: A * r ** 1.5, lambda r: 1.5 * A * r ** 0.5),
             "0.06·A_S": (lambda r: 0.06 * A * r ** 1.5, lambda r: 0.09 * A * r ** 0.5)}
    for p in rs.frozen_points():
        forms[f"saturante ρ*={p['rho_star']:.3g} ({p['label']})"] = (
            lambda r, p=p: eps_c(r, p["eps_max"], p["rho_star"]), lambda r, p=p: deps_c_drho(r, p["eps_max"], p["rho_star"]))
    rows = {}
    for lm in (10.5, 11.0, 11.5, 12.0):
        for size in SIZES:
            gal, tr = synthetic_galaxy(lm, size)
            if gal is None:
                continue
            fit = {"sigma0": tr["sigma0"], "cfg": {"ups_disk": 0.5}}
            rows[f"{lm}-{size}"] = {"sigma0": tr["sigma0"], "rho_mid0": ms.midplane_density_center(gal, 0.5),
                                    **{k: ms.q_cj_cross_reading(fit, gal, *fn) for k, fn in forms.items()}}
    summary = {}
    for k in forms:
        for cat, sizes in (("normales", ("normal", "extendido")), ("compactos", ("compacto",))):
            sel = [v[k] for key, v in rows.items() if key.split("-")[1] in sizes]
            summary[f"{k} — {cat}"] = {"q_CJ0_1.5c2eps": [min(s["q_CJ0"] for s in sel), max(s["q_CJ0"] for s in sel)],
                                       "q_CJ0_c2_rho_deps": [min(s["q_CJ0_general"] for s in sel), max(s["q_CJ0_general"] for s in sel)]}
    return {"por_galaxia": rows, "resumen": summary}


def check_lapse_collapse_age() -> dict:
    """C5 (ΔN(0)), C7 (t_c con σ_x/m constante, c mediana y +2σ) y t_edad desde la concentración (c₁ del autor)."""
    masses = [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]
    lapse, coll = {}, {"mediana": [], "+2σ": []}
    age = {str(c1): {} for c1 in (3.0, 3.5, 4.1)}
    for lm in masses:
        c = float(ms.c200_dm14(10 ** lm))
        h = nfw_halo(10 ** lm, c, mb.H0)
        lapse[str(lm)] = 4 * math.pi * mb.G_KPC * h["rho_s"] * h["r_s"] ** 2 / mb.C_KMS ** 2
        for tag, cc in (("mediana", c), ("+2σ", c * 10 ** 0.22)):
            hh = nfw_halo(10 ** lm, cc, mb.H0)
            coll[tag] += [gravothermal_collapse_time(s, hh["rho_s"], hh["r_s"], 0.75) for s in (1.0, 10.0)]
        for c1 in (3.0, 3.5, 4.1):
            age[str(c1)][str(lm)] = float(mb.formation_age_gyr(c, c1))
    with_core = {}
    for lm in (10.5, 11.0, 11.5, 12.0):
        gal, tr = synthetic_galaxy(lm, "normal")
        if gal is None:
            continue
        _, _, _, bar = ms.galaxy_baryons(gal, ms.nuisance_point(gal, 0.0, 0.0, 0.0))
        m = mb.inverse_map(tr["rho0"], tr["sigma0"], 1.0, 10.0, bar)
        fit = {**m, "t_age": 10.0, "cfg": ms.nuisance_point(gal, 0.0, 0.0, 0.0)}
        with_core[str(lm)] = ms.lapse_depth(fit, gal)["Delta_N0"]
    return {"C5_DeltaN0_nfw": lapse, "C5_DeltaN0_con_nucleo_y_bariones": with_core,
            "C7_t_c_gyr": {k: [min(v), max(v)] for k, v in coll.items()}, "edad_desde_c_gyr": age}


def check_step() -> dict:
    """S_pred de cada punto congelado sobre discos sintéticos con un halo NFW debajo (sin escalón en los datos), con las
    dos formas de δ_pred (hallazgo H2), y la fracción de galaxias con las dos ventanas según la extensión de la curva."""
    out = {}
    for rmax in (6.0, 25.0):
        gals = [g for lm in (10.5, 11.0, 11.5, 12.0) for size in SIZES
                for g in [synthetic_galaxy(lm, size, som=None, n=60, r_max_over_rd=rmax)[0]]]
        for p in rs.frozen_points():
            for form in rs.DELTA_PRED_FORMS:
                st = rs.step_statistics(gals, p, 0.15, 1.2e-10, form=form)
                out[f"R_max={rmax:g}R_d | ρ*={p['rho_star']:.3g} ({p['label']}) | {form}"] = {
                    "n_con_dos_ventanas": st["n"], "de": len(gals), "S_pred_dex": st["S_pred"]}
    return out


def _fit_pair(args):
    lm, size, seed, zoom = args
    gal, tr = synthetic_galaxy(lm, size, noise=1.0, seed=seed)
    if gal is None:
        return None
    lr, ls = math.log10(tr["rho0"]), math.log10(tr["sigma0"])
    spec = {**ms.FIT_SPEC, "log_rho0": [lr - 0.75, lr + 0.75], "log_sigma0": [ls - 0.35, ls + 0.35],
            "zoom": {**ms.FIT_SPEC["zoom"], "enabled": zoom}}
    st = [{"sigma100": 1.0, "a": 0.0}]
    a = ms.fit_mcv(gal, st, spec)[0]["chi2"]
    b = ms.fit_mcv(gal, st, spec, refine=2)[0]["chi2"]
    return {"galaxia": gal.name, "zoom": zoom, "chi2": a, "chi2_rejilla_doble": b}


def check_resolution(processes: int) -> dict:
    """Hallazgo H1: χ²_mín con la rejilla del documento frente a la rejilla doble, con y sin el refinado local, en
    galaxias sintéticas con errores del 3 % y ruido (rangos de (ρ₀, σ₀) acotados alrededor de la verdad: lo que se mide
    es el paso)."""
    tasks = [(lm, size, i, z) for z in (False, True) for i, (lm, size) in
             enumerate([(lm, s) for lm in (10.5, 11.0, 11.5, 12.0) for s in ("compacto", "normal", "extendido")])]
    with Pool(processes) as pool:
        res = [r for r in pool.map(_fit_pair, tasks) if r is not None]
    out = {}
    for z in (False, True):
        sel = [r for r in res if r["zoom"] is z]
        shift = np.array([abs(r["chi2_rejilla_doble"] - r["chi2"]) for r in sel])
        out["rejilla y refinado" if z else "rejilla sola"] = {
            "n": len(sel), "fraccion_desplazamiento_menor_0.5": float(np.mean(shift < 0.5)),
            "desplazamiento_max": float(shift.max()), "mediana": float(np.median(shift)), "filas": sel}
    return out


def check_cost() -> dict:
    """Coste por galaxia de la MCV (c) con la rejilla completa: un ajuste global frente a ocho (lo que crece con la rejilla
    global es el mapa, no la integración de los núcleos)."""
    gal, _ = synthetic_galaxy(11.0, "normal")
    sts = [{"sigma100": s, "a": 0.0} for s in (0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)]
    t0 = time.time()
    ms.fit_mcv(gal, sts[:1])
    t1 = time.time()
    ms.fit_mcv(gal, sts)
    t2 = time.time()
    per = (t2 - t1 - (t1 - t0)) / 7.0
    n_settings = len(sb.BANK_SPEC["sigma100_grid"]) * len(sb.BANK_SPEC["a_grid"])
    return {"un_ajuste_global_s": t1 - t0, "por_ajuste_global_extra_s": per,
            "estimacion_por_galaxia_70_ajustes_s": (t1 - t0) + per * (n_settings - 1),
            "estimacion_175_galaxias_h_un_proceso": ((t1 - t0) + per * (n_settings - 1)) * 175 / 3600.0,
            "nota": "sin bulbo y con la rejilla de molestias 5×5; con bulbo, ×5; el modo con t_edad desde la "
                    "concentración cuesta más por ajuste global (barrido del punto fijo)"}


def checks(processes: int | None = None) -> int:
    n = processes or min(4, os.cpu_count() or 1)
    t0 = time.time()
    doc = {"generator": "scripts/run_pr_s.py checks", "code_commit": _git_head(),
           "generated_utc": datetime.now(timezone.utc).isoformat(),
           "estatuto": "comprobaciones sin datos ni letra, con galaxias sintéticas declaradas (E8; ninguna cifra es señal)",
           "documentos": {"26-oct": DOC_26, "27-oct": DOC_27}}
    for key, fn in (("cifras_de_los_documentos", check_document_numbers), ("ecv_y_agujero_negro", check_ecv_and_bh),
                    ("q_CJ0_C3_H3", check_q_cj), ("lapse_colapso_edad", check_lapse_collapse_age), ("escalon_H2", check_step),
                    ("coste", check_cost)):
        doc[key] = fn()
        print(f"{key}: hecho ({time.time() - t0:.0f} s)", flush=True)
    doc["resolucion_H1"] = check_resolution(n)
    print(f"resolucion_H1: hecho ({time.time() - t0:.0f} s)", flush=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CHECKS.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"checks: {CHECKS} sha256 {_sha(CHECKS)}")
    return 0


# ------------------------------------------------------------------ borrador de la preinscripción
def draft_document() -> dict:
    """El BORRADOR de la preinscripción de PR-S con las enmiendas del 27 y del 28-oct: todo lo que el código hace y lo que
    el autor tiene que decidir. No es una preinscripción: no se congela ni lleva sha256 propio."""
    return {
        "estado": "BORRADOR — no congelado; no es una preinscripción. Se congela solo con las decisiones del autor "
                  "(decisiones_autor.json) y antes de cualquier byte de SPARC o de Li+2020.",
        "fecha": "2026-10-28", "generator": "scripts/run_pr_s.py draft", "code_commit": _git_head(),
        "ordenado_por": {"26-oct": f"PR-S desde la ontología ({DOC_26})", "27-oct": f"PR-S frente a la ontología ({DOC_27})",
                         "28-oct": f"revisión de la evaluación cosmológica y ajuste fino de PR-S ({DOC_28}); su bloque "
                                   "sustituye al del 27-oct"},
        "respaldo": "rama mcv/pr-s-sparc-instrument-wip (commit «borrador sin congelar», paso 0 del bloque del 28-oct); "
                    "la congelación se hace en mcv/pr-s-sparc-instrument, con el texto final y antes de los bytes",
        "cosmologia_fija": {"fondo": "PLANCK18 de dynamics/ecv_border.py", **ms.PLANCK18_DECLARED,
                            "rho_E": "Ω_Λ·ρ_crit,0, constante", "H0": "H0_DEFAULT = 67.4 de dynamics/mcv_equilibrium.py",
                            "leyes_de_Q_o_de_H": "ninguna (ni las de #65 ni la constante de la evaluación)",
                            "t_edad": "con el t(a) de ese fondo; retroceso hasta z = 1 = 7.95 Gyr",
                            "motivo": "con la ley constante de la evaluación (ξ_0 = 0.06), R₀ en z = 1 cambiaría ×0.986 y "
                                      "t_edad entre +0.15 % y +0.74 %, frente al ±29 % de la c–M y al 30–55 % que la "
                                      "edad mueve el núcleo"},
        "S_local_operativa": ms.S_LOCAL_TEXT,
        "simbolos": "ε solo con subíndice (ε_c, ε_max); el q₀ del 27-oct es q_CJ(0), el del criterio de Cronos–Jeans; q no "
                    "se usa para la deceleración",
        "no_se_toca": {k: {"fichero": str(v.relative_to(ROOT)), "sha256": _sha(v)} for k, v in FROZEN_UNTOUCHED.items()}
        | {"A_Sculptor": frozen_config().A_sculptor, "ρ*, q_S y la RAR congelada": "intactos"},
        "ingesta": {"ficheros": list(sd.FILES), "raw_dir": "data/raw/sparc", "manifest": "data/manifests/sparc.json (SCHEMA_UNVERIFIED)",
                    "schema_report": "data/manifests/sparc.schema_report.md, antes del parser",
                    "fallo_cerrado": "sin el dataset AVAILABLE no hay catálogo (dynamics.sparc_data)"},
        "sigma_x_m": "parámetro efectivo de la tasa de autointercambio Γ_x de la energía sellada, no la sección eficaz de "
                     "una partícula nueva (enmienda 1); σ_x/m(v) = σ₁₀₀·(⟨v_rel⟩/100 km/s)^(−a), ⟨v_rel⟩ = 4σ₀/√π",
        "antecedentes_del_corpus": "Maestro §5.1.1 (la MCV como ECV atrapada) y §3.20.2 (núcleos por saturación central y "
                                   "autointeracciones), antecedentes de la decisión (c); el perfil pseudoisotermo se "
                                   "llama ρ_M, no ρ_id",
        "brazos": {k: {"halo": v["params"][0], "parametros_de_halo": v["params"][1], "previa": v["prior"],
                       "termino_ECV": v["ecv"]} for k, v in sb.ARMS.items()} | {"MCV (c)": "dynamics/mcv_sparc.fit_mcv"},
        "molestias": "Υ_d y Υ_b lognormales (0.5 y 0.7, 0.1 dex), D e i gaussianas con los errores del catálogo, "
                     "idénticas en todos los brazos; i perfilada",
        "especificaciones": {"FIT_SPEC": ms.FIT_SPEC, "BANK_SPEC": sb.BANK_SPEC, "STEP_SPEC": rs.STEP_SPEC},
        "letras_MCV": "sobre B frente a ΛCDM-2: A — mediana de Δχ² < 0 con IC95 bootstrap que excluye 0 y más de la mitad "
                      "negativas; C — mediana > 0 con IC95 que excluye 0 y menos de un tercio negativas; B — el resto; "
                      "INDETERMINADO — puertas (E8-Q, n_B < 40, más del 10 % de B sin empalme, calibración en el borde, "
                      "bytes sin manifiesto). Orden: puertas, A, C, B.",
        "alcance_de_cada_letra": sb.LETTER_SCOPE,
        "escalon": "letras por punto (ε_max, ρ*) de la predicción 9566a2ac0457 con el estadístico de ventanas; la "
                   "lectura literal, aparte; las dos formas de δ_pred (hallazgo H2)",
        "lecturas_sin_letra": [
            "globalidad de σ_x/m (calibración con B dentro de la región del 95 % de A)",
            "ramas: fracción en la rama densa y con c a más de 2σ, frente al brillo superficial",
            "r_c(M200) frente a J.6.2 (control), la pendiente 0.35 de U (166), «∼0.4 en SPARC» y la Tabla 45 del Unificado",
            "SPARC → Sculptor: σ_x/m(v) calibrado en el ⟨v_rel⟩ de Sculptor",
            "memoria del halo: t_edad por galaxia desde la concentración (c₁ congelado), recalibración en A, prueba en B, "
            "Δχ² frente a t_edad = 10 Gyr y cambio de la calibración; o 6/10/13 Gyr",
            "rama densa y colapso: t_c por galaxia (β = 0.75 y 0.45) con el σ_x/m calibrado y la envoltura ajustada; la "
            "rama densa con t_c > t_edad, marcada; variante con la rama densa solo si t_c ≤ t_edad",
            "lapse: ΔN(0) = |Φ(0)|/c² y σ₀ por galaxia",
            "q_CJ(0) cruzado (antes q0): si el 5E o el escalón reciben A, q_CJ(0) = 1.5c²ε_c/σ₀² (y c²ρε_c'/σ₀²) de cada galaxia",
            "banco: BIC con k completo, preferencias por pares, ΔBIC frente a ΛCDM-2, a₀ por galaxia frente al brillo"],
        "fuera_y_declarado": {"cuspide_del_agujero_negro": ms.BH_OUT_OF_SCOPE, "gap": ms.GAP_OUT_OF_SCOPE,
                              "cronos/halo_profile.py": f"fuera de PR-S: {ms.HALO_PROFILE_LABEL} (12.6 kpc en 1e11 M☉)",
                              "intercambio_con_la_ECV": "no se ve en SPARC (C1); se prueba en el fondo y en el borde"},
        "expectativas_E13": [
            "MCV (c): A o B (Ren et al. 2019 ajustan 135 galaxias con σ/m = 3 cm²/g y la c–M de Planck).",
            "Riesgo: si SPARC calibra σ_x/m ~ 2–3 cm²/g con a ≥ 0, la rama con núcleo de Sculptor se queda corta; en la "
            "rama densa, Sculptor tendría t_c de 150–740 Gyr (incoherente salvo que σ_x/m crezca mucho a baja velocidad).",
            "Escalón: con ρ* = 0.001, comprobable; con ρ* = 0.01–0.03, INDETERMINADO por no comprobable.",
            "Lectura literal del escalón: C."],
        "decisiones_pendientes": pr.PENDING_DECISIONS,
        "decisiones_28oct_que_no_bloquean": pr.DECISIONS_28OCT,
        "hallazgos_del_ejecutor": pr.EXECUTOR_FINDINGS,
        "codigo": ["dynamics/mcv_baryons.py", "dynamics/mcv_sparc.py", "dynamics/sparc_bank.py",
                   "dynamics/rar_step_letters.py", "dynamics/sparc_pr_s.py", "dynamics/sparc_data.py",
                   "scripts/run_pr_s.py", "scripts/download_data.py"],
    }


def _md(doc: dict) -> str:
    lines = ["# PR-S — BORRADOR de la preinscripción (no congelado)", "", f"> {doc['estado']}", "",
             f"Generado por `{doc['generator']}` en el commit `{doc['code_commit']}` ({doc['fecha']}).", ""]
    lines += ["## Decisiones pendientes del autor", "", "| Id | Decisión | Recomendación | Alternativa | En el código |",
              "| --- | --- | --- | --- | --- |"]
    lines += [f"| {d['id']} | {d['decision']} | {d['recomendacion']} | {d['alternativa']} | {d['en_el_codigo']} |"
              for d in doc["decisiones_pendientes"]]
    lines += ["", "## Decisiones del 28-oct (no bloquean PR-S)", "", "| Id | Decisión | Recomendación | En PR-S |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {d['id']} | {d['decision']} | {d['recomendacion']} | {d['en_PR_S']} |"
              for d in doc["decisiones_28oct_que_no_bloquean"]]
    lines += ["", "## Hallazgos del ejecutor", "", "| Id | Hallazgo | Opciones | Efecto |", "| --- | --- | --- | --- |"]
    lines += [f"| {h['id']} | {h['hallazgo']} | {h['opciones']} | {h['efecto']} |" for h in doc["hallazgos_del_ejecutor"]]
    lines += ["", "## Alcance de cada letra", ""]
    for k, v in doc["alcance_de_cada_letra"].items():
        lines += [f"- **{k}.** Prueba {v['prueba']}. No puede decir: {v['no_puede_decir']}."]
    lines += ["", "## Lecturas sin letra", ""] + [f"- {x}" for x in doc["lecturas_sin_letra"]]
    lines += ["", "## Cosmología fija y declarada", ""] + [f"- {k}: {v}" for k, v in doc["cosmologia_fija"].items()]
    lines += ["", "## S_local operativa", "", doc["S_local_operativa"] + ".", "", "## Símbolos", "", doc["simbolos"] + "."]
    lines += ["", "## Respaldo", "", doc["respaldo"] + "."]
    lines += ["", "## Fuera y declarado", ""] + [f"- {k}: {v}" for k, v in doc["fuera_y_declarado"].items()]
    lines += ["", "## Expectativas E13 (antes de congelar)", ""] + [f"- {x}" for x in doc["expectativas_E13"]]
    lines += ["", "## No se toca", ""] + [f"- {k}: {v}" for k, v in doc["no_se_toca"].items()]
    lines += ["", "El resto (brazos, especificaciones, letras y escalón) está en `preregistration_draft.json`.", ""]
    return "\n".join(lines)


def draft() -> int:
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)})")
    doc = draft_document()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DRAFT.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    DRAFT.with_suffix(".md").write_text(_md(doc), encoding="utf-8")
    print(f"borrador: {DRAFT} y {DRAFT.with_suffix('.md')} (no congelado)")
    return 0


def pending(decisions_path: Path = DECISIONS) -> list[str]:
    """Decisiones sin resolver: todas, si no hay fichero del autor; si lo hay, las que no traen «elegida»."""
    ids = [d["id"] for d in pr.PENDING_DECISIONS] + [h["id"] for h in pr.EXECUTOR_FINDINGS]
    if not decisions_path.exists():
        return ids
    got = json.loads(decisions_path.read_text(encoding="utf-8"))
    return [i for i in ids if not str(got.get(i, {}).get("elegida", "")).strip()]


def freeze(decisions_path: Path = DECISIONS) -> int:
    """Congela la preinscripción solo con todas las decisiones del autor y sin bytes; se niega en cualquier otro caso."""
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)}); no se sobrescribe")
    left = pending(decisions_path)
    if left:
        raise SystemExit("FALLO CERRADO: decisiones del autor pendientes, no se congela: " + ", ".join(left))
    _refuse_if_bytes()
    doc = draft_document()
    doc["estado"] = "congelada antes de los bytes"
    doc["decisiones_del_autor"] = json.loads(decisions_path.read_text(encoding="utf-8"))
    doc["decisiones_del_autor_sha256"] = _sha(decisions_path)
    doc["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {_sha(PREREG)}")
    return 0


def analyze() -> int:
    """FALLA CERRADO: sin preinscripción congelada no hay análisis; con ella, sin el dataset sparc AVAILABLE tampoco."""
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: PR-S no está congelada (faltan las decisiones del autor); no se analiza nada")
    from dynamics.sparc_data import load_sparc_catalogue
    load_sparc_catalogue()
    raise SystemExit("FALLO CERRADO: el análisis con bytes se escribe tras congelar, con las decisiones del autor")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "checks":
        raise SystemExit(checks(int(sys.argv[2]) if len(sys.argv) > 2 else None))
    raise SystemExit({"draft": draft, "freeze": freeze, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
