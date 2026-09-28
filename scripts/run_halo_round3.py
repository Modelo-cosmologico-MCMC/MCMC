#!/usr/bin/env python
"""Frente 5 (b), RONDA 3 del halo con capas esféricas (orden del autor del 28-sep, PR-5): integrador corregido,
barrido de suavizado y brazo de control saturante — bajo preinscripción nueva con puertas heredadas de la
cualificación E8-Q (results/2026-09-28_qualification_shells/).

    python scripts/run_halo_round3.py prereg     # congela (en un commit posterior al generador y a la cualificación)
    python scripts/run_halo_round3.py run [--arms a,b]   # corridas reanudables (una por fichero)
    python scripts/run_halo_round3.py analyze    # regla congelada; falla cerrado sin ella

Las dos lecturas de la ronda 2 (INDETERMINADA): el brief dice que el cuello es el integrador y la resolución del
interior; la adenda del autor, que es la ley (la cúspide NFW es Cronos–Jeans-inestable para toda A > 0 y solo el
suavizado o una saturación la detienen). Se deciden con datos, no discutiendo:
  * BARRIDO DE SUAVIZADO ε_soft ∈ {0.2, 0.1, 0.05, 0.025} kpc a A_Sculptor y 0.05·A_Sculptor, N = 1e6, ds = 0.04:
    predicción t_weak ∝ ε_soft^p con p > 0 (la ley: el colapso es más rápido cuanto más adentro se resuelve) frente a
    t_weak independiente de ε_soft (el integrador).
  * SERIE DE RESOLUCIÓN ds ∈ {0.08, 0.04, 0.02} a ε_soft = 0.1 con rango M(<r) actualizado: predicción de no
    convergencia a A_Sculptor (firma UV: t_weak decrece al refinar).
  * BRAZO DE CONTROL SATURANTE (etiquetado control, no decisión): ε_c = ε_max ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}) en dos
    puntos declarados del 58 % viable de dictionary_eps_c — ρ* = ρ_NFW(0.1 kpc) y ε_max = f·ε_max_crit con
    ε_max_crit el techo para el que máx_r c²ρε_c'(ρ)/σ_r² = 1 sobre la tabla de Jeans del halo (f = 0.5:
    sub-umbral en TODA la cúspide; f = 2: justo por encima); predicción: el primero NO sale del régimen débil y
    converge con ds (contracción confinada a ρ ≈ ρ*), el segundo sale.
Puertas heredadas de la cualificación (regla del programa: solo se congelan puertas que una cualificación haya
mostrado alcanzables, assert_gate_attainable): energía = 3 × el suelo medido en el brazo newtoniano con rango
actualizado (y el suelo de los brazos de Cronos se publica: la preinscripción dice a priori qué brazos pueden
cumplirla); estacionariedad newtoniana 0.03 dex (≥ el máximo medido); ≥ 2000 capas dentro de 0.4 kpc; control de
N dentro de ×2; dt_min = 1e-5 Myr (≤ el mínimo con que salen las corridas de A_Sculptor de la cualificación).
Desenlaces: A (barrido con p > 0, serie no convergente a A_S, control saturante como predicho), B (p > 0 pero el
control sub-umbral también sale), C (t_weak independiente de ε_soft: el integrador manda y la lectura de la
adenda se retira), INDETERMINADO (puertas). La ronda 2 queda intacta.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.qualification import (  # noqa: E402
    assert_gate_attainable,
    load_qualification,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-28_halo_round3"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"
QUALIFICATION = ROOT / "results" / "2026-09-28_qualification_shells"
DICTIONARY = ROOT / "results" / "2026-09-22_dictionary_eps_c" / "eps_c_saturating.json"

H0 = 67.86705532886631
EPS_SWEEP = (0.2, 0.1, 0.05, 0.025)
DS_SERIES = (0.02, 0.04, 0.08)
N_MAIN, N_CTRL = 1_000_000, 100_000
DS_MAIN, EPS_MAIN = 0.04, 0.1
DT_MAX_MYR, DT_MIN_MYR, ETA_DT, ETA_FIELD, K_MAX = 0.15, 1e-5, 0.05, 0.02, 10
T_END_GYR = {"newton": 0.05, "AS": 0.02, "b005": 0.05, "sat": 0.05}
SNAPSHOTS = [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.03, 0.04, 0.05]
SAT_RHO_STAR_RULE = "ρ* = ρ_NFW(r = 0.1 kpc) de la tabla de Jeans del halo (M200 = 1e11, c = 10)"
SAT_FACTORS = {"sub": 0.5, "above": None}                  # 'above': el mayor de SAT_ABOVE_CANDIDATES que cumple Oort y las filas externas
SAT_ABOVE_CANDIDATES = (2.0, 1.5, 1.25)
RULES = {"energy_gate_factor": 3.0, "stationarity_dex": 0.03, "n_min_shells_within_0p4": 2000, "N_control_factor": 2.0,
         "p_power_law_min": 0.25, "p_independent_max": 0.10, "fit_r2_min": 0.9,
         "sat_convergence_dex": 0.03,
         "letters": {"A": "barrido: p ≥ p_power_law_min con r² ≥ fit_r2_min en las dos amplitudes; serie ds a A_Sculptor sin convergencia (todas salen y "
                          "t_weak decrece estrictamente al refinar); control saturante: 'sub' no sale en ninguna ds y M(<0.4) converge con ds "
                          "(≤ sat_convergence_dex), 'above' sale",
                     "B": "barrido con p ≥ p_power_law_min pero el control saturante 'sub' TAMBIÉN sale del régimen débil",
                     "C": "barrido con |p| ≤ p_independent_max en A_Sculptor (t_weak independiente de ε_soft: el integrador manda; la lectura de la adenda se retira)",
                     "INDETERMINADO": "puerta violada (energía, capas interiores, estacionariedad, control de N, corridas ausentes) o p entre "
                                      "p_independent_max y p_power_law_min o r² < fit_r2_min; ningún otro patrón se clasifica"},
         "order": "puertas → C → B → A; ningún umbral se toca tras ver los números"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _log(path: Path):
    def log(line: str):
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def saturating_points() -> dict:
    """Los dos puntos (ε_max, ρ*) del brazo de control, por la regla declarada, comprobados contra el 58 % viable."""
    from cronos.halo_nbody import C_KMS
    from dynamics.cronos_jeans import nfw_jeans_table
    from dynamics.epsilon_c_saturating import (
        deps_c_drho,
        oort_bound_on_deps,
        q_local,
        stability_systems,
    )
    tab = nfw_jeans_table()
    rows = tab["rows"]
    rho_star = float(next(r["rho_msun_pc3"] for r in rows if abs(r["r_kpc"] - 0.1) < 1e-9))
    # ε_max_crit: máx_r q(r) = 1 sobre la cúspide (q ∝ ε_max)
    q_unit = max(float(C_KMS ** 2 * r["rho_msun_pc3"] * deps_c_drho(r["rho_msun_pc3"], 1.0, rho_star) / r["sigma_r2_kms2"]) for r in rows)
    eps_crit = 1.0 / q_unit
    oort = oort_bound_on_deps()
    systems = stability_systems()

    def point(f: float) -> dict:
        em = f * eps_crit
        q_prof = [{"r_kpc": r["r_kpc"], "q": float(q_local(r["rho_msun_pc3"], r["sigma_r2_kms2"], em, rho_star))} for r in rows]
        ok_oort = bool(deps_c_drho(oort["rho0_msun_pc3"], em, rho_star) <= oort["deps_max_at_rho0"])
        ok_stab = all(bool(q_local(s["rho"], s["sigma2"], em, rho_star) < 1.0) for s in systems)
        # la tabla de estabilidad del diccionario incluye el propio halo NFW: un punto «justo por encima» del umbral de la
        # cúspide viola esas filas POR DEFINICIÓN; su viabilidad se comprueba con las filas externas (plano solar, Sculptor)
        ok_stab_ext = all(bool(q_local(s["rho"], s["sigma2"], em, rho_star) < 1.0) for s in systems if not s["system"].startswith("halo NFW"))
        return {"eps_max": float(em), "rho_star": rho_star, "factor_over_crit": f, "A_weak_equivalent": float(em / rho_star ** 1.5),
                "q_profile": q_prof, "q_max_cusp": max(x["q"] for x in q_prof),
                "in_viable_region": {"oort": ok_oort, "stability_table": ok_stab, "stability_table_external_rows": ok_stab_ext,
                                     "both": bool(ok_oort and ok_stab), "oort_and_external": bool(ok_oort and ok_stab_ext)}}

    pts = {"sub": point(SAT_FACTORS["sub"])}
    tried = {}
    for f in SAT_ABOVE_CANDIDATES:                         # de mayor a menor: el primero viable (Oort + filas externas) es 'above'
        cand = point(f)
        tried[str(f)] = cand["in_viable_region"]
        if cand["in_viable_region"]["oort_and_external"]:
            pts["above"] = cand
            break
    if "above" not in pts:
        raise SystemExit(f"FALLO CERRADO: ningún candidato 'above' {SAT_ABOVE_CANDIDATES} cumple Oort y las filas externas: {tried}")
    return {"rule": SAT_RHO_STAR_RULE, "eps_max_crit": float(eps_crit), "rho_star": rho_star, "points": pts, "above_candidates_tried": tried,
            "viability_rule": "'sub' debe estar en el 58 % viable completo; 'above' viola las filas del halo NFW por definición y es el mayor factor de "
                              f"{list(SAT_ABOVE_CANDIDATES)} que cumple Oort y las filas externas",
            "dictionary_artifact": str(DICTIONARY.relative_to(ROOT)), "dictionary_sha256": _sha(DICTIONARY) if DICTIONARY.exists() else None}


def arms_table() -> dict:
    arms = {"newton_N1e6": {"label": "newtoniano, N = 1e6", "form": "weak", "cronos": False, "amplitude": 0.0, "ds": DS_MAIN, "eps_soft": EPS_MAIN, "N": N_MAIN, "t_end_gyr": T_END_GYR["newton"]}}
    for amp, tag in ((1.0, "AS"), (0.05, "b005")):
        for e in EPS_SWEEP:
            arms[f"sweep_{tag}_eps{e}"] = {"label": f"barrido ε_soft = {e} kpc, {tag}", "form": "weak", "cronos": True, "amplitude": amp, "ds": DS_MAIN,
                                          "eps_soft": e, "N": N_MAIN, "t_end_gyr": T_END_GYR[tag]}
    for d in DS_SERIES:
        if d != DS_MAIN:
            arms[f"series_AS_ds{d}"] = {"label": f"serie ds = {d}, A_Sculptor, ε_soft = {EPS_MAIN}", "form": "weak", "cronos": True, "amplitude": 1.0, "ds": d,
                                       "eps_soft": EPS_MAIN, "N": N_MAIN, "t_end_gyr": T_END_GYR["AS"]}
    for name in SAT_FACTORS:
        for d in (DS_SERIES if name == "sub" else (DS_MAIN,)):
            arms[f"sat_{name}_ds{d}"] = {"label": f"control saturante '{name}', ds = {d}", "form": "saturating", "cronos": True, "amplitude": None, "ds": d,
                                        "eps_soft": EPS_MAIN, "N": N_MAIN, "t_end_gyr": T_END_GYR["sat"], "sat_point": name}
    for amp, tag in ((1.0, "AS"), (0.05, "b005")):
        arms[f"ctrlN_{tag}_N1e5"] = {"label": f"control de N, {tag}, N = 1e5", "form": "weak", "cronos": True, "amplitude": amp, "ds": DS_MAIN,
                                     "eps_soft": EPS_MAIN, "N": N_CTRL, "t_end_gyr": T_END_GYR[tag]}
    return arms


def prereg(_args) -> int:
    from cronos.halo_nbody import A_SCULPTOR, nfw_structural
    from dynamics.cronos_jeans import nfw_jeans_table
    qual = load_qualification(QUALIFICATION)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # puertas heredadas: fallo cerrado si la cualificación no las muestra alcanzables
    floor_newton = qual["limits"]["energy_floor_newton_rank_update1"]["value"]
    tol_e_newton = RULES["energy_gate_factor"] * floor_newton
    gate_newton = assert_gate_attainable(qual, "energy_floor_newton_rank_update1", tol_e_newton, RULES["energy_gate_factor"])
    st_max = qual["limits"]["newton_stationarity_dex_max"]["value"]
    if RULES["stationarity_dex"] < st_max:
        raise SystemExit(f"FALLO CERRADO: puerta de estacionariedad {RULES['stationarity_dex']} < máximo medido {st_max}")
    cronos_floors = {k: v for k, v in qual["limits"].items() if k.startswith("energy_floor_") and "newton" not in k}
    cronos_arms_can_meet = {k: bool(v["value"] <= tol_e_newton) for k, v in cronos_floors.items()}
    dtmin_ok = any(k.startswith("AS_t_weak") for k in qual["limits"])
    sat = saturating_points()
    for name, p in sat["points"].items():
        need = "both" if name == "sub" else "oort_and_external"
        if not p["in_viable_region"][need]:
            raise SystemExit(f"FALLO CERRADO: el punto saturante '{name}' no cumple la regla de viabilidad ({need}): {p['in_viable_region']}")
    cj = nfw_jeans_table(A=A_SCULPTOR)
    doc = {
        "title": "Frente 5 (b), ronda 3: integrador con rango M(<r) actualizado, barrido de suavizado ε_soft, serie ds y brazo de control saturante, bajo puertas heredadas de la cualificación E8-Q",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_halo_round3.py prereg", "generator_commit_must_precede_freeze": True,
        "qualification": {"path": str(QUALIFICATION.relative_to(ROOT)), "sha256": qual["_sha256"], "code_commit": qual["code_commit"],
                          "energy_gate_record": gate_newton, "cronos_floors_measured": {k: v["value"] for k, v in cronos_floors.items()},
                          "cronos_arms_can_meet_energy_gate_a_priori": cronos_arms_can_meet,
                          "dt_min_rule_satisfied": dtmin_ok, "newton_stationarity_dex_max_measured": st_max},
        "round2": "results/2026-09-22_halo_shells (INDETERMINADO por energía y suelo de dt): intacta",
        "question": "¿Depende t_weak de ε_soft como una potencia (la ley) o es independiente (el integrador)? ¿Converge la serie ds a A_Sculptor? "
                    "¿Se comporta la forma saturante sub-umbral como predice la ecuación 1 (no sale, converge) y la supra-umbral sale?",
        "prediction_from_criterion": {"r_CJ_kpc_A_sculptor": cj["r_CJ_kpc"], "r_CJ_kpc_0p05": cj["r_CJ_for_A_fraction"].get("0.05"), "expected": "A"},
        "amplitudes": {"A_sculptor": A_SCULPTOR, "fraction_arm": 0.05},
        "system": {"M200_msun": 1e11, "c": 10.0, "H0": H0, "seed": 1, "r_decay_factor": 0.3, "refine_r_kpc": 2.0, "refine_beta": 1.5,
                   "structural": nfw_structural(1e11, 10.0, H0)},
        "instrument": {"code": "cronos/halo_shells.py (capas esféricas, rango M(<r) ACTUALIZADO en los subpasos: rank_update = True)",
                       "dt_max_myr": DT_MAX_MYR, "dt_min_myr": DT_MIN_MYR, "eta_dt": ETA_DT, "eta_field": ETA_FIELD, "k_max": K_MAX,
                       "saturating_form": "ε_c = ε_max ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}); fuerza = gradiente exacto de U = −c²Σ_b V_b G(ρ_b), G' = ε_c; Γ = ε_c'(ρ)ρ̇Θ(ρ̇)",
                       "max_wall_hours_per_run": 4.0, "weak_regime_eps_max": 1e-3},
        "saturating_control": sat,
        "arms": arms_table(), "snapshots_gyr": SNAPSHOTS,
        "gates": {"tol_E_newton": tol_e_newton, "tol_E_cronos": tol_e_newton, "stationarity_dex": RULES["stationarity_dex"],
                  "n_min_shells_within_0p4": RULES["n_min_shells_within_0p4"], "N_control_factor": RULES["N_control_factor"],
                  "energy_gate_meaning": "|Δ(K+W)/E| (newton) y |ΔE_self/E| (Cronos) ≤ 3 × el suelo newtoniano medido con rango actualizado; los brazos de Cronos "
                                         "cuyo suelo medido en la cualificación supera la puerta quedan declarados a priori como no cualificados para ella: si "
                                         "fallan la puerta el desenlace es INDETERMINADO y se publica que estaba anunciado"},
        "rules": RULES,
        "metric": "t_weak = instante en que ε_c máx supera weak_regime_eps_max (parada declarada); r_exit; M(<0.1), M(<0.4) en la salida; None si llega a t_end",
        "sweep_fit": "p = pendiente de ln t_weak frente a ln ε_soft por mínimos cuadrados sobre las cuatro ε_soft (solo brazos que salen); r² de la recta",
        "expectations_E13": {"letter": "A", "p": "p > 0: r_CJ ∝ A^{2/5}-ish y la cúspide resuelta más adentro colapsa antes (adenda del autor); el brief espera independencia (C)",
                             "sat_sub": "no sale; M(<0.4) converge con ds", "sat_above": "sale"},
        "development_declaration": {"pilots": "ninguno específico de la ronda 3: la cualificación E8-Q de las capas midió el suelo de energía por N, ds, ε_soft, dt_min y modo de rango "
                                              "(results/2026-09-28_qualification_shells); el instrumento saturante se probó solo en tests (gradiente exacto, energía a 1e-5 en 4 Myr)",
                                    "what_they_fixed": "solo el instrumento y las puertas alcanzables; ninguna letra"},
        "what_this_cannot_decide": ["si la Ley de Cronos débil o la saturante es la ley (el brazo saturante es CONTROL: prueba la ecuación 1 como física solo si se cumple la predicción)",
                                    "la amplitud (A_Sculptor es hipótesis del 5E)", "los modos no radiales y el interior por debajo de ε_soft"],
        "prohibitions": {"no_threshold_tuning": True, "no_data": True, "no_change_to_A_sculptor": True, "round2_untouched": True, "qualification_untouched": True},
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador). sha256 de `preregistration.json`: `{sha}`. "
          f"Cualificación citada: `{qual['_sha256'][:12]}…` ({doc['qualification']['path']}).", "", f"**Pregunta**: {doc['question']}", "",
          f"**Puertas heredadas**: energía ≤ {tol_e_newton:.2e} (3 × suelo newtoniano {floor_newton:.2e}); brazos de Cronos que la cualificación dice que pueden cumplirla: "
          f"{cronos_arms_can_meet}; estacionariedad ≤ {RULES['stationarity_dex']} dex (máximo medido {st_max:.4f}).", "",
          f"**Control saturante**: {SAT_RHO_STAR_RULE}: ρ* = {sat['rho_star']:.4f} M☉/pc³, ε_max_crit = {sat['eps_max_crit']:.3e}; puntos "
          + "; ".join(f"'{k}' ε_max = {p['eps_max']:.3e} (×{p['factor_over_crit']}, q_máx = {p['q_max_cusp']:.2f}, viable {p['in_viable_region']['both']})" for k, p in sat["points"].items()) + ".", "",
          "## Brazos", "", "| brazo | forma | A/A_S | ds | ε_soft | N | t_end [Gyr] |", "|---|---|---|---|---|---|---|"]
    for k, a in doc["arms"].items():
        md.append(f"| {k} | {a['form']} | {a['amplitude'] if a['amplitude'] is not None else a.get('sat_point')} | {a['ds']} | {a['eps_soft']} | {a['N']} | {a['t_end_gyr']} |")
    md += ["", "## Reglas (congeladas)", ""] + [f"- **{k}**: {v}" for k, v in RULES.items()] + ["", "## Expectativas (E13)", ""] + \
          [f"- **{k}**: {v}" for k, v in doc["expectations_E13"].items()] + ["", "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}; puerta de energía {tol_e_newton:.2e}; saturante ρ* = {sat['rho_star']:.4f}, ε_max_crit = {sat['eps_max_crit']:.3e}")
    return 0


def _load_prereg():
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada de la ronda 3")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


_IC_CACHE: dict = {}


def run_one(pre: dict, arm: str, log) -> dict:
    from cronos.halo_shells import A_SCULPTOR, ShellRun, equilibrium_shells
    sysm, spec, inst = pre["system"], pre["arms"][arm], pre["instrument"]
    t0 = time.time()
    if spec["N"] not in _IC_CACHE:
        _IC_CACHE[spec["N"]] = equilibrium_shells(sysm["M200_msun"], sysm["c"], sysm["H0"], spec["N"], sysm["seed"], r_decay_factor=sysm["r_decay_factor"],
                                                  refine_r_kpc=sysm["refine_r_kpc"], refine_beta=sysm["refine_beta"])
    ic = _IC_CACHE[spec["N"]]
    log(f"  capas N = {spec['N']} ({time.time()-t0:.0f}s), N(<0.4) = {int((ic['r'] < 0.4).sum())}")
    kw = {"cronos": spec["cronos"], "ds": spec["ds"], "eps_soft": spec["eps_soft"], "dt_myr": inst["dt_max_myr"], "eta_dt": inst["eta_dt"],
          "eta_field": inst["eta_field"], "k_max": inst["k_max"], "dt_min_myr": inst["dt_min_myr"], "rank_update": True, "stop_speed_kms": None,
          "weak_max": inst["weak_regime_eps_max"], "max_wall_s": inst["max_wall_hours_per_run"] * 3600.0}
    if spec["form"] == "saturating":
        p = pre["saturating_control"]["points"][spec["sat_point"]]
        kw.update({"eps_form": "saturating", "eps_max": p["eps_max"], "rho_star": p["rho_star"], "A": A_SCULPTOR})
    else:
        kw["A"] = spec["amplitude"] * A_SCULPTOR
    run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], **kw)
    log(f"  brazo {arm}: {run.events[0]}")
    out = run.run(spec["t_end_gyr"], pre["snapshots_gyr"], log=log)
    out.update({"arm": arm, "N_within_0p4_initial": int((ic["r"] < 0.4).sum())})
    return out


def cmd_run(args) -> int:
    pre, psha = _load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    log = _log(OUTDIR / "run.log")
    log(f"[prereg] {psha[:12]}… — inicio de corridas")
    for arm in (args.arms.split(",") if args.arms else list(pre["arms"])):
        path = RUNS / f"{arm}.json"
        if path.exists():
            log(f"  {path.name} ya existe: se omite")
            continue
        log(f"== brazo {arm} ({pre['arms'][arm]['label']})")
        out = run_one(pre, arm, log)
        out["preregistration_sha256"] = psha
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log(f"  guardado {path.name} ({out['wall_s']} s, {out['n_steps']} pasos; parada temprana: {out['stopped_early']} {out['stop_reason'] or ''})")
    log("fin de corridas")
    return 0


def _exit(d: dict, weak_max: float):
    left = any(ev.get("kind") == "régimen débil violado" for ev in d["events"])
    snap = d["snapshots"][-1]
    if left and snap["eps_c_max"] > weak_max:
        return snap["t_gyr"], snap["r_eps_max_kpc"], snap
    return None, None, snap


def _power_fit(eps: list, tw: list) -> dict:
    if len(eps) < 3:
        return {"p": None, "r2": None, "n": len(eps)}
    x, y = np.log(np.asarray(eps)), np.log(np.asarray(tw))
    p, b = np.polyfit(x, y, 1)
    res = y - (p * x + b)
    r2 = 1.0 - float(np.sum(res ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300))
    return {"p": float(p), "intercept": float(b), "r2": r2, "n": len(eps)}


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    G, R, arms = pre["gates"], pre["rules"], pre["arms"]
    weak_max = pre["instrument"]["weak_regime_eps_max"]
    runs = {}
    for arm in arms:
        p = RUNS / f"{arm}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("preregistration_sha256") != psha:
                raise SystemExit(f"FALLO CERRADO: {p.name} cita otra preinscripción")
            runs[arm] = d
    missing = [a for a in arms if a not in runs]
    gates = {"energy": True, "inner_shells": True, "newton_stationarity": True, "N_control": True, "runs_complete": not missing}
    per_arm, energy_failures = {}, []
    for arm, d in runs.items():
        e = [s["energy"] for s in d["snapshots"]]
        key = "E_self" if arms[arm]["cronos"] else "E_grav_only"
        dE = max(abs(x[key] - e[0][key]) / abs(e[0][key]) for x in e)
        t_weak, r_exit, snap = _exit(d, weak_max)
        per_arm[arm] = {"dE_rel_max": dE, "t_weak_gyr": t_weak, "r_exit_kpc": r_exit, "M01_exit": snap["M_within"]["0.1"], "M04_exit": snap["M_within"]["0.4"],
                        "M04_final": d["snapshots"][-1]["M_within"]["0.4"], "eps_max_final": snap["eps_c_max"], "t_final_gyr": d["t_final_gyr"],
                        "stopped_early": d["stopped_early"], "stop_reason": d["stop_reason"], "n_steps": d["n_steps"], "wall_s": d["wall_s"],
                        "N_within_0p4_initial": d["N_within_0p4_initial"], "dt_adaptive": d.get("dt_adaptive"), "eps_form": d.get("eps_form")}
        tol = G["tol_E_cronos"] if arms[arm]["cronos"] else G["tol_E_newton"]
        if dE > tol:
            gates["energy"] = False
            energy_failures.append(arm)
        if arms[arm]["N"] >= N_MAIN and d["N_within_0p4_initial"] < G["n_min_shells_within_0p4"]:
            gates["inner_shells"] = False
    stationarity = None
    if "newton_N1e6" in runs:
        m0 = runs["newton_N1e6"]["snapshots"][0]["M_within"]["0.4"]
        stationarity = max(abs(np.log10(s["M_within"]["0.4"] / m0)) for s in runs["newton_N1e6"]["snapshots"])
        gates["newton_stationarity"] = bool(stationarity <= G["stationarity_dex"])
    # barrido de suavizado
    sweep = {}
    for tag in ("AS", "b005"):
        pts = [(e, per_arm[f"sweep_{tag}_eps{e}"]["t_weak_gyr"]) for e in EPS_SWEEP if f"sweep_{tag}_eps{e}" in per_arm]
        exits = [(e, t) for e, t in pts if t is not None]
        fit = _power_fit([e for e, _ in exits], [t for _, t in exits])
        sweep[tag] = {"t_weak_by_eps": {str(e): t for e, t in pts}, "n_exit": len(exits), "n_arms": len(pts), "fit": fit,
                      "power_law": bool(fit["p"] is not None and fit["p"] >= R["p_power_law_min"] and fit["r2"] >= R["fit_r2_min"] and len(exits) == len(EPS_SWEEP)),
                      "independent": bool(fit["p"] is not None and abs(fit["p"]) <= R["p_independent_max"] and len(exits) == len(EPS_SWEEP))}
    # serie ds a A_Sculptor (ds = 0.04 es sweep_AS_eps0.1)
    ds_arms = {0.02: "series_AS_ds0.02", 0.04: "sweep_AS_eps0.1", 0.08: "series_AS_ds0.08"}
    tw_ds = {d: per_arm[a]["t_weak_gyr"] for d, a in ds_arms.items() if a in per_arm}
    series_AS = {"complete": len(tw_ds) == 3, "t_weak_by_ds": {str(k): v for k, v in tw_ds.items()},
                 "all_exit": len(tw_ds) == 3 and all(v is not None for v in tw_ds.values()),
                 "UV_no_convergence": len(tw_ds) == 3 and all(v is not None for v in tw_ds.values()) and tw_ds[0.02] < tw_ds[0.04] < tw_ds[0.08]}
    # control saturante
    sub_arms = [f"sat_sub_ds{d}" for d in DS_SERIES if f"sat_sub_ds{d}" in per_arm]
    sub_exit = any(per_arm[a]["t_weak_gyr"] is not None for a in sub_arms)
    m04 = [per_arm[a]["M04_final"] for a in sub_arms]
    sub_conv = len(sub_arms) == 3 and (max(abs(np.log10(m / m04[0])) for m in m04) <= R["sat_convergence_dex"])
    above = per_arm.get(f"sat_above_ds{DS_MAIN}")
    sat = {"sub_arms": sub_arms, "sub_any_exit": sub_exit, "sub_M04_final_by_arm": {a: per_arm[a]["M04_final"] for a in sub_arms},
           "sub_converges_with_ds": bool(sub_conv), "above_exits": bool(above is not None and above["t_weak_gyr"] is not None),
           "above_t_weak_gyr": None if above is None else above["t_weak_gyr"], "complete": len(sub_arms) == 3 and above is not None,
           "as_predicted": bool(len(sub_arms) == 3 and not sub_exit and sub_conv and above is not None and above["t_weak_gyr"] is not None)}
    # control de N
    n_ctrl = {}
    for tag in ("AS", "b005"):
        a6, a5 = f"sweep_{tag}_eps{EPS_MAIN}", f"ctrlN_{tag}_N1e5"
        if a6 in per_arm and a5 in per_arm:
            t6, t5 = per_arm[a6]["t_weak_gyr"], per_arm[a5]["t_weak_gyr"]
            ratio = (t5 / t6) if (t6 and t5) else None
            ok = None if ratio is None else bool(1.0 / G["N_control_factor"] <= ratio <= G["N_control_factor"])
            n_ctrl[tag] = {"t_weak_1e6": t6, "t_weak_1e5": t5, "ratio": ratio, "within_factor": ok}
            if ok is False:
                gates["N_control"] = False
    # regla congelada
    if not all(gates.values()) or not series_AS["complete"] or not sat["complete"] or sweep["AS"]["n_arms"] < len(EPS_SWEEP):
        letter = "INDETERMINADO"
    elif sweep["AS"]["independent"]:
        letter = "C"
    elif sweep["AS"]["power_law"] and sweep["b005"]["power_law"]:
        if series_AS["UV_no_convergence"] and sat["as_predicted"]:
            letter = "A"
        elif sat["sub_any_exit"]:
            letter = "B"
        else:
            letter = "INDETERMINADO"
    else:
        letter = "INDETERMINADO"
    announced = [a for a in energy_failures if any(k in a for k in ("AS", "b005")) and not pre["qualification"]["cronos_arms_can_meet_energy_gate_a_priori"].get(
        f"energy_floor_{'AS' if 'AS' in a else 'b005'}_rank_update1", True)]
    res = {"preregistration_sha256": psha, "qualification_sha256": pre["qualification"]["sha256"], "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "verdict": letter, "gates": gates, "energy_failures": energy_failures,
           "energy_failures_announced_by_qualification": announced, "missing_runs": missing, "per_arm": per_arm, "newton_stationarity_dex": stationarity,
           "sweep": sweep, "series_A_sculptor": series_AS, "saturating_control": sat, "N_control": n_ctrl,
           "reading": ("letra bajo la regla congelada; A: t_weak ∝ ε_soft^p con p > 0 en las dos amplitudes, la serie ds a A_Sculptor no converge y el control "
                       "saturante se comporta como predice la ecuación 1 (sub-umbral estable y convergente, supra-umbral inestable) — primera letra A "
                       "positiva posible del frente 5 y la ecuación 1 como física; B: p > 0 pero el sub-umbral también sale (la saturante pierde su candidatura); "
                       "C: t_weak independiente de ε_soft (el integrador manda; la lectura de la adenda se retira). Nada decide la ley ni la amplitud (E8, E13)")}
    (OUTDIR / "halo_round3.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Frente 5 (b), ronda 3 — capas esféricas: desenlace **{letter}**", "",
          f"Preinscripción `{psha[:12]}`; cualificación `{pre['qualification']['sha256'][:12]}`; análisis en `{res['code_commit_analysis'][:9]}`. Puertas: {gates}. "
          f"Fallos de energía: {energy_failures or 'ninguno'} (anunciados por la cualificación: {announced or 'ninguno'}). Corridas ausentes: {missing or 'ninguna'}. "
          f"Estacionariedad newtoniana: {stationarity if stationarity is None else round(stationarity, 4)} dex.", "",
          "| brazo | forma | t_weak [Myr] | r_exit [kpc] | M(<0.1) salida | M(<0.4) fin | ε_máx fin | |ΔE/E| | N(<0.4) ini | pasos | parada |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in per_arm.items():
        tw = "—" if e["t_weak_gyr"] is None else f"{e['t_weak_gyr'] * 1000:.3f}"
        rx = "—" if e["r_exit_kpc"] is None else f"{e['r_exit_kpc']:.3f}"
        md.append(f"| {arm} | {e['eps_form']} | {tw} | {rx} | {e['M01_exit']:.3e} | {e['M04_final']:.3e} | {e['eps_max_final']:.2e} | {e['dE_rel_max']:.2e} | "
                  f"{e['N_within_0p4_initial']} | {e['n_steps']} | {e['stop_reason'] or '—'} |")
    for tag, s in sweep.items():
        md += ["", f"Barrido ε_soft ({tag}): t_weak por ε_soft {s['t_weak_by_eps']}; ajuste p = {s['fit'].get('p')}, r² = {s['fit'].get('r2')}; potencia {s['power_law']}; independiente {s['independent']}"]
    md += ["", f"Serie ds a A_Sculptor: {series_AS}", "", f"Control saturante: {sat}", "", f"Control de N: {n_ctrl}", "", res["reading"] + ".",
           "", "## Lo que no decide", ""] + [f"- {s_}" for s_ in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"desenlace: {letter}; puertas: {gates}; barrido AS p = {sweep['AS']['fit'].get('p')}; saturante como predicho: {sat['as_predicted']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prereg")
    r = sub.add_parser("run"); r.add_argument("--arms", type=str, default="")
    sub.add_parser("analyze")
    args = ap.parse_args()
    return {"prereg": prereg, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
