#!/usr/bin/env python
"""Frente 5 (b), RONDA 2 (orden del autor del 22-sep, §7): la serie de
resolución del campo de Cronos con un código de CAPAS ESFÉRICAS
(cronos/halo_shells.py) — bajo preinscripción nueva (la ronda 1,
results/2026-09-21_halo_resolution/, queda intacta con su INDETERMINADO).

    python scripts/run_halo_shells.py prereg --dt-myr <dt> --refine-r-kpc <r> --refine-beta <β> --tol-e-newton <x> --tol-e-cronos <y> --pilot "<texto>"
    python scripts/run_halo_shells.py run [--arms a_N1e6,...]
    python scripts/run_halo_shells.py analyze

Diagnóstico de la ronda 1 (§3.29): con 18–87 partículas dentro de 0.4 kpc
la ε_c ∝ ρ^{3/2} reconstruida fluctuaba en O(1) entre actualizaciones y
los brazos a A_Sculptor inyectaban energía (2.4 %, 4.0 %) y VACIABAN el
interior. Un código de capas resuelve el interior con 1e5–1e6 capas sin
ruido de conteo en el campo (M(<r) exacta), a coste O(N log N) por paso,
y captura exactamente los modos radiales del criterio (los inestables);
deja fuera los no radiales: declarado.

Brazos: a (newtoniano) × N ∈ {1e5, 1e6}; A_Sculptor y 0.05·A_Sculptor ×
resolución del campo ds ∈ {0.02, 0.04, 0.08} (Δln r ×½, ×1, ×2) a N = 1e6,
y ds = 0.04 a N = 1e5 (control de N). Misma semilla en todos. N es el
número de capas de masa m₀ del halo sin refinar; con el refinamiento de
masa declarado (m ∝ (r/r_ref)^β dentro de r_ref) el número real de capas
es mayor y el interior queda resuelto sin tocar la distribución de masa.

Regla congelada (antes de correr):
    métrico: t_weak, el instante en que ε_c máx supera 1e-3 (la ley DÉBIL
        deja de ser aplicable: parada declarada), r_exit (radio de la celda
        de ε_c máx en ese instante), M(<0.1) y M(<0.4) en la salida;
    firma UV: t_weak decrece estrictamente al refinar ds (0.08 → 0.04 →
        0.02) — el γ ∝ k del criterio (ronda 2 del test);
    A — las dos amplitudes salen del régimen débil en las tres
        resoluciones, con firma UV en ambas, r_exit < r_CJ(A) en todas las
        celdas y t_weak(0.05) > t_weak(A_Sculptor) a cada ds (el criterio
        ordena las amplitudes): predicción del criterio;
    B — A_Sculptor NO sale del régimen débil antes de t_end en ninguna
        resolución (saturación no lineal: el criterio no describe ese
        régimen);
    C — A_Sculptor sale pero sin firma UV o con r_exit ≥ r_CJ, o
        0.05·A_Sculptor no sale en ninguna resolución mientras A_Sculptor sí;
    INDETERMINADO — puertas: energía hasta la parada (|Δ(K+W)/E| ≤
        tol_E_newton en el control; |ΔE_self/E| ≤ tol_E_cronos en Cronos: el
        cruce de capas durante el colapso de la cúspide es el suelo del
        instrumento, declarado del piloto),
        ≥ n_min capas dentro de 0.4 kpc en los brazos de 1e6, estacionariedad
        del control newtoniano, control de N (t_weak a 1e5 y 1e6 dentro de un
        factor declarado), corridas ausentes.
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

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-22_halo_shells"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"
DS_SERIES = (0.02, 0.04, 0.08)         # resolución del campo ds (= Δln r lejos del centro): ×½ / ×1 / ×2
H0 = 67.86705532886631
ETA_DT, ETA_FIELD, K_MAX, DT_MIN_MYR = 0.05, 0.02, 10, 1e-4   # subpasos por capa (η·escala orbital, 2^k ≤ 2^K_MAX), cambio máx. de ε_c por paso global, suelo


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


def _log(path: Path):
    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def prereg(dt_myr: float, refine_r_kpc: float, refine_beta: float, tol_e_newton: float, tol_e_cronos: float, pilot_text: str) -> int:
    from cronos.halo_nbody import A_SCULPTOR, nfw_structural
    from dynamics.cronos_jeans import nfw_jeans_table
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cj = nfw_jeans_table(A=A_SCULPTOR)
    arms = {"a_N1e6": {"label": "newtoniano, N = 1e6", "cronos": False, "amplitude": 0.0, "ds": 0.04, "N": 1_000_000},
            "a_N1e5": {"label": "newtoniano, N = 1e5", "cronos": False, "amplitude": 0.0, "ds": 0.04, "N": 100_000}}
    for d in DS_SERIES:
        arms[f"bAS_ds{d}_N1e6"] = {"label": f"Cronos v3 a A_Sculptor, ds = {d}, N = 1e6", "cronos": True, "amplitude": 1.0, "ds": d, "N": 1_000_000}
    for d in DS_SERIES:
        arms[f"b005_ds{d}_N1e6"] = {"label": f"Cronos v3 a 0.05·A_Sculptor, ds = {d}, N = 1e6", "cronos": True, "amplitude": 0.05, "ds": d, "N": 1_000_000}
    arms["bAS_ds0.04_N1e5"] = {"label": "Cronos v3 a A_Sculptor, ds = 0.04, N = 1e5 (control de N)", "cronos": True, "amplitude": 1.0, "ds": 0.04, "N": 100_000}
    arms["b005_ds0.04_N1e5"] = {"label": "Cronos v3 a 0.05·A_Sculptor, ds = 0.04, N = 1e5 (control de N)", "cronos": True, "amplitude": 0.05, "ds": 0.04, "N": 100_000}
    doc = {
        "title": "Frente 5 (b), ronda 2: serie de resolución del campo de Cronos con código de capas esféricas (ds = 0.02/0.04/0.08 = Δln r ×½/×1/×2, N = 1e5/1e6, refinamiento de masa interior, campo conservativo), balance K + W + (2/5)U_C + W_fric a precisión de máquina",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_halo_shells.py prereg", "generator_commit_must_precede_freeze": True,
        "round1": "results/2026-09-21_halo_resolution (INDETERMINADO por la puerta de energía a A_Sculptor; interior no resuelto): intacta",
        "question": "¿Converge el interior del halo al refinar la resolución del campo de Cronos cuando el interior está resuelto (≥ 2000 capas dentro de 0.4 kpc) y el campo no tiene ruido de conteo? El criterio predice que NO a A_Sculptor (q > 1 dentro de r_CJ) y que SÍ a 0.05·A_Sculptor.",
        "prediction_from_criterion": {"r_CJ_kpc_A_sculptor": cj["r_CJ_kpc"], "r_CJ_kpc_0p05": cj["r_CJ_for_A_fraction"].get("0.05"),
                                      "M_within_r_CJ": cj["M_within_r_CJ"], "expected": "A"},
        "amplitudes": {"A_sculptor": A_SCULPTOR, "fraction_arm": 0.05, "units": "(M_sol/pc^3)^(-3/2); epsilon_c = A * rho^(3/2)"},
        "system": {"M200_msun": 1e11, "c": 10.0, "H0": H0, "seed": 1, "eps_soft_kpc": 0.1, "r_decay_factor": 0.3,
                   "refine_r_kpc": refine_r_kpc, "refine_beta": refine_beta,
                   "refinement": "capas con r < r_ref llevan masa m = m₀(r/r_ref)^β y se muestrean ∝ ρr²/m (importancia sobre la masa): "
                                 "misma M(<r) y misma f(E), interior resuelto con capas ligeras; N fija m₀ = M_tot/N",
                   "structural": nfw_structural(1e11, 10.0, H0),
                   "ics": "NFW truncado (Kazantzidis+2004), Eddington isótropo con tablas densas hacia Ψ(0) y borde interior 1e-6·r_s (el borde 1e-4·r_s del N-cuerpos "
                          "enfría ~25 % las capas de r < 0.1 kpc: piloto), velocidades por CDF inversa (el método del N-cuerpos), capas (r, v_r, L²); MISMA semilla"},
        "instrument": {"code": "cronos/halo_shells.py (capas esféricas)",
                       "gravity": "M(<r) exacta por rango (Hénon: M_i = M_interior + ½m_i); gravedad y término centrífugo suavizados con la misma ε_soft "
                                  "(hamiltoniano por capa v_r²/2 + L²/(2(r² + ε²)) + Φ_Plummer)",
                       "field": "KernelCronosField: depósito B-spline cúbico de la masa sobre malla fija en s = ln √(r² + ε²) de paso ds, ρ_b = M_b/V_b "
                                "(V_b = volumen del núcleo), ε_b = Aρ_b^{3/2}, ε̃(r) interpolada con el MISMO núcleo ⟹ a = c²dε̃/dr es el gradiente "
                                "exacto de U = −(2/5)c²A Σ_b V_b ρ_b^{5/2} (conservativo por construcción); actualizado en CADA paso",
                       "why_not_spline": "el estimador spline del frente 5b no deriva de un funcional: piloto |ΔE_self/E| ≈ 1e-2 a A_Sculptor, independiente de dt y malla",
                       "law": "fuerza +c²dε_c/dr, fricción con compuerta Γ = (3/2)(ρ̇/ρ)ε_cΘ(ρ̇), lapso N = 1 + Φ_N/c² − ε_c",
                       "integrator": f"salto de rana KDK con paso GLOBAL dt_max = {dt_myr} Myr (campo congelado dentro del paso: rangos M(<r), ε̃, lapso, fricción) "
                                     f"y SUBPASOS por capa 2^k, k = ⌈log₂(dt/(η_dt·τ))⌉ ≤ {K_MAX} con τ = min(r/|v|, √(r³/GM), L/v²) y η_dt = {ETA_DT} "
                                     f"(el pericentro de las capas casi radiales, L²/r³ exacto, queda resuelto sin frenar al resto); con Cronos el paso global "
                                     f"se acorta para que ε_c máx cambie ≤ η_field = {ETA_FIELD} por paso; suelo dt_min = {DT_MIN_MYR} Myr (alcanzarlo es parada declarada)",
                       "eta_dt": ETA_DT, "eta_field": ETA_FIELD, "k_max": K_MAX, "dt_min_myr": DT_MIN_MYR,
                       "not_captured": "modos no radiales (declarado): el código captura exactamente los radiales, que son los inestables del criterio"},
        "energy_balance": "E_self = K + W + (2/5)Σm(−c²ε_c) + W_fric (la fuerza deriva de −(2/5)c²A∫ρ^{5/2}dV); K + W + Σm(−c²ε_c) solo con campo congelado",
        "t_end_gyr": 0.1, "snapshots_gyr": [0.0, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1],
        "arms": arms,
        "gates": {"tol_E_newton": tol_e_newton, "tol_E_cronos": tol_e_cronos, "n_min_shells_within_0p4_N1e6": 2000, "newton_stationarity_dex": 0.1, "N_control_factor": 2.0,
                  "stop_rules": {"weak_regime_eps_max": 1e-3, "max_wall_hours_per_run": 3.0, "dt_min_myr": DT_MIN_MYR}},
        "rules": {"metric": "t_weak = instante en que ε_c máx supera 1e-3 (la ley débil deja de ser aplicable: parada declarada), r_exit = radio de la celda "
                            "de ε_c máx en ese instante, M(<0.1) y M(<0.4) en ese instante; None si la corrida llega a t_end dentro del régimen débil",
                  "UV": "t_weak decrece estrictamente al refinar ds (0.08 → 0.04 → 0.02): la firma γ ∝ k del criterio (ronda 2 del test)",
                  "letters": {"A": "las dos amplitudes salen del régimen débil en las tres resoluciones, con UV en ambas, r_exit < r_CJ(A) en todas las celdas "
                                   "y t_weak(0.05) > t_weak(A_Sculptor) a cada ds (el criterio ordena las amplitudes): predicción del criterio",
                              "B": "A_Sculptor NO sale del régimen débil antes de t_end en ninguna resolución (saturación no lineal: el criterio no describe ese régimen)",
                              "C": "A_Sculptor sale pero sin UV (t_weak no decrece al refinar) o con r_exit ≥ r_CJ(A_Sculptor), o 0.05·A_Sculptor no sale en "
                                   "ninguna resolución mientras A_Sculptor sí (el criterio falla en la amplitud pequeña)",
                              "INDETERMINADO": "puerta violada (energía, capas interiores, estacionariedad newtoniana, control de N) o corridas ausentes; "
                                               "ningún otro patrón se clasifica"},
                  "gates_meaning": {"energy": "|Δ(K+W)/E| ≤ tol_E_newton en las corridas newtonianas y |ΔE_self/E| ≤ tol_E_cronos en las de Cronos hasta su parada "
                                              "(dos puertas: el piloto muestra que el error de cruce de capas durante el colapso de la cúspide no baja con η_dt ni η_field)",
                                    "inner_shells": "≥ n_min capas dentro de 0.4 kpc en los brazos de 1e6",
                                    "newton_stationarity": "|log10 M(<0.4)(t)/M(<0.4)(0)| ≤ newton_stationarity_dex en a_N1e6 durante todo el recorrido",
                                    "N_control": "t_weak(1e5)/t_weak(1e6) ∈ [1/N_control_factor, N_control_factor] a ds = 0.04 en las dos amplitudes (si ambas salen)"},
                  "publish": ["t_weak, r_exit, M(<0.1) y M(<0.4) en la salida por brazo", "|ΔE_self/E|, |Δ(K+W)/E|, W_fric/|E|, pasos y subpasos por brazo",
                              "cociente b/a por bandas [0.1,0.2], [0.2,0.4], [0.4,1], [1,2.3], [2.3,5] kpc en el instante común con a_N1e6",
                              "control de N: 1e5 frente a 1e6 a ds = 0.04", "M(<0.4)(t) del control newtoniano (estacionariedad)"],
                  "order": "puertas → letra (A, B, C en ese orden de comprobación; lo que no encaje es INDETERMINADO); ningún umbral se toca tras ver los números"},
        "pilot_declared": pilot_text,
        "what_this_cannot_decide": ["si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): diccionario)",
                                    "la amplitud (A_Sculptor es hipótesis del 5E; 0.05 es brazo)",
                                    "los modos no radiales y el interior por debajo de ε_soft = 0.1 kpc"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). sha256 de "
          f"`preregistration.json`: `{sha}`.", "", f"**Pregunta**: {doc['question']}", "",
          f"**Predicción del criterio**: r_CJ = {cj['r_CJ_kpc']:.3f} kpc a A_Sculptor, {cj['r_CJ_for_A_fraction'].get('0.05')} kpc a 0.05·A_Sculptor ⟹ letra esperada A.", "",
          "## Declarado", "", f"- Sistema: {doc['system']}", f"- Instrumento: {doc['instrument']}", f"- Balance: {doc['energy_balance']}",
          f"- t_end = {doc['t_end_gyr']} Gyr; instantáneas {doc['snapshots_gyr']}.", f"- Brazos: {list(arms)}.", "",
          "## Reglas (congeladas)", ""] + [f"- **{k}**: {v}" for k, v in doc["rules"].items()] + [f"- **puertas**: {doc['gates']}", "",
          "## Piloto declarado", "", pilot_text, "", "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg():
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def run_one(pre: dict, arm: str, log) -> dict:
    from cronos.halo_shells import A_SCULPTOR, ShellRun, equilibrium_shells
    sysm, spec, G = pre["system"], pre["arms"][arm], pre["gates"]
    dt_myr = float(pre["instrument"]["integrator"].split("dt_max = ")[1].split(" Myr")[0])
    t0 = time.time()
    ic = equilibrium_shells(sysm["M200_msun"], sysm["c"], sysm["H0"], spec["N"], sysm["seed"], r_decay_factor=sysm["r_decay_factor"],
                            refine_r_kpc=sysm["refine_r_kpc"], refine_beta=sysm["refine_beta"])
    log(f"  capas semilla {sysm['seed']}, N = {spec['N']} (capas reales {ic.get('n_shells', len(ic['r']))}): {time.time()-t0:.0f}s, "
        f"N(<0.4) = {int((ic['r'] < 0.4).sum())}, m_min/m₀ = {float(ic['m'].min() / ic['m'].max()):.2e}")
    inst = pre["instrument"]
    run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], A=spec["amplitude"] * A_SCULPTOR, cronos=spec["cronos"], ds=spec["ds"],
                   eta_dt=inst["eta_dt"], eta_field=inst["eta_field"], k_max=inst["k_max"], dt_min_myr=inst["dt_min_myr"],
                   eps_soft=sysm["eps_soft_kpc"], dt_myr=dt_myr, weak_max=G["stop_rules"]["weak_regime_eps_max"],
                   stop_speed_kms=None, max_wall_s=G["stop_rules"]["max_wall_hours_per_run"] * 3600.0)
    log(f"  brazo {arm}: {run.events[0]}")
    out = run.run(pre["t_end_gyr"], pre["snapshots_gyr"], log=log)
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


def _band_ratio(snap_b, snap_a, lo, hi):
    r = np.asarray(snap_a["r_mid_kpc"])
    rb, ra = np.asarray(snap_b["rho"]), np.asarray(snap_a["rho"])
    m = (r >= lo) & (r < hi) & (ra > 0) & (rb > 0)
    return float(np.mean(np.log10(rb[m] / ra[m]))) if m.any() else None


def _snap_at(d: dict, t_gyr: float) -> dict:
    return min(d["snapshots"], key=lambda s: abs(s["t_gyr"] - t_gyr))


def _exit(d: dict, weak_max: float):
    """(t_weak, r_exit, snapshot de salida) o (None, None, último) si la corrida no salió del régimen débil."""
    left = any(ev.get("kind") == "régimen débil violado" for ev in d["events"])
    snap = d["snapshots"][-1]
    if left and snap["eps_c_max"] > weak_max:
        return snap["t_gyr"], snap["r_eps_max_kpc"], snap
    return None, None, snap


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    runs = {}
    for arm in pre["arms"]:
        p = RUNS / f"{arm}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("preregistration_sha256") != psha:
            raise SystemExit(f"FALLO CERRADO: {p.name} cita otra preinscripción")
        runs[arm] = d
    missing = [a for a in pre["arms"] if a not in runs]
    G = pre["gates"]
    weak_max = G["stop_rules"]["weak_regime_eps_max"]
    r_cj = {"bAS": pre["prediction_from_criterion"]["r_CJ_kpc_A_sculptor"], "b005": pre["prediction_from_criterion"]["r_CJ_kpc_0p05"]}
    gates = {"energy": True, "inner_shells": True, "newton_stationarity": True, "N_control": True, "runs_complete": not missing}
    per_arm = {}
    for arm, d in runs.items():
        e = [s["energy"] for s in d["snapshots"]]
        key = "E_self" if pre["arms"][arm]["cronos"] else "E_grav_only"
        dE = max(abs(x[key] - e[0][key]) / abs(e[0][key]) for x in e)
        t_weak, r_exit, snap = _exit(d, weak_max)
        per_arm[arm] = {"dE_rel_max": dE, "dE_grav_rel_max": max(abs(x["E_grav_only"] - e[0]["E_grav_only"]) / abs(e[0]["E_grav_only"]) for x in e),
                        "W_fric_over_E_final": e[-1]["W_fric"] / abs(e[0][key]), "t_final_gyr": d["t_final_gyr"],
                        "stopped_early": d["stopped_early"], "stop_reason": d["stop_reason"], "n_steps": d["n_steps"], "wall_s": d["wall_s"],
                        "dt_adaptive": d.get("dt_adaptive"),
                        "N_within_0p4_initial": d["N_within_0p4_initial"], "N_within_0p4_final": d["snapshots"][-1]["N_within"]["0.4"],
                        "t_weak_gyr": t_weak, "r_exit_kpc": r_exit, "M01_exit": snap["M_within"]["0.1"], "M04_exit": snap["M_within"]["0.4"],
                        "eps_max_final": snap["eps_c_max"]}
        if dE > (G["tol_E_cronos"] if pre["arms"][arm]["cronos"] else G["tol_E_newton"]):
            gates["energy"] = False
        if pre["arms"][arm]["N"] >= 1_000_000 and d["N_within_0p4_initial"] < G["n_min_shells_within_0p4_N1e6"]:
            gates["inner_shells"] = False
    stationarity = None
    if "a_N1e6" in runs:
        m0 = runs["a_N1e6"]["snapshots"][0]["M_within"]["0.4"]
        stationarity = max(abs(np.log10(s["M_within"]["0.4"] / m0)) for s in runs["a_N1e6"]["snapshots"])
        if stationarity > G["newton_stationarity_dex"]:
            gates["newton_stationarity"] = False

    def series(prefix):
        arms_s = [f"{prefix}_ds{d}_N1e6" for d in DS_SERIES if f"{prefix}_ds{d}_N1e6" in runs]
        if len(arms_s) < 3:
            return {"complete": False}
        tw = {a: per_arm[a]["t_weak_gyr"] for a in arms_s}                     # orden DS_SERIES: 0.02, 0.04, 0.08 (fino → grueso)
        all_exit = all(v is not None for v in tw.values())
        vals = [tw[a] for a in arms_s]
        uv = all_exit and vals[0] < vals[1] < vals[2]
        r_ex = {a: per_arm[a]["r_exit_kpc"] for a in arms_s}
        inside = all_exit and all(r_ex[a] < r_cj[prefix] for a in arms_s)
        return {"complete": True, "t_weak_by_arm": tw, "all_exit": all_exit, "none_exit": all(v is None for v in tw.values()),
                "UV": bool(uv), "r_exit_by_arm": r_ex, "r_CJ_kpc": r_cj[prefix], "exit_inside_r_CJ": bool(inside),
                "M01_exit_by_arm": {a: per_arm[a]["M01_exit"] for a in arms_s}, "M04_exit_by_arm": {a: per_arm[a]["M04_exit"] for a in arms_s}}
    sAS, s005 = series("bAS"), series("b005")
    n_ctrl = {}
    for prefix in ("bAS", "b005"):
        a6, a5 = f"{prefix}_ds0.04_N1e6", f"{prefix}_ds0.04_N1e5"
        if a6 in runs and a5 in runs:
            t6, t5 = per_arm[a6]["t_weak_gyr"], per_arm[a5]["t_weak_gyr"]
            ratio = (t5 / t6) if (t6 and t5) else None
            n_ctrl[prefix] = {"t_weak_1e6": t6, "t_weak_1e5": t5, "ratio_1e5_over_1e6": ratio,
                              "within_factor": bool(ratio is not None and 1.0 / G["N_control_factor"] <= ratio <= G["N_control_factor"]) if (t6 and t5) else None}
            if ratio is not None and not n_ctrl[prefix]["within_factor"]:
                gates["N_control"] = False
    bands = {}
    if "a_N1e6" in runs:
        for arm in runs:
            if arm.startswith("a_"):
                continue
            t_c = min(runs[arm]["t_final_gyr"], runs["a_N1e6"]["t_final_gyr"])
            b_s, a_s = _snap_at(runs[arm], t_c), _snap_at(runs["a_N1e6"], t_c)
            bands[arm] = {"t_common_gyr": t_c, **{f"[{lo},{hi})": _band_ratio(b_s, a_s, lo, hi) for lo, hi in ((0.1, 0.2), (0.2, 0.4), (0.4, 1.0), (1.0, 2.3), (2.3, 5.0))}}
    # --- regla congelada ------------------------------------------------------
    if not all(gates.values()) or not sAS.get("complete") or not s005.get("complete"):
        letter = "INDETERMINADO"
    else:
        ordered = sAS["all_exit"] and s005["all_exit"] and all(
            s005["t_weak_by_arm"][f"b005_ds{d}_N1e6"] > sAS["t_weak_by_arm"][f"bAS_ds{d}_N1e6"] for d in DS_SERIES)
        if sAS["all_exit"] and s005["all_exit"] and sAS["UV"] and s005["UV"] and sAS["exit_inside_r_CJ"] and s005["exit_inside_r_CJ"] and ordered:
            letter = "A"
        elif sAS["none_exit"]:
            letter = "B"
        elif sAS["all_exit"] and ((not sAS["UV"]) or (not sAS["exit_inside_r_CJ"]) or s005["none_exit"]):
            letter = "C"
        else:
            letter = "INDETERMINADO"
    res = {"preregistration_sha256": psha, "code_commit_analysis": _git_head(), "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "verdict": letter, "gates": gates, "missing_runs": missing, "per_arm": per_arm, "newton_stationarity_dex": stationarity,
           "series_A_sculptor": sAS, "series_0p05": s005, "bands_log10_b_over_a": bands, "N_control": n_ctrl,
           "reading": ("letra bajo la regla congelada; A confirma la predicción del criterio (las dos amplitudes salen del régimen débil dentro de "
                       "r_CJ, antes al refinar — la firma γ ∝ k — y A_Sculptor antes que 0.05·A_Sculptor); B, saturación no lineal a A_Sculptor; "
                       "C, salida sin firma UV o fuera de r_CJ, o 0.05 estable con A_Sculptor inestable; ninguna letra es afirmación sobre el "
                       "tratado: sitúa la Ley de Cronos débil con A_Sculptor frente a su propia inestabilidad ultravioleta (E8); los modos no "
                       "radiales y el interior bajo ε_soft no están en el instrumento; el número no es señal (E13)")}
    (OUTDIR / "halo_shells.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Frente 5 (b), ronda 2 — capas esféricas: desenlace **{letter}**", "",
          f"Preinscripción `{psha[:12]}`; análisis en `{res['code_commit_analysis'][:9]}`. Puertas: {gates}. Corridas ausentes: {missing or 'ninguna'}. "
          f"Estacionariedad newtoniana (a_N1e6): {stationarity if stationarity is None else round(stationarity, 4)} dex.", "",
          "## Salida del régimen débil, energía y puertas por brazo", "",
          "| brazo | t_weak [Myr] | r_exit [kpc] | M(<0.1) salida | M(<0.4) salida | ε_máx final | |ΔE/E| | |Δ(K+W)/E| | W_fric/|E| | N(<0.4) ini → fin | pasos | parada |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, e in per_arm.items():
        tw = "—" if e["t_weak_gyr"] is None else f"{e['t_weak_gyr'] * 1000:.3f}"
        rx = "—" if e["r_exit_kpc"] is None else f"{e['r_exit_kpc']:.3f}"
        md.append(f"| {arm} | {tw} | {rx} | {e['M01_exit']:.3e} | {e['M04_exit']:.3e} | {e['eps_max_final']:.2e} | {e['dE_rel_max']:.2e} | {e['dE_grav_rel_max']:.2e} | "
                  f"{e['W_fric_over_E_final']:.2e} | {e['N_within_0p4_initial']} → {e['N_within_0p4_final']} | {e['n_steps']} | {e['stop_reason'] or '—'} |")
    for name, s_ in (("A_Sculptor", sAS), ("0.05·A_Sculptor", s005)):
        if s_.get("complete"):
            md += ["", f"## Serie {name} (r_CJ = {s_['r_CJ_kpc']:.3f} kpc)", "",
                   f"t_weak por brazo [Gyr]: {s_['t_weak_by_arm']}; todas salen: {s_['all_exit']}; firma UV (t_weak decrece al refinar): {s_['UV']}; "
                   f"r_exit: {s_['r_exit_by_arm']}; salida dentro de r_CJ: {s_['exit_inside_r_CJ']}; M(<0.1) en la salida: {s_['M01_exit_by_arm']}."]
    if bands:
        keys = [k for k in next(iter(bands.values())).keys() if k != "t_common_gyr"]
        md += ["", "## Cociente log10 b/a por bandas (instante común con a_N1e6)", "", "| brazo | t común [Myr] | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for arm, b in bands.items():
            md.append(f"| {arm} | {b['t_common_gyr'] * 1000:.3f} | " + " | ".join("—" if b[k] is None else f"{b[k]:+.3f}" for k in keys) + " |")
    if n_ctrl:
        md += ["", f"Control de N (ds = 0.04): {n_ctrl}"]
    md += ["", res["reading"] + ".", "", "## Lo que no decide", ""] + [f"- {s_}" for s_ in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"desenlace: {letter}; puertas: {gates}; A_S: {sAS.get('t_weak_by_arm')}; 0.05: {s005.get('t_weak_by_arm')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prereg")
    p.add_argument("--dt-myr", type=float, required=True)
    p.add_argument("--refine-r-kpc", type=float, required=True)
    p.add_argument("--refine-beta", type=float, required=True)
    p.add_argument("--tol-e-newton", type=float, required=True)
    p.add_argument("--tol-e-cronos", type=float, required=True)
    p.add_argument("--pilot", type=str, required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", type=str, default="")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "prereg":
        return prereg(args.dt_myr, args.refine_r_kpc, args.refine_beta, args.tol_e_newton, args.tol_e_cronos, args.pilot)
    return {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
