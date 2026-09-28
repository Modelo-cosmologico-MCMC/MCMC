#!/usr/bin/env python
"""Cualificación E8-Q del instrumento de capas esféricas (cronos/halo_shells.py) — PR-3 de la orden del 28-sep.

    python scripts/qualify_shells.py run [--only <clave,...>]   # corre la rejilla; escribe results/<fecha>_qualification_shells/
    python scripts/qualify_shells.py table                        # recompone el artefacto a partir de las corridas guardadas

Sin letras ni umbrales de desenlace. Mide el SUELO del error de energía |ΔE/E| (E_grav_only en el brazo
newtoniano, E_self en los de Cronos) en función de N, ds, ε_soft, dt_min y del modo de rango (M(<r) congelado
en el paso global — ronda 2 — o ACTUALIZADO en los subpasos — ronda 3), la estacionariedad del control
newtoniano, el instante de salida del régimen débil y el coste. Criterio de puerta que hereda la ronda 3
(regla del brief): puerta de energía = 3 × el suelo medido en el brazo newtoniano equivalente; la
cualificación publica además el suelo de los brazos de Cronos para que la preinscripción diga, antes de correr,
qué brazos pueden cumplirla y cuáles no.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.halo_shells import A_SCULPTOR, ShellRun, equilibrium_shells  # noqa: E402
from validation.qualification import write_qualification  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-28_qualification_shells"
RUNS = OUTDIR / "runs"
H0 = 67.86705532886631
SYSTEM = {"M200_msun": 1e11, "c": 10.0, "H0": H0, "seed": 1, "r_decay_factor": 0.3, "refine_r_kpc": 2.0, "refine_beta": 1.5}
GRID = {"N": [100_000], "ds": [0.02, 0.04, 0.08], "eps_soft": [0.2, 0.1, 0.05], "rank_update": [False, True],
        "dt_min_myr": [1e-4], "dt_max_myr": 0.15, "eta_dt": 0.05, "eta_field": 0.02, "k_max": 10,
        "arms": {"newton": {"cronos": False, "amplitude": 0.0, "t_end_gyr": 0.05},
                 "AS": {"cronos": True, "amplitude": 1.0, "t_end_gyr": 0.02},
                 "b005": {"cronos": True, "amplitude": 0.05, "t_end_gyr": 0.01}},
        "extra": [{"N": 300_000, "ds": 0.04, "eps_soft": 0.1, "rank_update": True, "dt_min_myr": 1e-4},
                  {"N": 100_000, "ds": 0.04, "eps_soft": 0.1, "rank_update": True, "dt_min_myr": 1e-5},
                  # celdas que la ronda 3 del halo (PR-5) necesita: ε_soft = 0.025 kpc y N = 10⁶
                  {"N": 100_000, "ds": 0.04, "eps_soft": 0.025, "rank_update": False, "dt_min_myr": 1e-4},
                  {"N": 100_000, "ds": 0.04, "eps_soft": 0.025, "rank_update": True, "dt_min_myr": 1e-4},
                  {"N": 1_000_000, "ds": 0.04, "eps_soft": 0.1, "rank_update": True, "dt_min_myr": 1e-4}],
        "max_wall_s_per_run": 1500}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


def configs():
    out = []
    for N in GRID["N"]:
        for ds in GRID["ds"]:
            for eps in GRID["eps_soft"]:
                for ru in GRID["rank_update"]:
                    for dtm in GRID["dt_min_myr"]:
                        out.append({"N": N, "ds": ds, "eps_soft": eps, "rank_update": ru, "dt_min_myr": dtm})
    out += GRID["extra"]
    return out


def key_of(cfg: dict, arm: str) -> str:
    return f"{arm}_N{cfg['N']}_ds{cfg['ds']}_eps{cfg['eps_soft']}_ru{int(cfg['rank_update'])}_dtmin{cfg['dt_min_myr']:g}"


_IC_CACHE: dict = {}


def run_one(cfg: dict, arm: str) -> dict:
    spec = GRID["arms"][arm]
    if cfg["N"] not in _IC_CACHE:
        _IC_CACHE[cfg["N"]] = equilibrium_shells(SYSTEM["M200_msun"], SYSTEM["c"], SYSTEM["H0"], cfg["N"], SYSTEM["seed"],
                                                 r_decay_factor=SYSTEM["r_decay_factor"], refine_r_kpc=SYSTEM["refine_r_kpc"], refine_beta=SYSTEM["refine_beta"])
    ic = _IC_CACHE[cfg["N"]]
    t0 = time.time()
    run = ShellRun(ic["r"], ic["vr"], ic["L2"], ic["m"], A=spec["amplitude"] * A_SCULPTOR, cronos=spec["cronos"], ds=cfg["ds"],
                   eps_soft=cfg["eps_soft"], dt_myr=GRID["dt_max_myr"], eta_dt=GRID["eta_dt"], eta_field=GRID["eta_field"],
                   k_max=GRID["k_max"], dt_min_myr=cfg["dt_min_myr"], rank_update=cfg["rank_update"], stop_speed_kms=None,
                   max_wall_s=GRID["max_wall_s_per_run"])
    out = run.run(spec["t_end_gyr"], [0.0, spec["t_end_gyr"] / 2, spec["t_end_gyr"]])
    key = "E_self" if spec["cronos"] else "E_grav_only"
    e0 = out["snapshots"][0]["energy"]
    dE = max(abs(s["energy"][key] - e0[key]) / abs(e0[key]) for s in out["snapshots"])
    m04 = [s["M_within"]["0.4"] for s in out["snapshots"]]
    left = any(ev.get("kind") == "régimen débil violado" for ev in out["events"])
    return {"key": key_of(cfg, arm), "arm": arm, **cfg, "n_shells": len(ic["r"]), "N_within_0p4": int((ic["r"] < 0.4).sum()),
            "dE_rel_max": dE, "stationarity_dex": float(max(abs(np.log10(m / m04[0])) for m in m04)),
            "t_final_myr": out["t_final_gyr"] * 1000.0, "t_weak_myr": out["t_final_gyr"] * 1000.0 if left else None,
            "eps_max_final": out["snapshots"][-1]["eps_c_max"], "M01_final": out["snapshots"][-1]["M_within"]["0.1"],
            "stop_reason": out["stop_reason"], "n_steps": out["n_steps"], "dt_adaptive": out["dt_adaptive"], "wall_s": round(time.time() - t0, 1)}


def cmd_run(args) -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    runs = []
    # --reverse: un segundo trabajador recorre la rejilla desde el final; ambos saltan las corridas ya guardadas
    for cfg in (configs()[::-1] if args.reverse else configs()):
        for arm in GRID["arms"]:
            key = key_of(cfg, arm)
            if only and key not in only:
                continue
            path = RUNS / f"{key}.json"
            if path.exists():
                runs.append(json.loads(path.read_text(encoding="utf-8"))); continue
            rec = run_one(cfg, arm)
            path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            runs.append(rec)
            print(f"  {key}: |ΔE/E| = {rec['dE_rel_max']:.1e}, t_fin = {rec['t_final_myr']:.2f} Myr, ε = {rec['eps_max_final']:.1e}, "
                  f"{rec['n_steps']} pasos, {rec['wall_s']} s, {rec['stop_reason'] or ''}", flush=True)
    return write_artifact([json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))])


def write_artifact(runs: list) -> int:
    limits = {}
    for arm in GRID["arms"]:
        for ru in (False, True):
            sel = [r for r in runs if r["arm"] == arm and r["rank_update"] == ru]
            if sel:
                worst = max(sel, key=lambda r: r["dE_rel_max"]); best = min(sel, key=lambda r: r["dE_rel_max"])
                limits[f"energy_floor_{arm}_rank_update{int(ru)}"] = {
                    "value": worst["dE_rel_max"], "meaning": "máximo de |ΔE/E| sobre la rejilla (suelo conservador)",
                    "best": best["dE_rel_max"], "best_key": best["key"], "worst_key": worst["key"]}
    newt = [r for r in runs if r["arm"] == "newton"]
    if newt:
        limits["newton_stationarity_dex_max"] = {"value": max(r["stationarity_dex"] for r in newt)}
    per_eps = {}
    for r in runs:
        if r["arm"] == "AS" and r["t_weak_myr"] is not None:
            per_eps.setdefault(f"AS_t_weak_myr_eps{r['eps_soft']}_ru{int(r['rank_update'])}", []).append(r["t_weak_myr"])
    for k, v in per_eps.items():
        limits[k] = {"value": min(v), "all": v}
    criteria = {"energy_gate_rule": "puerta de energía de una preinscripción = 3 × el suelo medido aquí en el brazo newtoniano equivalente "
                                    "(mismo N, ds, ε_soft, modo de rango); los brazos de Cronos cuyo suelo medido supere esa puerta quedan "
                                    "declarados a priori como no cualificados para ella",
                "stationarity_rule": "la puerta de estacionariedad newtoniana no puede ser menor que newton_stationarity_dex_max",
                "dt_floor_rule": "una corrida que alcance dt_min antes de salir del régimen débil no tiene t_weak: la preinscripción debe "
                                 "declarar dt_min ≤ el mínimo con el que las corridas de A_Sculptor de esta rejilla salen"}
    notes = ["El suelo del error de energía en los brazos de Cronos durante el colapso de la cúspide es el que la ronda 2 no pudo bajar con η_dt ni η_field; aquí se mide con y sin rango actualizado en los subpasos.",
             "Los tiempos de salida t_weak se publican como límites del instrumento (dependencia de ε_soft, ds y N), no como resultado físico: E8-Q.",
             "Nada de esta cualificación decide una letra."]
    rows = ["## Corridas", "", "| clave | |ΔE/E| | estacionariedad [dex] | t_fin [Myr] | t_weak [Myr] | ε_máx fin | pasos | reordenaciones | pared [s] | parada |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        tw = "—" if r["t_weak_myr"] is None else f"{r['t_weak_myr']:.3f}"
        rows.append(f"| {r['key']} | {r['dE_rel_max']:.1e} | {r['stationarity_dex']:.3f} | {r['t_final_myr']:.2f} | {tw} | {r['eps_max_final']:.1e} | {r['n_steps']} | {r['dt_adaptive'].get('n_full_sorts', 0)} | {r['wall_s']} | {r['stop_reason'] or '—'} |")
    sha = write_qualification(OUTDIR, "capas esféricas (halo_shells)", _git_head(), GRID | {"system": SYSTEM}, runs, limits, criteria, notes, md_extra=rows)
    print(f"cualificación escrita: {OUTDIR} sha256 {sha}")
    return 0


def cmd_table(_args) -> int:
    return write_artifact([json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))])


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--only", type=str, default=""); r.add_argument("--reverse", action="store_true")
    sub.add_parser("table")
    args = ap.parse_args()
    return {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
