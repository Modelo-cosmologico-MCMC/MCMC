#!/usr/bin/env python
"""Frente 5 (b): el halo NFW con Cronos a A_Sculptor y a 0.05·A_Sculptor,
campo INSTANTÁNEO que sigue a las partículas, balance de energía
autoconsistente K + W + (2/5)U_C + W_fric y la serie de resolución del
campo k_inner ∈ {64, 128, 256} — bajo preinscripción congelada.

    python scripts/run_halo_resolution.py prereg
    python scripts/run_halo_resolution.py run [--arms a,bAS_k64,...]
    python scripts/run_halo_resolution.py analyze

La pregunta que el criterio de Cronos–Jeans (§3.21) deja: con q > 1
dentro de r_CJ ≈ 0.7 kpc a A_Sculptor, la ley local es
ultravioleta-inestable y el N-cuerpos NO debe converger al refinar la
resolución del campo (la escala más pequeña resuelta crece más deprisa);
con 0.05·A_Sculptor, r_CJ ≈ 0.18 kpc ≲ 2·ε_soft, y la serie debe
converger. El Nivel A (§3.19) vio «no convergido» en k_inner 64 → 128
pero con un campo de malla fija + media móvil cuyo estimador vaciaba el
interior y ganaba energía; aquí el campo se reconstruye de las partículas
en cada actualización y el balance de energía es el correcto para una
fuerza que deriva de la propia densidad: U_self = (2/5)·Σm(−c²ε_c).

Brazos: a (newtoniano, referencia), bAS_k{64,128,256} (A_Sculptor),
b005_k{64,128,256} (0.05·A_Sculptor); semilla única; N y t_end acotados
por el presupuesto del entorno (declarado).

Regla congelada. Convergencia de una serie: |log10 M(<0.4 kpc)| entre
k_inner consecutivos ≤ tol_conv en el instante final común (los dos
saltos 64→128 y 128→256) — la serie converge si ambos saltos ≤ tol_conv.
    A — predicción del criterio: la serie a A_Sculptor NO converge y la
        serie a 0.05·A_Sculptor SÍ.
    B — ambas convergen (la catástrofe ultravioleta no se manifiesta a
        estas resoluciones: el Nivel A era artefacto del estimador).
    C — ninguna converge (el instrumento no distingue: se publica como
        límite del instrumento, no como física).
    D — solo converge la de A_Sculptor (contrario a la predicción: sospecha
        de error primero).
    INDETERMINADO — puertas: |Δ(K + W)/E| ≤ tol_E_newton en a; |ΔE_self/E|
        ≤ tol_E_self en TODOS los brazos Cronos (con el balance correcto el
        campo dinámico debe conservar; si no, el instrumento no sirve);
        régimen débil; corridas ausentes.
Se publica además el cociente b/a por bandas, M(<0.4)(t) y W_fric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_halo_resolution"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"
K_SERIES = (64, 128, 256)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


def _log(path: Path):
    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def prereg(tol_E_self: float, pilot_text: str) -> int:
    from cronos.halo_nbody import A_SCULPTOR, nfw_structural
    from dynamics.cronos_jeans import nfw_jeans_table
    OUTDIR.mkdir(parents=True, exist_ok=True)
    H0 = 67.86705532886631
    cj = nfw_jeans_table(A=A_SCULPTOR)
    arms = {"a": {"label": "newtoniano (referencia)", "cronos": False, "amplitude": 1.0, "k_inner": 64}}
    for k in K_SERIES:
        arms[f"bAS_k{k}"] = {"label": f"Cronos v3 a A_Sculptor, campo instantáneo, k_inner = {k}", "cronos": True, "amplitude": 1.0, "k_inner": k}
    for k in K_SERIES:
        arms[f"b005_k{k}"] = {"label": f"Cronos v3 a 0.05·A_Sculptor, campo instantáneo, k_inner = {k}", "cronos": True, "amplitude": 0.05, "k_inner": k}
    doc = {
        "title": "Frente 5 (b): serie de resolución del campo de Cronos (k_inner 64/128/256) a A_Sculptor y a 0.05·A_Sculptor con campo instantáneo y balance K + W + (2/5)U_C + W_fric",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_halo_resolution.py prereg", "generator_commit_must_precede_freeze": True,
        "question": "¿Converge el interior del halo al refinar la resolución del campo de Cronos? El criterio de Cronos–Jeans predice que NO a A_Sculptor (q > 1 dentro de r_CJ) y que SÍ a 0.05·A_Sculptor.",
        "prediction_from_criterion": {"r_CJ_kpc_A_sculptor": cj["r_CJ_kpc"], "r_CJ_kpc_0p05": cj["r_CJ_for_A_fraction"].get("0.05"),
                                      "M_within_r_CJ": cj["M_within_r_CJ"], "expected": "A"},
        "amplitudes": {"A_sculptor": A_SCULPTOR, "fraction_arm": 0.05, "units": "(M_sol/pc^3)^(-3/2); epsilon_c = A * rho^(3/2)"},
        "system": {"M200_msun": 1e11, "c": 10.0, "H0": H0, "N": 200000, "seed": 1, "soft_plummer_kpc": 0.1, "r_decay_factor": 0.3,
                   "structural": nfw_structural(1e11, 10.0, H0),
                   "ics": "NFW truncado (Kazantzidis+2004), Eddington isótropo; la MISMA semilla en todos los brazos"},
        "integrator": {"theta": 0.7, "eta_acc": 0.025, "eta_dyn": 0.02, "eta_cross": 0.1, "dt_max_gyr": 0.0512, "n_levels": 12,
                       "brute_max": 500, "profile_every_ticks": 8, "field_tau_avg_myr": 0.0, "field_follow_particles": True,
                       "field_smooth": 1.0,
                       "scheme": "como el Nivel A, con el campo de Cronos INSTANTÁNEO: malla logarítmica reconstruida desde la partícula "
                                 "k_inner-ésima actual en cada actualización (sin malla fija ni media móvil)"},
        "energy_balance": "K + W + U_self + W_fric con U_self = (2/5)·Σm(−c²ε_c) (la fuerza +c²∇ε_c deriva del funcional "
                          "−(2/5)c²A∫ρ^{5/2}dV) y W_fric = trabajo acumulado de la fricción con compuerta; el Nivel A usaba "
                          "K + W + Σm(−c²ε_c), correcto solo con el campo congelado",
        "t_end_gyr": 1.0, "snapshots_gyr": [0.0, 0.25, 0.5, 0.75, 1.0],
        "arms": arms,
        "gates": {"tol_E_newton": 5e-3, "tol_E_self": tol_E_self, "weak_regime_eps_max": 1e-3,
                  "stop_rules": {"runaway_well_speed_kms": 1000.0, "max_wall_hours_per_run": 2.5}},
        "rules": {"convergence_metric": "|log10(M_k'(<0.4 kpc)/M_k(<0.4 kpc))| en el instante final común para los saltos 64→128 y 128→256",
                  "tol_conv": 0.15,
                  "series_converges": "ambos saltos ≤ tol_conv",
                  "letters": {"A": "A_Sculptor NO converge y 0.05·A_Sculptor SÍ (predicción del criterio)",
                              "B": "ambas convergen", "C": "ninguna converge (límite del instrumento)",
                              "D": "solo converge A_Sculptor (contrario a la predicción: sospecha de error primero)",
                              "INDETERMINADO": "puerta violada (energía, régimen) o corridas ausentes"},
                  "publish": ["cociente b/a por bandas [0.1,0.4], [0.4,1], [1,2.3], [2.3,5], [5,20] kpc", "M(<0.4 kpc)(t) por brazo",
                              "W_fric/|E| por brazo", "|ΔE_self/E| y |ΔE_naive/E| por brazo"],
                  "order": "puertas → letra; ningún umbral se toca tras ver los números"},
        "pilot_declared": pilot_text,
        "what_this_cannot_decide": ["si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): diccionario)",
                                    "la amplitud (A_Sculptor es hipótesis del 5E; 0.05 es brazo)",
                                    "el interior por debajo de ε_soft = 0.1 kpc"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). sha256 de "
          f"`preregistration.json`: `{sha}`.", "", f"**Pregunta**: {doc['question']}", "",
          f"**Predicción del criterio**: r_CJ = {cj['r_CJ_kpc']:.3f} kpc a A_Sculptor (M(<r_CJ) = {cj['M_within_r_CJ']:.2e} M☉), "
          f"r_CJ = {cj['r_CJ_for_A_fraction'].get('0.05')} kpc a 0.05·A_Sculptor ⟹ letra esperada A.", "",
          "## Declarado", "", f"- Sistema: {doc['system']}", f"- Integrador: {doc['integrator']}", f"- Balance de energía: {doc['energy_balance']}",
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
    from cronos.halo_nbody import A_SCULPTOR, HaloRun, sample_equilibrium_nfw
    sysm, integ, spec = pre["system"], pre["integrator"], pre["arms"][arm]
    A = spec["amplitude"] * A_SCULPTOR
    t0 = time.time()
    ic = sample_equilibrium_nfw(sysm["M200_msun"], sysm["c"], sysm["H0"], sysm["N"], sysm["seed"], r_decay_factor=sysm["r_decay_factor"])
    log(f"  ICs semilla {sysm['seed']}: {time.time()-t0:.0f}s, M_tot = {ic['M_tot']:.3e}")
    run = HaloRun(ic["pos"], ic["vel"], ic["mass"], A=A, cronos=spec["cronos"], soft_plummer_kpc=sysm["soft_plummer_kpc"],
                  theta=integ["theta"], eta_acc=integ["eta_acc"], eta_dyn=integ["eta_dyn"], eta_cross=integ["eta_cross"],
                  dt_max_gyr=integ["dt_max_gyr"], n_levels=integ["n_levels"], brute_max=integ["brute_max"],
                  profile_every=integ["profile_every_ticks"], weak_max=pre["gates"]["weak_regime_eps_max"],
                  stop_well_speed_kms=pre["gates"]["stop_rules"]["runaway_well_speed_kms"] if spec["cronos"] else None,
                  max_wall_s=pre["gates"]["stop_rules"]["max_wall_hours_per_run"] * 3600.0,
                  tau_avg_myr=integ["field_tau_avg_myr"],
                  field_kwargs={"k_inner": spec["k_inner"], "smooth": integ["field_smooth"], "follow_particles": integ["field_follow_particles"]})
    ev0 = run.events[0]
    log(f"  brazo {arm}: A = {A:.3e}, k_inner = {spec['k_inner']}, r_inner = {ev0['r_inner_field_kpc']:.3f} kpc, D_F(ε_soft) = {ev0['D_F_at_soft']:.2f}")
    out = run.run(pre["t_end_gyr"], pre["snapshots_gyr"], log=log)
    out.update({"arm": arm, "A": A, "k_inner": spec["k_inner"], "N": sysm["N"]})
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
        log(f"  guardado {path.name} ({out['wall_s']} s; parada temprana: {out['stopped_early']} {out['stop_reason'] or ''})")
    log("fin de corridas")
    return 0


def _band_ratio(snap_b, snap_a, lo, hi):
    r = np.asarray(snap_a["r_mid_kpc"])
    rb, ra = np.asarray(snap_b["rho"]), np.asarray(snap_a["rho"])
    m = (r >= lo) & (r < hi) & (ra > 0) & (rb > 0)
    return float(np.mean(np.log10(rb[m] / ra[m]))) if m.any() else None


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
    G, R = pre["gates"], pre["rules"]
    energy, gates = {}, {"energy_newton": True, "energy_self_cronos": True, "weak_regime": True, "runs_complete": not missing}
    for arm, d in runs.items():
        e = [s["energy"] for s in d["snapshots"] if "energy" in s]
        E0 = e[0]["E_grav_only"] if not pre["arms"][arm]["cronos"] else e[0]["E_self"]
        dE_self = max(abs(x["E_self"] - e[0]["E_self"]) / abs(e[0]["E_self"]) for x in e)
        dE_naive = max(abs(x["E"] - e[0]["E"]) / abs(e[0]["E"]) for x in e)
        dE_grav = max(abs(x["E_grav_only"] - e[0]["E_grav_only"]) / abs(e[0]["E_grav_only"]) for x in e)
        energy[arm] = {"dE_self_rel_max": dE_self, "dE_naive_rel_max": dE_naive, "dE_grav_rel_max": dE_grav,
                       "W_fric_over_E_final": e[-1]["W_fric"] / abs(E0), "t_final_gyr": d["t_final_gyr"],
                       "stopped_early": d["stopped_early"], "stop_reason": d["stop_reason"],
                       "weak_ok": all(ev["weak_regime_ok"] for ev in d["events"])}
        if not pre["arms"][arm]["cronos"] and dE_grav > G["tol_E_newton"]:
            gates["energy_newton"] = False
        if pre["arms"][arm]["cronos"] and dE_self > G["tol_E_self"]:
            gates["energy_self_cronos"] = False
        if not energy[arm]["weak_ok"]:
            gates["weak_regime"] = False

    def m04_final(arm):
        return runs[arm]["snapshots"][-1]["M_within"]["0.4"]

    def series(prefix):
        ks = [k for k in K_SERIES if f"{prefix}_k{k}" in runs]
        jumps = {}
        for k1, k2 in pairwise(ks):
            jumps[f"{k1}->{k2}"] = float(np.log10(m04_final(f"{prefix}_k{k2}") / m04_final(f"{prefix}_k{k1}")))
        conv = (len(jumps) == 2 and all(abs(v) <= R["tol_conv"] for v in jumps.values())) if jumps else None
        return {"jumps_log10_M04": jumps, "converges": conv,
                "M04_by_k": {str(k): m04_final(f"{prefix}_k{k}") for k in ks},
                "M04_t_by_k": {str(k): [s["M_within"]["0.4"] for s in runs[f"{prefix}_k{k}"]["snapshots"]] for k in ks}}
    sAS, s005 = series("bAS"), series("b005")
    all_gates = all(gates.values())
    if not all_gates or sAS["converges"] is None or s005["converges"] is None:
        letter = "INDETERMINADO"
    elif (not sAS["converges"]) and s005["converges"]:
        letter = "A"
    elif sAS["converges"] and s005["converges"]:
        letter = "B"
    elif (not sAS["converges"]) and (not s005["converges"]):
        letter = "C"
    else:
        letter = "D"
    bands = {}
    if "a" in runs:
        a_final = runs["a"]["snapshots"][-1]
        for arm in runs:
            if arm == "a":
                continue
            b_final = runs[arm]["snapshots"][-1]
            bands[arm] = {f"[{lo},{hi})": _band_ratio(b_final, a_final, lo, hi) for lo, hi in ((0.1, 0.4), (0.4, 1.0), (1.0, 2.3), (2.3, 5.0), (5.0, 20.0))}
    res = {"preregistration_sha256": psha, "code_commit_analysis": _git_head(), "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "verdict": letter, "gates": gates, "missing_runs": missing, "energy": energy,
           "series_A_sculptor": sAS, "series_0p05": s005, "bands_log10_b_over_a": bands,
           "reading": ("letra bajo la regla congelada; A confirma la predicción del criterio (no convergencia a A_Sculptor, "
                       "convergencia a 0.05·A_Sculptor); ninguna letra es afirmación sobre el tratado: sitúa la Ley de Cronos "
                       "débil con A_Sculptor frente a su propia inestabilidad ultravioleta (E8)")}
    (OUTDIR / "halo_resolution.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Frente 5 (b) — serie de resolución del campo: desenlace **{letter}**", "",
          f"Preinscripción `{psha[:12]}`; análisis en `{res['code_commit_analysis'][:9]}`. Puertas: {gates}. Corridas ausentes: {missing or 'ninguna'}.", "",
          "## Energía y puertas por brazo", "", "| brazo | k_inner | |ΔE_self/E| | |ΔE_naive/E| | |Δ(K+W)/E| | W_fric/|E| | t_final | parada |", "|---|---|---|---|---|---|---|---|"]
    for arm, e in energy.items():
        md.append(f"| {arm} | {pre['arms'][arm]['k_inner']} | {e['dE_self_rel_max']:.2e} | {e['dE_naive_rel_max']:.2e} | {e['dE_grav_rel_max']:.2e} | "
                  f"{e['W_fric_over_E_final']:.2e} | {e['t_final_gyr']:.3f} | {e['stop_reason'] or '—'} |")
    for name, s in (("A_Sculptor", sAS), ("0.05·A_Sculptor", s005)):
        md += ["", f"## Serie {name}: M(<0.4 kpc) final por k_inner", "", f"M04 = {s['M04_by_k']}; saltos log10: {s['jumps_log10_M04']}; converge: {s['converges']}."]
    md += ["", "## Cociente b/a por bandas (t final)", "", "| brazo | " + " | ".join(next(iter(bands.values())).keys() if bands else []) + " |",
           "|---|" + "---|" * (len(next(iter(bands.values()))) if bands else 0)]
    for arm, b in bands.items():
        md.append(f"| {arm} | " + " | ".join("—" if v is None else f"{v:+.3f}" for v in b.values()) + " |")
    md += ["", res["reading"] + ".", "", "## Lo que no decide", ""] + [f"- {s}" for s in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"desenlace: {letter}; puertas: {gates}; saltos A_S: {sAS['jumps_log10_M04']}; saltos 0.05: {s005['jumps_log10_M04']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prereg")
    p.add_argument("--tol-E-self", type=float, required=True)
    p.add_argument("--pilot", type=str, required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", type=str, default="")
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "prereg":
        return prereg(args.tol_E_self, args.pilot)
    return {"run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
