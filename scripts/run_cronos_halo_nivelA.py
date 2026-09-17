#!/usr/bin/env python
"""Nivel A del frente 5 — ejecuta bajo la preinscripción congelada los
brazos del halo aislado (a: newtoniano; b: Cronos v3 con A_Sculptor;
c: control de exclusión con el cierre cosmológico; c′: control de
respuesta con 10·A_Sculptor) y, con `--analyze`, aplica la regla
congelada y publica el artefacto.

Uso:
    python scripts/run_cronos_halo_nivelA.py run [--arms a,b,c,cprime] [--seeds 1,2,3]
    python scripts/run_cronos_halo_nivelA.py analyze

Cada corrida se guarda en results/<dir>/runs/<arm>_seed<k>.json y no se
repite si ya existe (reanudable). La preinscripción se lee ANTES de
generar cualquier condición inicial.
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

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-17_cronos_halo_nivelA"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción del Nivel A")
    return (json.loads(PREREG.read_text(encoding="utf-8")),
            hashlib.sha256(PREREG.read_bytes()).hexdigest())


def _log(path: Path):
    def log(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def run_one(prereg: dict, arm: str, seed: int, log) -> dict:
    from cronos.halo_nbody import A_SCULPTOR, HaloRun, sample_equilibrium_nfw
    sysm, integ, arms = prereg["system"], prereg["integrator"], prereg["arms"]
    spec = arms[arm]
    A = {"A_sculptor": A_SCULPTOR, "A_cosmological": prereg["amplitudes"]["A_cosmological"],
         "ten_A_sculptor": 10.0 * A_SCULPTOR}[spec["amplitude"]]
    t0 = time.time()
    ic = sample_equilibrium_nfw(sysm["M200_msun"], sysm["c"], sysm["H0"], sysm["N"], seed,
                                r_decay_factor=sysm["r_decay_factor"])
    log(f"  ICs semilla {seed}: {time.time()-t0:.0f}s, M_tot = {ic['M_tot']:.3e}, f<0 = {ic['f_negative_fraction']:.3f}")
    run = HaloRun(ic["pos"], ic["vel"], ic["mass"], A=A, cronos=spec["cronos"],
                  soft_plummer_kpc=sysm["soft_plummer_kpc"], theta=integ["theta"],
                  eta_acc=integ["eta_acc"], eta_dyn=integ["eta_dyn"], eta_cross=integ["eta_cross"],
                  dt_max_gyr=integ["dt_max_gyr"], n_levels=integ["n_levels"],
                  brute_max=integ["brute_max"], profile_every=integ["profile_every_ticks"],
                  stop_on_weak_violation=spec.get("stop_on_weak_violation", False),
                  weak_max=prereg["gates"]["weak_regime_eps_max"],
                  stop_well_speed_kms=prereg["stop_rules"]["runaway_well_speed_kms"] if spec["cronos"] else None,
                  max_wall_s=prereg["stop_rules"]["max_wall_hours_per_run"] * 3600.0,
                  field_static=spec.get("field_static", False),
                  tau_avg_myr=integ["field_tau_avg_myr"],
                  field_kwargs={"k_inner": spec.get("field_k_inner", integ["field_k_inner"]),
                                "smooth": integ["field_smooth"]})
    ev0 = run.events[0]
    log(f"  brazo {arm} semilla {seed}: A = {A:.3e}, D_F(ε_soft) = {ev0['D_F_at_soft']:.2f}, "
        f"r(D_F = 1) = {ev0['r_DF_equals_one_kpc']}, ε_c,max = {ev0['eps_c_max']:.2e}, v_well = {ev0['v_well_kms']:.0f} km/s, "
        f"r_inner = {ev0['r_inner_field_kpc']:.3f} kpc, régimen débil = {ev0['weak_regime_ok']}")
    out = run.run(spec["t_end_gyr"], prereg["snapshots_gyr"], log=log)
    out.update({"arm": arm, "seed": seed, "A": A, "N": sysm["N"], "ic_diag": {
        "M_tot": ic["M_tot"], "r_max": ic["r_max"], "f_negative_fraction": ic["f_negative_fraction"]},
        "levels_t0": {str(k): int(v) for k, v in zip(*np.unique(run.s_old, return_counts=True))}})
    return out


def cmd_run(args) -> int:
    prereg, psha = load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    log = _log(OUTDIR / "run.log")
    log(f"[prereg] {psha[:12]}… — inicio de corridas")
    arms = args.arms.split(",") if args.arms else list(prereg["arms"])
    for arm in arms:
        spec = prereg["arms"][arm]
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else spec["seeds"]
        for seed in seeds:
            path = RUNS / f"{arm}_seed{seed}.json"
            if path.exists():
                log(f"  {path.name} ya existe: se omite")
                continue
            log(f"== brazo {arm} ({spec['label']}), semilla {seed}")
            out = run_one(prereg, arm, seed, log)
            out["preregistration_sha256"] = psha
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            log(f"  guardado {path.name} ({out['wall_s']} s; parada temprana: {out['stopped_early']} {out['stop_reason'] or ''})")
    log("fin de corridas")
    return 0


# ---------------------------------------------------------------- análisis

def _profile(snap: dict):
    return np.array(snap["r_mid_kpc"]), np.array(snap["rho"]), np.array(snap["count"])


def _band_mean_log_ratio(rb, ra, r, lo, hi):
    sel = (r >= lo) & (r < hi) & (ra > 0) & (rb > 0)
    return float(np.mean(np.log10(rb[sel] / ra[sel]))) if sel.any() else float("nan")


def cmd_analyze(_args) -> int:
    from cronos.profile_fit import shape_comparison
    prereg, psha = load_prereg()
    rules, gates = prereg["rules"], prereg["gates"]
    runs = {}
    for p in sorted(RUNS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("preregistration_sha256") != psha:
            raise SystemExit(f"FALLO CERRADO: {p.name} pertenece a otra preinscripción")
        runs.setdefault(d["arm"], {})[d["seed"]] = d
    missing = [(a, s) for a, spec in prereg["arms"].items() for s in spec["seeds"] if s not in runs.get(a, {})]
    final_t = prereg["snapshots_gyr"][-1]

    def final_snap(d):
        return d["snapshots"][-1]

    def a_snap_at(t_gyr):
        """Instantánea del brazo a (por semilla) más cercana a t_gyr."""
        out = {}
        for s, d in runs.get("a", {}).items():
            i = int(np.argmin([abs(sn["t_gyr"] - t_gyr) for sn in d["snapshots"]]))
            out[s] = d["snapshots"][i]
        return out

    # --- puertas -------------------------------------------------------
    gate = {"missing_runs": missing, "regime": {}, "numerical": {}}
    for arm, byseed in runs.items():
        for seed, d in byseed.items():
            ev_ok = all(e["weak_regime_ok"] and e["lapse_min"] > 0 for e in d["events"])
            gate["regime"][f"{arm}_seed{seed}"] = {"weak_regime_all_checks": bool(ev_ok),
                                                   "stopped_early": d["stopped_early"],
                                                   "stop_reason": d["stop_reason"],
                                                   "runaway_stop": any(e.get("runaway_stop") for e in d["events"])}
            e0, e1 = d["snapshots"][0]["energy"], final_snap(d)["energy"]
            dE = abs(e1["E"] - e0["E"]) / abs(e0["E"])
            # la puerta de energía rige en a (K + W) y en b_static (K + W + U con campo
            # congelado); en los brazos con campo dinámico se PUBLICA, no es puerta
            gated = arm in ("a", "b_static")
            tol = gates["energy_rel_tol_newton"] if arm == "a" else gates["energy_rel_tol_frozen"]
            gate["numerical"][f"{arm}_seed{seed}"] = {"dE_over_E": dE, "gated": gated,
                                                      "pass": bool(dE <= tol) if gated else None,
                                                      "t_final_gyr": d["t_final_gyr"]}
    # equilibrio del brazo a: |Δlog10 ρ| en [1, 10] kpc entre t = 0 y t_final (media de semillas)
    eq = []
    for seed, d in runs.get("a", {}).items():
        r, rho0, _ = _profile(d["snapshots"][0])
        _, rho1, _ = _profile(final_snap(d))
        eq.append(abs(_band_mean_log_ratio(rho1, rho0, r, 1.0, 10.0)))
    gate["equilibrium_a_dlog10rho_1_10kpc"] = float(np.mean(eq)) if eq else float("nan")
    gate["equilibrium_pass"] = bool(eq and np.mean(eq) <= gates["equilibrium_dlog10_max"])

    # --- perfiles finales, medias de semillas, cocientes b/a -------------
    def seed_mean_profile(arm):
        prof = [_profile(final_snap(d))[1] for d in runs[arm].values()]
        r = _profile(final_snap(next(iter(runs[arm].values()))))[0]
        return r, np.mean(prof, axis=0), np.std(prof, axis=0), np.array(prof)

    res: dict = {"bands": {}, "fits": {}}
    a_ok = "a" in runs and all(s in runs["a"] for s in prereg["arms"]["a"]["seeds"])
    b_ok = "b" in runs and all(s in runs["b"] for s in prereg["arms"]["b"]["seeds"])
    if a_ok and b_ok:
        # comparación en el instante final de cada corrida b (si paró antes por
        # runaway, con la instantánea de a más cercana a ese instante)
        t_b = {s: d["t_final_gyr"] for s, d in runs["b"].items()}
        prof_b = np.array([_profile(final_snap(d))[1] for d in runs["b"].values()])
        prof_a = np.array([_profile(a_snap_at(t_b[s])[s])[1] for s in runs["b"]])
        r = _profile(final_snap(next(iter(runs["b"].values()))))[0]
        rho_a = prof_a.mean(axis=0)
        rho_b = prof_b.mean(axis=0)
        res["t_compare_gyr"] = t_b
        for name, (lo, hi) in rules["bands_kpc"].items():
            sel = (r >= lo) & (r < hi) & (rho_a > 0) & (rho_b > 0)
            lr_pairs = [np.mean(np.log10(pb[sel] / pa[sel])) for pa, pb in zip(prof_a, prof_b)]
            sigma_seed_a = float(np.std([np.mean(np.log10(pa[sel] / rho_a[sel])) for pa in prof_a]))
            res["bands"][name] = {"log10_ratio_b_over_a_mean": float(np.mean(lr_pairs)),
                                  "log10_ratio_pairs": [float(x) for x in lr_pairs],
                                  "sigma_seed_a_log10": sigma_seed_a,
                                  "N_a_in_band_mean": float(np.mean([np.sum(np.array(final_snap(d)["count"])[sel]) for d in runs["a"].values()])),
                                  "significant": bool(abs(np.mean(lr_pairs)) > max(rules["band_min_log10_shift"], rules["band_sigma_factor"] * sigma_seed_a))}
        fit_lo, fit_hi = rules["fit_window_kpc"]
        sel = (r >= fit_lo) & (r <= fit_hi) & (rho_b > 0)
        res["fits"]["b_seed_mean"] = shape_comparison(r[sel], rho_b[sel])
        sel_a = (r >= fit_lo) & (r <= fit_hi) & (rho_a > 0)
        res["fits"]["a_seed_mean"] = shape_comparison(r[sel_a], rho_a[sel_a])
        res["fits"]["b_per_seed"] = {str(s): shape_comparison(r[(r >= fit_lo) & (r <= fit_hi) & (pb > 0)], pb[(r >= fit_lo) & (r <= fit_hi) & (pb > 0)])
                                     for s, pb in zip(runs["b"], prof_b)}
        # colapso progresivo: M_b(<0.4)/M_a(<0.4) por instantánea (a en los mismos instantes)
        ratios, t_axis = [], None
        for s, d in runs["b"].items():
            row = []
            for sb in d["snapshots"]:
                sa = a_snap_at(sb["t_gyr"])[s]
                row.append(sb["M_within"]["0.4"] / max(sa["M_within"]["0.4"], 1.0))
            ratios.append(row)
            t_axis = [sb["t_gyr"] for sb in d["snapshots"]] if t_axis is None or len(row) < len(t_axis) else t_axis
        n_common = min(len(r_) for r_ in ratios)
        ratios = np.array([r_[:n_common] for r_ in ratios])
        mean_ratio = ratios.mean(axis=0)
        res["M_within_0p4_ratio_b_over_a_by_snapshot"] = {"t_gyr": t_axis[:n_common], "mean": mean_ratio.tolist()}
        mono = bool(n_common >= 3 and np.all(np.diff(mean_ratio[1:]) > 0))
        stopped = [gate["regime"][f"b_seed{s}"]["runaway_stop"] for s in runs["b"]]
        res["runaway"] = {"monotonic": mono, "final_ratio": float(mean_ratio[-1]),
                          "runaway_stops": stopped,
                          "t_stop_gyr": [d["t_final_gyr"] for d in runs["b"].values()],
                          "flag": bool(all(stopped) or (mono and mean_ratio[-1] > rules["runaway_final_ratio_min"]))}
        res["N_a_within_0p4_final"] = float(np.mean([final_snap(d)["N_within_0p4"] for d in runs["a"].values()]))
        # control de convergencia del campo: b_res (k_inner = 128) frente a b (semilla 1)
        if "b_res" in runs and 1 in runs["b"] and 1 in runs["b_res"]:
            db, dr = runs["b"][1], runs["b_res"][1]
            t_ref = min(db["t_final_gyr"], dr["t_final_gyr"])
            ib = int(np.argmin([abs(sn["t_gyr"] - t_ref) for sn in db["snapshots"]]))
            ir = int(np.argmin([abs(sn["t_gyr"] - t_ref) for sn in dr["snapshots"]]))
            mb, mr = db["snapshots"][ib]["M_within"]["0.4"], dr["snapshots"][ir]["M_within"]["0.4"]
            res["field_resolution_control"] = {"t_ref_gyr": t_ref, "M_within_0p4_b": mb, "M_within_0p4_b_res": mr,
                                               "log10_ratio": float(np.log10(mr / mb)) if mb > 0 and mr > 0 else None,
                                               "t_stop_b": db["t_final_gyr"], "t_stop_b_res": dr["t_final_gyr"],
                                               "converged": bool(mb > 0 and mr > 0 and abs(np.log10(mr / mb)) <= rules["resolution_control_max_log10"])}
        res["friction"] = {arm: {str(s): {"gamma_active_fraction_final": final_snap(d)["gamma_active_fraction"],
                                          "gamma_mean_inner_per_gyr_final": final_snap(d)["gamma_mean_inner_per_gyr"]}
                                 for s, d in byseed.items()} for arm, byseed in runs.items() if arm != "a"}
    # control c′
    ctrl = None
    if "cprime" in runs and a_ok:
        d = next(iter(runs["cprime"].values()))
        t_c = d["t_final_gyr"]
        r, rho_c, _ = _profile(final_snap(d))
        rho_a_tc = np.mean([_profile(sn)[1] for sn in a_snap_at(t_c).values()], axis=0)
        lo, hi = rules["bands_kpc"]["inner_0p4_1"]
        ctrl = {"t_gyr": t_c, "log10_ratio_cprime_over_a_0p4_1kpc": _band_mean_log_ratio(rho_c, rho_a_tc, r, lo, hi),
                "responds": None}
        ctrl["responds"] = bool(abs(ctrl["log10_ratio_cprime_over_a_0p4_1kpc"]) >= rules["control_min_log10_shift"])
    res["control_cprime"] = ctrl
    res["control_c"] = ({s: {"stopped_early": d["stopped_early"], "stop_reason": d["stop_reason"],
                             "eps_c_max_t0": d["events"][0]["eps_c_max"]} for s, d in runs["c"].items()}
                        if "c" in runs else None)

    # --- regla congelada: puertas → C → B → A → INDETERMINADO ----------------
    reasons = []
    outcome = None
    if missing:
        outcome, reasons = "INDETERMINADO", [f"corridas ausentes: {missing}"]
    else:
        reg_b = all(v["weak_regime_all_checks"] for k, v in gate["regime"].items() if k.startswith("b_seed"))
        num_ok = all(v["pass"] for v in gate["numerical"].values() if v["gated"])
        if not reg_b:
            outcome, reasons = "INDETERMINADO", ["régimen débil violado en el brazo b"]
        elif not num_ok or not gate["equilibrium_pass"]:
            outcome, reasons = "INDETERMINADO", ["puerta numérica: energía o equilibrio del brazo a fuera de tolerancia"]
        elif ctrl is not None and not ctrl["responds"]:
            outcome, reasons = "INDETERMINADO", ["el control c′ (10·A_Sculptor) no altera el interior: implementación en duda"]
        else:
            fb, fa = res["fits"]["b_seed_mean"], res["fits"]["a_seed_mean"]
            cored_b = (fb["preferred"] == "cored" and fb["delta_rmse_log"] >= rules["cored_delta_rmse_min"]
                       and fb["cored"]["scale"] >= rules["cored_rc_min_kpc"])
            cored_a = (fa["preferred"] == "cored" and fa["delta_rmse_log"] >= rules["cored_delta_rmse_min"]
                       and fa["cored"]["scale"] >= rules["cored_rc_min_kpc"])
            if cored_b and not cored_a:
                outcome, reasons = "C", ["núcleo cored resuelto en b (no en a): sospecha de error primero"]
            elif cored_b and cored_a:
                outcome, reasons = "INDETERMINADO", ["ambos brazos cored: relajación numérica, no Cronos"]
            else:
                sig = [n for n, b in res["bands"].items() if b["significant"]]
                if sig:
                    sign = "más concentrado" if res["bands"][sig[0]]["log10_ratio_b_over_a_mean"] > 0 else "menos concentrado"
                    outcome = "B"
                    conv = res.get("field_resolution_control")
                    reasons = [f"interior modificado en {sig} con signo «{sign}»"
                               + (" — colapso progresivo (runaway)" if res["runaway"]["flag"] else "")
                               + ("" if conv is None else
                                  (" — ritmo convergido en resolución del campo" if conv["converged"]
                                   else " — ritmo DEPENDIENTE de la resolución del campo (k_inner 64 vs 128): sin predicción convergida del instrumento"))]
                elif res["N_a_within_0p4_final"] < rules["interior_resolved_N_min"]:
                    outcome, reasons = "A", ["b indistinguible de a en las bandas; interior ≤ 0.4 kpc no resuelto"]
                else:
                    outcome, reasons = "A", ["b indistinguible de a en todas las bandas, interior resuelto: sin efecto"]

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome": outcome, "reasons": reasons, "preregistration_sha256": psha,
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "final_t_gyr": final_t, "gates": gate, "results": res,
           "runs": {arm: {str(s): {"wall_s": d["wall_s"], "t_final_gyr": d["t_final_gyr"], "stopped_early": d["stopped_early"],
                                   "levels_t0": d["levels_t0"], "force_calls": final_snap(d)["force_calls"],
                                   "events_t0": d["events"][0]} for s, d in byseed.items()} for arm, byseed in runs.items()}}
    (OUTDIR / "cronos_halo_nivelA.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    word = {"C": "TENSIÓN: núcleo cored resuelto en el brazo Cronos y no en el newtoniano — sospecha de error primero",
            "B": "INTERIOR MODIFICADO por Cronos v3 a A_Sculptor (signo publicado) — predicción falsable del modelo operativo",
            "A": "a la amplitud galáctica congelada, Cronos v3 no produce núcleos kpc; interior ≤ 0.4 kpc indeterminado por resolución",
            "INDETERMINADO": "puerta violada o control sin efecto: el resultado se retiene"}[outcome]
    md = [f"# Nivel A — halo aislado con y sin Cronos v3: desenlace **{outcome}**\n",
          f"Preinscripción `{psha[:12]}` (congelada antes de generar ninguna condición inicial); commit `{sha[:9]}`.", "",
          f"**Lectura obligatoria**: {word}. Motivo: {'; '.join(reasons)}.", "",
          "## Puertas", "",
          f"Equilibrio del brazo a (|Δlog10 ρ| en [1, 10] kpc, media de semillas): {gate['equilibrium_a_dlog10rho_1_10kpc']:.3f} dex "
          f"(tolerancia {gates['equilibrium_dlog10_max']}) → {'pasa' if gate['equilibrium_pass'] else 'FALLA'}.", "",
          "| corrida | ΔE/E | puerta | régimen débil | t_final [Gyr] | parada temprana |", "|---|---|---|---|---|---|"]
    for k in sorted(gate["numerical"]):
        g, rg = gate["numerical"][k], gate["regime"][k]
        md.append(f"| {k} | {g['dE_over_E']:.2e} | {('pasa' if g['pass'] else 'FALLA') if g['gated'] else 'publicada'} | "
                  f"{'sí' if rg['weak_regime_all_checks'] else 'NO'} | {g['t_final_gyr']:.3f} | {rg['stop_reason'] or '—'} |")
    if res.get("bands"):
        md += ["", f"## Perfiles finales (t = {final_t} Gyr): cociente b/a por bandas (media de semillas)", "",
               "| banda [kpc] | log10(ρ_b/ρ_a) | σ_semillas(a) | N_a en banda | significativo |", "|---|---|---|---|---|"]
        for n, b in res["bands"].items():
            md.append(f"| {n} {rules['bands_kpc'][n]} | {b['log10_ratio_b_over_a_mean']:+.3f} | {b['sigma_seed_a_log10']:.3f} | "
                      f"{b['N_a_in_band_mean']:.0f} | {'sí' if b['significant'] else 'no'} |")
        fb, fa = res["fits"]["b_seed_mean"], res["fits"]["a_seed_mean"]
        md += ["", f"Ajuste de forma en {rules['fit_window_kpc']} kpc — brazo b: preferida {fb['preferred']}, Δrmse = {fb['delta_rmse_log']:+.3f} dex, "
               f"r_c(cored) = {fb['cored']['scale']:.2f} kpc, r_s(NFW) = {fb['nfw']['scale']:.2f} kpc; brazo a: preferida {fa['preferred']}, "
               f"Δrmse = {fa['delta_rmse_log']:+.3f} dex.", "",
               f"M_b(<0.4 kpc)/M_a(<0.4 kpc) por instantánea: {[round(x, 2) for x in res['M_within_0p4_ratio_b_over_a_by_snapshot']['mean']]} "
               f"(t = {[round(t, 3) for t in res['M_within_0p4_ratio_b_over_a_by_snapshot']['t_gyr']]} Gyr); colapso progresivo: {res['runaway']['flag']} "
               f"(paradas por runaway: {res['runaway']['runaway_stops']}, t_stop = {[round(t, 3) for t in res['runaway']['t_stop_gyr']]} Gyr). "
               f"N_a(<0.4 kpc) final = {res['N_a_within_0p4_final']:.0f} (resuelto si ≥ {rules['interior_resolved_N_min']}).", ""]
        conv = res.get("field_resolution_control")
        if conv:
            md += [f"Control de resolución del campo (k_inner 128 frente a 64, semilla 1, t = {conv['t_ref_gyr']:.3f} Gyr): "
                   f"log10(M_res/M_b)(<0.4 kpc) = {conv['log10_ratio']:+.3f} → {'convergido' if conv['converged'] else 'NO convergido'} "
                   f"(t_stop: b {conv['t_stop_b']:.3f}, b_res {conv['t_stop_b_res']:.3f} Gyr).", ""]
    if ctrl:
        md += [f"Control c′ (10·A_Sculptor, t = {ctrl['t_gyr']} Gyr): log10(ρ_c′/ρ_a) en [0.4, 1) kpc = {ctrl['log10_ratio_cprime_over_a_0p4_1kpc']:+.3f} → "
               f"{'responde' if ctrl['responds'] else 'NO responde'}.", ""]
    if res.get("control_c"):
        md += ["Control c (cierre cosmológico): " + "; ".join(f"semilla {s}: parada = {v['stopped_early']} ({v['stop_reason']}), ε_c,max(t=0) = {v['eps_c_max_t0']:.2e}"
                                                            for s, v in res["control_c"].items()), ""]
    md += ["Estatuto: experimento numérico interno (E8) bajo preinscripción, sin datos observacionales; campo de Cronos "
           "en aproximación de campo medio esférico; N y duración limitados por el presupuesto del entorno (declarado en la preinscripción)."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Desenlace {outcome}: {'; '.join(reasons)} → {OUTDIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", default=None)
    r.add_argument("--seeds", default=None)
    sub.add_parser("analyze")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
