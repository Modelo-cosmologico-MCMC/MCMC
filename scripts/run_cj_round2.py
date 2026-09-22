#!/usr/bin/env python
"""Test del criterio de Cronos–Jeans, RONDA 2 (orden del autor del 22-sep,
§6): siembra del modo propio creciente exacto, modos {4, 8, 16, 32},
q ∈ {0.8, 1.2, 2.0}, puertas más estrictas — bajo preinscripción nueva
(no un retoque de la ronda 1: `results/2026-09-21_cj_criterion_test/`
queda intacta con su INDETERMINADO).

    python scripts/run_cj_round2.py prereg     # congela (en un commit posterior al generador)
    python scripts/run_cj_round2.py run        # corridas reanudables
    python scripts/run_cj_round2.py analyze    # regla congelada; falla cerrado sin ella

Diagnóstico de la ronda 1 (§3.28): la puerta de fase lineal falló en tres
modos bajos con q > 1. Dos pilotos de instrumento (22-sep, declarados en
el JSON) mostraron la causa: NO era (solo) la proyección de la siembra en
densidad sobre el continuo de van Kampen — sembrar el modo propio exacto
por haz (amplitud |v|/√(v² + y'²) y fase arg(v + iy') por haz) mejoraba
r² de 0.39 a 0.67 en (q = 2, n = 4) pero no lo resolvía. La causa era el
BATIDO RETÍCULO/MALLA del arranque silencioso: con N/n_beams no múltiplo
entero de ng (1953 = 3.81·512 en la ronda 1), el depósito CIC de cada haz
deja un rizado a k = 2π·|per_beam − p·ng| que, con γ ∝ k (la catástrofe
ultravioleta que se quiere medir), crece desde ~1e-6 y se traga al modo
bajo antes de que su ventana lineal cierre. Con N/n_beams = p·ng entero el
depósito es EXACTAMENTE uniforme (partición de la unidad de la B-spline
lineal) y (q = 2, n = 4) da r² = 1.0000, γ/k = 0.6086 frente a 0.6089.
Los pilotos fijaron SOLO el instrumento (regla del retículo, siembra del
modo propio, amplitudes, muestreo); ninguna tolerancia ni desenlace se
ajustó a ellos.
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

from cronos.cronos_jeans_1d import fit_growth, run_sheets  # noqa: E402
from cronos.cronos_jeans_kinetic import (  # noqa: E402
    gamma_over_k_instrument,
    prediction_table,
    transfer_function_cic,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-22_cj_criterion_round2"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"

Q_GRID = [0.8, 1.2, 2.0]
MODES = [4, 8, 16, 32]
NG, N_BEAMS, CELLS_PER_BEAM = 512, 1024, 4
INSTRUMENT = {
    "N": N_BEAMS * NG * CELLS_PER_BEAM, "ng": NG, "dt": 1e-3, "sample_every": 1, "full_modes_every": 50,
    "quiet_start": True, "n_beams": N_BEAMS, "cells_per_beam": CELLS_PER_BEAM,
    "lattice_rule": "N/n_beams = p·ng con p ENTERO (p = 4): depósito CIC de cada haz exactamente uniforme; sin batido retículo/malla "
                    "(la regla de la ronda 1, N/n_beams ≥ 2·ng, no bastaba)",
    "seed_eigenmode": True,
    "seed_eigenmode_rule": "cada haz j se desplaza ξ_j = −Im[c_j e^{ikx}]/k con c_j = κ v_j (v_j + iy')/(σ²(v_j² + y'²))·δρ̂ e y' = γ/k "
                           "de la predicción cinética PARA EL INSTRUMENTO (q·W(k)); q ≤ 1: siembra en densidad (no hay modo creciente)",
    "seed_amp_by_mode": {"4": 2e-5, "8": 2e-5, "16": 2e-6, "32": 2e-6},
    "seed_amp_rule": "|δ_k|(0) = seed_amp/2: 1e-5 en los modos bajos y 1e-6 en los altos (los armónicos de la siembra escalan como δ²: "
                     "una siembra menor baja el suelo de los modos altos, que crecen más deprisa)",
    "seed": 1,
    "T_rule": "q > 1: T = 1.5·ln(1e-3/|δ_k|(0))/(γ_inst/k · k) + 0.1; q < 1: T = 0.6 L/σ",
    "T_stable": 0.6, "T_safety": 1.5, "delta_end": 1e-3,
    "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8},
}
RULES = {
    "linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)]: una década de amplitud, tras el transitorio",
    "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0,
    "min_points_linear": 8, "min_r2": 0.98,
    "rate_rel_tol": 0.25,                    # |γ_med/γ_inst − 1| ≤ 0.25 por celda (γ_inst = cinética con W(k))
    "k_independence_rel_spread": 0.25,       # (max − min)/media del cociente γ_med/γ_inst entre los cuatro modos, por q
    "stable_max_growth_factor": 3.0,         # q = 0.8: max|δ_k|/|δ_k|(0) ≤ 3
    "beam_convergence_rel_tol": 0.02,        # |γ/k(512 haces) − γ/k(1024 haces)|/γ/k(1024) ≤ 0.02
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


def prereg(_args) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred = prediction_table(Q_GRID)
    h = 1.0 / NG
    windows = {}
    for r in pred:
        for mode in MODES:
            k = 2.0 * np.pi * mode
            W = transfer_function_cic(k, h)
            amp0 = 0.5 * INSTRUMENT["seed_amp_by_mode"][str(mode)]
            if r["q"] > 1.0:
                g_inst = gamma_over_k_instrument(r["q"], k, h)
                T = INSTRUMENT["T_safety"] * np.log(INSTRUMENT["delta_end"] / amp0) / (g_inst * k) + 0.1
                windows[f"q{r['q']}_n{mode}"] = {"window": [RULES["linear_window_lo_factor"] * amp0, RULES["linear_window_hi_factor"] * amp0],
                                                 "T": float(T), "W_k": W, "q_eff": r["q"] * W, "gamma_over_k_instrument": g_inst,
                                                 "gamma_over_k_continuum": r["gamma_over_k_kinetic"], "delta_k_initial": amp0,
                                                 "expected_points_in_window": float(np.log(10.0) / (g_inst * k * INSTRUMENT["dt"] * INSTRUMENT["sample_every"]))}
            else:
                windows[f"q{r['q']}_n{mode}"] = {"window": None, "T": INSTRUMENT["T_stable"], "W_k": W, "q_eff": r["q"] * W,
                                                 "gamma_over_k_instrument": 0.0, "gamma_over_k_continuum": 0.0, "delta_k_initial": amp0}
    doc = {
        "title": "Test del criterio de Cronos–Jeans, ronda 2: siembra del modo propio exacto, retículo exacto, modos 4/8/16/32",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_cj_round2.py prereg", "generator_commit_must_precede_freeze": True,
        "round1": "results/2026-09-21_cj_criterion_test (INDETERMINADO por la puerta de fase lineal en q1.2_n4, q2.0_n4, q2.0_n8): intacta",
        "question": "¿Reproduce el instrumento de láminas, con fase lineal resuelta en TODAS las celdas, el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q·W(k)) proporcional a k de la ley local ε_c = A·ρ^(3/2)?",
        "system": {"sigma": 1.0, "rho0": 1.0, "L": 1.0, "gravity": False, "force": "+c²∂_x ε_c(ρ), ε_c = A·ρ^(3/2), con (3/2)c²A·ρ0^(3/2) ≡ q·σ²"},
        "instrument": INSTRUMENT, "q_grid": Q_GRID, "modes": MODES, "prediction_kinetic": pred, "windows_and_T": windows,
        "rules": RULES,
        "gates": {"linear_phase_resolved": "≥ min_points_linear puntos en la ventana con r² ≥ min_r2 en CADA celda con q > 1",
                  "seed_noise": "|δ_k|(0) de los modos NO sembrados < 0.1·|δ_k|(0) sembrado",
                  "lattice_exact": "cada corrida declara lattice_exact = true (N/n_beams múltiplo entero de ng)",
                  "beam_convergence": "γ/k en (q = 1.2, n = 8) con 512 haces (p = 8) difiere del de 1024 haces (p = 4) en ≤ 2 %"},
        "outcomes": {
            "order": "puertas → C → B → A; INDETERMINADO si una puerta falla; ningún umbral se ajusta tras ver los números",
            "C_criterion_fails": "crecimiento (factor > 3) en q = 0.8, o ausencia de crecimiento en alguna q > 1, o tasa NO proporcional a k "
                                 "(dispersión del cociente medido/predicho entre modos > 25 % en alguna q > 1)",
            "B_finite_size": "umbral y proporcionalidad correctos pero γ/k dentro del 25 % solo para n ≥ 8 (el modo n = 4 fuera de tolerancia "
                             "en alguna q > 1): efecto de tamaño finito real, a entender, no un fallo del criterio",
            "A_kinetic_reproduced": "todas las celdas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k), dispersión entre modos ≤ 25 % y "
                                    "estabilidad en q = 0.8: el instrumento reproduce el criterio con su predicción convergida",
            "INDETERMINADO": "puerta violada de nuevo (fase lineal, ruido de siembra, retículo, convergencia en haces) o corridas ausentes: "
                             "entonces se cambia de instrumento (Vlasov euleriano 1D), no de siembra"},
        "expectations_E13": "A (la ronda 1 dio 3/3 celdas resueltas al 1 % de la cinética; el piloto del retículo exacto dio 0.6086 frente a 0.6089 en q2_n4)",
        "development_declaration": {
            "pilots": ("22-sep, N = 5e5–2e6, ng = 256, q = 2, n ∈ {4, 8}: (i) siembra del modo propio frente a siembra en densidad con "
                       "N/n_beams = 488 y 976 (no múltiplo de ng): r² 0.39 → 0.67 en n = 4, ambas con el mismo crecimiento explosivo de "
                       "modos altos a t ≈ 0.15–0.18 dominado al final por el modo 8 ó 4 y con los modos altos partiendo de ~1e-10; la "
                       "siembra conjugada (−y') decae al principio, confirmando el signo del modo; (ii) N/n_beams = 4·ng: los modos no "
                       "sembrados arrancan en 1.5e-10 y solo se hacen relevantes tras cerrar la ventana; r² = 1.0000 y γ/k = 0.6086 "
                       "frente a 0.6089. Los pilotos mostraron tasas y se declaran como calibración del instrumento."),
            "what_they_fixed": "solo el instrumento: regla del retículo exacto, siembra del modo propio, amplitudes por modo, muestreo; "
                               "NINGUNA tolerancia ni desenlace (25 %, 25 %, 2 %, 8 puntos, r² 0.98) se ajustó a ellos",
            "budget": "13 corridas (3 q × 4 modos + control de haces) con N = 2 097 152: ~2–5 min cada una"},
        "prohibitions": {"no_threshold_tuning": True, "no_data": True, "no_change_to_A_sculptor": True, "round1_untouched": True},
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador), ANTES de ninguna corrida de "
          f"producción. sha256 de `preregistration.json`: `{sha}`.", "", f"Pregunta: {doc['question']}", "",
          "| q | y(q) | γ/k cinético | γ/k fluido | inestable |", "|---|---|---|---|---|"]
    for r in pred:
        md.append(f"| {r['q']} | {r['y']:.4f} | {r['gamma_over_k_kinetic']:.4f} | {r['gamma_over_k_fluid']:.4f} | {'sí' if r['unstable'] else 'no'} |")
    md += ["", "| celda | W(k) | γ/k instrumento | ventana | T | puntos esperados en la ventana |", "|---|---|---|---|---|---|"]
    for key, w in windows.items():
        pts = "—" if w["window"] is None else "%.0f" % w["expected_points_in_window"]
        md.append(f"| {key} | {w['W_k']:.4f} | {w['gamma_over_k_instrument']:.4f} | {w['window']} | {w['T']:.3f} | {pts} |")
    md += ["", f"Instrumento: {json.dumps(INSTRUMENT, ensure_ascii=False)}", "", f"Reglas: {json.dumps(RULES, ensure_ascii=False)}", "",
           "Desenlaces: " + json.dumps(doc["outcomes"], ensure_ascii=False), "", "Pilotos declarados: " + doc["development_declaration"]["pilots"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción de la ronda 2")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def _run_cell(pre: dict, q: float, mode: int, n_beams: int, cells_per_beam: int) -> dict:
    ins = pre["instrument"]
    wt = pre["windows_and_T"][f"q{q}_n{mode}"]
    N = n_beams * ins["ng"] * cells_per_beam
    eig = wt["gamma_over_k_instrument"] if (ins["seed_eigenmode"] and q > 1.0) else None
    return run_sheets(q, N=N, ng=ins["ng"], T=wt["T"], dt=ins["dt"], seed=ins["seed"], nmodes=max(pre["modes"]),
                      sample_every=ins["sample_every"], quiet_start=True, n_beams=n_beams, seed_mode=mode,
                      seed_amp=ins["seed_amp_by_mode"][str(mode)], seed_eigen_gamma_over_k=eig, q_seed=q,
                      full_modes_every=ins["full_modes_every"])


def cmd_run(_args) -> int:
    pre, psha = _load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    ins = pre["instrument"]
    for q in pre["q_grid"]:
        for mode in pre["modes"]:
            path = RUNS / f"q{q}_n{mode}.json"
            if path.exists():
                print(f"  {path.name} ya existe: se omite", flush=True)
                continue
            t0 = time.time()
            res = _run_cell(pre, q, mode, ins["n_beams"], ins["cells_per_beam"])
            res["preregistration_sha256"] = psha
            res["wall_s"] = round(time.time() - t0, 1)
            path.write_text(json.dumps(res) + "\n", encoding="utf-8")
            amp = [s["delta_k"][mode - 1] for s in res["samples"]]
            print(f"  q = {q}, modo {mode}: {res['wall_s']} s; retículo exacto {res['lattice_exact']}; |δ_k| {amp[0]:.2e} → máx {max(amp):.2e}", flush=True)
    bc = ins["beam_convergence_run"]
    path = RUNS / f"q{bc['q']}_n{bc['mode']}_beams{bc['n_beams_alt']}.json"
    if not path.exists():
        t0 = time.time()
        res = _run_cell(pre, bc["q"], bc["mode"], bc["n_beams_alt"], bc["cells_per_beam_alt"])
        res["preregistration_sha256"] = psha
        res["wall_s"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(res) + "\n", encoding="utf-8")
        print(f"  control de haces ({bc['n_beams_alt']}): {res['wall_s']} s", flush=True)
    return 0


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    R, ins = pre["rules"], pre["instrument"]
    pred = {str(r["q"]): r for r in pre["prediction_kinetic"]}
    runs, missing = {}, []
    for q in pre["q_grid"]:
        for mode in pre["modes"]:
            p = RUNS / f"q{q}_n{mode}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("preregistration_sha256") != psha:
                raise SystemExit(f"FALLO CERRADO: {p.name} pertenece a otra preinscripción")
            runs[(q, mode)] = d
    table, gate_linear, gate_seed, gate_lattice = [], [], [], []
    for (q, mode), d in runs.items():
        wt = pre["windows_and_T"][f"q{q}_n{mode}"]
        win = wt["window"]
        lo, hi = win if win is not None else (1e-9, 1.0)
        fit = fit_growth(d, mode, lo, hi)
        others = [d["samples"][0]["delta_k"][m - 1] for m in pre["modes"] if m != mode]
        noise_ok = all(a < 0.1 * wt["delta_k_initial"] for a in others)
        if not noise_ok:
            gate_seed.append(f"q{q}_n{mode}")
        if not d.get("lattice_exact"):
            gate_lattice.append(f"q{q}_n{mode}")
        growth = fit["amp_max"] / max(fit["amp_initial"], 1e-300)
        row = {"q": q, "mode": mode, "k": fit["k"], "gamma_over_k_measured": fit["gamma_over_k"], "r2": fit["r2"], "n_points": fit["n_points"],
               "growth_factor": growth, "gamma_over_k_instrument": wt["gamma_over_k_instrument"], "W_k": wt["W_k"],
               "gamma_over_k_kinetic": pred[str(q)]["gamma_over_k_kinetic"], "gamma_over_k_fluid": pred[str(q)]["gamma_over_k_fluid"],
               "seed_noise_ok": noise_ok, "lattice_exact": d.get("lattice_exact"), "eigen_dispersion_residual": d.get("eigen_dispersion_residual"),
               "wall_s": d.get("wall_s")}
        if q > 1.0:
            row["linear_resolved"] = bool(fit["n_points"] >= R["min_points_linear"] and np.isfinite(fit["r2"]) and fit["r2"] >= R["min_r2"])
            row["ratio_measured_over_instrument"] = float(fit["gamma_over_k"] / wt["gamma_over_k_instrument"]) if np.isfinite(fit["gamma_over_k"]) else None
            row["rate_within_tol"] = (bool(abs(row["ratio_measured_over_instrument"] - 1.0) <= R["rate_rel_tol"]) if row["linear_resolved"] else None)
            row["closer_to"] = (None if not np.isfinite(fit["gamma_over_k"]) else
                                ("kinetic" if abs(fit["gamma_over_k"] - wt["gamma_over_k_instrument"]) <= abs(fit["gamma_over_k"] - row["gamma_over_k_fluid"]) else "fluid"))
            if not row["linear_resolved"]:
                gate_linear.append(f"q{q}_n{mode}")
        else:
            row["stable"] = bool(growth <= R["stable_max_growth_factor"])
        table.append(row)
    k_indep = {}
    for q in pre["q_grid"]:
        if q > 1.0:
            vals = [r["ratio_measured_over_instrument"] for r in table if r["q"] == q and r.get("ratio_measured_over_instrument") is not None]
            spread = float((max(vals) - min(vals)) / np.mean(vals)) if len(vals) >= 2 else None
            k_indep[str(q)] = {"values": vals, "rel_spread": spread, "pass": bool(spread is not None and spread <= R["k_independence_rel_spread"])}
    bc = ins["beam_convergence_run"]
    bpath = RUNS / f"q{bc['q']}_n{bc['mode']}_beams{bc['n_beams_alt']}.json"
    beam_ctrl = None
    if bpath.exists() and (bc["q"], bc["mode"]) in runs:
        dalt = json.loads(bpath.read_text(encoding="utf-8"))
        if dalt.get("preregistration_sha256") != psha:
            raise SystemExit("FALLO CERRADO: el control de haces pertenece a otra preinscripción")
        win = pre["windows_and_T"][f"q{bc['q']}_n{bc['mode']}"]["window"]
        f_alt, f_ref = fit_growth(dalt, bc["mode"], *win), fit_growth(runs[(bc["q"], bc["mode"])], bc["mode"], *win)
        rel = (abs(f_alt["gamma_over_k"] - f_ref["gamma_over_k"]) / abs(f_ref["gamma_over_k"])
               if np.isfinite(f_alt["gamma_over_k"]) and np.isfinite(f_ref["gamma_over_k"]) else None)
        beam_ctrl = {"gamma_over_k_1024": f_ref["gamma_over_k"], "gamma_over_k_alt": f_alt["gamma_over_k"], "n_beams_alt": bc["n_beams_alt"],
                     "rel_diff": rel, "lattice_exact_alt": dalt.get("lattice_exact"), "pass": bool(rel is not None and rel <= R["beam_convergence_rel_tol"])}
    else:
        missing.append(bpath.name)
    # --- regla congelada: puertas → C → B → A -------------------------------
    stable_rows = [r for r in table if r["q"] < 1.0]
    unstable_rows = [r for r in table if r["q"] > 1.0]
    grew_below = [f"q{r['q']}_n{r['mode']}" for r in stable_rows if not r["stable"]]
    no_growth_q = [q for q in pre["q_grid"] if q > 1.0 and all(r["growth_factor"] <= R["stable_max_growth_factor"] for r in unstable_rows if r["q"] == q)]
    if missing:
        outcome, reasons = "INDETERMINADO", [f"corridas ausentes: {missing}"]
    elif gate_lattice:
        outcome, reasons = "INDETERMINADO", [f"puerta del retículo exacto violada: {gate_lattice}"]
    elif gate_seed:
        outcome, reasons = "INDETERMINADO", [f"puerta de ruido de siembra violada: {gate_seed}"]
    elif beam_ctrl is None or not beam_ctrl["pass"]:
        outcome, reasons = "INDETERMINADO", [f"puerta de convergencia en haces violada: {beam_ctrl}"]
    elif gate_linear:
        outcome, reasons = "INDETERMINADO", [f"fase lineal no resuelta: {gate_linear} — cambiar de instrumento (Vlasov euleriano), no de siembra"]
    elif grew_below or no_growth_q or not all(v["pass"] for v in k_indep.values()):
        reasons = []
        if grew_below:
            reasons.append(f"crecimiento con q = 0.8: {grew_below}")
        if no_growth_q:
            reasons.append(f"sin crecimiento con q > 1: {no_growth_q}")
        bad = [q for q, v in k_indep.items() if not v["pass"]]
        if bad:
            reasons.append(f"tasa no proporcional a k (dispersión > {R['k_independence_rel_spread']}): q = {bad}")
        outcome = "C"
    else:
        off = [r for r in unstable_rows if not r["rate_within_tol"]]
        if not off:
            outcome, reasons = "A", ["umbral correcto; todas las celdas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k); dispersión entre modos ≤ 25 %"]
        elif all(r["mode"] == 4 for r in off):
            outcome, reasons = "B", ["acuerdo solo para n ≥ 8; n = 4 fuera de tolerancia en %s (tamaño finito, a entender)" % ["q%s" % r["q"] for r in off]]
        else:
            outcome, reasons = "C", ["tasa fuera de tolerancia en modos n ≥ 8: %s" % ["q%s_n%s" % (r["q"], r["mode"]) for r in off]]
    sha = _git_head()
    doc = {"outcome": outcome, "reasons": reasons, "preregistration_sha256": psha, "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "table": table, "k_independence": k_indep, "beam_convergence": beam_ctrl, "missing_runs": missing}
    (OUTDIR / "cj_criterion_round2.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    word = {"A": "el instrumento reproduce el criterio con su predicción cinética convergida en todas las celdas",
            "B": "umbral y proporcionalidad correctos; el modo n = 4 queda fuera de tolerancia: tamaño finito, a entender",
            "C": "el criterio falla bajo la regla: crecimiento bajo el umbral, sin crecimiento sobre él o tasa no ∝ k",
            "INDETERMINADO": "puerta violada: el resultado se retiene; el paso siguiente es otro instrumento, no otra siembra"}[outcome]
    md = [f"# Test del criterio de Cronos–Jeans, ronda 2: desenlace **{outcome}**\n",
          f"Preinscripción `{psha[:12]}`; commit `{sha[:9]}`. **Lectura obligatoria**: {word}. Motivo: {'; '.join(reasons)}.", "",
          "| q | n | k | γ/k medido | γ/k instrumento (W(k)) | γ/k continuo | γ/k fluido | más cerca de | r² | puntos | crecimiento | retículo exacto | fila |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        verdict = (("estable" if r["stable"] else "CRECE") if r["q"] < 1.0 else
                   ("no resuelto" if not r["linear_resolved"] else ("dentro de tol." if r["rate_within_tol"] else "fuera de tol.")))
        g = r["gamma_over_k_measured"]
        g_txt = "—" if not np.isfinite(g) else "%.4f" % g
        r2_txt = "—" if not np.isfinite(r["r2"]) else "%.4f" % r["r2"]
        md.append(f"| {r['q']} | {r['mode']} | {r['k']:.1f} | {g_txt} | {r['gamma_over_k_instrument']:.4f} (W = {r['W_k']:.3f}) | "
                  f"{r['gamma_over_k_kinetic']:.4f} | {r['gamma_over_k_fluid']:.4f} | {r.get('closer_to') or '—'} | "
                  f"{r2_txt} | {r['n_points']} | ×{r['growth_factor']:.1f} | {r['lattice_exact']} | {verdict} |")
    md.append("")
    for q, v in k_indep.items():
        spread = "—" if v["rel_spread"] is None else "%.3f" % v["rel_spread"]
        md.append(f"Independencia de k en q = {q}: dispersión relativa {spread} → {'pasa' if v['pass'] else 'FALLA'}")
    if beam_ctrl:
        md.append(f"Convergencia en haces (q = {bc['q']}, n = {bc['mode']}): γ/k = {beam_ctrl['gamma_over_k_1024']:.4f} (1024) frente a "
                  f"{beam_ctrl['gamma_over_k_alt']:.4f} ({beam_ctrl['n_beams_alt']}): diferencia relativa "
                  f"{'—' if beam_ctrl['rel_diff'] is None else '%.4f' % beam_ctrl['rel_diff']} → {'pasa' if beam_ctrl['pass'] else 'FALLA'}")
    md += ["", "Estatuto: test numérico interno (E8) bajo preinscripción nueva; predicción de la teoría cinética lineal (solo q entra); sin datos; "
           "A_Sculptor, la fila criterio-cronos-jeans y la ronda 1 no se tocan."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Desenlace {outcome}: {'; '.join(reasons)} → {OUTDIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "run", "analyze"):
        sub.add_parser(c)
    args = ap.parse_args()
    return {"prereg": prereg, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
