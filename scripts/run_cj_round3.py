#!/usr/bin/env python
"""Test del criterio de Cronos–Jeans, RONDA 3 (orden del autor del 28-sep, PR-4): banda ultravioleta controlada.

    python scripts/run_cj_round3.py prereg     # congela (en un commit posterior al generador y a la cualificación E8-Q)
    python scripts/run_cj_round3.py run        # corridas reanudables
    python scripts/run_cj_round3.py analyze    # regla congelada; falla cerrado sin ella

La ronda 2 (results/2026-09-22_cj_criterion_round2/, INDETERMINADO) resolvió la fase lineal en 11 de 12 celdas al
1–2 % de la cinética exacta y falló la puerta en q = 2.0, n = 4: la banda ultravioleta del depósito, que crece
desde el ruido de redondeo a la tasa γ_UV medida después por la cualificación E8-Q
(results/2026-09-28_qualification_sheets/), alcanza la amplitud no lineal antes de que cierre la ventana
lineal del modo más lento. La ronda 3 cambia tres cosas, todas declaradas antes de correr:
  1. filtro espectral declarado en el depósito: k_c = ½ k_Nyquist (paso bajo de Fourier sobre la densidad antes
     de la no linealidad); la corrección de W(k) correspondiente es γ_inst(q, k) = γ_cin(q·W(k)) para k ≤ k_c y
     0 para k > k_c — los cuatro modos medidos (n ≤ 32, k ≤ 201) quedan por debajo de k_c = 804: W(k) intacta;
  2. celdas excluidas A PRIORI por el criterio de la cualificación (γ_UV·t_ventana > ln(A_nl/A_ruido_ef)) con
     el instrumento de la ronda 2 (ng 512, 1024 haces, p = 4) y el filtro: la preinscripción las lista y no las
     corre; una preinscripción solo puede incluir celdas que la cualificación no excluya (regla del programa);
  3. T por celda = mín(regla de la ronda 2, t_uv_nonlinear de la cualificación): la corrida termina antes de que
     la banda UV sin siembra alcance A_nl; la ventana lineal cierra antes (puerta computable).
Tolerancias y desenlaces son los de la ronda 2 (25 %, ≥ 8 puntos con r² ≥ 0.98): A — todas las celdas no
excluidas con q > 1 dentro de tolerancia y umbral en q = 1 (expectativa E13 por las 11/12 de la ronda 2);
B — acuerdo solo para n ≥ 8; C — tasa no ∝ k o umbral desplazado; INDETERMINADO — puertas. La ronda 2 queda
intacta; la fila criterio-cronos-jeans solo cambia con la letra.
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
from validation.qualification import load_qualification  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-09-28_cj_criterion_round3"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"
QUALIFICATION = ROOT / "results" / "2026-09-28_qualification_sheets"

Q_GRID = [0.8, 1.2, 2.0]
MODES = [4, 8, 16, 32]
NG, N_BEAMS, CELLS_PER_BEAM = 512, 1024, 4
K_CUT_FRAC = 0.5
INSTRUMENT = {
    "N": N_BEAMS * NG * CELLS_PER_BEAM, "ng": NG, "dt": 1e-3, "sample_every": 1, "full_modes_every": 50,
    "quiet_start": True, "n_beams": N_BEAMS, "cells_per_beam": CELLS_PER_BEAM,
    "lattice_rule": "N/n_beams = p·ng con p ENTERO (p = 4): depósito CIC de cada haz exactamente uniforme",
    "k_cut_frac": K_CUT_FRAC, "uv_band_frac": 0.5,
    "filter_rule": "paso bajo de Fourier sobre la densidad depositada antes de la no linealidad: k ≤ k_cut_frac·k_Nyquist; "
                   "corrección de W(k): γ_inst(q, k) = γ_cin(q·W(k)) si k ≤ k_c, 0 si k > k_c",
    "seed_eigenmode": True,
    "seed_eigenmode_rule": "cada haz j se desplaza ξ_j = −Im[c_j e^{ikx}]/k con c_j = κ v_j (v_j + iy')/(σ²(v_j² + y'²))·δρ̂ e y' = γ/k "
                           "de la predicción cinética PARA EL INSTRUMENTO (q·W(k)); q ≤ 1: siembra en densidad (no hay modo creciente)",
    "seed_amp_by_mode": {"4": 2e-5, "8": 2e-5, "16": 2e-6, "32": 2e-6},
    "seed": 1,
    "T_rule": "q > 1: T = mín(1.5·ln(1e-3/|δ_k|(0))/(γ_inst/k · k) + 0.1, t_uv_nonlinear(ng 512, 1024 haces, k_c, q) de la cualificación); q < 1: T = 0.6 L/σ",
    "T_stable": 0.6, "T_safety": 1.5, "delta_end": 1e-3,
    "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8},
}
RULES = {
    "linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)]: una década de amplitud, tras el transitorio",
    "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0,
    "min_points_linear": 8, "min_r2": 0.98,
    "rate_rel_tol": 0.25,                    # |γ_med/γ_inst − 1| ≤ 0.25 por celda no excluida (γ_inst = cinética con W(k))
    "k_independence_rel_spread": 0.25,       # (max − min)/media del cociente γ_med/γ_inst entre los modos no excluidos, por q
    "stable_max_growth_factor": 3.0,         # q = 0.8: max|δ_k|/|δ_k|(0) ≤ 3
    "beam_convergence_rel_tol": 0.02,        # |γ/k(512 haces) − γ/k(1024 haces)|/γ/k(1024) ≤ 0.02
    "uv_amp_nonlinear": 1e-2,                # puerta UV: uv_rms < A_nl hasta que cierra la ventana lineal, en cada celda no excluida
    "exclusion_source": "results/2026-09-28_qualification_sheets (E8-Q): celda excluida si γ_UV·t_ventana > ln(A_nl/A_ruido_ef) con "
                        "ng 512, 1024 haces, k_c = ½ k_Nyq; las celdas excluidas no se corren ni cuentan",
}
QUAL_TAG = f"ng{NG}_nb{N_BEAMS}"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _qual_cell(qual: dict, q: float, mode: int) -> dict:
    key = f"{QUAL_TAG}_kc{K_CUT_FRAC}_q{q}_n{mode}"
    cells = qual["exclusion_criteria"]["cells"]
    if key not in cells:
        raise SystemExit(f"FALLO CERRADO: la cualificación no tiene la celda {key}")
    return cells[key]


def _qual_t_uv(qual: dict, q: float) -> float:
    key = f"t_uv_nonlinear_{QUAL_TAG}_kc{K_CUT_FRAC}_q{q}"
    if key not in qual["limits"]:
        raise SystemExit(f"FALLO CERRADO: la cualificación no tiene el límite {key}")
    return float(qual["limits"][key]["value"])


def prereg(_args) -> int:
    qual = load_qualification(QUALIFICATION)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pred = prediction_table(Q_GRID)
    h = 1.0 / NG
    k_c = K_CUT_FRAC * np.pi * NG
    windows, excluded = {}, []
    for r in pred:
        for mode in MODES:
            k = 2.0 * np.pi * mode
            W = transfer_function_cic(k, h)
            amp0 = 0.5 * INSTRUMENT["seed_amp_by_mode"][str(mode)]
            key = f"q{r['q']}_n{mode}"
            if r["q"] > 1.0:
                g_inst = gamma_over_k_instrument(r["q"], k, h) if k <= k_c else 0.0
                t_uv = _qual_t_uv(qual, r["q"])
                T_rule = INSTRUMENT["T_safety"] * np.log(INSTRUMENT["delta_end"] / amp0) / (g_inst * k) + 0.1
                t_window_close = float(np.log(RULES["linear_window_hi_factor"]) / (g_inst * k))       # instante en que |δ_k| = 30·|δ_k|(0)
                qc = _qual_cell(qual, r["q"], mode)
                windows[key] = {"window": [RULES["linear_window_lo_factor"] * amp0, RULES["linear_window_hi_factor"] * amp0],
                                "T": float(min(T_rule, t_uv)), "T_rule_round2": float(T_rule), "t_uv_nonlinear": t_uv,
                                "t_window_close": t_window_close, "window_closes_before_uv_nonlinear": bool(t_window_close < t_uv),
                                "W_k": W, "q_eff": r["q"] * W, "k_below_cut": bool(k <= k_c),
                                "gamma_over_k_instrument": g_inst, "gamma_over_k_continuum": r["gamma_over_k_kinetic"], "delta_k_initial": amp0,
                                "expected_points_in_window": float(np.log(10.0) / (g_inst * k * INSTRUMENT["dt"] * INSTRUMENT["sample_every"])),
                                "qualification_cell": qc, "excluded_a_priori": bool(qc["excluded"]) or not (t_window_close < t_uv)}
                if windows[key]["excluded_a_priori"]:
                    excluded.append(key)
            else:
                windows[key] = {"window": None, "T": INSTRUMENT["T_stable"], "W_k": W, "q_eff": r["q"] * W, "k_below_cut": bool(k <= k_c),
                                "gamma_over_k_instrument": 0.0, "gamma_over_k_continuum": 0.0, "delta_k_initial": amp0, "excluded_a_priori": False}
    doc = {
        "title": "Test del criterio de Cronos–Jeans, ronda 3: banda ultravioleta controlada (filtro k_c declarado, celdas excluidas a priori por la cualificación E8-Q)",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_cj_round3.py prereg", "generator_commit_must_precede_freeze": True,
        "qualification": {"path": str(QUALIFICATION.relative_to(ROOT)), "sha256": qual["_sha256"], "code_commit": qual["code_commit"],
                          "rule": "una preinscripción solo puede incluir celdas que la cualificación no excluya y pedir T ≤ t_uv_nonlinear"},
        "round2": "results/2026-09-22_cj_criterion_round2 (INDETERMINADO por la fase lineal de q2.0_n4): intacta",
        "question": "¿Reproduce el instrumento de láminas, con la banda ultravioleta controlada y la fase lineal resuelta en todas las celdas no excluidas, "
                    "el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q·W(k)) proporcional a k de la ley local ε_c = A·ρ^(3/2)?",
        "system": {"sigma": 1.0, "rho0": 1.0, "L": 1.0, "gravity": False, "force": "+c²∂_x ε_c(ρ_f), ε_c = A·ρ^(3/2), ρ_f = densidad filtrada (k ≤ k_c), con (3/2)c²A·ρ0^(3/2) ≡ q·σ²"},
        "instrument": INSTRUMENT, "q_grid": Q_GRID, "modes": MODES, "k_cut": float(k_c), "prediction_kinetic": pred, "windows_and_T": windows,
        "excluded_cells_a_priori": excluded, "rules": RULES,
        "gates": {"linear_phase_resolved": "≥ min_points_linear puntos en la ventana con r² ≥ min_r2 en CADA celda no excluida con q > 1",
                  "seed_noise": "|δ_k|(0) de los modos NO sembrados < 0.1·|δ_k|(0) sembrado",
                  "lattice_exact": "cada corrida declara lattice_exact = true",
                  "filter_declared": "cada corrida declara k_cut_frac = 0.5",
                  "uv_controlled": "uv_rms < uv_amp_nonlinear hasta t_window_close en cada celda no excluida con q > 1",
                  "beam_convergence": "γ/k en (q = 1.2, n = 8) con 512 haces (p = 8) difiere del de 1024 haces (p = 4) en ≤ 2 %"},
        "outcomes": {
            "order": "puertas → C → B → A; INDETERMINADO si una puerta falla; ningún umbral se ajusta tras ver los números; las celdas excluidas a priori no cuentan",
            "C_criterion_fails": "crecimiento (factor > 3) en q = 0.8, o ausencia de crecimiento en alguna q > 1, o tasa NO proporcional a k "
                                 "(dispersión del cociente medido/predicho entre los modos no excluidos > 25 % en alguna q > 1)",
            "B_finite_size": "umbral y proporcionalidad correctos pero γ/k dentro del 25 % solo para n ≥ 8 (el modo n = 4 fuera de tolerancia "
                             "en alguna q > 1 no excluida): efecto de tamaño finito real, a entender, no un fallo del criterio",
            "A_kinetic_reproduced": "todas las celdas no excluidas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k), dispersión entre modos ≤ 25 % y "
                                    "estabilidad en q = 0.8: el instrumento reproduce el criterio con su predicción convergida",
            "INDETERMINADO": "puerta violada (fase lineal, ruido de siembra, retículo, filtro, UV, convergencia en haces) o corridas ausentes"},
        "expectations_E13": "A: la ronda 2 dio 11/12 celdas al 1–2 % de la cinética y la celda que falló es la que la cualificación excluye a priori",
        "consequence_if_A": "la fila criterio-cronos-jeans pasa a «derivado y confirmado por el test numérico» (E8: comprobación interna, no demostración física)",
        "development_declaration": {
            "pilots": "ninguno específico de la ronda 3: el filtro k_c y la banda UV se cualificaron en results/2026-09-28_qualification_sheets (30 corridas "
                      "sin siembra), que fija la exclusión y las cotas de T; la ronda 2 fijó el retículo exacto y la siembra del modo propio",
            "what_they_fixed": "solo el instrumento; NINGUNA tolerancia ni desenlace (25 %, 25 %, 2 %, 8 puntos, r² 0.98) se ajustó",
            "budget": f"{sum(1 for k, w in windows.items() if not w['excluded_a_priori'])} celdas + control de haces, N = 2 097 152: ~2–5 min cada una"},
        "prohibitions": {"no_threshold_tuning": True, "no_data": True, "no_change_to_A_sculptor": True, "round2_untouched": True, "qualification_untouched": True},
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador), ANTES de ninguna corrida de "
          f"producción. sha256 de `preregistration.json`: `{sha}`. Cualificación citada: `{qual['_sha256'][:12]}…` ({doc['qualification']['path']}).", "",
          f"Pregunta: {doc['question']}", "", f"Celdas excluidas a priori: {excluded or 'ninguna'}", "",
          "| celda | W(k) | γ/k instrumento | ventana | T | t_ventana cierra | t_uv_nonlinear | excluida | puntos esperados |", "|---|---|---|---|---|---|---|---|---|"]
    for key, w in windows.items():
        pts = "—" if w["window"] is None else "%.0f" % w["expected_points_in_window"]
        md.append(f"| {key} | {w['W_k']:.4f} | {w['gamma_over_k_instrument']:.4f} | {w['window']} | {w['T']:.3f} | "
                  f"{w.get('t_window_close', float('nan')):.3f} | {w.get('t_uv_nonlinear', float('nan')):.3f} | {'sí' if w['excluded_a_priori'] else 'no'} | {pts} |")
    md += ["", f"Instrumento: {json.dumps(INSTRUMENT, ensure_ascii=False)}", "", f"Reglas: {json.dumps(RULES, ensure_ascii=False)}", "",
           "Desenlaces: " + json.dumps(doc["outcomes"], ensure_ascii=False), "", "Expectativa (E13): " + doc["expectations_E13"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}; celdas excluidas a priori: {excluded}")
    return 0


def _load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción de la ronda 3")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def _run_cell(pre: dict, q: float, mode: int, n_beams: int, cells_per_beam: int) -> dict:
    ins = pre["instrument"]
    wt = pre["windows_and_T"][f"q{q}_n{mode}"]
    N = n_beams * ins["ng"] * cells_per_beam
    eig = wt["gamma_over_k_instrument"] if (ins["seed_eigenmode"] and q > 1.0) else None
    return run_sheets(q, N=N, ng=ins["ng"], T=wt["T"], dt=ins["dt"], seed=ins["seed"], nmodes=max(pre["modes"]),
                      sample_every=ins["sample_every"], quiet_start=True, n_beams=n_beams, seed_mode=mode,
                      seed_amp=ins["seed_amp_by_mode"][str(mode)], seed_eigen_gamma_over_k=eig, q_seed=q,
                      full_modes_every=ins["full_modes_every"], k_cut_frac=ins["k_cut_frac"], uv_band_frac=ins["uv_band_frac"])


def cmd_run(_args) -> int:
    pre, psha = _load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    ins = pre["instrument"]
    for q in pre["q_grid"]:
        for mode in pre["modes"]:
            key = f"q{q}_n{mode}"
            if key in pre["excluded_cells_a_priori"]:
                print(f"  {key}: excluida a priori por la cualificación — no se corre", flush=True)
                continue
            path = RUNS / f"{key}.json"
            if path.exists():
                print(f"  {path.name} ya existe: se omite", flush=True)
                continue
            t0 = time.time()
            res = _run_cell(pre, q, mode, ins["n_beams"], ins["cells_per_beam"])
            res["preregistration_sha256"] = psha
            res["wall_s"] = round(time.time() - t0, 1)
            path.write_text(json.dumps(res) + "\n", encoding="utf-8")
            amp = [s["delta_k"][mode - 1] for s in res["samples"] if s["delta_k"][mode - 1] is not None]
            uv = max(s["uv_rms"] for s in res["samples"])
            print(f"  q = {q}, modo {mode}: {res['wall_s']} s; retículo exacto {res['lattice_exact']}; k_c {res['k_cut_frac']}; "
                  f"|δ_k| {amp[0]:.2e} → máx {max(amp):.2e}; UV máx {uv:.1e}", flush=True)
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
    excluded = set(pre["excluded_cells_a_priori"])
    runs, missing = {}, []
    for q in pre["q_grid"]:
        for mode in pre["modes"]:
            key = f"q{q}_n{mode}"
            if key in excluded:
                continue
            p = RUNS / f"{key}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("preregistration_sha256") != psha:
                raise SystemExit(f"FALLO CERRADO: {p.name} pertenece a otra preinscripción")
            runs[(q, mode)] = d
    table, gate_linear, gate_seed, gate_lattice, gate_filter, gate_uv = [], [], [], [], [], []
    for (q, mode), d in runs.items():
        wt = pre["windows_and_T"][f"q{q}_n{mode}"]
        win = wt["window"]
        lo, hi = win if win is not None else (1e-9, 1.0)
        fit = fit_growth(d, mode, lo, hi)
        others = [d["samples"][0]["delta_k"][m - 1] for m in pre["modes"] if m != mode]
        noise_ok = all(a < 0.1 * wt["delta_k_initial"] for a in others)
        key = f"q{q}_n{mode}"
        if not noise_ok:
            gate_seed.append(key)
        if not d.get("lattice_exact"):
            gate_lattice.append(key)
        if d.get("k_cut_frac") != ins["k_cut_frac"]:
            gate_filter.append(key)
        uv_until_close = max((s["uv_rms"] for s in d["samples"] if s["t"] <= wt.get("t_window_close", float("inf"))), default=0.0)
        uv_ok = bool(uv_until_close < R["uv_amp_nonlinear"])
        growth = fit["amp_max"] / max(fit["amp_initial"], 1e-300)
        row = {"q": q, "mode": mode, "k": fit["k"], "gamma_over_k_measured": fit["gamma_over_k"], "r2": fit["r2"], "n_points": fit["n_points"],
               "growth_factor": growth, "gamma_over_k_instrument": wt["gamma_over_k_instrument"], "W_k": wt["W_k"],
               "gamma_over_k_kinetic": pred[str(q)]["gamma_over_k_kinetic"], "gamma_over_k_fluid": pred[str(q)]["gamma_over_k_fluid"],
               "seed_noise_ok": noise_ok, "lattice_exact": d.get("lattice_exact"), "k_cut_frac": d.get("k_cut_frac"),
               "uv_rms_max_until_window_close": uv_until_close, "uv_ok": uv_ok, "uv_rms_max_total": max(s["uv_rms"] for s in d["samples"]),
               "eigen_dispersion_residual": d.get("eigen_dispersion_residual"), "wall_s": d.get("wall_s")}
        if q > 1.0:
            if not uv_ok:
                gate_uv.append(key)
            row["linear_resolved"] = bool(fit["n_points"] >= R["min_points_linear"] and np.isfinite(fit["r2"]) and fit["r2"] >= R["min_r2"])
            row["ratio_measured_over_instrument"] = float(fit["gamma_over_k"] / wt["gamma_over_k_instrument"]) if np.isfinite(fit["gamma_over_k"]) else None
            row["rate_within_tol"] = (bool(abs(row["ratio_measured_over_instrument"] - 1.0) <= R["rate_rel_tol"]) if row["linear_resolved"] else None)
            row["closer_to"] = (None if not np.isfinite(fit["gamma_over_k"]) else
                                ("kinetic" if abs(fit["gamma_over_k"] - wt["gamma_over_k_instrument"]) <= abs(fit["gamma_over_k"] - row["gamma_over_k_fluid"]) else "fluid"))
            if not row["linear_resolved"]:
                gate_linear.append(key)
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
    elif gate_lattice or gate_filter:
        outcome, reasons = "INDETERMINADO", [f"puerta del retículo/filtro violada: {gate_lattice + gate_filter}"]
    elif gate_seed:
        outcome, reasons = "INDETERMINADO", [f"puerta de ruido de siembra violada: {gate_seed}"]
    elif gate_uv:
        outcome, reasons = "INDETERMINADO", [f"puerta UV violada (banda ≥ A_nl antes de cerrar la ventana): {gate_uv}"]
    elif beam_ctrl is None or not beam_ctrl["pass"]:
        outcome, reasons = "INDETERMINADO", [f"puerta de convergencia en haces violada: {beam_ctrl}"]
    elif gate_linear:
        outcome, reasons = "INDETERMINADO", [f"fase lineal no resuelta: {gate_linear}"]
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
            outcome, reasons = "A", ["umbral correcto; todas las celdas no excluidas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k); dispersión entre modos ≤ 25 %"]
        elif all(r["mode"] == 4 for r in off):
            outcome, reasons = "B", ["acuerdo solo para n ≥ 8; n = 4 fuera de tolerancia en %s (tamaño finito, a entender)" % ["q%s" % r["q"] for r in off]]
        else:
            outcome, reasons = "C", ["tasa fuera de tolerancia en modos n ≥ 8: %s" % ["q%s_n%s" % (r["q"], r["mode"]) for r in off]]
    sha = _git_head()
    doc = {"outcome": outcome, "reasons": reasons, "preregistration_sha256": psha, "qualification_sha256": pre["qualification"]["sha256"],
           "excluded_cells_a_priori": sorted(excluded), "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "table": table, "k_independence": k_indep, "beam_convergence": beam_ctrl, "missing_runs": missing}
    (OUTDIR / "cj_criterion_round3.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    word = {"A": "el instrumento reproduce el criterio con su predicción cinética convergida en todas las celdas no excluidas",
            "B": "umbral y proporcionalidad correctos; el modo n = 4 queda fuera de tolerancia: tamaño finito, a entender",
            "C": "el criterio falla bajo la regla: crecimiento bajo el umbral, sin crecimiento sobre él o tasa no ∝ k",
            "INDETERMINADO": "puerta violada: el resultado se retiene"}[outcome]
    md = [f"# Test del criterio de Cronos–Jeans, ronda 3: desenlace **{outcome}**\n",
          f"Preinscripción `{psha[:12]}`; cualificación `{pre['qualification']['sha256'][:12]}`; commit `{sha[:9]}`. **Lectura obligatoria**: {word}. "
          f"Motivo: {'; '.join(reasons)}. Celdas excluidas a priori: {sorted(excluded) or 'ninguna'}.", "",
          "| q | n | k | γ/k medido | γ/k instrumento (W(k)) | γ/k continuo | γ/k fluido | más cerca de | r² | puntos | crecimiento | UV máx (ventana) | fila |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        verdict = (("estable" if r["stable"] else "CRECE") if r["q"] < 1.0 else
                   ("no resuelto" if not r["linear_resolved"] else ("dentro de tol." if r["rate_within_tol"] else "fuera de tol.")))
        g = r["gamma_over_k_measured"]
        g_txt = "—" if not np.isfinite(g) else "%.4f" % g
        r2_txt = "—" if not np.isfinite(r["r2"]) else "%.4f" % r["r2"]
        md.append(f"| {r['q']} | {r['mode']} | {r['k']:.1f} | {g_txt} | {r['gamma_over_k_instrument']:.4f} (W = {r['W_k']:.3f}) | "
                  f"{r['gamma_over_k_kinetic']:.4f} | {r['gamma_over_k_fluid']:.4f} | {r.get('closer_to') or '—'} | "
                  f"{r2_txt} | {r['n_points']} | ×{r['growth_factor']:.1f} | {r['uv_rms_max_until_window_close']:.1e} | {verdict} |")
    md.append("")
    for q, v in k_indep.items():
        spread = "—" if v["rel_spread"] is None else "%.3f" % v["rel_spread"]
        md.append(f"Independencia de k en q = {q}: dispersión relativa {spread} → {'pasa' if v['pass'] else 'FALLA'}")
    if beam_ctrl:
        md.append(f"Convergencia en haces (q = {bc['q']}, n = {bc['mode']}): γ/k = {beam_ctrl['gamma_over_k_1024']:.4f} (1024) frente a "
                  f"{beam_ctrl['gamma_over_k_alt']:.4f} ({beam_ctrl['n_beams_alt']}): diferencia relativa "
                  f"{'—' if beam_ctrl['rel_diff'] is None else '%.4f' % beam_ctrl['rel_diff']} → {'pasa' if beam_ctrl['pass'] else 'FALLA'}")
    md += ["", "Estatuto: test numérico interno (E8) bajo preinscripción nueva con celdas excluidas a priori por una cualificación E8-Q; predicción "
           "de la teoría cinética lineal (solo q entra); sin datos; A_Sculptor, la ronda 2 y la cualificación no se tocan."]
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
