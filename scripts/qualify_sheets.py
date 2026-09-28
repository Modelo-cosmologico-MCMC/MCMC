#!/usr/bin/env python
"""Cualificación E8-Q del instrumento de láminas (cronos/cronos_jeans_1d.py) — PR-3 de la orden del 28-sep.

    python scripts/qualify_sheets.py run       # corre la rejilla y escribe results/<fecha>_qualification_sheets/
    python scripts/qualify_sheets.py table     # imprime los límites y la tabla de exclusión de las celdas de la ronda 3

Sin letras ni umbrales de desenlace. Mide, para cada (ng, celdas por haz p, q, filtro k_c):
  * γ_UV: tasa de crecimiento de la banda ultravioleta (k ≥ ½ k_Nyquist) de la densidad depositada, que
    arranca del ruido de redondeo (~1e-15) sin ninguna siembra (arranque silencioso, retículo exacto);
  * A_ruido: amplitud UV en la primera muestra tras el arranque;
  * y publica, para cada celda (q, n) de la ronda 3, la ventana lineal t_ventana = ln(hi/lo)/γ_n con γ_n la
    tasa cinética del instrumento (q·W(k)), y el criterio de exclusión
        celda excluida si γ_UV · t_ventana > ln(A_nl / A_ruido)
    con A_nl declarada (la amplitud a la que la banda UV deja de ser lineal y contamina los modos bajos).
La comparación con y sin filtro k_c = ½ k_Nyquist dice si el filtro declarado suprime la banda (γ_UV → 0) sin
tocar los modos k ≤ k_c (W(k) intacta). Regla: la preinscripción de la ronda 3 solo puede incluir celdas que
este criterio no excluya, y ninguna tolerancia se decide aquí.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cronos.cronos_jeans_1d import run_sheets  # noqa: E402
from cronos.cronos_jeans_kinetic import (  # noqa: E402
    gamma_over_k_instrument,
    transfer_function_cic,
)
from validation.qualification import (  # noqa: E402
    sheet_cell_excluded,
    write_qualification,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-28_qualification_sheets"
RUNS = OUTDIR / "runs"
GRID = {"ng": [256, 512], "cells_per_beam": [2, 4], "q": [0.8, 1.2, 2.0], "k_cut_frac": [None, 0.5],
        "n_beams": 256, "T": 0.4, "dt": 1e-3, "sample_every": 5, "seed": 1,
        "uv_band_frac": 0.5, "amp_nonlinear_declared": 1e-2,
        "round3_cells": {"q": [1.2, 2.0], "modes": [4, 8, 16, 32]},
        "linear_window_factors": [3.0, 30.0], "seed_amp_by_mode": {"4": 2e-5, "8": 2e-5, "16": 2e-6, "32": 2e-6}}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


NOISE_FACTOR = 10.0        # la banda «ha crecido» cuando supera 10 × su amplitud en la primera muestra tras el arranque
UV_CEILING = 1e-4          # por encima, la banda deja de ser lineal: el ajuste se detiene antes


def fit_uv_growth(curve, noise_factor: float = NOISE_FACTOR, ceiling: float = UV_CEILING) -> dict:
    """γ_UV por ajuste log-lineal de uv_rms(t) en la racha MÁS LARGA de puntos con noise_factor·A_ruido < uv_rms < ceiling.

    `curve` es la lista [[t, uv_rms], ...] guardada en cada corrida (o la lista de muestras); la regla es única y se aplica
    en `write_artifact`, de modo que el artefacto se recompone siempre desde los datos brutos con la misma regla.
    Resultado: γ_UV = 0 si la banda nunca superó noise_factor·A_ruido; NaN si superó pero con menos de 4 puntos ajustables
    (crecimiento más rápido que el muestreo: la celda se trata como no cualificable, no como benigna).
    """
    if curve and isinstance(curve[0], dict):
        curve = [[s["t"], s["uv_rms"]] for s in curve]
    t = np.array([c[0] for c in curve], dtype=float); u = np.array([c[1] for c in curve], dtype=float)
    a_noise = float(u[1]) if len(u) > 1 else float("nan")
    base = {"uv_rms_initial": a_noise, "uv_rms_final": float(u[-1]), "uv_rms_max": float(u.max()), "fit_rule": "longest run"}
    ok = np.isfinite(u) & (u > noise_factor * max(a_noise, 1e-300)) & (u < ceiling)
    idx = np.nonzero(ok)[0]
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1) if idx.size else []
    idx = max(runs, key=len) if runs else np.array([], dtype=int)
    if idx.size < 4:
        grew = bool(u.max() >= noise_factor * max(a_noise, 1e-300))
        return {"gamma_uv": float("nan") if grew else 0.0, "n_points": int(idx.size), "r2": None, **base}
    y = np.log(u[idx]); p = np.polyfit(t[idx], y, 1); resid = y - np.polyval(p, t[idx])
    r2 = 1.0 - float(np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300))
    return {"gamma_uv": float(p[0]), "n_points": int(idx.size), "r2": r2, "t_window": [float(t[idx[0]]), float(t[idx[-1]])], **base}


def _r2(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def cmd_run(_args) -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    runs = []
    for ng in GRID["ng"]:
        for p in GRID["cells_per_beam"]:
            for q in GRID["q"]:
                for kc in GRID["k_cut_frac"]:
                    key = f"ng{ng}_p{p}_q{q}_kc{kc}"
                    path = RUNS / f"{key}.json"
                    if path.exists():
                        runs.append(json.loads(path.read_text(encoding="utf-8"))); print(f"  {key}: existe"); continue
                    t0 = time.time()
                    res = run_sheets(q, N=GRID["n_beams"] * ng * p, ng=ng, T=GRID["T"], dt=GRID["dt"], seed=GRID["seed"],
                                     nmodes=4, sample_every=GRID["sample_every"], quiet_start=True, n_beams=GRID["n_beams"],
                                     k_cut_frac=kc, uv_band_frac=GRID["uv_band_frac"], full_modes_every=1000)
                    fit = fit_uv_growth(res["samples"])
                    rec = {"key": key, "ng": ng, "cells_per_beam": p, "N": res["N"], "q": q, "k_cut_frac": kc,
                           "lattice_exact": res["lattice_exact"], "wall_s": round(time.time() - t0, 1), **fit,
                           "low_modes_final": res["samples"][-1]["delta_k"], "low_modes_initial": res["samples"][0]["delta_k"],
                           "uv_curve": [[s["t"], s["uv_rms"]] for s in res["samples"]]}
                    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
                    runs.append(rec)
                    print(f"  {key}: γ_UV = {fit['gamma_uv']}, A_ruido = {fit['uv_rms_initial']:.1e}, UV final {fit['uv_rms_final']:.1e} ({rec['wall_s']} s)", flush=True)
    return write_artifact(runs)


def write_artifact(runs: list) -> int:
    # límites: γ_UV máxima por (ng, filtro) y A_ruido máxima; tabla de exclusión para las celdas de la ronda 3
    for r in runs:                       # regla única de ajuste, aplicada a las curvas guardadas (recomponible)
        if "uv_curve" in r:
            r.update(fit_uv_growth(r["uv_curve"]))
    limits = {}
    for kc in GRID["k_cut_frac"]:
        for ng in GRID["ng"]:
            grp = [r for r in runs if r["k_cut_frac"] == kc and r["ng"] == ng]
            if not grp:
                continue
            limits[f"uv_noise_max_ng{ng}_kc{kc}"] = {"value": max(r["uv_rms_initial"] for r in grp), "runs": [r["key"] for r in grp]}
            for q in GRID["q"]:
                sel = [r for r in grp if r["q"] == q]
                if not sel:
                    continue
                unfit = [r["key"] for r in sel if not np.isfinite(r["gamma_uv"])]
                # fallo cerrado: una corrida que creció sin racha ajustable hace INFINITO el límite (todas sus celdas excluidas)
                g_max = math.inf if unfit else max(r["gamma_uv"] for r in sel)
                limits[f"gamma_uv_max_ng{ng}_kc{kc}_q{q}"] = {"value": g_max, "unit": "1/(L/σ)", "unfittable_runs": unfit,
                                                              "per_run": {r["key"]: r["gamma_uv"] for r in sel},
                                                              "uv_final_max": max(r["uv_rms_final"] for r in sel)}
                # instante en que la banda UV, desde el ruido de redondeo y sin siembra, alcanza A_nl: cota superior de T por celda
                a_noise = max(limits[f"uv_noise_max_ng{ng}_kc{kc}"]["value"], 1e-16)
                limits[f"t_uv_nonlinear_ng{ng}_kc{kc}_q{q}"] = {
                    "value": (math.log(GRID["amp_nonlinear_declared"] / a_noise) / g_max) if g_max > 0 else math.inf,
                    "meaning": "ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, k_c, q)"}
    # efecto del filtro declarado sobre la banda UV del depósito: cociente de tasas con y sin filtro, misma (ng, q)
    for ng in GRID["ng"]:
        for q in GRID["q"]:
            a, b = limits.get(f"gamma_uv_max_ng{ng}_kcNone_q{q}"), limits.get(f"gamma_uv_max_ng{ng}_kc0.5_q{q}")
            if a and b and np.isfinite(a["value"]) and a["value"] > 0 and np.isfinite(b["value"]):
                limits[f"filter_gamma_ratio_ng{ng}_q{q}"] = {"value": b["value"] / a["value"],
                                                             "meaning": "γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada"}
    excl = {}
    for ng in GRID["ng"]:
        h = 1.0 / ng
        for kc in GRID["k_cut_frac"]:
            n_key = f"uv_noise_max_ng{ng}_kc{kc}"
            if n_key not in limits:
                continue
            for q in GRID["round3_cells"]["q"]:
                g_key = f"gamma_uv_max_ng{ng}_kc{kc}_q{q}"
                if g_key not in limits:
                    continue
                g_uv = limits[g_key]["value"]
                for n in GRID["round3_cells"]["modes"]:
                    k = 2.0 * np.pi * n
                    g_inst = gamma_over_k_instrument(q, k, h) * k
                    lo, hi = GRID["linear_window_factors"]
                    t_win = math.log(hi / lo) / g_inst if g_inst > 0 else math.inf
                    # A_ruido efectiva: el mayor entre el ruido de redondeo medido y el segundo armónico de la siembra (A_semilla²)
                    seed = GRID["seed_amp_by_mode"][str(n)]
                    a_noise = max(limits[n_key]["value"], 1e-16, seed ** 2)
                    c = sheet_cell_excluded(g_uv, t_win, GRID["amp_nonlinear_declared"], a_noise)
                    reason = ("sin ventana lineal (γ_inst ≤ 0)" if not math.isfinite(t_win) else
                              "γ_UV no ajustable (fallo cerrado)" if not math.isfinite(g_uv) else
                              "la banda UV alcanza A_nl dentro de la ventana" if c["excluded"] else "")
                    excl[f"ng{ng}_kc{kc}_q{q}_n{n}"] = {"W_k": transfer_function_cic(k, h), "gamma_inst": g_inst, "t_window": t_win,
                                                        "gamma_uv": g_uv, "amp_noise_eff": a_noise, "reason": reason, **c}
    criteria = {"rule": "celda (q, n) excluida si γ_UV(ng, k_c, q) · t_ventana > ln(A_nl / A_ruido_ef), con t_ventana = ln(hi/lo)/γ_inst(q, k) "
                        "(hi/lo = factores de la ventana lineal), A_nl declarada antes de correr y A_ruido_ef = máx(ruido UV medido, A_semilla²); "
                        "γ_UV se toma de la MISMA q (máximo sobre celdas por haz); una corrida sin racha ajustable excluye todas sus celdas; "
                        "una celda sin ventana lineal (γ_inst ≤ 0) queda excluida por no tener qué medir",
                "amp_nonlinear_declared": GRID["amp_nonlinear_declared"], "n_cells": len(excl),
                "n_excluded": sum(1 for v in excl.values() if v["excluded"]), "cells": excl}
    notes = ["La banda UV se mide sobre la densidad DEPOSITADA (antes del filtro): con filtro, γ_UV mide lo que el filtro deja pasar al campo de fuerzas a través de la no linealidad, no lo que queda en el depósito.",
             f"γ_UV = 0 significa que la banda no creció por encima de {NOISE_FACTOR:g}× el ruido inicial en T; NaN, que creció sin racha ajustable de ≥ 4 muestras (fallo cerrado: excluye).",
             "Todas las corridas arrancan del retículo exacto sin siembra: γ_UV es la tasa a la que el instrumento amplifica su propio ruido de redondeo, y crece con q y con ng (no con las celdas por haz).",
             "Nada de esta cualificación decide una letra ni una tolerancia: fija qué celdas puede incluir la preinscripción de la ronda 3."]
    sha = write_qualification(OUTDIR, "láminas (cronos_jeans_1d)", _git_head(), GRID, runs, limits, criteria, notes,
                              md_extra=["## Exclusión de las celdas de la ronda 3", "", "| celda | W(k) | γ_inst | t_ventana | γ_UV | γ_UV·t | ln(A_nl/A_ruido_ef) | excluida | motivo |", "|---|---|---|---|---|---|---|---|---|"]
                              + [f"| {k} | {v['W_k']:.3f} | {v['gamma_inst']:.2f} | {v['t_window']:.3f} | {v['gamma_uv']:.1f} | {v['gamma_uv_times_t_window']:.2f} | {v['ln_budget']:.1f} | {'sí' if v['excluded'] else 'no'} | {v['reason'] or '—'} |" for k, v in excl.items()]
                              + ["", "## Corridas", "", "| clave | N | γ_UV | puntos | r² | A_ruido | UV final | pared [s] |", "|---|---|---|---|---|---|---|---|"]
                              + [f"| {r['key']} | {r['N']} | {r['gamma_uv']:.2f} | {r['n_points']} | {_r2(r['r2'])} | {r['uv_rms_initial']:.1e} | {r['uv_rms_final']:.1e} | {r['wall_s']} |" for r in runs])
    print(f"cualificación escrita: {OUTDIR} sha256 {sha}")
    return 0


def cmd_table(_args) -> int:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))]
    return write_artifact(runs)


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run"); sub.add_parser("table")
    args = ap.parse_args()
    return {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
