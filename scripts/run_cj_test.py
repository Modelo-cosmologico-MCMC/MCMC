#!/usr/bin/env python
"""Test del criterio de Cronos–Jeans bajo la preinscripción congelada:
`run` ejecuta las corridas (reanudable por runs/q<q>_n<mode>.json) y
`analyze` aplica la regla sin decisiones nuevas. Fallo cerrado sin
preinscripción.

Uso: python scripts/run_cj_test.py run | analyze
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

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_cj_criterion_test"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción del test del criterio")
    return json.loads(PREREG.read_text(encoding="utf-8")), hashlib.sha256(PREREG.read_bytes()).hexdigest()


def cmd_run(_args) -> int:
    prereg, psha = load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    ins = prereg["instrument"]
    for q in prereg["q_grid"]:
        for mode in prereg["modes"]:
            path = RUNS / f"q{q}_n{mode}.json"
            if path.exists():
                print(f"  {path.name} ya existe: se omite")
                continue
            t0 = time.time()
            T = prereg["windows_and_T"][f"q{q}_n{mode}"]["T"]
            res = run_sheets(q, N=ins["N"], ng=ins["ng"], T=T, dt=ins["dt"], seed=ins["seed"],
                             nmodes=max(prereg["modes"]), sample_every=ins["sample_every"],
                             quiet_start=ins["quiet_start"], n_beams=ins["n_beams"], seed_mode=mode, seed_amp=ins["seed_amp"])
            res["preregistration_sha256"] = psha
            res["wall_s"] = round(time.time() - t0, 1)
            path.write_text(json.dumps(res) + "\n", encoding="utf-8")
            print(f"  q = {q}, modo {mode}: {res['wall_s']} s; |δ_k| inicial {res['samples'][0]['delta_k'][mode - 1]:.2e} → "
                  f"máximo {max(s['delta_k'][mode - 1] for s in res['samples']):.2e}")
    bc = ins["beam_convergence_run"]
    path = RUNS / f"q{bc['q']}_n{bc['mode']}_beams{bc['n_beams_alt']}.json"
    if not path.exists():
        t0 = time.time()
        T = prereg["windows_and_T"][f"q{bc['q']}_n{bc['mode']}"]["T"]
        res = run_sheets(bc["q"], N=ins["N"], ng=ins["ng"], T=T, dt=ins["dt"], seed=ins["seed"], nmodes=max(prereg["modes"]),
                         sample_every=ins["sample_every"], quiet_start=True, n_beams=bc["n_beams_alt"], seed_mode=bc["mode"],
                         seed_amp=ins["seed_amp"])
        res["preregistration_sha256"] = psha
        res["wall_s"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(res) + "\n", encoding="utf-8")
        print(f"  control de haces ({bc['n_beams_alt']}): {res['wall_s']} s")
    return 0


def cmd_analyze(_args) -> int:
    prereg, psha = load_prereg()
    rules = prereg["rules"]
    pred = {str(r["q"]): r for r in prereg["prediction_kinetic"]}
    runs, missing = {}, []
    for q in prereg["q_grid"]:
        for mode in prereg["modes"]:
            p = RUNS / f"q{q}_n{mode}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("preregistration_sha256") != psha:
                raise SystemExit(f"FALLO CERRADO: {p.name} pertenece a otra preinscripción")
            runs[(q, mode)] = d
    table, gate_fail, seed_noise_fail = [], [], []
    for (q, mode), d in runs.items():
        win = prereg["windows_and_T"][f"q{q}_n{mode}"]["window"]
        lo, hi = win if win is not None else (1e-9, 1.0)
        fit = fit_growth(d, mode, lo, hi)
        fit["window"] = win
        others = [d["samples"][0]["delta_k"][m - 1] for m in prereg["modes"] if m != mode]
        noise_ok = all(a < 0.1 * prereg["delta_k_initial"] for a in others)
        if not noise_ok:
            seed_noise_fail.append(f"q{q}_n{mode}")
        growth_factor = fit["amp_max"] / max(fit["amp_initial"], 1e-300)
        wt = prereg["windows_and_T"][f"q{q}_n{mode}"]
        row = {"q": q, "mode": mode, "k": fit["k"], "gamma_over_k_measured": fit["gamma_over_k"], "r2": fit["r2"],
               "n_points": fit["n_points"], "growth_factor": growth_factor,
               "gamma_over_k_kinetic": pred[str(q)]["gamma_over_k_kinetic"], "gamma_over_k_instrument": wt["gamma_over_k_instrument"],
               "W_k": wt["W_k"], "gamma_over_k_fluid": pred[str(q)]["gamma_over_k_fluid"],
               "seed_noise_ok": noise_ok, "wall_s": d.get("wall_s")}
        if q > 1.0:
            row["linear_resolved"] = bool(fit["n_points"] >= rules["min_points_linear"] and np.isfinite(fit["r2"]) and fit["r2"] >= rules["min_r2"])
            row["ratio_measured_over_instrument"] = float(fit["gamma_over_k"] / row["gamma_over_k_instrument"]) if np.isfinite(fit["gamma_over_k"]) else None
            row["rate_within_tol"] = (bool(abs(row["ratio_measured_over_instrument"] - 1.0) <= rules["rate_rel_tol"])
                                      if row["linear_resolved"] else None)
            row["closer_to"] = (None if not np.isfinite(fit["gamma_over_k"]) else
                                ("kinetic" if abs(fit["gamma_over_k"] - row["gamma_over_k_instrument"]) <= abs(fit["gamma_over_k"] - row["gamma_over_k_fluid"]) else "fluid"))
            if not row["linear_resolved"]:
                gate_fail.append(f"q{q}_n{mode}")
        else:
            row["stable"] = bool(growth_factor <= rules["stable_max_growth_factor"])
        table.append(row)
    # dispersión entre modos del cociente medido/predicho(instrumento) por q
    k_indep = {}
    for q in prereg["q_grid"]:
        if q > 1.0:
            vals = [r["ratio_measured_over_instrument"] for r in table if r["q"] == q and r.get("ratio_measured_over_instrument") is not None]
            k_indep[str(q)] = {"values": vals, "rel_spread": float((max(vals) - min(vals)) / np.mean(vals)) if len(vals) >= 2 else None,
                               "pass": bool(len(vals) >= 2 and (max(vals) - min(vals)) / np.mean(vals) <= rules["k_independence_rel_spread"])}
    # control de convergencia en haces
    bc = prereg["instrument"]["beam_convergence_run"]
    bpath = RUNS / f"q{bc['q']}_n{bc['mode']}_beams{bc['n_beams_alt']}.json"
    beam_ctrl = None
    if bpath.exists() and (bc["q"], bc["mode"]) in runs:
        dalt = json.loads(bpath.read_text(encoding="utf-8"))
        if dalt.get("preregistration_sha256") != psha:
            raise SystemExit("FALLO CERRADO: el control de haces pertenece a otra preinscripción")
        win = prereg["windows_and_T"][f"q{bc['q']}_n{bc['mode']}"]["window"]
        f_alt = fit_growth(dalt, bc["mode"], *win)
        f_ref = fit_growth(runs[(bc["q"], bc["mode"])], bc["mode"], *win)
        rel = abs(f_alt["gamma_over_k"] - f_ref["gamma_over_k"]) / abs(f_ref["gamma_over_k"]) if np.isfinite(f_alt["gamma_over_k"]) and np.isfinite(f_ref["gamma_over_k"]) else None
        beam_ctrl = {"gamma_over_k_1024": f_ref["gamma_over_k"], "gamma_over_k_alt": f_alt["gamma_over_k"], "n_beams_alt": bc["n_beams_alt"],
                     "rel_diff": rel, "pass": bool(rel is not None and rel <= rules["beam_convergence_rel_tol"])}
    else:
        missing.append(bpath.name)
    # --- regla congelada -------------------------------------------------
    stable_rows = [r for r in table if r["q"] < 1.0]
    unstable_rows = [r for r in table if r["q"] > 1.0]
    grew_below = [f"q{r['q']}_n{r['mode']}" for r in stable_rows if not r["stable"]]
    no_growth_above = all(r["growth_factor"] <= rules["stable_max_growth_factor"] for r in unstable_rows) if unstable_rows else False
    if missing:
        outcome, reasons = "INDETERMINADO", [f"corridas ausentes: {missing}"]
    elif seed_noise_fail:
        outcome, reasons = "INDETERMINADO", [f"puerta de ruido de siembra violada: {seed_noise_fail}"]
    elif beam_ctrl is None or not beam_ctrl["pass"]:
        outcome, reasons = "INDETERMINADO", [f"puerta de convergencia en haces violada: {beam_ctrl}"]
    elif grew_below or no_growth_above:
        outcome, reasons = "C", [f"crecimiento con q < 1: {grew_below}" if grew_below else "sin crecimiento con q > 1 en todos los modos"]
    elif gate_fail:
        outcome, reasons = "INDETERMINADO", [f"fase lineal no resuelta: {gate_fail}"]
    else:
        rates_ok = all(r["rate_within_tol"] for r in unstable_rows)
        kind_ok = all(v["pass"] for v in k_indep.values())
        if rates_ok and kind_ok:
            outcome, reasons = "A", ["umbral correcto; γ/k dentro de ±25 % de la cinética en todos los q > 1; independencia de k dentro del 25 %"]
        else:
            outcome, reasons = "B", [("tasa fuera de tolerancia" if not rates_ok else "") + (" / " if not rates_ok and not kind_ok else "")
                                     + ("dependencia de k excesiva" if not kind_ok else "")]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome": outcome, "reasons": reasons, "preregistration_sha256": psha, "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "table": table, "k_independence": k_indep, "beam_convergence": beam_ctrl, "missing_runs": missing}
    (OUTDIR / "cj_criterion_test.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    word = {"A": "el instrumento reproduce el criterio con su predicción cinética convergida",
            "B": "umbral correcto pero la tasa medida no es la cinética (o depende de k): se publica",
            "C": "crecimiento bajo el umbral o ausencia de crecimiento sobre él: sospecha de error primero",
            "INDETERMINADO": "fase lineal no resuelta o puerta violada: el resultado se retiene"}[outcome]
    md = [f"# Test del criterio de Cronos–Jeans: desenlace **{outcome}**\n",
          f"Preinscripción `{psha[:12]}`; commit `{sha[:9]}`. **Lectura obligatoria**: {word}. Motivo: {'; '.join(reasons)}.", "",
          "| q | modo n | k | γ/k medido | γ/k cinético (instrumento, W(k)) | γ/k cinético (continuo) | γ/k fluido | más cerca de | r² | puntos | factor de crecimiento | veredicto de fila |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        verdict = (("estable" if r["stable"] else "CRECE") if r["q"] < 1.0 else
                   ("no resuelto" if not r["linear_resolved"] else ("dentro de tol." if r["rate_within_tol"] else "fuera de tol.")))
        g = r["gamma_over_k_measured"]
        g_txt = "—" if not np.isfinite(g) else "%.4f" % g
        r2_txt = "—" if not np.isfinite(r["r2"]) else "%.3f" % r["r2"]
        closer = r.get("closer_to") or "—"
        md.append(f"| {r['q']} | {r['mode']} | {r['k']:.1f} | {g_txt} | {r['gamma_over_k_instrument']:.4f} (W = {r['W_k']:.3f}) | "
                  f"{r['gamma_over_k_kinetic']:.4f} | {r['gamma_over_k_fluid']:.4f} | {closer} | {r2_txt} | "
                  f"{r['n_points']} | ×{r['growth_factor']:.1f} | {verdict} |")
    md.append("")
    for q, v in k_indep.items():
        spread = "—" if v["rel_spread"] is None else "%.3f" % v["rel_spread"]
        md.append(f"Independencia de k en q = {q}: dispersión relativa {spread} → {'pasa' if v['pass'] else 'FALLA'}")
    if beam_ctrl:
        md.append(f"Convergencia en haces (q = {bc['q']}, n = {bc['mode']}): γ/k = {beam_ctrl['gamma_over_k_1024']:.4f} (1024) frente a "
                  f"{beam_ctrl['gamma_over_k_alt']:.4f} ({beam_ctrl['n_beams_alt']}): diferencia relativa "
                  f"{'—' if beam_ctrl['rel_diff'] is None else '%.3f' % beam_ctrl['rel_diff']} → {'pasa' if beam_ctrl['pass'] else 'FALLA'}")
    md += ["", "Estatuto: test numérico interno (E8) bajo preinscripción; predicción de la teoría cinética lineal sin parámetros ajustados; "
           "sin datos observacionales; A_Sculptor y el criterio (fila criterio-cronos-jeans) no se tocan."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Desenlace {outcome}: {'; '.join(reasons)} → {OUTDIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("analyze")
    args = ap.parse_args()
    return cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
