#!/usr/bin/env python
"""Cribado preinscrito de candidatos a β en el reloj S (frente 2) — PR-6 de la orden del 28-sep.

    python scripts/run_beta_screening.py prereg    # congela (en un commit posterior al generador)
    python scripts/run_beta_screening.py run       # corridas reanudables
    python scripts/run_beta_screening.py analyze   # regla congelada; falla cerrado sin ella

Cualquier candidato a segundo nivel del Camino que entre en el flujo de acoplos se somete, ANTES de
discutir λ, a tres criterios (core/beta_candidates.py): (i) reduce D = B² − 4C₀M₀² de forma monótona sin
llevar M₀² a cero antes del primer cruce; (ii) produce tres cruces D = 0; (iii) se publican los S de los
cruces frente a 0.009/0.099/0.999 y los cocientes entre cruces sucesivos (la Década sería 10).

Primer candidato: la contribución del término κΣ̇ê_E (brazo (ii) del vacío 2D) a las β vía la Def. 4.4,
bajo dos cierres de proyección declarados ('plane' ⟹ Δβ ≡ 0, control negativo; 'quadrant'). El cribado
aísla la contribución a las β: la trayectoria corre con el Flujo del Camino canónico (κ_conv = 0) y la Δβ
usa el κ̂ declarado; un brazo aparte ('trajectory_current') lleva además la corriente en la trayectoria y
publica lo que ocurre (el piloto vio divergencia numérica: con B < 0 desaparece el vacío verdadero y Σ̇ crece
sin cota — límite del instrumento, no física).

Desenlaces (congelados, SIN letra sobre λ): «pasa» — alguna celda con κ̂ en la ventana plausible declarada
cumple (i) y (ii) con el cierre 'quadrant'; «no pasa» — ninguna; INDETERMINADO — alguna corrida de la ventana
no terminó (divergencia, max_steps) o falta. La fila decada-discriminante no cambia salvo que un candidato
pase (i)–(ii).

Piloto declarado (28-sep, antes de congelar): δ₀ = 0.01, τ ∈ {0.1, 1}, κ̂ ∈ {0.1, 1, 10, 100, 1000, 3000}.
Fijó solo el instrumento: (a) la regla de parada en la frontera del modo emergente (la velocidad proyectada
≈ 0 con S parado ya no agota max_steps: antes 400 000 pasos y ~70 s por corrida, ahora ~9 000 pasos); (b) la
separación entre la Δβ (κ̂ propio) y la corriente en la trayectoria (κ_conv), porque con las dos juntas el
reloj diverge para κ̂ ≥ 1; (c) la malla de κ̂ hasta 1e4. Lo que mostró — Δβ_C₀ ≈ 6.8e-4·κ̂ frente a
β_C₀ = −3.3e-3 y Δβ_{M₀²} ≈ 3e-8·κ̂ frente a β_{M₀²} = −0.12 en el punto inicial (δ₀ = 0.01); C₀ final 1.004
/ 1.041 / 1.136 para κ̂ = 100 / 1000 / 3000 sin ningún cruce — se declara como expectativa E13 y no mueve
ningún criterio ni la ventana.
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

from core.beta_candidates import delta_beta, screening_report  # noqa: E402
from core.fokker_planck_beta import FPClosure, beta_functions  # noqa: E402
from core.s_clock import DECLARED_FORMS, ClockConfig, SClock  # noqa: E402
from mcmc_ontology import constants as C  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-28_beta_screening"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"

CANDIDATE = "conversion_current"
DELTA0_GRID = [0.01, 0.03]
TAU_GRID = [0.01, 0.1, 1.0]
KAPPA_HAT_GRID = [float(x) for x in np.round(np.geomspace(0.1, 1e4, 11), 4)]
RESET_ARMS = [False, True]
TRAJECTORY_CURRENT_ARM = {"delta0": 0.01, "tau": 0.1, "kappa_hat": [0.5, 1.0, 2.11]}
RULES = {"kappa_hat_window": [0.5, 5.0], "closure_for_verdict": "quadrant", "tol_monotone": 0.0,
         "criteria": {"i": "D monótono (sin subidas > tol_monotone·|D₀|) hasta el primer cruce y M₀² > 0 en ese cruce",
                      "ii": "n_cruces ≥ 3 (con o sin escalón del potencial: se publican los dos brazos)",
                      "iii": "S de los cruces frente a 0.009/0.099/0.999 y cocientes sucesivos (Década = 10) — se publica, no se juzga"},
         "outcomes": {"pasa": "alguna celda con κ̂ en kappa_hat_window, cierre 'quadrant', cumple (i) y (ii)",
                      "no pasa": "ninguna celda de la ventana cumple (i) y (ii), con todas las corridas de la ventana terminadas",
                      "INDETERMINADO": "alguna corrida de la ventana no terminó (divergencia o max_steps) o falta"},
         "no_letter_on_lambda": True, "decada_discriminante_row_unchanged_unless_pass": True}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


def prereg(_args) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {"title": "Cribado del candidato «corriente de conversión» a las β del frente 2 en el reloj S (modo emergente)",
           "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_beta_screening.py prereg", "generator_commit_must_precede_freeze": True,
           "question": "¿Puede la contribución del término κΣ̇ê_E a las β (Def. 4.4, cierres declarados) producir tres cruces D = 0 "
                       "sin llevar M₀² a cero antes, para algún κ̂ en la ventana plausible?",
           "form": DECLARED_FORMS["beta_extra"],
           "declared": {"candidate": CANDIDATE, "delta0_grid": DELTA0_GRID, "tau_grid": TAU_GRID, "kappa_hat_grid": KAPPA_HAT_GRID,
                        "closures": ["quadrant (todas las κ̂)", "plane (control: Δβ ≡ 0, una vez por δ₀ y τ)", "canónica (sin candidato, control)"],
                        "reset_arms": RESET_ARMS, "trajectory_kappa_conv": 0.0,
                        "trajectory_current_arm": TRAJECTORY_CURRENT_ARM, "max_steps": 400000, "d_sigma_hat": 5e-4,
                        "kappa_definition": "κ = κ̂·ρ₊/T₀ (la misma del vacío 2D)"},
           "rules": RULES,
           "expectations_E13": {"outcome": "no pasa: en el punto inicial Δβ_C₀ ≈ 6.8e-4·κ̂ frente a β_C₀ = −3.3e-3 y Δβ_{M₀²} ≈ 3e-8·κ̂ frente a "
                                           "β_{M₀²} = −0.12; para que 4C₀M₀² alcance B² antes de M₀² = 0 haría falta κ̂ ~ 1e6, fuera de la malla",
                                "kappa_hat_plausible": "≈ 1 (E13 del vacío 2D); el cruce de la diagonal exigía κ̂ ≥ 2.11",
                                "trajectory_current_arm": "divergencia numérica (límite del instrumento, publicado)"},
           "development_declaration": {"pilot": "δ₀ = 0.01, τ ∈ {0.1, 1}, κ̂ ∈ {0.1, 1, 10, 100, 1000, 3000}: ningún cruce; C₀ final 1.004/1.041/1.136 "
                                                "para κ̂ = 100/1000/3000; con la corriente en la trayectoria (κ_conv = κ̂ ≥ 1) el reloj diverge",
                                       "what_it_fixed": "solo el instrumento: regla de parada en la frontera del modo emergente, separación Δβ/κ_conv, malla de κ̂; "
                                                        "ningún criterio ni la ventana"},
           "what_this_cannot_decide": ["λ (la Década): el cribado dice si el candidato PUEDE ser la β del frente 2, no que lo sea (E13)",
                                       "cuál cierre de proyección es el del tratado: los dos son declarados",
                                       "nada observable: recorrido interno del reloj (E8)"],
           "prohibitions": {"no_threshold_tuning": True, "no_data": True, "decada_discriminante_untouched_unless_pass": True}}
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador), ANTES de ninguna corrida de producción. "
          f"sha256 de `preregistration.json`: `{sha}`.", "", f"**Pregunta**: {doc['question']}", "", f"**Forma declarada**: {doc['form']}", "",
          "## Declarado", ""] + [f"- **{k}**: {v}" for k, v in doc["declared"].items()] + ["", "## Reglas (congeladas)", ""] + \
         [f"- **{k}**: {v}" for k, v in RULES.items()] + ["", "## Expectativas (E13)", ""] + [f"- **{k}**: {v}" for k, v in doc["expectations_E13"].items()] + \
         ["", "## Piloto declarado", "", doc["development_declaration"]["pilot"], "", doc["development_declaration"]["what_it_fixed"], "",
          "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción del cribado de β")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def _one(delta0: float, tau: float, kappa_hat: float | None, closure: str | None, reset: bool, kappa_conv: float, max_steps: int) -> dict:
    kw = {"delta0": delta0, "thresholds": "emergent", "couplings_flow": True, "fp": FPClosure(tau=tau), "max_steps": max_steps,
          "kappa_conv": kappa_conv, "reset_mass_on_collapse": reset}
    if closure is not None:
        kw.update({"beta_extra": CANDIDATE, "beta_extra_closure": closure, "beta_extra_kappa_hat": kappa_hat})
    clk = SClock(ClockConfig(**kw))
    t0 = time.time()
    r = clk.run()
    tr = r["trajectory"]
    crossings = [e["S"] for e in r["events"] if e["kind"].startswith("colapso")]
    rep = screening_report(crossings, tr["D"], tr["S"], tr["M0_sq"], C.decade_thresholds()[:3], RULES["tol_monotone"])
    kappa = (kappa_hat or 0.0) * np.sqrt(clk.x_plus0) / clk.T0
    db0 = delta_beta(CANDIDATE, *clk.lam0, kappa=kappa, x_plus=clk.x_plus0, closure=closure or "plane")
    b0 = beta_functions(*clk.lam0)
    return {"delta0": delta0, "tau": tau, "kappa_hat": kappa_hat, "closure": closure, "reset_mass_on_collapse": reset, "kappa_conv": kappa_conv,
            "finished": r["finished"], "diverged": r["diverged"], "stop_reason": r["stop_reason"], "steps": r["steps"], "wall_s": round(time.time() - t0, 2),
            "S_final": float(tr["S"][-1]), "f_final": float(tr["f"][-1]), "lam_initial": clk.lam0.tolist(),
            "lam_final": [float(tr["M0_sq"][-1]), float(tr["B"][-1]), float(tr["C0"][-1])],
            "D_initial": float(tr["D"][0]), "D_final": float(tr["D"][-1]), "D_min": float(np.min(tr["D"])),
            "delta_beta_initial": db0.tolist(), "beta_canonical_initial": b0.tolist(),
            "delta_over_canonical_initial": [float(d / b) if b != 0 else None for d, b in zip(db0, b0)],
            "n_mass_instabilities": len(r["mass_instability_events"]), "n_mass_resets": (r["beta_extra"] or {}).get("n_mass_resets", 0),
            "screening": rep}


def cmd_run(_args) -> int:
    pre, _ = _load_prereg()
    D = pre["declared"]
    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d0 in D["delta0_grid"]:
        for tau in D["tau_grid"]:
            jobs.append((f"canon_d{d0}_tau{tau}", d0, tau, None, None, False, 0.0))
            jobs.append((f"plane_d{d0}_tau{tau}", d0, tau, 1.0, "plane", False, 0.0))
            for kh in D["kappa_hat_grid"]:
                for reset in D["reset_arms"]:
                    jobs.append((f"quadrant_d{d0}_tau{tau}_kh{kh:g}_reset{int(reset)}", d0, tau, kh, "quadrant", reset, 0.0))
    tc = D["trajectory_current_arm"]
    for kh in tc["kappa_hat"]:
        jobs.append((f"trajcurrent_d{tc['delta0']}_tau{tc['tau']}_kh{kh:g}", tc["delta0"], tc["tau"], kh, "quadrant", True, kh))
    for key, d0, tau, kh, closure, reset, kconv in jobs:
        path = RUNS / f"{key}.json"
        if path.exists():
            continue
        rec = {"key": key, **_one(d0, tau, kh, closure, reset, kconv, D["max_steps"])}
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        s = rec["screening"]
        print(f"  {key}: {rec['stop_reason'][:38] if rec['stop_reason'] else '—':38s} cruces={len(s['S_crossings'])} i={s['criterion_i']} "
              f"ii={s['criterion_ii']} C0_fin={rec['lam_final'][2]:.4g} ({rec['wall_s']} s)", flush=True)
    return 0


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    rules = pre["rules"]
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*.json"))]
    if not runs:
        raise SystemExit("FALLO CERRADO: no hay corridas")
    lo, hi = rules["kappa_hat_window"]
    window = [r for r in runs if r["closure"] == rules["closure_for_verdict"] and r["kappa_conv"] == 0.0 and r["kappa_hat"] is not None and lo <= r["kappa_hat"] <= hi]
    expected_window = len(pre["declared"]["delta0_grid"]) * len(pre["declared"]["tau_grid"]) * len(pre["declared"]["reset_arms"]) * \
        sum(1 for k in pre["declared"]["kappa_hat_grid"] if lo <= k <= hi)
    unfinished = [r["key"] for r in window if not r["finished"] or r["diverged"]]
    passing_window = [r["key"] for r in window if r["screening"]["passes_i_and_ii"]]
    passing_any = [r for r in runs if r["closure"] == "quadrant" and r["kappa_conv"] == 0.0 and r["screening"]["passes_i_and_ii"]]
    if len(window) < expected_window or unfinished:
        outcome = "INDETERMINADO"
    elif passing_window:
        outcome = "pasa"
    else:
        outcome = "no pasa"
    first_crossing_any = [r for r in runs if r["kappa_conv"] == 0.0 and r["screening"]["n_crossings"] > 0]
    doc = {"outcome": outcome, "no_letter_on_lambda": True, "prereg_sha256": psha, "code_commit": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "n_runs": len(runs),
           "window": {"kappa_hat": [lo, hi], "n_cells": len(window), "expected": expected_window, "unfinished": unfinished, "passing": passing_window},
           "passing_anywhere_in_grid": [r["key"] for r in passing_any],
           "kappa_hat_min_passing": min((r["kappa_hat"] for r in passing_any), default=None),
           "runs_with_any_crossing": [{"key": r["key"], **{k: r["screening"][k] for k in ("S_crossings", "S_over_threshold", "ratios_successive", "criterion_i", "criterion_ii")}}
                                      for r in first_crossing_any],
           "delta_over_canonical_initial_by_kappa_hat": {
               f"d{r['delta0']}_kh{r['kappa_hat']:g}": r["delta_over_canonical_initial"]
               for r in runs if r["closure"] == "quadrant" and r["tau"] == pre["declared"]["tau_grid"][0] and not r["reset_mass_on_collapse"] and r["kappa_conv"] == 0.0},
           "controls": {"canonical": [{"key": r["key"], "stop_reason": r["stop_reason"], "n_crossings": r["screening"]["n_crossings"],
                                       "n_mass_instabilities": r["n_mass_instabilities"], "D_final": r["D_final"], "lam_final": r["lam_final"]}
                                      for r in runs if r["closure"] is None],
                        "plane": [{"key": r["key"], "identical_to_canonical_lam_final": None} for r in runs if r["closure"] == "plane"]},
           "trajectory_current_arm": [{"key": r["key"], "kappa_hat": r["kappa_hat"], "finished": r["finished"], "diverged": r["diverged"],
                                       "stop_reason": r["stop_reason"], "S_final": r["S_final"], "f_final": r["f_final"]}
                                      for r in runs if r["kappa_conv"] > 0.0],
           "reading": "cribado diagnóstico (E13): el desenlace dice si el candidato puede ser la β del frente 2 bajo los cierres declarados; "
                      "ninguna letra sobre λ; la fila decada-discriminante no cambia salvo «pasa»"}
    canon = {(r["delta0"], r["tau"]): r["lam_final"] for r in runs if r["closure"] is None}
    for row in doc["controls"]["plane"]:
        r = next(x for x in runs if x["key"] == row["key"])
        row["identical_to_canonical_lam_final"] = bool(np.allclose(r["lam_final"], canon[(r["delta0"], r["tau"])], rtol=0, atol=1e-12))
    (OUTDIR / "analysis.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Cribado del candidato «{CANDIDATE}» — desenlace: **{outcome}** (sin letra sobre λ)", "",
          f"Preinscripción `{psha[:12]}…`; commit `{doc['code_commit'][:9]}`; {len(runs)} corridas.", "",
          f"Ventana κ̂ ∈ [{lo}, {hi}] (cierre {rules['closure_for_verdict']}): {len(window)}/{expected_window} celdas, "
          f"{len(passing_window)} pasan (i)+(ii), {len(unfinished)} sin terminar.", "",
          f"κ̂ mínima que pasa en TODA la malla (hasta {max(pre['declared']['kappa_hat_grid']):g}): {doc['kappa_hat_min_passing']}.", "",
          "## Δβ/β canónica en el punto inicial (M₀², B, C₀), τ = " + str(pre["declared"]["tau_grid"][0]), "",
          "| celda | Δβ_{M₀²}/β | Δβ_B/β | Δβ_{C₀}/β |", "|---|---|---|---|"]
    for k, v in doc["delta_over_canonical_initial_by_kappa_hat"].items():
        md.append(f"| {k} | " + " | ".join("—" if x is None else f"{x:.3g}" for x in v) + " |")
    md += ["", "## Corridas con algún cruce D = 0", ""]
    md += ["- ninguna"] if not first_crossing_any else [f"- {r['key']}: S = {r['S_crossings']}, cocientes {r['ratios_successive']}" for r in doc["runs_with_any_crossing"]]
    md += ["", "## Controles", ""] + [f"- {c['key']}: {c['stop_reason']}; cruces {c['n_crossings']}; inestabilidades de masa {c['n_mass_instabilities']}; D_fin {c['D_final']:.3g}" for c in doc["controls"]["canonical"]]
    md += [f"- {c['key']}: idéntica a la canónica (Δβ ≡ 0): {c['identical_to_canonical_lam_final']}" for c in doc["controls"]["plane"]]
    md += ["", "## Brazo con la corriente en la trayectoria", ""] + [f"- {t['key']}: terminó {t['finished']}, divergió {t['diverged']}; {t['stop_reason']}; f_fin = {t['f_final']:.3g}" for t in doc["trajectory_current_arm"]]
    md += ["", f"**Lectura**: {doc['reading']}."]
    (OUTDIR / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"desenlace: {outcome}; ventana {len(passing_window)}/{len(window)} pasan; κ̂_min que pasa: {doc['kappa_hat_min_passing']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "run", "analyze"):
        sub.add_parser(c)
    args = ap.parse_args()
    return {"prereg": prereg, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
