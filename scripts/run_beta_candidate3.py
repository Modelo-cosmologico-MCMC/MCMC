#!/usr/bin/env python
"""Cribado del candidato 3 a las β del frente 2, «sentido del flujo» (bloque P1 del autor del 23-oct): las β canónicas del
reloj recorridas hacia el UV (τ < 0; FPClosure(flow_sign=−1) en el modo emergente), brazo nuevo del cribado de PR-6 con
sus criterios congelados (i)–(iii), en fichero propio (la preinscripción del frente 2 no se toca).

    python scripts/run_beta_candidate3.py prereg    # congela las letras (commit posterior al generador), antes de correr
    python scripts/run_beta_candidate3.py run       # flujo de acoplos (letra), controles IR y reloj completo (sin letra)
    python scripts/run_beta_candidate3.py analyze   # letra por δ₀ y sub-brazo, global la peor; falla cerrado sin preinscripción

Letra del autor sobre los dos primeros cocientes S_{k+1}/S_k (no dependen de τ), evaluada sobre el flujo de acoplos
(autónomo en S) hasta tres cruces o S·|τ| = 50: A, los dos en [7, 14]; B, tres cruces con algún cociente fuera de [7, 14]
pero dentro de [3, 30]; C, menos de tres cruces o algún cociente fuera de [3, 30]; INDETERMINADO, puertas numéricas o
truncamiento > 10 % en algún cociente. Letra por δ₀; la global es la peor. Ninguna letra decide λ ni el sentido del flujo.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.basal import B_BAR, C0_DEFAULT, M_BAR  # noqa: E402
from core.beta_candidate3 import (  # noqa: E402
    cell_letter,
    flow_crossings,
    ratios,
    worst,
)
from core.fokker_planck_beta import FPClosure  # noqa: E402
from core.s_clock import ClockConfig, SClock  # noqa: E402
from mcmc_ontology import constants as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-23_beta_candidate3"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"

DELTA0_GRID = [0.003, 0.01, 0.0554]
RESET_ARMS = [False, True]
TAU_CLOCK = [0.01, 0.1, 1.0]
RULES = {
    "evaluated_on": "flujo de acoplos dλ/ds = flow_sign·β(λ) con s = |τ|·S (autónomo en S; los cocientes no dependen de τ), "
                    "desde el escalado (3.2) de las formas por defecto, hasta tres cruces D = 0 o s = s_max",
    "s_max": 50.0, "n_crossings_max": 3,
    "A_interval": [7.0, 14.0], "C_outside": [3.0, 30.0], "truncation_max_rel": 0.10,
    "letters": {"A": "los dos primeros cocientes S_{k+1}/S_k en [7, 14]",
                "B": "tres cruces con algún cociente fuera de [7, 14] pero dentro de [3, 30]",
                "C": "menos de tres cruces o algún cociente fuera de [3, 30]",
                "INDETERMINADO": "puertas numéricas o truncamiento > 10 % en algún cociente"},
    "per_cell": "letra por celda (δ₀, sub-brazo); la global de cada sub-brazo es la peor sobre δ₀ con el orden A < B < "
                "INDETERMINADO < C (precisión del ejecutor: los sub-brazos son hipótesis distintas y se publican con el mismo "
                "peso; la peor sobre todas las celdas se publica también, como lectura)",
    "numerical_gates": ["el integrador termina en los tres flujos de la celda (cúbico, cúbico con tolerancia estricta, cuártico)",
                        "convergencia: mismo número de cruces, mismo final y cada cruce a ≤ convergence_rel entre rtol y rtol_tight"],
    "truncation_gate": "con tres cruces cúbicos, el cuártico (beta_functions_quartic, E₄(0) = 0) debe dar tres cruces y mover cada "
                       "uno de los dos cocientes ≤ 10 %; con menos de tres cruces cúbicos, el cuártico no debe dar tres "
                       "(si los da, la C depende del truncamiento): si no, INDETERMINADO",
    "ends": "un flujo termina en tres cruces, en s = s_max, en la divergencia de un acoplo (|λ_i| > blowup·escala_i: final "
            "resuelto del séxtico truncado, no puerta) o con C₀ ≤ 0 (fuera del dominio)",
    "convergence_rel": 1e-6, "rtol": 1e-10, "rtol_tight": 1e-12, "blowup": 1e8, "method": "LSODA",
    "order": "puertas → C → B → A",
    "no_letter_on_lambda": True, "front2_prereg_untouched": True,
}
E13 = ("C en los dos sub-brazos, por razonamiento y sin cálculo de cruces. Hacia el UV, dM₀²/ds = 8aB + 2bM₀⁴ > 0: M₀² crece sin "
       "cota. Sin escalón, D no vuelve a cruzar tras el primer cruce: menos de tres cruces. Con escalón, M₀² vuelve a su valor "
       "inicial en cada cruce mientras B y C₀ siguen creciendo, y nada en las β hace crecer los intervalos entre cruces en "
       "progresión geométrica: lo esperable son cocientes de orden 1–2, fuera de [3, 30]. Una A sería sorprendente y no "
       "decidiría λ ni el sentido del flujo (decisión 1 del autor); el S del primer cruce escala con 1/|τ| y no prueba nada")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _lam0(delta0: float) -> tuple:
    return (M_BAR ** 2 * delta0 ** 2, B_BAR * delta0, C0_DEFAULT)


def prereg(_args) -> int:
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)}); no se sobrescribe")
    if RUNS.exists() and any(RUNS.iterdir()):
        raise SystemExit("FALLO CERRADO: hay corridas; la preinscripción se congela antes de correr")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {"title": "Cribado del candidato 3 a las β del frente 2: el sentido del flujo (β canónicas del reloj hacia el UV, τ < 0)",
           "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_beta_candidate3.py prereg", "generator_commit_must_precede_freeze": True,
           "ordered_by": "bloque P1 del autor del 23-oct-2026, «Gatillo, candidato 3 "
                         "«sentido del flujo»»: corre aunque el autor no haya decidido el sentido; su resultado es parte de lo que "
                         "necesita para decidir",
           "question": "¿Producen las β canónicas del reloj, recorridas hacia el UV, tres cruces D = 0 con cocientes de Década?",
           "declared": {"flow_sign": -1, "closure": "Polchinski en dimensión cero, a = b = 1/2 (core/fokker_planck_beta.py)",
                        "delta0_grid": DELTA0_GRID, "reset_arms": RESET_ARMS,
                        "initial_couplings": "escalado (3.2): (M₀², B, C₀) = (m̄²δ₀², b̄δ₀, C₀) con m̄ = 1, b̄ = 3, C₀ = 1 (R₀ = 9/4)",
                        "truncation_cost": "variante cuártica beta_functions_quartic con E₄ como cuarto acoplo (E₄(0) = 0); D sigue "
                                           "siendo B² − 4C₀M₀²",
                        "clock_complement": {"tau_grid": TAU_CLOCK, "mode": "emergent, couplings_flow, FPClosure(flow_sign=−1, τ)",
                                             "reading": "sin letra: el reloj se detiene en S = 1.001 (S·|τ| ≤ 1.001·τ)"},
                        "controls": "el mismo flujo hacia el IR (flow_sign = +1), sin letra (resultado publicado: sin cruces)",
                        "criteria_pr6": "(i) D monótono hasta el primer cruce con M₀² > 0 en él, y su forma integral (para "
                                        "preinscripciones desde el 18-oct); (ii) n_cruces ≥ 3; (iii) S de los cruces frente a "
                                        "0.009/0.099/0.999 y cocientes sucesivos — se publican junto a la letra"},
           "rules": RULES,
           "author_exploration_without_letter": "con τ < 0 se cumple el criterio (i) en los cuatro δ₀ probados (los flujos llegan a "
                                                "T₀ = 0 y a D = 0); (ii) y (iii) no se han calculado, para no ver el resultado antes "
                                                "de preinscribirlo",
           "pre_freeze_checks": "el integrador se comprobó con flujos de juguete con cruces analíticos y con el control hacia el IR "
                                "(sin cruces, resultado ya publicado); ningún cruce del flujo hacia el UV se calculó antes de congelar",
           "expectation_E13": E13,
           "what_this_cannot_decide": ["λ (la Década) ni el sentido del flujo en S: decisión 1 del autor",
                                       "la ecuación de Polchinski completa hacia el UV (difusión hacia atrás): vale para el séxtico "
                                       "truncado; el cuártico mide solo el primer escalón del truncamiento",
                                       "nada observable: recorrido interno del reloj (E8)"],
           "prohibitions": {"no_threshold_tuning": True, "front2_prereg_untouched": True, "no_runs_before_freeze": True}}
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador), ANTES de ninguna corrida. "
          f"sha256 de `preregistration.json`: `{sha}`.", "", f"**Pregunta**: {doc['question']}", "", "## Declarado", ""] + \
         [f"- **{k}**: {v}" for k, v in doc["declared"].items()] + ["", "## Reglas (congeladas)", ""] + \
         [f"- **{k}**: {v}" for k, v in RULES.items()] + ["", "## Exploración del autor, sin letra", "",
                                                          doc["author_exploration_without_letter"] + ".", "",
                                                          "## Comprobaciones antes de congelar", "", doc["pre_freeze_checks"] + ".", "",
                                                          "## Expectativa (E13)", "", E13 + ".", "", "## Lo que no puede decidir", ""] + \
         [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción del candidato 3")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def _flows(delta0: float, reset: bool, flow_sign: int, R: dict) -> dict:
    kw = {"flow_sign": flow_sign, "reset_mass": reset, "s_max": R["s_max"], "n_max": R["n_crossings_max"],
          "blowup": R["blowup"], "method": R["method"]}
    return {"cubic": flow_crossings(_lam0(delta0), rtol=R["rtol"], **kw),
            "cubic_tight": flow_crossings(_lam0(delta0), rtol=R["rtol_tight"], **kw),
            "quartic": flow_crossings(_lam0(delta0), quartic=True, rtol=R["rtol"], **kw)}


def _clock(delta0: float, tau: float, reset: bool) -> dict:
    cfg = ClockConfig(delta0=delta0, thresholds="emergent", couplings_flow=True, fp=FPClosure(flow_sign=-1, tau=tau),
                      reset_mass_on_collapse=reset)
    t0 = time.time()
    r = SClock(cfg).run()
    tr = r["trajectory"]
    cross = [e["S"] for e in r["events"] if e["kind"].startswith("colapso")]
    return {"delta0": delta0, "tau": tau, "reset": reset, "finished": r["finished"], "diverged": r["diverged"],
            "stop_reason": r["stop_reason"], "steps": r["steps"], "wall_s": round(time.time() - t0, 2), "S_crossings": cross,
            "S_times_tau_crossings": [s * tau for s in cross], "ratios": ratios(cross),
            "n_mass_instabilities": len(r["mass_instability_events"]), "S_final": float(tr["S"][-1]),
            "lam_final": [float(tr["M0_sq"][-1]), float(tr["B"][-1]), float(tr["C0"][-1])]}


def cmd_run(_args) -> int:
    pre, _ = _load_prereg()
    R, D = pre["rules"], pre["declared"]
    RUNS.mkdir(parents=True, exist_ok=True)
    for d0 in D["delta0_grid"]:
        for reset in D["reset_arms"]:
            for name, sign in (("uv", -1), ("ir_control", +1)):
                path = RUNS / f"flow_{name}_d{d0}_reset{int(reset)}.json"
                if not path.exists():
                    rec = {"delta0": d0, "reset": reset, "flow_sign": sign, **_flows(d0, reset, sign, R)}
                    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(f"  {path.name}: cruces {rec['cubic']['crossings']} ({rec['cubic']['stop']})", flush=True)
            for tau in D["clock_complement"]["tau_grid"]:
                path = RUNS / f"clock_d{d0}_tau{tau}_reset{int(reset)}.json"
                if not path.exists():
                    rec = _clock(d0, tau, reset)
                    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(f"  {path.name}: {rec['stop_reason']}; cruces S = {rec['S_crossings']} ({rec['wall_s']} s)", flush=True)
    return 0


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    R, D = pre["rules"], pre["declared"]
    cells, controls, clocks = [], [], []
    for d0 in D["delta0_grid"]:
        for reset in D["reset_arms"]:
            path = RUNS / f"flow_uv_d{d0}_reset{int(reset)}.json"
            if not path.exists():
                cells.append({"delta0": d0, "reset": reset, "letter": "INDETERMINADO", "reason": "falta la corrida"})
                continue
            rec = json.loads(path.read_text(encoding="utf-8"))
            L = cell_letter(rec["cubic"], rec["cubic_tight"], rec["quartic"], R)
            c = rec["cubic"]["crossings"]
            th = C.decade_thresholds()[:3]
            tau_star = c[0] / th[0] if c else None
            cells.append({"delta0": d0, "reset": reset, **L, "s_crossings": c, "stop": rec["cubic"]["stop"],
                          "s_crossings_quartic": rec["quartic"]["crossings"], "criterion_i": rec["cubic"].get("criterion_i"),
                          "criterion_i_integral": rec["cubic"].get("criterion_i_integral"),
                          "criterion_ii": len(c) >= 3,
                          "criterion_iii": {"tau_placing_S1_at_0.009": tau_star,
                                            "S_at_that_tau": [s / tau_star for s in c] if tau_star else [],
                                            "decade_thresholds": th, "ratios_successive": ratios(c)}})
            ctl = json.loads((RUNS / f"flow_ir_control_d{d0}_reset{int(reset)}.json").read_text(encoding="utf-8"))
            controls.append({"delta0": d0, "reset": reset, "crossings": ctl["cubic"]["crossings"], "stop": ctl["cubic"]["stop"]})
            for tau in D["clock_complement"]["tau_grid"]:
                p = RUNS / f"clock_d{d0}_tau{tau}_reset{int(reset)}.json"
                if p.exists():
                    clocks.append(json.loads(p.read_text(encoding="utf-8")))
    per_arm = {f"reset_{int(r)}": worst([c["letter"] for c in cells if c["reset"] == r]) for r in D["reset_arms"]}
    doc = {"prereg_sha256": psha, "code_commit": _git_head(), "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "letters_by_sub_arm": per_arm, "worst_over_all_cells_reading": worst([c["letter"] for c in cells]),
           "cells": cells, "ir_controls": controls, "clock_complement_without_letter": clocks,
           "reading": "cribado diagnóstico (E8, E13): ninguna letra decide λ ni el sentido del flujo (decisión 1 del autor)"}
    (OUTDIR / "analysis.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Candidato 3 «sentido del flujo» — sin escalón: **{per_arm['reset_0']}**; con escalón: **{per_arm['reset_1']}**", "",
          f"Preinscripción `{psha[:12]}…`; commit `{doc['code_commit'][:9]}`. Letra por δ₀; la global de cada sub-brazo es la peor.", "",
          "| δ₀ | escalón | cruces en s = S·abs(τ) | cocientes | cuártico | letra | motivo |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        md.append(f"| {c['delta0']:g} | {'sí' if c['reset'] else 'no'} | {', '.join(f'{x:.4g}' for x in c.get('s_crossings', []))} | "
                  f"{', '.join(f'{x:.3g}' for x in c.get('ratios', []))} | {', '.join(f'{x:.3g}' for x in c.get('ratios_quartic', []))} | "
                  f"{c['letter']} | {c['reason']} |")
    md += ["", "## Controles hacia el IR (sin letra)", ""] + [f"- δ₀ = {c['delta0']:g}, escalón {'sí' if c['reset'] else 'no'}: "
                                                            f"cruces {c['crossings']} ({c['stop']})" for c in controls]
    md += ["", "## Reloj completo (complemento, sin letra)", ""] + [
        f"- δ₀ = {k['delta0']:g}, τ = {k['tau']:g}, escalón {'sí' if k['reset'] else 'no'}: {k['stop_reason']}; cruces S = "
        f"{[round(x, 6) for x in k['S_crossings']]}; inestabilidades de masa {k['n_mass_instabilities']}" for k in clocks]
    md += ["", f"**Lectura**: {doc['reading']}."]
    (OUTDIR / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"sin escalón: {per_arm['reset_0']}; con escalón: {per_arm['reset_1']}")
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
