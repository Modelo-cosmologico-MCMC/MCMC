#!/usr/bin/env python
"""El vacío 2D fuera del polo de masa: dos cierres compatibles con el
tratado como brazos preinscritos, con J∇C de control (orden del autor
del 22-sep, §5).

    python scripts/run_vacuum_2d.py prereg    # congela (en un commit posterior al generador)
    python scripts/run_vacuum_2d.py run       # corridas reanudables
    python scripts/run_vacuum_2d.py analyze   # regla congelada; falla cerrado sin ella

§3.25 demostró que con C = V el término J∇C se anula en los puntos
críticos y el vacío verdadero 2D queda sobre el polo de masa: θ sube,
cruza la diagonal solo si |J| ≥ |J|_min y VUELVE a θ = 0 — el cruce tiene
un S máximo < 1 y subir |J| lo adelanta. Hace falta un término que actúe
DURANTE la descarga y no en el vacío. Los dos cierres (§5):
  (ii) corriente de conversión  dΦ/dσ = −G⁻¹∇V + κ·Σ̇·ê_E,  κ = κ̂·ρ₊/T₀
       (la conversión Mp → Ep es creación de espacio; Σ̇ ≥ 0 por el Teo. 4.5);
  (i)  rotación de la inclinación  η_eff = η₀·(1 − f/f_×), con un empujón
       declarado |J| = 0.1 (C = V) para que el gradiente rote θ una vez que
       el polo de espacio es el mínimo.
Letras (por brazo y global, congeladas): A — cruce de la diagonal en
S ∈ [0.95, 1.05] para un intervalo de κ̂ (o f_×) de anchura ≥ 20 %
(máx/mín ≥ 1.2) con θ monótona (caída máxima ≤ tol_theta) y, en (ii),
Monotonía 4.5; B — la ventana se alcanza pero solo con ajuste fino
(intervalo no vacío de anchura < 20 %); C — ninguna corrida cruza en la
ventana con θ monótona (o, en (ii), sin romper la Monotonía);
INDETERMINADO — puertas (identidad S = f − f₀ + ΣW/T₀, descenso
terminado, corridas ausentes). Global = la mejor letra de los dos brazos
(A > B > C); ambas se publican. Control: J∇C con |J| = |J|_min de §3.25.
El panel Mp/Ep del visor (cos²θ, sin²θ) pasa de dibujado a derivado.

Piloto de instrumento (22-sep, declarado): δ₀ = 0.01, κ̂ ∈ {0.1 … 3} y
f_× ∈ {0.5, 0.8, 0.9, 0.99} con |J| = 0.1. Fijó solo el instrumento: la
malla de κ̂ y de f_×, el empujón, la regla de parada del brazo (i)
(∇V_eff ≈ 0, porque f de referencia no alcanza 1 − ε_res con η rotada) y
la tolerancia de identidad (5.6e-7 medido en el peor caso). Lo que mostró
(θ no es monótona en (ii): sube y vuelve; (i) cruza en S ≈ 0.88 para
f_× ≤ 0.8 y no cruza para f_× ≥ 0.9) se declara y NO movió ninguna letra
ni tolerancia: la estructura A/B/C es la del texto del autor.
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

from core.s_clock import DECLARED_FORMS, diagonal_crossing_S  # noqa: E402
from mass_program.B7_empalme import delta0_required_full  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-22_vacuum_2d"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"

DELTA0_GRID = [0.003, 0.01, 0.03, "delta_H_full"]
KAPPA_GRID = [float(x) for x in np.round(np.geomspace(0.05, 20.0, 25), 5)]
FCROSS_GRID = [float(x) for x in np.round(np.linspace(0.30, 0.99, 24), 4)]
J_PUSH = 0.1
J_CONTROL = {"0.003": 1.363, "0.01": 1.370, "0.03": 1.389}      # |J|_min de §3.25 (control; δ_H_full sin valor congelado)
RULES = {"S_window": [0.95, 1.05], "width_ratio_A": 1.2, "tol_theta_drop_rad": 0.01, "tol_identity": 1e-6,
         "monotonia_required_arm_ii": True, "monotonia_required_arm_i": False,
         "letters": {"A": "cruce en la ventana para un intervalo de κ̂ (o f_×) con máx/mín ≥ width_ratio_A, θ monótona (caída ≤ tol_theta_drop) y, en (ii), Monotonía 4.5",
                     "B": "intervalo no vacío pero máx/mín < width_ratio_A (ajuste fino)",
                     "C": "ninguna corrida cruza en la ventana con θ monótona (y Monotonía en (ii))",
                     "INDETERMINADO": "identidad > tol_identity en alguna corrida, descenso no terminado, o corridas ausentes"},
         "global": "la mejor letra de los dos brazos (A > B > C); INDETERMINADO si cualquiera de los dos lo es"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()


def _delta0s() -> dict:
    return {str(d): (delta0_required_full() if d == "delta_H_full" else float(d)) for d in DELTA0_GRID}


def prereg(_args) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "title": "El vacío 2D fuera del polo de masa: corriente de conversión κΣ̇ê_E y rotación de la inclinación η_eff = η₀(1 − f/f_×) como brazos, J∇C de control",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_vacuum_2d.py prereg", "generator_commit_must_precede_freeze": True,
        "question": "¿Sitúa alguno de los dos cierres compatibles con el tratado el cruce de la diagonal θ = π/4 en S = 1 ± 0.05 con θ monótona, para un intervalo de su único número declarado (κ̂ o f_×) de anchura ≥ 20 %?",
        "form": DECLARED_FORMS["vacuum_2d"],
        "declared": {"delta0_grid": _delta0s(), "kappa_hat_grid": KAPPA_GRID, "f_cross_grid": FCROSS_GRID, "J_push_arm_i": J_PUSH,
                     "circulation_C": "V", "control_J_min_from_3_25": J_CONTROL, "G": 1.0, "theta_nuc": 0.0,
                     "kappa_definition": "κ = κ̂·ρ₊/T₀ (κ̂ adimensional); espacio acumulado ∫κΣ̇dσ = κ̂·ρ₊·f",
                     "stop_rule_arm_i": "descenso terminado cuando f ≥ 1 − ε_res o |∇V_eff| < 1e-6·T₀/ρ₊",
                     "identity": "S = f − f₀ + (W_J + W_conv + W_tilt)/T₀ con W_tilt = ∫(∇V_ref − ∇V_eff)·dΦ"},
        "rules": RULES,
        "expectations_E13": {"kappa_hat": "≈ 1 (κ ≈ ρ₊/T₀ en orden de magnitud, §5 del texto del autor); geométricamente ≈ 0.7 para que κ̂ρ₊f ≈ ρ₊/√2",
                             "arm_ii": "A (θ monótona por construcción, según el texto del autor); el piloto vio que θ vuelve al polo",
                             "arm_i": "B o C (el cruce depende de un empujón externo y de f_×)"},
        "development_declaration": {"pilot": "δ₀ = 0.01: κ̂ ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3} → cruce solo para κ̂ ≥ 2 (S_cross 0.81, 0.66), θ sube y "
                                             "vuelve al polo (caída ≈ θ_max), Monotonía conservada, identidad ≤ 3e-10; f_× ∈ {0.5, 0.8, 0.9, 0.99} con "
                                             "|J| = 0.1 → cruce en S = 0.880 para 0.5 y 0.8 (θ → π/2 y 1.16), sin cruce para 0.9 y 0.99, identidad "
                                             "≤ 5.6e-7, descenso sin terminar en f (⟹ regla de parada ∇V_eff ≈ 0).",
                                    "what_it_fixed": "solo el instrumento: mallas, empujón, regla de parada, tolerancia de identidad; ninguna letra ni ventana",
                                    "budget": "4 δ₀ × (25 κ̂ + 24 f_× + 1 control) ≈ 200 corridas del reloj, 0.2–20 s cada una"},
        "what_this_cannot_decide": ["κ̂ o f_× son calibraciones con significado, no derivaciones (frente 2)",
                                    "cuál de los dos cierres es el del tratado: la letra dice cuál sitúa el cruce, no cuál es verdadero",
                                    "nada observable: recorrido interno (E8)"],
        "prohibitions": {"no_threshold_tuning": True, "no_data": True, "J_circulation_results_untouched": True},
    }
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
        raise SystemExit("FALLO CERRADO: falta la preinscripción del vacío 2D")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def cmd_run(_args) -> int:
    pre, psha = _load_prereg()
    RUNS.mkdir(parents=True, exist_ok=True)
    D = pre["declared"]
    for key, d0 in D["delta0_grid"].items():
        path = RUNS / f"delta0_{key}.json"
        if path.exists():
            print(f"  {path.name} ya existe: se omite", flush=True)
            continue
        t0 = time.time()
        rows_ii = [dict(kappa_hat=k, **diagonal_crossing_S(d0, 0.0, kappa_conv=k)) for k in D["kappa_hat_grid"]]
        rows_i = [dict(f_cross=fx, **diagonal_crossing_S(d0, D["J_push_arm_i"], tilt_cross_f=fx)) for fx in D["f_cross_grid"]]
        ctrl = None
        if key in D["control_J_min_from_3_25"]:
            ctrl = diagonal_crossing_S(d0, D["control_J_min_from_3_25"][key])
        out = {"preregistration_sha256": psha, "delta0_key": key, "delta0": d0, "arm_ii_conversion": rows_ii, "arm_i_tilt": rows_i,
               "control_J": ctrl, "wall_s": round(time.time() - t0, 1), "code_commit": _git_head()}
        path.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
        n_ii = sum(r["crossed"] for r in rows_ii)
        n_i = sum(r["crossed"] for r in rows_i)
        print(f"  δ₀ = {d0:.5f} ({key}): {out['wall_s']} s; cruces (ii) {n_ii}/{len(rows_ii)}, (i) {n_i}/{len(rows_i)}", flush=True)
    return 0


def _sc(r: dict) -> str:
    return "—" if r["S_cross"] is None else "%.3f" % r["S_cross"]


def _letter_for_arm(rows: list, x_key: str, R: dict, require_monotonia: bool) -> dict:
    lo, hi = R["S_window"]
    ok = [r for r in rows if r["crossed"] and r["S_cross"] is not None and lo <= r["S_cross"] <= hi
          and r["theta_max_drop"] <= R["tol_theta_drop_rad"] and (r["monotonia_pass"] or not require_monotonia)]
    xs = sorted(r[x_key] for r in ok)
    gates_bad = [r[x_key] for r in rows if r["S_equals_f_max_diff"] > R["tol_identity"] or not r["finished"]]
    if gates_bad:
        return {"letter": "INDETERMINADO", "in_window": xs, "gate_failures": gates_bad}
    if not xs:
        return {"letter": "C", "in_window": [], "n_crossed": sum(r["crossed"] for r in rows),
                "S_cross_range": [min((r["S_cross"] for r in rows if r["crossed"]), default=None), max((r["S_cross"] for r in rows if r["crossed"]), default=None)],
                "theta_monotone_any": bool(any(r["theta_max_drop"] <= R["tol_theta_drop_rad"] and r["crossed"] for r in rows))}
    ratio = max(xs) / min(xs) if min(xs) > 0 else float("inf")
    return {"letter": "A" if ratio >= R["width_ratio_A"] else "B", "in_window": xs, "width_ratio": ratio}


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    R, D = pre["rules"], pre["declared"]
    per = {}
    missing = []
    for key in D["delta0_grid"]:
        p = RUNS / f"delta0_{key}.json"
        if not p.exists():
            missing.append(p.name)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["preregistration_sha256"] != psha:
            raise SystemExit(f"FALLO CERRADO: {p.name} pertenece a otra preinscripción")
        per[key] = {"delta0": d["delta0"],
                    "arm_ii": _letter_for_arm(d["arm_ii_conversion"], "kappa_hat", R, R["monotonia_required_arm_ii"]),
                    "arm_i": _letter_for_arm(d["arm_i_tilt"], "f_cross", R, R["monotonia_required_arm_i"]),
                    "control_J": d["control_J"],
                    "table_ii": [{"kappa_hat": r["kappa_hat"], "crossed": r["crossed"], "S_cross": r["S_cross"], "theta_max": r["theta_max"],
                                  "theta_max_drop": r["theta_max_drop"], "W_conv_over_T0": r["W_conv_over_T0"], "monotonia": r["monotonia_pass"],
                                  "identity": r["S_equals_f_max_diff"], "finished": r["finished"]} for r in d["arm_ii_conversion"]],
                    "table_i": [{"f_cross": r["f_cross"], "crossed": r["crossed"], "S_cross": r["S_cross"], "theta_max": r["theta_max"],
                                 "theta_final": r["theta_final"], "theta_max_drop": r["theta_max_drop"], "W_tilt_over_T0": r["W_tilt_over_T0"],
                                 "monotonia": r["monotonia_pass"], "identity": r["S_equals_f_max_diff"], "finished": r["finished"]} for r in d["arm_i_tilt"]]}
    order = {"A": 0, "B": 1, "C": 2, "INDETERMINADO": 3}
    if missing:
        global_letter = "INDETERMINADO"
    else:
        letters = [per[k][arm]["letter"] for k in per for arm in ("arm_ii", "arm_i")]
        if "INDETERMINADO" in letters:
            global_letter = "INDETERMINADO"
        else:
            # global por brazo = la peor letra sobre δ₀ (la ventana debe alcanzarse en todos los δ₀); global = mejor de los dos brazos
            arm_letter = {arm: max((per[k][arm]["letter"] for k in per), key=lambda L: order[L]) for arm in ("arm_ii", "arm_i")}
            global_letter = min(arm_letter.values(), key=lambda L: order[L])
    sha = _git_head()
    res = {"preregistration_sha256": psha, "code_commit_analysis": sha, "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "verdict": global_letter, "missing_runs": missing, "per_delta0": per,
           "arm_letters_by_delta0": {k: {"arm_ii": per[k]["arm_ii"]["letter"], "arm_i": per[k]["arm_i"]["letter"]} for k in per},
           "reading": ("letra bajo la regla congelada; A diría que uno de los dos cierres sitúa el cruce en S = 1 ± 0.05 con su único número "
                       "declarado y sin ajuste fino; C que ninguno lo hace con θ monótona: entonces el término que mueve el vacío 2D no es "
                       "ninguno de los dos y el perfil θ_imp(S) sigue impuesto (E8; κ̂ y f_× son calibraciones, no derivaciones)")}
    (OUTDIR / "vacuum_2d.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = [f"# El vacío 2D fuera del polo de masa — desenlace **{global_letter}**", "",
          f"Preinscripción `{psha[:12]}`; análisis en `{sha[:9]}`. Corridas ausentes: {missing or 'ninguna'}.", ""]
    for key, pd in per.items():
        md += [f"## δ₀ = {pd['delta0']:.5f} ({key}) — (ii) conversión: **{pd['arm_ii']['letter']}**; (i) rotación: **{pd['arm_i']['letter']}**", "",
               "| κ̂ | cruza | S_cross | θ_max | caída de θ | W_conv/T₀ | Monotonía | identidad | fin |", "|---|---|---|---|---|---|---|---|---|"]
        for r in pd["table_ii"]:
            md.append(f"| {r['kappa_hat']:g} | {'sí' if r['crossed'] else 'no'} | {_sc(r)} | {r['theta_max']:.3f} | "
                      f"{r['theta_max_drop']:.3f} | {r['W_conv_over_T0']:+.3f} | {'sí' if r['monotonia'] else 'no'} | {r['identity']:.1e} | {'sí' if r['finished'] else 'no'} |")
        md += ["", "| f_× | cruza | S_cross | θ_max | θ_final | caída de θ | W_tilt/T₀ | Monotonía | identidad | fin |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in pd["table_i"]:
            md.append(f"| {r['f_cross']:g} | {'sí' if r['crossed'] else 'no'} | {_sc(r)} | {r['theta_max']:.3f} | "
                      f"{r['theta_final']:.3f} | {r['theta_max_drop']:.3f} | {r['W_tilt_over_T0']:+.3f} | {'sí' if r['monotonia'] else 'no'} | {r['identity']:.1e} | {'sí' if r['finished'] else 'no'} |")
        if pd["control_J"]:
            c = pd["control_J"]
            md += ["", f"Control J∇C (|J| = {c['J']}): cruza {c['crossed']}, S_cross {c['S_cross']}, θ_max {c['theta_max']:.3f}, caída de θ {c['theta_max_drop']:.3f}."]
        md.append("")
    md += [res["reading"] + ".", "", "## Lo que no decide", ""] + [f"- {s}" for s in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"desenlace: {global_letter}; por δ₀: {res['arm_letters_by_delta0']}")
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
