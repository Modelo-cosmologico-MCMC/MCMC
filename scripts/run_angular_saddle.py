#!/usr/bin/env python
"""Cuarta ecuación, brazo «término angular»: el collado de salida de S0 y el umbral A_c (revisión profunda del
10-nov-2026, hallazgo 2; PR R2 del plan de acción del 11-nov), SIN letra.

    python scripts/run_angular_saddle.py    # escribe results/2026-11-12_angular_saddle/

Para δ₀ ∈ {0.003, 0.01, 0.0554} publica:

- las formas cerradas de `core.angular_term` con A = 1: t, A_c, el collado interior (ángulo, distancia a la diagonal,
  signatura del hessiano), el vacío (θ*) y el cociente entre la barrera del rayo del polo y la del collado;
- la corrida de CONSISTENCIA del reloj con `theta_nuc` en el ángulo del collado, junto a la del rayo declarado
  (θ = 0): qué queda de la rotación del 23-oct si la salida es por el collado más bajo;
- la salida por el polo con A = 0.8·A_c: sin collado interior, el campo sale del punto de escape del polo y el vacío
  queda a arcsin(t/4A) de la diagonal (el descenso se integra hasta un residuo de 1e-9 para ver dónde acaba).

Nada de esto cambia el comportamiento por defecto del reloj ni las cifras del 23-oct. Qué mitad de la Prop. 3.5 se
conserva es la decisión 2 del autor.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.angular_term import (  # noqa: E402
    critical_A,
    min_diagonal_distance_with_pole_exit,
    saddle_distance_for_vacuum_distance,
    saddle_landscape,
    saddle_ray_run,
    tilt_weight,
)
from core.s_clock import ClockConfig, SClock  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-11-12_angular_saddle"
DELTA0 = (0.003, 0.01, 0.0554)
A_REF = 1.0                     # el A de las cifras del 23-oct
BELOW = 0.8                     # fracción de A_c para la salida por el polo
EPS_RES_CONVERGED = 1e-9        # residuo de descarga de la corrida que se deja llegar al vacío
VACUUM_TARGETS = (0.2, 0.1, 0.05)   # distancias del vacío a la diagonal para la tabla vacío ↔ collado (rad)

# comprobaciones «derivadas» del reloj escritas para el rayo del polo sin término angular: en este brazo no aplican
CHECKS_OF_THE_DEFAULT_RAY = {
    "S0/T0_tilt_first_order": "la corrección de la inclinación a T₀ a primer orden es la del Basal sin término angular",
    "S0/delta0_max_closed_form": "la bisección de δ₀_max busca hasta δ₀ = 2 y junto a la diagonal el máximo es mayor",
    "descent/exit_to_mass_pole_3_5": "exige θ final < 0.35: con el término angular el campo no se queda en el polo",
}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def pole_ray_run(delta0: float, A: float, eps_res: float | None = None) -> dict:
    """El reloj desde el rayo declarado θ = 0 con el término angular; con eps_res pequeño el descenso llega al vacío."""
    kw = {} if eps_res is None else {"eps_res": eps_res, "max_steps": 2_000_000}
    clk = SClock(ClockConfig(delta0=delta0, angular_A=A, **kw))
    out = clk.run()
    theta = np.asarray(out["trajectory"]["theta"])
    chk = out["checks"]["descent"]
    return {"delta0": delta0, "A": A, "eps_res": clk.cfg.eps_res, "theta_start": float(theta[0]), "theta_final": float(theta[-1]),
            "theta_star": float(clk.true_vacuum_2d["theta"]), "distance_to_diagonal_final": float(math.pi / 4.0 - theta[-1]),
            "distance_to_diagonal_vacuum": float(math.pi / 4.0 - clk.true_vacuum_2d["theta"]),
            "theta_max_drop": float(np.max(np.maximum.accumulate(theta) - theta)),
            "crosses_diagonal": bool(theta.max() > math.pi / 4.0), "S_final": float(out["trajectory"]["S"][-1]),
            "S_equals_f_pass": chk["S_equals_f_identity"]["pass"], "monotonia_pass": chk["monotonia_4_5"]["pass"],
            "finished": out["finished"], "stop_reason": out["stop_reason"], "steps": out["steps"],
            "derived_checks_failed": out["ledger"]["derived_checks_failed"]}


def one(delta0: float) -> dict:
    A_c = critical_A(delta0)
    land = saddle_landscape(delta0, A_REF)
    below = saddle_landscape(delta0, BELOW * A_c)
    return {"delta0": delta0, "t": tilt_weight(delta0), "A_c": A_c,
            "A_ref": {"A": A_REF, "landscape": land, "run_from_saddle_ray": saddle_ray_run(delta0, A_REF),
                      "run_from_pole_ray": pole_ray_run(delta0, A_REF)},
            "below_A_c": {"A": BELOW * A_c, "landscape": below,
                          "distance_closed_form": math.asin(tilt_weight(delta0) / (4.0 * BELOW * A_c)),
                          "run_from_pole_ray_converged": pole_ray_run(delta0, BELOW * A_c, eps_res=EPS_RES_CONVERGED)}}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for d0 in DELTA0:
        rows.append(one(d0))
        r = rows[-1]
        ld, sr, pr, bl = r["A_ref"]["landscape"], r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"], r["below_A_c"]
        print(f"δ₀ = {d0:<6} t = {r['t']:.4f}, A_c = {r['A_c']:.4f}; A = 1: collado en θ = {ld['saddle']['theta']:.4f} (a "
              f"{ld['saddle']['distance_to_diagonal']:.4f} de la diagonal), barreras polo/collado = {ld['barriers']['ratio']:.3f}, "
              f"θ* = {ld['vacuum']['theta']:.4f}; desde el collado θ: {sr['theta_start']:.4f} → {sr['theta_final']:.4f}, queda "
              f"{sr['fraction_of_pole_rotation_left']:.1%} de la rotación del polo ({pr['theta_final']:.4f} desde θ = 0); "
              f"A = 0.8·A_c: acaba a {bl['run_from_pole_ray_converged']['distance_to_diagonal_final']:.4f} de la diagonal "
              f"(forma cerrada {bl['distance_closed_form']:.4f})", flush=True)
    d_min = min_diagonal_distance_with_pole_exit()
    seen = sorted({c for r in rows for run in (r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"],
                                               r["below_A_c"]["run_from_pole_ray_converged"]) for c in run["derived_checks_failed"]})
    doc = {"title": "Cuarta ecuación, brazo «término angular»: el collado de salida de S₀ y el umbral A_c, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_angular_saddle.py",
           "form": "V = V₀ + ½μ²χ², μ² = 4A(c̄/κ₊)δ₀² (ClockConfig.angular_A); puntos críticos interiores en R(ρ²) = 0 con χ = η/μ²",
           "closed_forms": {"t": "η·ρ₊/T₀ = ē·√(κ₊δ₀)/c̄", "A_c": "(t/4)·√(2κ₊/κ₋)",
                            "vacuum": "anillo κ₊, a arcsin(t/4A) de la diagonal",
                            "saddle": "anillo κ₋, a arcsin((t/4A)·√(κ₊/κ₋)) de la diagonal; existe si A ≥ A_c",
                            "bound": "con salida por el polo el vacío queda a arcsin√(κ₋/2κ₊) o más de la diagonal",
                            "V_saddle": "V₀_radial(x₋) − η²/(2μ²)"},
           "min_distance_with_pole_exit_rad": d_min, "min_distance_with_pole_exit_deg": math.degrees(d_min),
           "vacuum_vs_saddle": [{"vacuum_distance": d, "saddle_distance": saddle_distance_for_vacuum_distance(d),
                                 "saddle_angle_from_pole": math.pi / 4.0 - saddle_distance_for_vacuum_distance(d)}
                                for d in VACUUM_TARGETS],
           "grid": f"δ₀ ∈ {list(DELTA0)}; A = {A_REF:g} (cifras del 23-oct) y A = {BELOW:g}·A_c; modo impuesto (S ≡ f)",
           "rows": rows,
           "derived_checks_failed_seen": {c: CHECKS_OF_THE_DEFAULT_RAY.get(c, "SIN EXPLICACIÓN DECLARADA") for c in seen},
           "summary": {"interior_saddle_at_A_ref": all(r["A_ref"]["landscape"]["saddle"] is not None for r in rows),
                       "pole_ray_barrier_is_a_saddle_at_A_ref": any(r["A_ref"]["landscape"]["pole_ray_barrier_is_a_saddle"] for r in rows),
                       "pole_exit_below_A_c": all(r["below_A_c"]["landscape"]["exit_route"] == "polo"
                                                  and r["below_A_c"]["landscape"]["pole_ray_barrier_is_a_saddle"] for r in rows),
                       "any_crossing": any(run["crosses_diagonal"] for r in rows for run in (
                           r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"], r["below_A_c"]["run_from_pole_ray_converged"])),
                       "identities_pass_all": all(run["S_equals_f_pass"] and run["monotonia_pass"] and run["finished"] for r in rows for run in (
                           r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"], r["below_A_c"]["run_from_pole_ray_converged"])),
                       "unexplained_failed_checks": [c for c in seen if c not in CHECKS_OF_THE_DEFAULT_RAY]},
           "reading": "sin letra (E8, E13): con ½μ²χ², salir por el polo de masa y llegar junto a la diagonal se excluyen. Con A "
                      "= 1 el collado más bajo ya está cerca de la diagonal y la barrera del rayo del polo es de 1.8 a 2.6 veces "
                      "más alta; con A ≤ A_c la salida es por el polo y el vacío queda a 15.7° o más de la diagonal. Las cifras "
                      "del 23-oct valen como descenso desde el rayo θ = 0, que es la forma declarada del reloj. Qué mitad de la "
                      "Prop. 3.5 se conserva es la decisión 2 del autor"}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. {doc['form']}.", "",
          "## Formas cerradas con A = 1 (las cifras del 23-oct)", "",
          "| δ₀ | t | A_c | ángulo del collado | del collado a la diagonal | autovalores del hessiano en el collado (÷δ₀²) | barrera del polo / barrera del collado | θ* |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ld = r["A_ref"]["landscape"]
        ev = ", ".join(f"{e / r['delta0'] ** 2:+.3f}" for e in ld["hessian_eigenvalues_saddle"])
        md.append(f"| {r['delta0']:g} | {r['t']:.4f} | {r['A_c']:.4f} | {ld['saddle']['theta']:.4f} | "
                  f"{ld['saddle']['distance_to_diagonal']:.4f} | {ev} | {ld['barriers']['ratio']:.3f} | {ld['vacuum']['theta']:.4f} |")
    md += ["", ("Con A = 1 la barrera del rayo del polo no es un collado en ninguna de las tres: allí ∂V/∂φ_E < 0 y el campo baja "
                "hacia el interior del cuadrante." if not doc["summary"]["pole_ray_barrier_is_a_saddle_at_A_ref"] else
                "Con A = 1 la barrera del rayo del polo es un collado en alguna de las tres (∂V/∂φ_E ≥ 0 allí)."), "",
           "## Corrida de consistencia: el reloj desde el rayo del collado frente al rayo declarado (A = 1)", "",
           "| δ₀ | θ inicial (collado) | θ final | θ* | rotación que queda desde el collado | fracción de la del polo | θ final desde θ = 0 | θ monótona | S ≡ f y Monotonía |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sr, pr = r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"]
        md.append(f"| {r['delta0']:g} | {sr['theta_start']:.4f} | {sr['theta_final']:.4f} | {sr['theta_star']:.4f} | "
                  f"{sr['rotation_left_from_saddle']:.4f} | {sr['fraction_of_pole_rotation_left']:.1%} | {pr['theta_final']:.4f} | "
                  f"{'sí' if sr['theta_max_drop'] <= 1e-12 else 'no'} | {'sí' if sr['S_equals_f_pass'] and sr['monotonia_pass'] else 'no'} |")
    left = [r["A_ref"]["run_from_saddle_ray"]["fraction_of_pole_rotation_left"] for r in rows]
    residual = all(run["stop_reason"] == "residuo de descarga alcanzado" for r in rows
                   for run in (r["A_ref"]["run_from_saddle_ray"], r["A_ref"]["run_from_pole_ray"]))
    md += ["", ("Las dos corridas terminan en el residuo de descarga (f = 1 − ΔS), antes de que θ alcance θ*: el último tramo de "
                "la rotación es lento. " if residual else "") +
           f"Desde el collado, la «rotación continua desde el polo» del 23-oct se reduce a su último {min(left):.0%}–{max(left):.0%}.", "",
           f"## Salida por el polo con A = {BELOW:g}·A_c", "",
           "| δ₀ | A | collado interior | la barrera del polo es un collado | del vacío a la diagonal (forma cerrada) | donde acaba el descenso | θ monótona |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        bl = r["below_A_c"]
        run = bl["run_from_pole_ray_converged"]
        md.append(f"| {r['delta0']:g} | {bl['A']:.4f} | {'sí' if bl['landscape']['saddle'] is not None else 'no'} | "
                  f"{'sí' if bl['landscape']['pole_ray_barrier_is_a_saddle'] else 'no'} | {bl['distance_closed_form']:.4f} | "
                  f"{run['distance_to_diagonal_final']:.4f} | {'sí' if run['theta_max_drop'] <= 1e-12 else 'no'} |")
    md += ["", f"Con salida por el polo el vacío queda a arcsin√(κ₋/2κ₊) = {d_min:.4f} rad ({math.degrees(d_min):.1f}°) o más de la "
           "diagonal, sea cual sea δ₀.", "", "## Acercar el vacío a la diagonal aleja el collado del polo", "",
           "sin(d_collado) = √(κ₊/κ₋)·sin(d_vacío), sin δ₀, A ni ē:", "",
           "| del vacío a la diagonal (rad) | del collado a la diagonal (rad) | ángulo del collado desde el polo (rad) |", "|---|---|---|"]
    md += [f"| {v['vacuum_distance']:g} | {v['saddle_distance']:.4f} | {v['saddle_angle_from_pole']:.4f} |" for v in doc["vacuum_vs_saddle"]]
    md += ["", "## Comprobaciones «derivadas» del reloj que estos brazos no superan", "",
           "Están escritas para el rayo del polo sin término angular; se publican, no se ocultan:", ""]
    md += [f"- `{c}`: {why}." for c, why in doc["derived_checks_failed_seen"].items()]
    s = doc["summary"]
    md += ["", f"Cruza la diagonal en alguna corrida: **{'sí' if s['any_crossing'] else 'no'}**; identidades del descenso en todas: "
           f"**{'sí' if s['identities_pass_all'] else 'no'}**.", "", f"**Lectura**: {doc['reading']}."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
