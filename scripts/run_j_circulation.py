#!/usr/bin/env python
"""La diagonal y el |J| requerido — el término J∇C del Camino de dos niveles
en el reloj S (21-sep-2026, tarde; punto (3) del orden).

    python scripts/run_j_circulation.py prereg    # congela la preinscripción (sha256)
    python scripts/run_j_circulation.py run       # ejecuta bajo la preinscripción
    python scripts/run_j_circulation.py analyze   # aplica la regla congelada; falla cerrado sin ella

La pregunta del autor: ¿qué |J| sitúa el cruce de la diagonal θ = π/4 en
S = 1 ± 0.05? El flujo de gradiente puro deja θ = 0; la circulación J∇C
(J antisimétrica, |J| declarado) es el único término del programa que
rota θ. Generador C DECLARADO: C = V (J∇V ⊥ ∇V: Monotonía 4.5 y S ≡ f
exactas) como forma principal; C = δ₀²ρ²/2 (rotación rígida) como
alternativa que trabaja contra la inclinación (coste publicado).

Desenlaces (por δ₀, con C = V; regla congelada):
    A(δ₀) — el cruce puede situarse en la ventana [0.95, 1.05]: S_max(δ₀) ≥ 0.95,
            donde S_max es el S del cruce en el umbral |J|_min (el más alto
            posible: subir |J| ADELANTA el cruce); se publica |J| ∈ [|J|_min,
            |J|(S_cross = 0.95)].
    B(δ₀) — no puede: S_max(δ₀) < 0.95 y el flujo vuelve hacia θ = 0 (el vacío
            verdadero 2D está sobre el polo de masa) completando el descenso en
            el interior; se publica |J|_min y S_max.
    F(δ₀) — frontera: con |J|_min el flujo vuelve a la frontera φ_E = 0 y la
            componente tangencial de la circulación (que la ligadura normal no
            cancela) desplaza el equilibrio: W_J/T₀ ≫ 0 y el descenso no
            completa (f estancado bajo 1 − ε_res). Se publica W_J y f final.
    Global: A si A(δ₀) para todo δ₀; B si B(δ₀) para todo δ₀; F si F para todo
            δ₀; MIXTO si cambia con δ₀ (se publica el δ₀* mayor con A y el
            menor con F).
    C     — la alternativa rígida C = δ₀²ρ²/2 sitúa el cruce en la ventana
            para algún |J| pero a costa de la Monotonía (W_J/T₀ > 1e-6) o de
            S ≡ f: se publica como coste, no como solución.
    Invariancia de escala (publicada con umbral a priori): |J|_min(δ₀)
            varía menos del 10 % sobre la malla (σ̂ = σδ₀² hace |J| adimensional).
    E13  — cociente |J|_min / [(π/4)/∫|∇C|dσ] (la expectativa del autor)
            publicado sin veredicto.
    INDETERMINADO — alguna puerta falla (sin umbral en el intervalo; identidad
            S = f − f₀ + W_J/T₀ rota por encima de 1e-6 en una corrida interior
            con C = V; Monotonía rota con C = V).

Piloto declarado (ANTES de congelar, δ₀ ∈ {0.003, 0.01, 0.03}): |J|_min ≈
1.36–1.39, S del cruce en el umbral 0.93 / 0.88 / 0.81, retorno hacia θ = 0;
en 0.03 el retorno llega a la frontera (W_J/T₀ = 5.5, descenso sin
completar). La estructura A/B/F se escribió conociéndolo; lo preinscrito
es el barrido completo (δ₀ ∈ {0.001, 0.003, 0.01, 0.03, δ_H}), la
invariancia, los δ₀ de cambio y la alternativa rígida. Ninguna tolerancia
se ajustó al piloto.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.s_clock import (  # noqa: E402
    DECLARED_FORMS,
    J_min_threshold,
    J_required,
    diagonal_crossing_S,
)
from mass_program.B7_empalme import delta0_required  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_j_circulation"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


def prereg() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    delta_H = float(delta0_required())
    doc = {
        "title": "La diagonal y el |J| requerido: circulación J∇C del Camino de dos niveles en el reloj S",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_j_circulation.py prereg", "generator_commit_must_precede_freeze": True,
        "declared": {
            "form": DECLARED_FORMS["J_circulation"],
            "C_primary": "V", "C_alternative": "rho2 (C = δ₀²ρ²/2; |J| adimensional en σ̂)",
            "orientation": "θ crece hacia la diagonal durante el descenso",
            "G": 1.0, "delta0_grid": [0.001, 0.003, 0.01, 0.03, delta_H],
            "window": [0.95, 1.05], "S_end_of_descent": 0.999,
            "J_bracket": [0.05, 50.0], "bisection_rtol": 2e-3,
            "shapes": "por defecto (m̄ = 1, b̄ = 3, ē = 1, C0 = 1)",
        },
        "rules": {
            "F_per_delta0": "con |J|_min (C = V) el flujo vuelve a la frontera φ_E = 0 (θ_final ≤ 1e-9) — se evalúa ANTES que A/B",
            "A_per_delta0": "no F y S_max_cross(δ₀; C = V) ≥ 0.95",
            "B_per_delta0": "no F y S_max_cross(δ₀; C = V) < 0.95",
            "global": "A/B/F si todo δ₀ comparte la letra; MIXTO si cambia (publicar el mayor δ₀ con A y el menor con F)",
            "C_rigid": "el umbral rígido (rho2) tiene S_max ∈ ventana Y (W_J/T₀ > 1e-6 o |S − f − W_J/T₀|_max > 1e-6 o "
                       "Monotonía rota): la alternativa rígida sitúa el cruce pero con coste publicado",
            "scale_invariance": {"quantity": "max(J_min)/min(J_min) − 1 sobre la malla (C = V)", "tol": 0.10},
            "E13_estimate": "cociente |J|_min / [(π/4)/∫|∇V|dσ] publicado sin veredicto",
            "gates": {"bracket_ok_all": True, "identity_interior_C_V": 1e-6, "monotonia_C_V": True},
            "verdict": "letra global + tabla por δ₀ + coste de C rígida + invariancia; INDETERMINADO si falla una puerta; "
                       "|J| es calibración declarada, no derivación (frente 2, E8)",
        },
        "pilot_declared": "δ₀ ∈ {0.003, 0.01, 0.03} con C = V antes de congelar: |J|_min ≈ 1.362 / 1.369 / 1.389, S del cruce "
                          "en el umbral 0.930 / 0.879 / 0.806, θ vuelve hacia 0 (vacío verdadero 2D en el polo de masa); en "
                          "0.03 llega a la frontera φ_E = 0 con W_J/T₀ = 5.5 y el descenso no completa (letra F). Con C = V y "
                          "|J| = 2.5 (0.01) llega a la frontera φ_M = 0. C rígida con |J| ∈ {0.5, 1, 2} (0.01): cruce en "
                          "0.88 / 0.88 / 0.32, W_J/T₀ = 6 / 12 / 24, |S − f| ≈ 1.3. La estructura A/B/F/MIXTO se escribió "
                          "conociendo esto; lo no medido y preinscrito: 0.001, δ_H, la invariancia, los δ₀ de cambio y el "
                          "umbral rígido.",
        "what_this_cannot_decide": ["J desde la ontología (frente 2)", "si el tratado quiere la diagonal como estado final "
                                    "(requiere que el vacío 2D no esté en el polo de masa: otro término, no |J|)"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). "
          f"sha256 de `preregistration.json`: `{sha}`.", "",
          "## Declarado", "", f"- {doc['declared']['form']}",
          f"- C principal: V; alternativa: {doc['declared']['C_alternative']}. Orientación: {doc['declared']['orientation']}.",
          f"- Malla δ₀: {doc['declared']['delta0_grid']}; ventana {doc['declared']['window']}; fin del descenso S = 0.999; "
          f"intervalo de |J| {doc['declared']['J_bracket']}; bisección al {doc['declared']['bisection_rtol']}.", "",
          "## Reglas (congeladas)", ""] + [f"- **{k}**: {v}" for k, v in doc["rules"].items()] + [
          "", "## Piloto declarado", "", doc["pilot_declared"], "",
          "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada")
    return json.loads(PREREG.read_text(encoding="utf-8"))


def run() -> int:
    pre = _load_prereg()
    d = pre["declared"]
    lo, hi = d["J_bracket"]
    rows = []
    for d0 in d["delta0_grid"]:
        r = J_min_threshold(d0, "V", J_lo=lo, J_hi=hi, rtol=d["bisection_rtol"])
        print(f"δ₀ = {d0:.4g}: |J|_min = {r['J_min']}, S_max = {r.get('S_max_cross')}, θ_final = {r.get('theta_final_at_J_min')}, "
              f"frontera = {r.get('returns_to_boundary')}, W_J/T₀ = {r.get('W_J_over_T0')}", flush=True)
        # ventana: |J| con S_cross = 0.95 si S_max ≥ 0.95 (cota superior del intervalo publicado)
        jw = J_required(d0, d["window"][0], "V", J_lo=lo, J_hi=hi) if r.get("S_max_cross") and r["S_max_cross"] >= d["window"][0] else None
        rig = J_min_threshold(d0, "rho2", J_lo=lo, J_hi=hi, rtol=d["bisection_rtol"])
        print(f"   rígida: |J|_min = {rig['J_min']}, S_max = {rig.get('S_max_cross')}, W_J/T₀ = {rig.get('W_J_over_T0')}, "
              f"|S − f − W_J/T₀| = {rig.get('S_equals_f_max_diff')}, monotonía = {rig.get('monotonia_pass')}", flush=True)
        rows.append({"delta0": d0, "V": r, "V_window_upper": jw, "rho2": rig,
                     "baseline_J0": diagonal_crossing_S(d0, 0.0, "V")})
    RUNS.write_text(json.dumps({"preregistration_sha256": _sha(PREREG), "code_commit": _git_head(),
                                "executed_utc": datetime.now(timezone.utc).isoformat(), "rows": rows},
                               ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"corridas guardadas en {RUNS}")
    return 0


def analyze() -> int:
    pre = _load_prereg()
    if not RUNS.exists():
        raise SystemExit("FALLO CERRADO: no hay corridas")
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    if runs["preregistration_sha256"] != _sha(PREREG):
        raise SystemExit("FALLO CERRADO: la preinscripción cambió después de las corridas")
    R, D = pre["rules"], pre["declared"]
    w_lo, w_hi = D["window"]
    per = []
    gates = {"bracket_ok_all": True, "identity_interior_C_V_ok": True, "monotonia_C_V_ok": True}
    for row in runs["rows"]:
        v = row["V"]
        if not v["bracket_ok"]:
            gates["bracket_ok_all"] = False
            per.append({"delta0": row["delta0"], "letter": None, "J_min": None, "S_max": None})
            continue
        boundary = bool(v["returns_to_boundary"])
        if not boundary and v["S_equals_f_max_diff"] > R["gates"]["identity_interior_C_V"]:
            gates["identity_interior_C_V_ok"] = False
        if not v["monotonia_pass"]:
            gates["monotonia_C_V_ok"] = False
        letter = "F" if boundary else ("A" if v["S_max_cross"] >= w_lo else "B")
        rig = row["rho2"]
        rig_ok = rig["bracket_ok"]
        rigid_in_window = bool(rig_ok and rig["S_max_cross"] is not None and w_lo <= rig["S_max_cross"] <= w_hi)
        rigid_cost = bool(rig_ok and (abs(rig.get("W_J_over_T0") or 0.0) > 1e-6 or (rig.get("S_equals_f_max_diff") or 0.0) > 1e-6
                                      or not rig.get("monotonia_pass", True)))
        per.append({"delta0": row["delta0"], "letter": letter, "J_min": v["J_min"], "S_max": v["S_max_cross"],
                    "theta_final_at_J_min": v["theta_final_at_J_min"], "W_J_over_T0": v["W_J_over_T0"], "finished": v["finished"],
                    "J_window_upper": (row["V_window_upper"] or {}).get("J_required"),
                    "J_estimate_E13": v["J_estimate_pi4_over_int_gradC"], "ratio_J_min_over_estimate": v["ratio_J_min_over_estimate"],
                    "rigid_J": rig.get("J_min"), "rigid_S_cross": rig.get("S_max_cross"), "rigid_W_J_over_T0": rig.get("W_J_over_T0"),
                    "rigid_S_equals_f_max_diff": rig.get("S_equals_f_max_diff"), "rigid_monotonia": rig.get("monotonia_pass"),
                    "rigid_in_window": rigid_in_window, "rigid_cost": rigid_cost, "C_rigid_letter": bool(rigid_in_window and rigid_cost)})
    letters = [p["letter"] for p in per if p["letter"]]
    all_gates = all(gates.values())
    if all_gates and letters:
        glob = letters[0] if all(x == letters[0] for x in letters) else "MIXTO"
    else:
        glob = "INDETERMINADO"
    dstar = max([p["delta0"] for p in per if p["letter"] == "A"], default=None)
    dF = min([p["delta0"] for p in per if p["letter"] == "F"], default=None)
    jmins = [p["J_min"] for p in per if p["J_min"]]
    inv = (max(jmins) / min(jmins) - 1.0) if jmins else None
    res = {"preregistration_sha256": runs["preregistration_sha256"], "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "per_delta0": per,
           "verdict_global": glob, "delta0_star_largest_A": dstar, "delta0_smallest_F": dF,
           "scale_invariance": {"spread": inv, "tol": R["scale_invariance"]["tol"],
                                "pass": (inv is not None and inv <= R["scale_invariance"]["tol"])},
           "C_rigid": any(p["C_rigid_letter"] for p in per),
           "reading": ("|J| es una calibración declarada con significado (frente 2 debe reproducirla junto con λ), no una "
                       "derivación; con C = V el vacío 2D está en el polo de masa y θ vuelve a 0: la diagonal como estado final "
                       "no es cuestión de |J| sino de otro término (E8, E13)")}
    (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    md = [f"# La diagonal y el |J| requerido — resultado bajo la preinscripción `{runs['preregistration_sha256'][:12]}`", "",
          f"Corridas en `{runs['code_commit'][:9]}`; análisis en `{res['code_commit_analysis'][:9]}`. Puertas: {gates}.", "",
          f"## Veredicto global: **{glob}**" + (f" (mayor δ₀ con A: {dstar}; menor δ₀ con F: {dF})" if glob == "MIXTO" else ""), "",
          res["reading"] + ".", "",
          "| δ₀ | letra | \\|J\\|_min (C = V) | S_max del cruce | θ_final | W_J/T₀ | descenso completo | \\|J\\| para S = 0.95 | E13: \\|J\\|_est | cociente | rígida \\|J\\|_min | rígida S_max | rígida W_J/T₀ | rígida \\|S − f − W_J/T₀\\| |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in per:
        f = lambda x, fmt="{:.4g}": "—" if x is None else fmt.format(x)  # noqa: E731
        md.append(f"| {p['delta0']:.4g} | {p['letter']} | {f(p['J_min'])} | {f(p['S_max'])} | {f(p.get('theta_final_at_J_min'))} | "
                  f"{f(p.get('W_J_over_T0'), '{:.2e}')} | {'sí' if p.get('finished') else 'no'} | "
                  f"{f(p.get('J_window_upper'))} | {f(p.get('J_estimate_E13'))} | {f(p.get('ratio_J_min_over_estimate'), '{:.3f}')} | "
                  f"{f(p.get('rigid_J'))} | {f(p.get('rigid_S_cross'))} | {f(p.get('rigid_W_J_over_T0'))} | {f(p.get('rigid_S_equals_f_max_diff'), '{:.2e}')} |")
    md += ["", f"Invariancia de escala de |J|_min: dispersión {inv if inv is None else round(inv, 4)} (tol {R['scale_invariance']['tol']}): "
           f"{'sí' if res['scale_invariance']['pass'] else 'no'}. C rígida en ventana con coste: {'sí' if res['C_rigid'] else 'no'}.", "",
           "## Lo que no decide", ""] + [f"- {s}" for s in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"veredicto: {glob}; δ₀* = {dstar}; invariancia = {inv}; puertas = {gates}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"prereg": prereg, "run": run, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
