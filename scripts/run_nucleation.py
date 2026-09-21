#!/usr/bin/env python
"""La salida de S₀: bounce de Coleman sobre el Basal (B(δ₀), Γ₀/A, ley de
escala), escape de Kramers del Flujo del Camino con difusión entrópica
(Def. 4.4) y el reloj S arrancando en el estado que la nucleación
entrega, con la integración conjunta Φ_Ad ⊗ λ_i (modo emergente) y sus
eventos S_imposed / S_emergent publicados. Publica
results/2026-09-21_nucleation/.

Cálculo de consistencia (E8) con convenciones declaradas (dimensión d
del instantón, prefactor A, D_ent, normalización de la acción); ninguna
preinscripción: no decide nada.

Uso: python scripts/run_nucleation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fokker_planck_beta import FPClosure  # noqa: E402
from core.nucleation import STATUS, kramers_scaling, scaling_law  # noqa: E402
from core.s_clock import ClockConfig, SClock, delta0_metastability_max  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_nucleation"
DELTA0_GRID = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064]
DELTA0_CLOCK = 0.01
D_ENT_GRID = [1e-9, 1e-8, 1e-7]
TAU_GRID = [1e-3, 1e-2, 1e-1]


def _slim(run: dict) -> dict:
    return {k: v for k, v in run.items() if k != "trajectory"}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    scal = {str(d): scaling_law(DELTA0_GRID, d=d) for d in (3, 4)}
    kram = {str(D): kramers_scaling(DELTA0_GRID, D_ent=D) for D in D_ENT_GRID}
    clocks = {}
    for d in (3, 4):
        run = SClock(ClockConfig(delta0=DELTA0_CLOCK, nucleation="bounce", bounce_d=d)).run()
        clocks[f"imposed_bounce_d{d}"] = _slim(run)
    joint = {}
    for d in (3, 4):
        for tau in TAU_GRID:
            run = SClock(ClockConfig(delta0=DELTA0_CLOCK, nucleation="bounce", bounce_d=d, thresholds="emergent",
                                     couplings_flow=True, fp=FPClosure(tau=tau))).run()
            joint[f"d{d}_tau{tau:g}"] = {"stop_reason": run["stop_reason"], "S_final": run["trajectory"]["S"][-1],
                                         "f0": run["checks"]["descent"]["S_equals_f_identity"]["f0_nucleation"],
                                         "events": [{"kind": e["kind"], "S_emergent": e["S_emergent"], "sigma": e["sigma"]}
                                                    for e in run["events"] if e["kind"].startswith("colapso")],
                                         "D_min": min(run["trajectory"]["D"]), "couplings_out_of_domain": run["couplings_out_of_domain"]}
    doc = {"kind": "cálculo de consistencia (E8) con convenciones declaradas; sin preinscripción",
           "status": STATUS, "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "delta0_metastability_max": delta0_metastability_max(), "bounce_scaling": scal, "kramers_scaling": kram,
           "clock_with_bounce": clocks, "joint_emergent_from_bounce": joint,
           "declared": ["dimensión d del instantón (3 y 4 publicadas; el tramo pre-geométrico no tiene espacio-tiempo)",
                        "prefactor A de Γ₀ = A·e^{−B} (dimensional, no está en el tratado): se publica e^{−B}",
                        "normalización de la acción euclidiana (G = 1, ħ = 1)",
                        "D_ent de la difusión entrópica (el diccionario t ↔ σ de la Def. 4.4 no la fija)",
                        "corte radial θ = 0 (salida al polo de masa, Prop. 3.5)"],
           "what_is_not_claimed": ["ninguna de las dos nucleaciones es «la del tratado»: decidirlo es del diccionario",
                                   "ningún umbral emerge en la integración conjunta (los eventos S_emergent se publican vacíos si D no cruza cero)",
                                   "no es demostración física"]}
    (OUTDIR / "nucleation.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = ["# La salida de S₀: bounce de Coleman, escape de Kramers y el reloj S arrancando en el estado nucleado (21-sep-2026)\n",
          f"Commit `{sha[:9]}`. Formas del Basal por defecto; δ₀_max de metastabilidad = {doc['delta0_metastability_max']:.4f}.", "",
          f"**Lectura obligatoria**: {STATUS}.", "",
          "## Bounce O(d) sobre el corte radial (paisaje completo)", "",
          "| δ₀ | barrera/T₀ | d = 4: B | e^{−B} | f₀ | d = 3: B | e^{−B} | f₀ |", "|---|---|---|---|---|---|---|---|"]
    for r4, r3 in zip(scal["4"]["rows"], scal["3"]["rows"]):
        md.append(f"| {r4['delta0']} | {r4['barrier_over_T0']:.4f} | {r4['B']:.4g} | {r4['Gamma_over_A']:.2e} | {r4['f0']:.4f} | "
                  f"{r3['B']:.4g} | {r3['Gamma_over_A']:.2e} | {r3['f0']:.4f} |")
    md += ["", f"Ley de escala medida: B ∝ δ₀^p con p = {scal['4']['exponent_measured']:.3f} (d = 4; argumento 3 − d = −1) y "
           f"p = {scal['3']['exponent_measured']:.3f} (d = 3; argumento 0). Γ₀(0) = 0 sin prefactor para d = 4 (B → ∞); para d = 3 "
           "B tiende a una constante y el axioma exige A(δ₀) → 0. f₀ es la fracción de T₀ ya descargada en el centro del bounce: "
           "en este régimen (ε = T₀ ≫ altura de barrera, pared gruesa) el instantón entrega el campo casi en el vacío verdadero "
           "para d = 4 y parcialmente descargado para d = 3 — la descarga y sus umbrales caen DENTRO de la nucleación, no a lo "
           "largo del Flujo del Camino. La estimación de pared delgada no es aplicable (barrera ≪ ε) y se publica solo como referencia.", "",
           "## Escape de Kramers del Flujo del Camino con difusión entrópica (Def. 4.4)", "",
           "| δ₀ | ΔV_b | √(V''_fv|V''_b|)/2π | Γ_K (D = 1e-9) | Γ_K (1e-8) | Γ_K (1e-7) |", "|---|---|---|---|---|---|"]
    for i, d0 in enumerate(DELTA0_GRID):
        r = kram[str(D_ENT_GRID[0])]["rows"][i]
        md.append(f"| {d0} | {r['barrier_height']:.2e} | {r['prefactor']:.2e} | " + " | ".join(
            f"{kram[str(D)]['rows'][i]['Gamma_K']:.2e}" for D in D_ENT_GRID) + " |")
    k0 = kram[str(D_ENT_GRID[0])]
    md += ["", f"Exponentes medidos: prefactor ∝ δ₀^{k0['prefactor_exponent_measured']:.2f} (argumento 2), barrera ∝ "
           f"δ₀^{k0['barrier_exponent_measured']:.2f} (argumento 3): Γ_K(0) = 0 por el PREFACTOR (el paisaje plano no tiene "
           "curvatura que fije un ritmo), mecanismo opuesto al bounce. Cuál es la nucleación del tratado — instantón conservativo "
           "o escape disipativo de la dinámica de primer orden del Axioma 4 — es decisión del diccionario.", "",
           "## El reloj S arrancando en el centro del bounce (δ₀ = 0.01, umbrales impuestos)", "",
           "| d | f₀ | B | eventos en σ = 0 (dentro de la nucleación) | σ̂ del descenso restante | S ≡ f − f₀ |", "|---|---|---|---|---|---|"]
    for d in (3, 4):
        c = clocks[f"imposed_bounce_d{d}"]
        chk = c["checks"]["descent"]
        inside = [e["kind"] for e in c["events"][:3] if "dentro de la nucleación" in e["trigger"]]
        md.append(f"| {d} | {chk['S_equals_f_identity']['f0_nucleation']:.4f} | {c['checks']['S0']['nucleation']['bounce']['B']:.4g} | "
                  f"{inside or '—'} | {chk['descent_finished']['sigma_hat_final']:.3f} | {chk['S_equals_f_identity']['max_abs_diff']:.1e} |")
    md += ["", "## Integración conjunta Φ_Ad ⊗ λ_i desde el bounce (modo emergente, diagnóstico)", "",
           "| d | τ | f₀ | S final | cruces D = 0 (S_emergent) | D mínimo | fin |", "|---|---|---|---|---|---|---|"]
    for key, j in joint.items():
        d, tau = key.split("_tau")
        md.append(f"| {d[1:]} | {tau} | {j['f0']:.3f} | {j['S_final']:.4f} | {[e['S_emergent'] for e in j['events']] or '—'} | "
                  f"{j['D_min']:.2e} | {j['stop_reason']} |")
    md += ["", "Ningún cruce D = 0 en el rango explorado: los eventos S_emergent se publican vacíos (sin estatuto; depende de τ).", "",
           "## Declarado / lo que NO afirma", ""] + [f"- {s}" for s in doc["declared"]] + [""] + [f"- {s}" for s in doc["what_is_not_claimed"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"B(0.01): d4 {scal['4']['rows'][3]['B']:.4g}, d3 {scal['3']['rows'][3]['B']:.4g}; p4 = {scal['4']['exponent_measured']:.3f}, "
          f"p3 = {scal['3']['exponent_measured']:.3f}; Kramers prefactor exp {k0['prefactor_exponent_measured']:.2f}; "
          f"f0(d4) = {clocks['imposed_bounce_d4']['checks']['descent']['S_equals_f_identity']['f0_nucleation']:.4f}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
