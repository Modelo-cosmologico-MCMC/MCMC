#!/usr/bin/env python
"""Cuarta ecuación, brazo «término angular» (bloque P1 del autor del 23-oct), SIN letra: V_ang = ½μ²χ² con
μ² = 4A(c̄/κ₊)δ₀² en el reloj (ClockConfig.angular_A), cierre declarado junto a los dos del vacío 2D.

    python scripts/run_angular_term.py    # escribe results/2026-10-23_angular_term/

Barrido de A dentro de la ventana [t/4, 1 + t/√2], con t = ηρ₊/T₀ (T₀ = c̄δ₀³, ρ₊ = √(κ₊δ₀)): nueve valores en el punto
medio logarítmico de nueve tramos y A = 1, para δ₀ ∈ {0.003, 0.01, 0.0554}. Se publican θ final, monotonía de θ, S con medio
giro (primer θ ≥ θ*/2), S al final de la fase radial (ρ ≥ 0.99ρ_vv), si cruza π/4 y T₀ del paisaje completo frente a la
ley. Sin letra: la exploración del 23-oct ya mostró que no cruza la diagonal, así que la predicción del 22-sep («cruce en
S = 1 ± 0.05») no se puede someter a prueba; la redacción de la Prop. 3.5 es la decisión 2 del autor.
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

from core.basal import E_BAR, c_bar, kappa_plus  # noqa: E402
from core.s_clock import ClockConfig, SClock  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-23_angular_term"
DELTA0 = (0.003, 0.01, 0.0554)
N_A = 9


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def tilt_weight(delta0: float) -> float:
    """t = ηρ₊/T₀ con la ley T₀ = c̄δ₀³ y ρ₊ = √(κ₊δ₀): mide el peso de la inclinación."""
    return E_BAR * math.sqrt(kappa_plus() * delta0) / c_bar()


def exact_upper_edge(delta0: float) -> float:
    """El borde superior exacto de la ventana: el mayor A con punto de escape en el rayo θ = 0 (el rayo conserva un vacío
    más profundo que el falso). 1 + t/√2 es su aproximación de primer orden."""
    from core.s_clock import radial_landscape

    def has_escape(A):
        mu2 = 4.0 * A * c_bar() / kappa_plus() * delta0 ** 2
        return radial_landscape(delta0, 0.0, mass_extra=0.5 * mu2).get("rho_esc") is not None
    t = tilt_weight(delta0)
    lo, hi = 1.0 + t / math.sqrt(2.0) * 0.5, 2.0 * (1.0 + t / math.sqrt(2.0))
    if not has_escape(lo) or has_escape(hi):
        raise ValueError("borde superior fuera del intervalo de búsqueda")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if has_escape(mid) else (lo, mid)
    return lo


def a_grid(delta0: float, n: int = N_A) -> list[float]:
    t = tilt_weight(delta0)
    lo, hi = math.log(t / 4.0), math.log(1.0 + t / math.sqrt(2.0))
    return sorted({*(math.exp(lo + (k + 0.5) * (hi - lo) / n) for k in range(n)), 1.0})


def one(delta0: float, A: float) -> dict:
    clk = SClock(ClockConfig(delta0=delta0, angular_A=A))
    r = clk.run()
    tr = r["trajectory"]
    th, S, rho = np.asarray(tr["theta"]), np.asarray(tr["S"]), np.asarray(tr["rho"])
    tv = clk.true_vacuum_2d
    half = th >= 0.5 * tv["theta"]
    radial = rho >= 0.99 * tv["rho"]
    t = tilt_weight(delta0)
    return {"delta0": delta0, "A": A, "t": t, "mu2": clk.mu2, "finished": r["finished"], "stop_reason": r["stop_reason"],
            "steps": r["steps"], "theta_star": tv["theta"], "pi4_minus_theta_star": math.pi / 4.0 - tv["theta"],
            "delta_theta_analytic": math.asin(min(1.0, t / (4.0 * A))), "theta_final": float(th[-1]), "theta_max": float(th.max()),
            "crosses_diagonal": bool(th.max() > math.pi / 4.0), "theta_max_drop": float(np.max(np.maximum.accumulate(th) - th)),
            "S_half_turn": float(S[int(np.argmax(half))]) if half.any() else None,
            "S_end_radial": float(S[int(np.argmax(radial))]) if radial.any() else None, "S_final": float(S[-1]),
            "T0_full_over_law": clk.T0 / clk.T0_law, "rho_true_vacuum_over_rho_plus": tv["rho"] / math.sqrt(kappa_plus() * delta0)}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for d0 in DELTA0:
        for A in a_grid(d0):
            rows.append(one(d0, A))
            x = rows[-1]
            print(f"δ₀ = {d0:<6} A = {A:.4f}: θ* = {x['theta_star']:.4f}, θ_fin = {x['theta_final']:.4f}, cruza π/4: "
                  f"{x['crosses_diagonal']}, caída de θ = {x['theta_max_drop']:.1e}, S(medio giro) = {x['S_half_turn']}, "
                  f"T₀/ley = {x['T0_full_over_law']:.4f}", flush=True)
    doc = {"title": "Cuarta ecuación, brazo «término angular» en el reloj: barrido de A en la ventana, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_angular_term.py", "form": "V_ang = ½μ²χ², μ² = 4A(c̄/κ₊)δ₀² (ClockConfig.angular_A)",
           "window": "A ∈ [t/4, 1 + t/√2], t = ηρ₊/T₀ con T₀ = c̄δ₀³ y ρ₊ = √(κ₊δ₀)",
           "grid": "nueve puntos medios logarítmicos de la ventana y A = 1, δ₀ ∈ {0.003, 0.01, 0.0554}; modo impuesto (S ≡ f)",
           "rows": rows,
           "window_edges": {str(d0): {"t": tilt_weight(d0), "lower_t_over_4": tilt_weight(d0) / 4.0,
                                      "upper_first_order": 1.0 + tilt_weight(d0) / math.sqrt(2.0),
                                      "upper_exact": exact_upper_edge(d0)} for d0 in DELTA0},
           "summary": {"any_crossing": any(r["crosses_diagonal"] for r in rows),
                       "theta_monotone_all": all(r["theta_max_drop"] <= 1e-12 for r in rows),
                       "all_finished": all(r["finished"] for r in rows)},
           "reading": "sin letra (E8, E13): el término acerca el campo a la diagonal sin cruzarla; la llegada coincide con S → 1 "
                      "por construcción; la Prop. 3.5 es la decisión 2 del autor"}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. {doc['form']}; ventana {doc['window']}.", "",
          "| δ₀ | t | A | θ* | π/4 − θ* | arcsin(t/4A) | θ final | cruza π/4 | caída de θ | S medio giro | S fin radial | T₀/ley |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sh = "—" if r["S_half_turn"] is None else f"{r['S_half_turn']:.3f}"
        sr = "—" if r["S_end_radial"] is None else f"{r['S_end_radial']:.3f}"
        md.append(f"| {r['delta0']:g} | {r['t']:.3f} | {r['A']:.4f} | {r['theta_star']:.4f} | {r['pi4_minus_theta_star']:.4f} | "
                  f"{r['delta_theta_analytic']:.4f} | {r['theta_final']:.4f} | {'sí' if r['crosses_diagonal'] else 'no'} | "
                  f"{r['theta_max_drop']:.1e} | {sh} | {sr} | {r['T0_full_over_law']:.4f} |")
    md += ["", "Bordes de la ventana (el superior exacto: el mayor A con punto de escape en el rayo θ = 0):", ""] + [
        f"- δ₀ = {d}: t = {e['t']:.4f}; t/4 = {e['lower_t_over_4']:.4f}; 1 + t/√2 = {e['upper_first_order']:.4f} (primer orden); "
        f"exacto {e['upper_exact']:.4f}" for d, e in doc["window_edges"].items()]
    s = doc["summary"]
    md += ["", f"Cruza la diagonal en alguna celda: **{'sí' if s['any_crossing'] else 'no'}**; θ monótona en todas: "
           f"**{'sí' if s['theta_monotone_all'] else 'no'}**; descensos terminados: **{'sí' if s['all_finished'] else 'no'}**.", "",
           f"**Lectura**: {doc['reading']}."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
