#!/usr/bin/env python
"""Candidato 2 del cribado de β (revisión del autor del 30-sep): β a un lazo del séxtico en d = 1, 2, 3 — derivación y
verificación numérica, SIN letra. Escribe results/2026-10-18_beta_one_loop_sextic/.

    python scripts/run_beta_one_loop.py derive       # tablas analíticas y flujos desde las condiciones iniciales del reloj
    python scripts/run_beta_one_loop.py candidate1   # candidato 1 bajo el criterio (i) integral (informativo, re-corrida)
    python scripts/run_beta_one_loop.py report       # report.md desde los dos JSON

Qué es y qué no es. La derivación (core/beta_one_loop.py) es cerrada: el cruce D = 0 depende solo de R = B²/(4C₀M₀²), que es
invariante bajo toda reescala, y el teorema de signo decide el candidato para el campo del reloj (N = 2) sin correr nada. Los
flujos que se publican aquí son la verificación numérica de esa derivación desde el escalado (3.2) de las formas por defecto,
en tiempo de RG s; en el reloj s = τ·S y el flujo de acoplos es autónomo, así que la conclusión vale para todo τ. No es una
corrida preinscrita del reloj y no lleva letra (E8, E13). La re-corrida del candidato 1 publica, para las celdas de su ventana,
los números que pide la revisión (el criterio (i) integral, S(T₀ = 0) y S(D = 0)); su letra congelada «no pasa» no cambia.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.basal import B_BAR, C0_DEFAULT, M_BAR  # noqa: E402
from core.beta_candidates import integral_criterion  # noqa: E402
from core.beta_one_loop import (  # noqa: E402
    F_wetterich,
    R_c_wetterich,
    R_star_polchinski,
    c_d,
    integrate_flow,
    wetterich_window,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-18_beta_one_loop_sextic"
SCREENING = ROOT / "results" / "2026-09-28_beta_screening"
DELTA0 = (0.01, 0.03)
DIMS = (1.0, 2.0, 3.0)
NS = (1, 2)
SCHEMES = ("wetterich_litim", "polchinski")
S_MAX = 40.0


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def cmd_derive(_args) -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    R0 = B_BAR ** 2 / (4.0 * C0_DEFAULT * M_BAR ** 2)
    thresholds = {str(N): {"R_c_wetterich": R_c_wetterich(N), "R_star_polchinski": R_star_polchinski(N)} for N in (1, 2, 3, 8)}
    thresholds["inf"] = {"R_c_wetterich": 13.0 / 12.0, "R_star_polchinski": 1.0}
    windows = {f"N{N}_R{R:.4f}": wetterich_window(R, N) for N in NS for R in (4.0 / 3.0, 1.5, 2.0, R0)}
    Mg = np.concatenate([np.linspace(-0.99, 2.0, 300001), np.geomspace(2.0, 1e5, 4001)])
    F_min_at_R0 = {str(N): float(np.min(F_wetterich(R0, Mg, N))) for N in NS}
    flows = []
    for scheme in SCHEMES:
        for N in NS:
            for d in DIMS:
                for delta0 in DELTA0:
                    lam0 = (M_BAR ** 2 * delta0 ** 2, B_BAR * delta0, C0_DEFAULT)
                    r = integrate_flow(scheme, lam0, d, N, s_max=S_MAX)
                    r["delta0"] = delta0
                    flows.append(r)
    doc = {"title": "Candidato 2 del cribado de β: β a un lazo (LPA) del séxtico O(N) en d = 1, 2, 3 — derivación y verificación, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "module": "core/beta_one_loop.py", "default_forms": {"m_bar": M_BAR, "b_bar": B_BAR, "C0": C0_DEFAULT, "R0": R0,
                                                                "required_integral": -math.log(R0)},
           "c_d_litim": {str(int(d)): c_d(d) for d in (1.0, 2.0, 3.0, 4.0)},
           "thresholds": thresholds, "wetterich_windows": windows, "F_min_over_all_mass_at_R0": F_min_at_R0,
           "flows_from_clock_initial_conditions": flows,
           "declared": {"schemes": "Wetterich con regulador de Litim (c_d absorbido) y Polchinski normalizado (a = b = 1/2, el del reloj)",
                        "truncation": "séxtico en x = |φ|² alrededor del origen (el del reloj)", "N": "2 = campo (φ_M, φ_E) del reloj; 1 = «un escalar»",
                        "initial_conditions": "escalado (3.2): M₀² = m̄²δ₀², B = b̄δ₀, C₀; acoplos adimensionales en k = Λ (convención)",
                        "time": "s = ln(Λ/k); en el reloj s = τ·S (diccionario τ no derivado): el flujo de acoplos es autónomo"},
           "status": "derivación (E8) y verificación numérica; sin letra; no es una corrida preinscrita del reloj"}
    (OUTDIR / "derivation.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_cross = sum(1 for f in flows if f["s_D_zero"] is not None)
    print(f"derivación escrita: {len(flows)} flujos, {n_cross} con D = 0; F mínima en R₀ = {F_min_at_R0}")
    return 0


def _clock_one(delta0: float, tau: float, kappa_hat, closure) -> dict:
    from core.fokker_planck_beta import FPClosure
    from core.s_clock import ClockConfig, SClock
    kw = {"delta0": delta0, "thresholds": "emergent", "couplings_flow": True, "fp": FPClosure(tau=tau), "max_steps": 400000,
          "kappa_conv": 0.0, "reset_mass_on_collapse": False}
    if closure is not None:
        kw.update({"beta_extra": "conversion_current", "beta_extra_closure": closure, "beta_extra_kappa_hat": kappa_hat})
    t0 = time.time()
    r = SClock(ClockConfig(**kw)).run()
    tr = r["trajectory"]
    crit = integral_criterion(tr["S"], tr["M0_sq"], tr["B"], tr["C0"])
    return {"delta0": delta0, "tau": tau, "kappa_hat": kappa_hat, "closure": closure, "finished": r["finished"],
            "stop_reason": r["stop_reason"], "wall_s": round(time.time() - t0, 1), "S_final": float(tr["S"][-1]),
            "n_crossings": sum(1 for e in r["events"] if e["kind"].startswith("colapso")), "criterion_i_integral": crit}


def cmd_candidate1(_args) -> int:
    """Celdas de la ventana del candidato 1 (κ̂ ∈ [0.5, 5], cierre 'quadrant', sin escalón) y los controles canónicos."""
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pre = json.loads((SCREENING / "preregistration.json").read_text(encoding="utf-8"))
    lo, hi = pre["rules"]["kappa_hat_window"]
    kappas = [k for k in pre["declared"]["kappa_hat_grid"] if lo <= k <= hi]
    out_path = OUTDIR / "candidate1_integral.json"
    done = json.loads(out_path.read_text(encoding="utf-8"))["runs"] if out_path.exists() else []
    keys = {(r["delta0"], r["tau"], r["kappa_hat"], r["closure"]) for r in done}
    jobs = [(d0, tau, None, None) for d0 in pre["declared"]["delta0_grid"] for tau in pre["declared"]["tau_grid"]]
    jobs += [(d0, tau, k, "quadrant") for d0 in pre["declared"]["delta0_grid"] for tau in pre["declared"]["tau_grid"] for k in kappas]
    for job in jobs:
        if job in keys:
            continue
        rec = _clock_one(*job)
        done.append(rec)
        out_path.write_text(json.dumps({"screening_prereg_sha256_cited": __import__("hashlib").sha256((SCREENING / "preregistration.json").read_bytes()).hexdigest(),
                                        "frozen_outcome_unchanged": "no pasa", "kappa_hat_window": [lo, hi], "runs": done},
                                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        c = rec["criterion_i_integral"]
        print(f"  δ₀ {job[0]} τ {job[1]} κ̂ {job[2]}: fracción {c['fraction_of_required']:.3g}, S(T₀=0) {c['S_T0_zero']}, S(D=0) {c['S_D_zero']}, "
              f"S(M₀²=0) {c['S_mass_zero']} ({rec['wall_s']} s)", flush=True)
    return 0


def _fmt(v, nd=3):
    return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def cmd_report(_args) -> int:
    der = json.loads((OUTDIR / "derivation.json").read_text(encoding="utf-8"))
    c1_path = OUTDIR / "candidate1_integral.json"
    c1 = json.loads(c1_path.read_text(encoding="utf-8")) if c1_path.exists() else None
    df = der["default_forms"]
    md = [f"# {der['title']}", "", f"Commit `{der['code_commit'][:9]}`; módulo `{der['module']}`. Estatuto: {der['status']}.", "",
          "## Umbrales del flujo de R = B²/(4C₀M₀²)", "",
          f"Formas por defecto: m̄ = {df['m_bar']}, b̄ = {df['b_bar']}, C₀ = {df['C0']} ⟹ R₀ = {df['R0']}; la espinodal exige "
          f"∫[2β_B/B − β_C/C − β_M/M] = {df['required_integral']:.3f} antes de M₀² = 0. T₀ = 0 en R = 4/3, D = 0 en R = 1.", "",
          "| N | R_c (Wetterich): F > 0 para todo m̃² si R > R_c | R* (Polchinski): crece si R > R* y M₀² > 0 |", "|---|---|---|"]
    for N, t in der["thresholds"].items():
        md.append(f"| {N} | {t['R_c_wetterich']:.4f} | {t['R_star_polchinski']:.4f} |")
    md += ["", "Ventanas de m̃² en las que R decrece (Wetterich):", ""] + \
          [f"- {k}: {('(%.4f, %.4f)' % tuple(v)) if v else 'ninguna'}" for k, v in der["wetterich_windows"].items()]
    md += ["", f"Mínimo de F sobre todo m̃² en R₀ = {df['R0']}: " + ", ".join(f"N = {k}: {v:.3f}" for k, v in der["F_min_over_all_mass_at_R0"].items()), "",
           "## Flujos desde las condiciones iniciales del reloj (tiempo de RG s; en el reloj s = τ·S)", "",
           "| esquema | N | d | δ₀ | s(M₀² = 0) | s(T₀ = 0) | s(D = 0) | fracción del exigido | R crece monótonamente con M₀² > 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for f in der["flows_from_clock_initial_conditions"]:
        ic = f["integral_criterion"]
        md.append(f"| {f['scheme']} | {f['N']} | {int(f['d'])} | {f['delta0']} | {_fmt(f['s_mass_zero'])} | {_fmt(f['s_T0_zero'])} | "
                  f"{_fmt(f['s_D_zero'])} | {_fmt(ic['fraction_of_required'])} | {'sí' if ic['R_monotone_increasing_while_M_positive'] else 'no'} |")
    if c1:
        md += ["", "## Candidato 1 bajo el criterio (i) integral (informativo)", "",
               f"Re-corrida de las celdas de la ventana κ̂ ∈ {c1['kappa_hat_window']} (cierre 'quadrant', sin escalón) y de los controles "
               f"canónicos, con el código del reloj de #45. La letra congelada del cribado («{c1['frozen_outcome_unchanged']}») no cambia.", "",
               "| δ₀ | τ | κ̂ | fracción del exigido | R crece monótonamente con M₀² > 0 | S(T₀ = 0) | S(D = 0) | S(M₀² = 0) |", "|---|---|---|---|---|---|---|---|"]
        for r in c1["runs"]:
            c = r["criterion_i_integral"]
            md.append(f"| {r['delta0']} | {r['tau']} | {_fmt(r['kappa_hat'])} | {_fmt(c['fraction_of_required'])} | "
                      f"{'sí' if c['R_monotone_increasing_while_M0sq_positive'] else 'no'} | {_fmt(c['S_T0_zero'])} | {_fmt(c['S_D_zero'])} | {_fmt(c['S_mass_zero'])} |")
    md += ["", "## Lectura", "",
           "Con el campo del reloj (N = 2) y las formas por defecto (R₀ = 9/4 > R_c = 25/12), R solo puede crecer en los dos esquemas y en "
           "toda d: el candidato 2 no puede producir D = 0 y el criterio (i) integral no se alcanza. La dimensión entra en la trayectoria "
           "de los acoplos pero no en el flujo instantáneo de R. Con un escalar (N = 1) hay una ventana de masa en la que R decrece, "
           "pero desde el escalado (3.2) la inestabilidad de masa llega antes de entrar en ella. Derivación, no corrida preinscrita: "
           "sin letra sobre λ ni sobre la ley."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"report.md escrito en {OUTDIR}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("derive", "candidate1", "report"):
        sub.add_parser(c)
    args = ap.parse_args()
    return {"derive": cmd_derive, "candidate1": cmd_candidate1, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
