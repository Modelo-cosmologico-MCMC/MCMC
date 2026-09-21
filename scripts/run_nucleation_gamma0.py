#!/usr/bin/env python
"""Γ₀(δ₀) condicional a n_dim — ronda 2 de la nucleación (21-sep-2026, tarde).

Tres modos, en este orden y en commits distintos:

    python scripts/run_nucleation_gamma0.py prereg    # congela la preinscripción (sha256)
    python scripts/run_nucleation_gamma0.py run       # ejecuta el barrido bajo la preinscripción
    python scripts/run_nucleation_gamma0.py analyze   # aplica la regla congelada; falla cerrado sin ella

La pregunta: ¿qué ley Γ₀(δ₀) entrega el Basal completo para cada
dimensionalidad declarada del instantón, n_dim ∈ {1, 3, 4}? El tratado
exige Γ₀(0) = 0 (inercia eterna del perfecto, Prop. 3.5) pero no fija
n_dim (decisión B del autor). Desenlaces, evaluados CADA UNO con su
regla congelada y publicados juntos (no son excluyentes: son
condicionales a n_dim):

    A  — ley LINEAL por prefactor (n = 1): Γ₀ = (ω_fv/2π)·e^{−B₁} con
         B₁ = b₁·δ₀² → 0; el exponente medido de Γ₀(δ₀) en la década baja
         cae en 1 ± tol_A y B₁/δ₀² tiende a b₁ (sin inclinación) al 10 %.
    B  — supresión exponencial ∝ δ₀⁻¹ (n = 4): el exponente medido de
         B(δ₀) en la década baja cae en −1 ± tol_B.
    C  — la deformación del camino en (ρ, χ) cambia B₁ en más de ×2
         (B_min/B_ray < 0.5) en algún (δ₀, ē) del barrido.
    INDETERMINADO — alguna puerta falla (bounce no convergido, cuadratura
         no convergida, residuo del ajuste > tol_fit, optimizador sin
         convergencia) o la preinscripción no está.

Nunca se ajustan tolerancias tras ver los números (candado:
tests/test_nucleation_gamma0_lock.py). El analizador lee TODAS las
reglas del JSON congelado, no de este fichero.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.basal import B_BAR, C0_DEFAULT, M_BAR  # noqa: E402
from core.nucleation import (  # noqa: E402
    N_DIM_DECLARED,
    b1_no_tilt,
    gamma0,
    path_deformation,
)
from core.s_clock import delta0_metastability_max_analytic  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_nucleation_gamma0"
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=OUTDIR.parent.parent).stdout.strip()


# ----------------------------------------------------------------- prereg
def prereg() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    e_bars = [0.5, 1.0, 2.0]
    grids = {}
    for e in e_bars:
        d0max = delta0_metastability_max_analytic(M_BAR, B_BAR, e, C0_DEFAULT)
        grids[str(e)] = {"delta0_max": d0max,
                         "delta0_grid": [float(x) for x in np.geomspace(1e-4, 0.9 * d0max, 8)],
                         "delta0_deformation": [0.01, float(0.5 * d0max)]}
    b1 = b1_no_tilt(M_BAR, B_BAR, C0_DEFAULT)
    doc = {
        "title": "Γ₀(δ₀) condicional a n_dim ∈ {1, 3, 4}: túnel de Gamow (n = 1), bounce O(n) (n = 3, 4), "
                 "deformación de camino en (ρ, χ) como segunda pasada",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/run_nucleation_gamma0.py prereg",
        "generator_commit_must_precede_freeze": True,
        "code_commit": _git_head(),
        "declared": {
            "n_dim": list(N_DIM_DECLARED),
            "G": 1.0, "hbar": 1.0, "theta_cut": 0.0,
            "shapes": {"m_bar": M_BAR, "b_bar": B_BAR, "C0": C0_DEFAULT},
            "e_bar": e_bars,
            "grids": grids,
            "prefactor_n1": "Gamow: ω_fv/(2π), ω_fv = √(V''(φ_fv)/G) = m̄·δ₀ (frecuencia de intento)",
            "prefactor_n34": "A dimensional, NO fijado: se publica Γ₀/A = e^{−B}",
            "path_deformation": {"n_modes": 2, "n_s": 1201, "parametrization":
                                 "θ(s) = θ_end·s + Σ a_k sin(kπs) recortado a [0, π/2]; ρ(s) lineal ρ_fv → ρ_esc(θ_end); "
                                 "B = 2∫√(2G·max(V − V_fv, 0))|dΦ/ds|ds; Nelder–Mead desde el rayo θ = 0"},
        },
        "predictions": {
            "n1": {"law": "B₁ = b₁·δ₀²·(1 + O(√δ₀)); Γ₀ = (m̄δ₀/2π)·e^{−B₁} → lineal en δ₀",
                   "b1_no_tilt": b1, "exponent_Gamma0_argument": 1.0, "exponent_B1_argument": 2.0},
            "n3": {"law": "B₃ → constante (argumento 3 − d = 0): Γ₀(0) = 0 solo si A(δ₀) → 0",
                   "exponent_B_argument": 0.0},
            "n4": {"law": "B₄ ∝ δ₀^{−1} (argumento 3 − d = −1): Γ₀(0) = 0 sin prefactor",
                   "exponent_B_argument": -1.0},
            "deformation": {"expectation_E13": "cociente B_min/B_ray ≃ 1: la inclinación −η·χ es máxima sobre θ = 0 "
                                               "dentro del dominio φ ≥ 0, el rayo recto es el candidato natural"},
            "note": "la inclinación corrige las leyes puras en O(√δ₀) relativo, AMPLIFICADO por T₀/altura de barrera "
                    "(la barrera vale 1–8 % de T₀): por eso los exponentes se miden en la década BAJA del barrido",
        },
        "rules": {
            "fit_window": "década baja: los TRES puntos más bajos de cada malla (δ₀ ∈ [1e-4, ~1e-3])",
            "fit_window_n_points": 3,
            "fit": "mínimos cuadrados en log–log; residuo relativo máximo del ajuste = tol_fit",
            "A": {"exponent_Gamma0_target": 1.0, "tol_A": 0.05, "b1_rel_tol": 0.10,
                  "text": "A si, para TODO ē, |p_Γ₀ − 1| ≤ tol_A y |B₁/δ₀²(δ₀ mín) − b₁| ≤ 0.10·b₁"},
            "B": {"exponent_B_target": -1.0, "tol_B": 0.25,
                  "text": "B si, para TODO ē, |p_B4 + 1| ≤ tol_B"},
            "C": {"ratio_threshold": 0.5, "text": "C si en algún (δ₀, ē) B_min/B_ray < 0.5 con el optimizador convergido"},
            "gates": {"bounce_converged_all": True, "gamow_quadrature_converged_all": True, "tol_fit": 0.10,
                      "optimizer_converged_all": True},
            "verdict": "tupla (A: sí/no, B: sí/no, C: sí/no) publicada entera; INDETERMINADO si alguna puerta falla; "
                       "la fila nucleacion-gamma0 queda condicional a n_dim (decisión B del autor); ningún desenlace "
                       "se convierte en afirmación sobre el tratado (E8, E13)",
        },
        "pilots_declared": {
            "what": "tres comprobaciones de instrumento ANTES de congelar, con n_grid = 4001 y trapecio: "
                    "B₁/δ₀² = 0.4966 en δ₀ = 1e-4 (ē = 1), bounce n = 4 en δ₀ = 1e-4 convergido en 43 disparos (~2 s), "
                    "deformación en δ₀ ∈ {0.01, 0.05} con cociente 1.000. Fijaron la cuadratura (quad adaptativa por la "
                    "singularidad √ en los extremos) y la ventana de ajuste; NINGUNA tolerancia se ajustó a ellos "
                    "(tol_A = 0.05 y tol_B = 0.25 se declaran a priori por el tamaño esperado de la corrección O(√δ₀))",
        },
        "what_this_cannot_decide": [
            "n_dim (decisión B del autor)", "el prefactor A de n ∈ {3, 4}", "cuál nucleación es la del tratado "
            "(instantón conservativo, túnel 0+1 o escape de Kramers)", "el diccionario t ↔ σ"],
    }
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (que contiene el generador). "
          f"sha256 de `preregistration.json`: `{sha}`.", "",
          "## Declarado", "",
          f"- n_dim ∈ {doc['declared']['n_dim']}; G = ħ = 1; corte θ = 0; formas m̄ = {M_BAR}, b̄ = {B_BAR}, C0 = {C0_DEFAULT}; ē ∈ {e_bars}.",
          "- Mallas δ₀ (8 puntos log entre 1e-4 y 0.9·δ₀_max(ē) = 0.9·0.1028·ē⁻²): " +
          "; ".join(f"ē = {e}: [{g['delta0_grid'][0]:.1e}, {g['delta0_grid'][-1]:.3e}]" for e, g in grids.items()) + ".",
          f"- Prefactor n = 1: {doc['declared']['prefactor_n1']}. n = 3, 4: {doc['declared']['prefactor_n34']}.",
          f"- Deformación: {doc['declared']['path_deformation']['parametrization']}.", "",
          "## Predicciones (argumento de escala)", "",
          f"- n = 1: {doc['predictions']['n1']['law']}; b₁(sin inclinación) = {b1:.4f}.",
          f"- n = 3: {doc['predictions']['n3']['law']}.", f"- n = 4: {doc['predictions']['n4']['law']}.",
          f"- Deformación: {doc['predictions']['deformation']['expectation_E13']}.",
          f"- {doc['predictions']['note']}.", "",
          "## Reglas (congeladas)", "",
          f"- Ventana de ajuste: {doc['rules']['fit_window']}. {doc['rules']['fit']}.",
          f"- **A**: {doc['rules']['A']['text']} (tol_A = {doc['rules']['A']['tol_A']}).",
          f"- **B**: {doc['rules']['B']['text']} (tol_B = {doc['rules']['B']['tol_B']}).",
          f"- **C**: {doc['rules']['C']['text']}.",
          f"- Puertas: bounce convergido en todos los puntos; cuadratura convergida; residuo del ajuste ≤ {doc['rules']['gates']['tol_fit']}; optimizador convergido.",
          f"- Veredicto: {doc['rules']['verdict']}.", "",
          "## Pilotos declarados", "", doc["pilots_declared"]["what"] + ".", "",
          "## Lo que no puede decidir", ""] + [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


# -------------------------------------------------------------------- run
def _load_prereg() -> dict:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción congelada (ejecuta prereg y COMMÍTELA antes)")
    return json.loads(PREREG.read_text(encoding="utf-8"))


def run() -> int:
    pre = _load_prereg()
    dec = pre["declared"]
    sh = dec["shapes"]
    kw = {"m_bar": sh["m_bar"], "b_bar": sh["b_bar"], "C0": sh["C0"]}
    rows, deform = [], []
    for e in dec["e_bar"]:
        g = dec["grids"][str(e)]
        for d0 in g["delta0_grid"]:
            for n in dec["n_dim"]:
                r = gamma0(d0, n, e_bar=e, **kw)
                rows.append({"e_bar": e, "delta0": d0, "n_dim": n, "B": r["B"], "Gamma0": r["Gamma0"],
                             "Gamma0_over_A": r.get("Gamma0_over_A"), "converged": r["converged"],
                             "b1_over_delta0_sq": r["detail"].get("b1_over_delta0_sq"),
                             "f0": r["detail"].get("f0")})
                print(f"ē = {e}, δ₀ = {d0:.3e}, n = {n}: B = {r['B']:.4e}, conv = {r['converged']}")
        for d0 in g["delta0_deformation"]:
            pd = path_deformation(d0, e_bar=e, n_modes=dec["path_deformation"]["n_modes"],
                                  n_s=dec["path_deformation"]["n_s"], **kw)
            deform.append(pd)
            print(f"ē = {e}, δ₀ = {d0:.3e}: B_min/B_ray = {pd['ratio_min_over_ray']:.4f}")
    RUNS.write_text(json.dumps({"preregistration_sha256": _sha(PREREG), "code_commit": _git_head(),
                                "executed_utc": datetime.now(timezone.utc).isoformat(),
                                "rows": rows, "deformation": deform}, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"corridas guardadas en {RUNS}")
    return 0


# ---------------------------------------------------------------- analyze
def _fit(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    p, c = np.polyfit(lx, ly, 1)
    resid = np.max(np.abs(np.exp(c + p * lx) / np.asarray(y, float) - 1.0))
    return float(p), float(resid)


def analyze() -> int:
    pre = _load_prereg()
    if not RUNS.exists():
        raise SystemExit("FALLO CERRADO: no hay corridas")
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    if runs["preregistration_sha256"] != _sha(PREREG):
        raise SystemExit("FALLO CERRADO: la preinscripción cambió después de las corridas")
    R = pre["rules"]
    rows = runs["rows"]
    per_e = {}
    gates = {"bounce_converged_all": all(r["converged"] for r in rows if r["n_dim"] in (3, 4)),
             "gamow_quadrature_converged_all": all(r["converged"] for r in rows if r["n_dim"] == 1),
             "optimizer_converged_all": all(d["optimizer_converged"] for d in runs["deformation"]),
             "fit_residual_ok": True}
    b1 = pre["predictions"]["n1"]["b1_no_tilt"]
    A_ok, B_ok = True, True
    for e in pre["declared"]["e_bar"]:
        sub = [r for r in rows if r["e_bar"] == e]
        low = sorted({r["delta0"] for r in sub})[:R["fit_window_n_points"]]
        out = {"delta0_low_window": low}
        for n in (1, 3, 4):
            pts = sorted([r for r in sub if r["n_dim"] == n], key=lambda r: r["delta0"])
            xs = [r["delta0"] for r in pts if r["delta0"] in low]
            yB = [r["B"] for r in pts if r["delta0"] in low]
            pB, resB = _fit(xs, yB)
            out[f"n{n}_exponent_B"] = pB
            out[f"n{n}_fit_residual_B"] = resB
            out[f"n{n}_B_all"] = [(r["delta0"], r["B"]) for r in pts]
            if resB > R["gates"]["tol_fit"]:
                gates["fit_residual_ok"] = False
            if n == 1:
                yG = [r["Gamma0"] for r in pts if r["delta0"] in low]
                pG, resG = _fit(xs, yG)
                out["n1_exponent_Gamma0"], out["n1_fit_residual_Gamma0"] = pG, resG
                b1_min = pts[0]["b1_over_delta0_sq"]
                out["n1_b1_at_delta0_min"], out["n1_b1_no_tilt"] = b1_min, b1
                ok = abs(pG - R["A"]["exponent_Gamma0_target"]) <= R["A"]["tol_A"] and abs(b1_min - b1) <= R["A"]["b1_rel_tol"] * b1
                out["A_this_e"] = bool(ok)
                A_ok &= ok
                if resG > R["gates"]["tol_fit"]:
                    gates["fit_residual_ok"] = False
            if n == 4:
                ok = abs(pB - R["B"]["exponent_B_target"]) <= R["B"]["tol_B"]
                out["B_this_e"] = bool(ok)
                B_ok &= ok
        per_e[str(e)] = out
    ratios = [d["ratio_min_over_ray"] for d in runs["deformation"]]
    C_ok = any(r < R["C"]["ratio_threshold"] for r in ratios)
    all_gates = all(gates.values())
    verdict = ({"A": bool(A_ok), "B": bool(B_ok), "C": bool(C_ok)} if all_gates else "INDETERMINADO")
    res = {"preregistration_sha256": runs["preregistration_sha256"], "code_commit_analysis": _git_head(),
           "analyzed_utc": datetime.now(timezone.utc).isoformat(), "gates": gates, "per_e_bar": per_e,
           "deformation": runs["deformation"], "deformation_ratio_min": float(min(ratios)),
           "verdict": verdict,
           "reading": ("cada letra es condicional a su n_dim: A habla de n = 1, B de n = 4, C del camino; ninguna decide "
                       "n_dim ni convierte el resultado en afirmación sobre el tratado (E8, E13)")}
    (OUTDIR / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Γ₀(δ₀) condicional a n_dim — resultado bajo la preinscripción `{runs['preregistration_sha256'][:12]}`", "",
          f"Corridas en el commit `{runs['code_commit'][:9]}`; análisis en `{res['code_commit_analysis'][:9]}`. "
          f"Puertas: {gates}.", "",
          f"## Veredicto: **{verdict}**", "", res["reading"] + ".", "",
          "| ē | p(Γ₀), n = 1 | B₁/δ₀² en δ₀ mín (b₁ = %.4f) | p(B), n = 3 | p(B), n = 4 | A | B |" % b1,
          "|---|---|---|---|---|---|---|"]
    for e, o in per_e.items():
        md.append(f"| {e} | {o['n1_exponent_Gamma0']:+.4f} | {o['n1_b1_at_delta0_min']:.4f} | {o['n3_exponent_B']:+.4f} | "
                  f"{o['n4_exponent_B']:+.4f} | {'sí' if o['A_this_e'] else 'no'} | {'sí' if o['B_this_e'] else 'no'} |")
    md += ["", "## Deformación del camino (segunda pasada, n = 1)", "", "| ē | δ₀ | B_ray | B_min | cociente | θ_end óptimo |", "|---|---|---|---|---|---|"]
    for d in runs["deformation"]:
        md.append(f"| {d['e_bar']} | {d['delta0']:.4g} | {d['B_ray']:.4e} | {d['B_min']:.4e} | {d['ratio_min_over_ray']:.4f} | {d['theta_end_opt']:.3f} |")
    md += ["", "## Barrido completo (B por n_dim)", "", "| ē | δ₀ | B₁ (n = 1) | Γ₀ (n = 1) | B₃ | B₄ | f₀(n = 4) |", "|---|---|---|---|---|---|---|"]
    for e in pre["declared"]["e_bar"]:
        for d0 in pre["declared"]["grids"][str(e)]["delta0_grid"]:
            rr = {r["n_dim"]: r for r in rows if r["e_bar"] == e and r["delta0"] == d0}
            md.append(f"| {e} | {d0:.3e} | {rr[1]['B']:.3e} | {rr[1]['Gamma0']:.3e} | {rr[3]['B']:.3e} | {rr[4]['B']:.3e} | {rr[4]['f0']:.4f} |")
    md += ["", "## Lo que no decide", ""] + [f"- {s}" for s in pre["what_this_cannot_decide"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"veredicto: {verdict}; puertas: {gates}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"prereg": prereg, "run": run, "analyze": analyze}.get(mode, lambda: print(__doc__) or 2)())
