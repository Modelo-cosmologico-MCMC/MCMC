#!/usr/bin/env python
"""E3_Atlas — ejecuta, bajo la preinscripción congelada, la escalera
exacta del modo creciente (E3a: orden dominante + residuos del polo) y el
arnés numérico con ICs adiabáticas (E3b), clasifica ambos desenlaces por
separado y publica el artefacto.

Uso: python scripts/run_mu_eta_atlas_tail.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_atlas import (  # noqa: E402
    eta_tail_coefficient,
    tail_pole_residues,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / \
    "2026-09-13_mu_eta_atlas_tail"
PREREG = OUTDIR / "preregistration.json"
H0_OVER_C = 3.336e-4          # h/Mpc — solo para las magnitudes ilustrativas


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción E3_Atlas")
    return (json.loads(PREREG.read_text(encoding="utf-8")),
            hashlib.sha256(PREREG.read_bytes()).hexdigest())


def R(x) -> sp.Rational:
    return sp.Rational(str(x))


def within(x, ref, tol) -> bool:
    return bool(np.isfinite(x) and abs(x / ref - 1.0) <= tol)


def main() -> int:
    T0 = time.time()
    prereg, psha = load_prereg()
    rules = prereg["outcomes_enumerated"]["rules"]
    print(f"[prereg] {psha[:12]}…", flush=True)

    from validation.atlas_derivation import build_quadratic_action
    from validation.atlas_tail_derivation import (
        ladder_point,
        ladder_summary,
        pole_residues,
        run_adiabatic,
        run_qs_ic_control,
    )

    S = build_quadratic_action()
    t_sym = time.time() - T0
    print(f"  acción cuadrática en {t_sym:.0f}s", flush=True)

    # ------------------------------------------------------------ E3a
    e3a = prereg["E3a_ladder"]
    tw = e3a["tower"]
    grid_rows, ladders = [], {}
    for lam, al in e3a["grid_lambda_alpha"]:
        r = ladder_point(S, R(lam), R(al), nlow=tw["nlow"], nhigh=tw["nhigh"],
                         prof=tw["prof"])
        ladders[(lam, al)] = r
        s = ladder_summary(r)
        s["leading_pass"] = bool(s["mu0_equals_1_over_1_minus_alpha_half"]
                                 and s["eta0_is_1"] and all(s["checks"].values()))
        grid_rows.append(s)
        print(f"  escalera ({lam}, {al}): coef_η = {s['coef_eta_e2']:.6f}, "
              f"coef_µ_loc = {s['coef_mu_local_e2']:.6f}, orden dominante "
              f"{'OK' if s['leading_pass'] else 'FALLA'}", flush=True)
    tol_r = rules["E3a_residue_rel_tol"]
    res_rows = []
    for al in e3a["residues"]["alphas"]:
        rr = pole_residues(S, R(al), h_steps=[R(h) for h in e3a["residues"]["h_steps"]],
                           nlow=tw["nlow"], nhigh=tw["nhigh"], prof=tw["prof"])
        rr["pass_P_eta"] = rr["P_eta_rel_err"] <= tol_r
        rr["pass_P_mu"] = rr["P_mu_rel_err"] <= tol_r
        rr["pass"] = bool(rr["pass_P_eta"] and rr["pass_P_mu"])
        res_rows.append(rr)
        print(f"  residuos α = {al}: P_η = {rr['P_eta']:.8f} (cerrado "
              f"{rr['P_eta_closed']:.8f}, err {rr['P_eta_rel_err']:.1e}); "
              f"P_µ = {rr['P_mu_local']:.8f} (cerrado {rr['P_mu_local_closed']:.8f}, "
              f"err {rr['P_mu_rel_err']:.1e}); Q_η = {rr['Q_eta']:.3f}, "
              f"Q_µ = {rr['Q_mu_local']:.3f}", flush=True)
    e3a_pass = all(g["leading_pass"] for g in grid_rows) and all(
        r["pass"] for r in res_rows)
    outcome_a = "A" if e3a_pass else "B"

    # ------------------------------------------------------------ E3b
    e3b = prereg["E3b_adiabatic_harness"]
    pt = e3b["point"]
    lam, al = pt["lambda_K"], pt["alpha_a"]
    r_h = ladders[(lam, al)]
    coef_eta_l = float(sp.N(r_h["coef_eta"], 30))
    coef_mu_l = float(sp.N(r_h["coef_mu_relloc"], 30))
    coef_eta_qs = eta_tail_coefficient(lam, 1.0)
    coef_mu_qs = -3 * (3 * lam - 1) / (2 - al)
    sv = e3b["solver"]
    print(f"  arnés adiabático en ({lam}, {al}), k = {e3b['k_over_H0']}…", flush=True)
    ad = run_adiabatic(S, r_h, e3b["k_over_H0"], a_start=e3b["a_start"],
                       e_fit_max=e3b["e_fit_max"], rtol=sv["rtol"], atol=sv["atol"],
                       n_eval=sv["n_eval"])
    tol_s = rules["E3b_slope_rel_tol"]
    lo, hi = rules["E3b_loglog_exponent_range"]
    for row in ad["rows"]:
        row["ratio_eta_to_ladder"] = row["slope_eta_e2"] / coef_eta_l
        row["ratio_mu_to_ladder"] = row["slope_mu_local_e2"] / coef_mu_l
        row["match_ladder"] = bool(within(row["slope_eta_e2"], coef_eta_l, tol_s)
                                   and within(row["slope_mu_local_e2"], coef_mu_l, tol_s))
        row["match_qs"] = bool(within(row["slope_eta_e2"], coef_eta_qs, tol_s)
                               and within(row["slope_mu_local_e2"], coef_mu_qs, tol_s))
        row["exponent_ok"] = bool(lo <= row["loglog_exponent_eta"] <= hi)
        row["purity_ok"] = bool(row["p_spread_std"] <= rules["E3b_purity_std_max"])
        row["pass"] = bool(row["match_ladder"] and row["exponent_ok"]
                           and row["purity_ok"] and row["solver_success"])
        print(f"    k = {row['k_over_H0']:.1f}: s_η = {row['slope_eta_e2']:.3f} "
              f"(escalera {coef_eta_l:.3f}, ×{row['ratio_eta_to_ladder']:.3f}), "
              f"s_µ = {row['slope_mu_local_e2']:.3f} (escalera {coef_mu_l:.3f}, "
              f"×{row['ratio_mu_to_ladder']:.3f}), exp {row['loglog_exponent_eta']:.3f}"
              f" → {'PASS' if row['pass'] else 'FAIL'}", flush=True)
    if all(r["pass"] for r in ad["rows"]):
        outcome_b = "A"
    elif all(r["match_qs"] for r in ad["rows"]):
        outcome_b = "B"
    else:
        outcome_b = "C"
    print("  control con ICs QS-consistentes (no vinculante)…", flush=True)
    control = run_qs_ic_control(S, lam, al, e3b["k_over_H0"])
    for c in control:
        print(f"    k = {c['k_over_H0']:.1f}: s_η = {c['slope_eta_e2']:.3f}, "
              f"exp {c['loglog_exponent_eta']:.2f}, η−1 ∈ [{c['eta_minus_one_min']:.1e}, "
              f"{c['eta_minus_one_max']:.1e}]", flush=True)

    # magnitudes ilustrativas (convención declarada: e = H0/(c k) en z = 0)
    mags = []
    for (lam_, al_), r in ladders.items():
        ce = float(sp.N(r["coef_eta"], 30))
        for kk in (0.02, 0.05, 0.1):
            e = H0_OVER_C / kk
            mags.append({"lambda_K": lam_, "alpha_a": al_, "k_h_over_Mpc": kk,
                         "e": e, "eta_minus_one_full_ladder": ce * e ** 2,
                         "eta_minus_one_pole_only": tail_pole_residues(al_)["P_eta"]
                         / (lam_ - 1) * e ** 2,
                         "eta_minus_one_qs_truncated": eta_tail_coefficient(lam_) * e ** 2})

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome_E3a": outcome_a, "outcome_E3b": outcome_b,
           "preregistration_sha256": psha,
           "executed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "runtime_seconds": round(time.time() - T0, 1),
           "symbolic_action_seconds": round(t_sym, 1),
           "E3a": {"pass": bool(e3a_pass), "grid": grid_rows, "residues": res_rows},
           "E3b": {"pass": outcome_b == "A", "point": pt,
                   "ladder_coefficients": {"coef_eta_e2": coef_eta_l,
                                           "coef_mu_local_e2": coef_mu_l},
                   "qs_truncated_coefficients": {"coef_eta_e2": coef_eta_qs,
                                                 "coef_mu_local_e2": coef_mu_qs},
                   "rows": ad["rows"], "pooled": ad["pooled"],
                   "a_start": ad["a_start"], "e_fit_max": ad["e_fit_max"],
                   "control_qs_ics_non_gating": control},
           "observational_magnitudes_illustrative": {
               "convention": "z = 0, e = (H0/c)/k con H0/c = 3.336e-4 h/Mpc; "
                             "materia dominante; ilustrativo, no contraste",
               "rows": mags},
           "rules": rules}
    (OUTDIR / "mu_eta_atlas_tail.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    word_b = {
        "A": "la cola física queda CONFIRMADA por integración completa con ICs "
             "adiabáticas; la discrepancia del arnés E2 de #18 era de condiciones "
             "iniciales (ver control), no de la derivación",
        "B": "ganan los coeficientes QS truncados: la escalera es errónea — "
             "SOSPECHA DE ERROR EN LA DERIVACIÓN PRIMERO; las formas cerradas se retiran",
        "C": "discrepancia ABIERTA: la escalera queda como derivación exacta "
             "pendiente de confirmación numérica independiente",
    }[outcome_b]
    md = [f"# E3_Atlas — E3a: desenlace **{outcome_a}** · E3b: desenlace **{outcome_b}**\n",
          f"Preinscripción `{psha[:12]}` (congelada antes de ejecutar); commit "
          f"`{sha[:9]}`; {doc['runtime_seconds']} s.", "",
          f"**Lectura obligatoria E3b**: {word_b}.", "",
          "## E3a — escalera exacta (ξ = 1)", "",
          "| λ_K | α_a | µ₀ = 1/(1−α/2) | η₀ = 1 | coef η−1 (e²) | coef µ_loc−1 (e²) | cabeceras |",
          "|---|---|---|---|---|---|---|"]
    for g in grid_rows:
        md.append(f"| {g['lambda_K']} | {g['alpha_a']} | "
                  f"{'✓' if g['mu0_equals_1_over_1_minus_alpha_half'] else '✗'} | "
                  f"{'✓' if g['eta0_is_1'] else '✗'} | {g['coef_eta_e2']:.6f} | "
                  f"{g['coef_mu_local_e2']:.6f} | "
                  f"{'✓' if all(g['checks'].values()) else '✗'} |")
    md += ["", "| α_a | P_η | 3α/(2−α) | err | P_µ | −P_η·p(2p−1)/3 | err | Q_η | Q_µ | QS/físico |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res_rows:
        md.append(f"| {r['alpha_a']} | {r['P_eta']:.8f} | {r['P_eta_closed']:.8f} | "
                  f"{r['P_eta_rel_err']:.1e} | {r['P_mu_local']:.8f} | "
                  f"{r['P_mu_local_closed']:.8f} | {r['P_mu_rel_err']:.1e} | "
                  f"{r['Q_eta']:.3f} | {r['Q_mu_local']:.3f} | "
                  f"{r['qs_over_physical_residue']:.2f} |")
    md += ["", "## E3b — arnés con ICs adiabáticas", "",
           f"Punto {pt}; escalera: coef η = {coef_eta_l:.4f}, coef µ_loc = "
           f"{coef_mu_l:.4f}; QS truncada: {coef_eta_qs:.3f}, {coef_mu_qs:.3f}.", "",
           "| k/H0 | e ajustado | s_η | s_η/escalera | s_µ | s_µ/escalera | exp log-log | std p | PASS |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in ad["rows"]:
        md.append(f"| {r['k_over_H0']:.1f} | [{r['e_range_fit'][0]:.4f}, "
                  f"{r['e_range_fit'][1]:.4f}] | {r['slope_eta_e2']:.3f} | "
                  f"{r['ratio_eta_to_ladder']:.3f} | {r['slope_mu_local_e2']:.3f} | "
                  f"{r['ratio_mu_to_ladder']:.3f} | {r['loglog_exponent_eta']:.3f} | "
                  f"{r['p_spread_std']:.1e} | {'sí' if r['pass'] else 'no'} |")
    md += ["", f"Agrupado: s_η = {ad['pooled']['slope_eta_e2']:.3f}, s_µ = "
           f"{ad['pooled']['slope_mu_local_e2']:.3f} ({ad['pooled']['n_points']} puntos).",
           "", "### Control no vinculante — ICs QS-consistentes de #18", "",
           "| k/H0 | s_η | exp log-log | η−1 min | η−1 max | fracción < 0 |",
           "|---|---|---|---|---|---|"]
    for c in control:
        md.append(f"| {c['k_over_H0']:.1f} | {c['slope_eta_e2']:.3f} | "
                  f"{c['loglog_exponent_eta']:.2f} | {c['eta_minus_one_min']:.1e} | "
                  f"{c['eta_minus_one_max']:.1e} | {c['fraction_negative']:.2f} |")
    md += ["", "## Magnitudes ilustrativas (z = 0, e = (H0/c)/k, ξ = 1)", "",
           "| λ_K | α_a | k [h/Mpc] | η−1 escalera | η−1 solo polo | η−1 QS truncada |",
           "|---|---|---|---|---|---|"]
    for mg in mags:
        md.append(f"| {mg['lambda_K']} | {mg['alpha_a']} | {mg['k_h_over_Mpc']} | "
                  f"{mg['eta_minus_one_full_ladder']:.2e} | "
                  f"{mg['eta_minus_one_pole_only']:.2e} | "
                  f"{mg['eta_minus_one_qs_truncated']:.2e} |")
    md += ["", "Estatuto: derivación interna (E8) sin datos; ξ = 1; partes "
           "regulares numéricas; fronteras declaradas en la preinscripción."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nE3a = {outcome_a}, E3b = {outcome_b}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
