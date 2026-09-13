#!/usr/bin/env python
"""Ejecuta la re-derivación simbólica (E1) y el arnés numérico (E2) del
canal Atlas BAJO la preinscripción congelada y publica el artefacto.

Falla cerrado sin preinscripción. Requiere sympy (extra [derivation]).

Uso: python scripts/run_mu_eta_atlas.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_atlas import (  # noqa: E402
    eta_tail_coefficient,
    growth_index_matter_era,
    mu_atlas_subhorizon,
)
from validation.atlas_derivation import run_all_symbolic, run_numeric  # noqa: E402

OUTDIR = (Path(__file__).resolve().parent.parent / "results"
          / "2026-09-13_mu_eta_atlas")
PREREG = OUTDIR / "preregistration.json"


def main() -> int:
    if not PREREG.exists():
        raise SystemExit("PREREGISTRATION_MISSING: congela primero con "
                         "scripts/run_mu_eta_atlas_prereg.py")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    e2 = prereg["E2_numerical_integration"]
    hp = e2["harness_point"]
    lam, xi_, al = hp["lambda_K"], hp["xi"], hp["alpha_a"]
    tol_mu = float(e2["rules"]["mu"].split("≤")[1])
    tol_eta = float(e2["rules"]["eta"].split("≤")[1])
    tol_p = float(e2["rules"]["growth_index"].split("≤")[1].split("con")[0])
    purity = float(e2["rules"]["mode_purity"].split("≤")[1].split("y")[0])

    print("[E1] re-derivación simbólica desde la acción…", flush=True)
    sym = run_all_symbolic(verbose=True)
    S = sym.pop("S")
    e1_flags = {
        "sello_newton_reproduced": sym["background"]["sello_newton_reproduced"],
        "delta_is_j0_plus_3phi": sym["quasi_static"]["delta_is_j0_plus_3phi"],
        "mu_matches_closed_form": sym["quasi_static"]["mu_matches_closed_form"],
        "eta_matches_closed_form": sym["quasi_static"]["eta_matches_closed_form"],
        "mu_sub_is_1_over_1_minus_alpha_over_2xi":
            sym["quasi_static"]["mu_sub_is_1_over_1_minus_alpha_over_2xi"],
        "eta_sub_is_1": sym["quasi_static"]["eta_sub_is_1"],
        "gr_limit_mu_eta_1": all(sym["quasi_static"]["gr_limit_mu_eta_1"]),
        "G_growth_equals_G_local": sym["static_G"]["G_growth_equals_G_local"],
        "G_growth_is_GB_over_xi_minus_alpha_half":
            sym["static_G"]["G_growth_is_GB_over_xi_minus_alpha_half"],
        "cs2_matches_closed_form": sym["khronon"]["cs2_matches_closed_form"],
        "cT2_is_xi": sym["tensor"]["cT2_is_xi"],
    }
    e1_pass = all(e1_flags.values())
    print(f"[E1] {'PASS' if e1_pass else 'FAIL'}: {e1_flags}", flush=True)

    print("[E2] integración numérica completa…", flush=True)
    rows = run_numeric(S, lam, xi_, al, e2["k_over_H0"],
                       a_start=float(e2["a_start"]),
                       window=tuple(e2["window_a"]))
    e2_flags = []
    for r in rows:
        ok = (abs(r["mu_ratio_minus_one"]) <= tol_mu
              and abs(r["eta_num_minus_one"]) <= tol_eta
              and abs(r["p_rel_err"]) <= tol_p
              and r["p_spread_std"] <= purity
              and r["delta_sign_flips_in_window"] == 0
              and r["solver_success"])
        r["pass"] = bool(ok)
        e2_flags.append(bool(ok))
        print(f"    k/H0 = {r['k_over_H0']:7.1f}: µ_num/µ_QS − 1 = "
              f"{r['mu_ratio_minus_one']:+.2e}, η_num − 1 = "
              f"{r['eta_num_minus_one']:+.2e} (cola QS {r['eta_qs_tail_at_window']:+.2e}), "
              f"p = {r['p_num']:.4f} vs {r['p_qs']:.4f} "
              f"(std {r['p_spread_std']:.1e}) → {'PASS' if ok else 'FAIL'}",
              flush=True)
    e2_pass = all(e2_flags)
    outcome = "A" if (e1_pass and e2_pass) else "B"

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {
        "outcome": outcome,
        "preregistration_sha256": prereg_sha,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "symbolic_seconds": sym["seconds"],
        "E1": {"pass": e1_pass, "flags": e1_flags,
               "background": sym["background"],
               "quasi_static": sym["quasi_static"],
               "static_G": sym["static_G"], "khronon": sym["khronon"],
               "tensor": sym["tensor"]},
        "E2": {"pass": e2_pass, "harness_point": hp, "rows": rows,
               "tolerances": {"mu": tol_mu, "eta": tol_eta, "p": tol_p,
                              "purity_std": purity}},
        "closed_form_numbers": {
            "mu_subhorizon_harness": mu_atlas_subhorizon(lam, xi_, al),
            "p_qs_harness": growth_index_matter_era(lam, xi_, al),
            "eta_tail_coefficient_harness": eta_tail_coefficient(lam, xi_),
            "eta_tail_coefficient_treatise_epsK": eta_tail_coefficient(
                1.0 + 0.012, 1.0)},
        "atlas_status_if_A": ("DERIVADO-NULO al orden dominante: la firma "
                              "sub-horizonte de (µ, η) es solo la del canal "
                              "Cronos"),
    }
    (OUTDIR / "mu_eta_atlas.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# Canal Atlas de µ/η derivado desde la acción — desenlace **{outcome}** "
        f"(E1 {'PASS' if e1_pass else 'FAIL'}, E2 {'PASS' if e2_pass else 'FAIL'})\n",
        f"Preinscripción: `{prereg_sha[:12]}` (congelada ANTES de ejecutar); "
        f"ejecución: commit `{sha[:9]}`; re-derivación simbólica en "
        f"{sym['seconds']} s. Sin datos: teoría pura.",
        "",
        "## E1 — identidades reproducidas desde la acción",
        "",
        "| identidad | reproducida |", "|---|---|",
    ] + [f"| {kk} | {'✓' if v else '✗'} |" for kk, v in e1_flags.items()] + [
        "",
        f"- µ_QS(e) = `{sym['quasi_static']['mu_qs']}`",
        f"- η_QS(e) = `{sym['quasi_static']['eta_qs']}`",
        f"- G_growth/G_B = `{sym['static_G']['G_growth_over_GB']}` = "
        f"G_local/G_B = `{sym['static_G']['G_local_over_GB']}`",
        f"- c_s² = `{sym['khronon']['cs2']}`; coef. cinético = "
        f"`{sym['khronon']['kinetic_coefficient']}`",
        f"- c_T² = `{sym['tensor']['cT2']}`",
        f"- η_QS en GR estricto a e finito: `{sym['quasi_static']['eta_qs_at_lamK_1_finite_e']}` "
        "— la degeneración 0/0 en λ_K = 1 (precisión (i)): el límite GR toma "
        "e → 0 primero",
        "",
        "## E2 — integración numérica completa (sin QS)",
        "",
        f"Punto λ_K = {lam}, ξ = {xi_}, α_a = {al}: µ_sub = "
        f"{doc['closed_form_numbers']['mu_subhorizon_harness']:.6f}, p_QS = "
        f"{doc['closed_form_numbers']['p_qs_harness']:.6f}.",
        "",
        "| k/H0 | e_fin | µ_num/µ_QS − 1 | η_num − 1 | cola QS de η (no puerta) | "
        "p_num | p_QS | std(p) | cambios de signo | veredicto |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ] + [
        f"| {r['k_over_H0']:g} | {r['e_end']:.2e} | {r['mu_ratio_minus_one']:+.2e} | "
        f"{r['eta_num_minus_one']:+.2e} | {r['eta_qs_tail_at_window']:+.2e} | "
        f"{r['p_num']:.4f} | {r['p_qs']:.4f} | {r['p_spread_std']:.1e} | "
        f"{r['delta_sign_flips_in_window']} | {'PASS' if r['pass'] else 'FAIL'} |"
        for r in rows
    ] + [
        "",
        "## Lectura",
        "",
        ("**Desenlace A**: la re-derivación desde la acción reproduce las "
         "formas cerradas y el sistema completo converge al orden dominante "
         "QS. El offset α_a/(2ξ) se cancela contra la G local: el canal "
         "Atlas no deja firma sub-horizonte en (µ, η) al orden dominante y "
         "la cola k² del canal Cronos queda como la ÚNICA firma sub-horizonte "
         "del sector perturbativo. ATLAS_STATUS pasa a DERIVADO-NULO."
         if outcome == "A" else
         "**Desenlace B**: alguna identidad o umbral no se cumple — error de "
         "implementación o de derivación, a discriminar por el procedimiento "
         "preinscrito; los umbrales no se tocan."),
        "",
        "**Precisión (ii)**: la cola QS de η lleva el polo 1/(λ_K−1) — "
        f"coeficiente {doc['closed_form_numbers']['eta_tail_coefficient_harness']:.1f} "
        "en el punto del arnés y "
        f"{doc['closed_form_numbers']['eta_tail_coefficient_treatise_epsK']:.0f} "
        "con ε_K = 0.012 — y el arnés NO la reproduce (esperado: la QS "
        "descarta términos del mismo orden). Los coeficientes completos de "
        "las colas son frontera declarada (sector de velocidades); el "
        "parámetro pequeño efectivo es e/√(λ_K−1).",
        "",
        "**Residuos refinados**: G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1) ≈ "
        "1 − (3/2)ε_K − α_a/2 — BBN acota la combinación. **Erratum "
        "candidata H.2.2** (decisión del autor): c_s² diverge, no se anula, "
        "cuando α_a → 0 a λ_K fijo.",
        "",
        "**Lo que NO queda derivado**: coeficientes de las colas; régimen "
        "superhorizonte; cotas PPN sobre (ε_K, α_a); acoplamiento fuerte.",
    ]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nDESENLACE {outcome} → {OUTDIR / 'mu_eta_atlas.json'}")
    return 0 if outcome == "A" else 1


if __name__ == "__main__":
    raise SystemExit(main())
