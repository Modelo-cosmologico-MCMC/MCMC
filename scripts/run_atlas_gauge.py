#!/usr/bin/env python
"""E5_Atlas — ejecuta bajo la preinscripción congelada: E5a (escalera
exacta en invariantes de gauge: η_N, µ_Δ, identidad de torre Φ_N = Ψ_N)
y E5b (arnés adiabático midiendo η_N y µ_Δ). Publica el artefacto.

Uso: python scripts/run_atlas_gauge.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_atlas import mu_delta_tail_leading  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-15_mu_eta_atlas_gauge"
PREREG = OUTDIR / "preregistration.json"
H0_OVER_C = 3.336e-4


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción E5_Atlas")
    return (json.loads(PREREG.read_text(encoding="utf-8")),
            hashlib.sha256(PREREG.read_bytes()).hexdigest())


def R(x) -> sp.Rational:
    return sp.Rational(str(x))


def main() -> int:
    T0 = time.time()
    prereg, psha = load_prereg()
    print(f"[prereg] {psha[:12]}…", flush=True)
    from validation.atlas_derivation import build_quadratic_action
    from validation.atlas_tail_derivation import (
        GAUGE_TRANSFORMATION,
        gauge_invariant_coefficients,
        gauge_invariant_summary,
        ladder_point,
        run_adiabatic_gauge_invariant,
    )

    S = build_quadratic_action()
    print(f"  acción cuadrática en {time.time()-T0:.0f}s", flush=True)

    # ------------------------------------------------------------ E5a
    a5 = prereg["E5a_ladder"]
    ra = a5["rules"]
    tw = a5["tower"]
    ladders = []
    ladder_cache = {}
    for lamK, alv in a5["points"]:
        r = ladder_point(S, R(lamK), R(alv), nlow=tw["nlow"], nhigh=tw["nhigh"], prof=tw["prof"])
        ladder_cache[(lamK, alv)] = r
        gi = gauge_invariant_coefficients(r, n_identity_max=tw["reliable_n_max"])
        s = gauge_invariant_summary(gi)
        e = H0_OVER_C / ra["tail_k_h_over_Mpc"]
        s["muD_minus_one_at_k002"] = s["coef_muD_e2"] * e ** 2
        s["muD_minus_one_at_k002_closed"] = float(mu_delta_tail_leading(e, lamK, alv))
        s["etaN_minus_one_at_k002"] = s["coef_etaN_e2"] * e ** 2
        s["closed_form_applies"] = bool(alv <= ra["muD_closed_form_alpha_max"])
        s["closed_form_pass"] = bool(abs(s["ratio_muD_to_closed"] - 1.0) <= ra["muD_closed_form_rel_tol"]) \
            if s["closed_form_applies"] else None
        ladders.append(s)
        print(f"  escalera ({lamK}, {alv:g}): η_N,0 = {s['etaN0']:.6g}, coef η_N(e²) = "
              f"{s['coef_etaN_e2']:.3e} [{'0 exacto' if s['coef_etaN_e2_is_zero'] else 'NO nulo'}]; "
              f"coef µ_Δ(e²) = {s['coef_muD_e2']:.6g} (forma cerrada {s['coef_muD_e2_closed_leading']:.6g}, "
              f"cociente {s['ratio_muD_to_closed']:.4f}); unitario η: {s['unitary_coef_eta_e2']:.4f}",
              flush=True)
    deep = []
    dd = a5["deep_identity"]
    for lamK, alv in dd["points"]:
        r16 = ladder_point(S, R(lamK), R(alv), nlow=tw["nlow"], nhigh=dd["nhigh"], prof=dd["prof"])
        gi16 = gauge_invariant_coefficients(r16, n_identity_max=max(dd["n_check"]))
        ident = {str(n): gi16["phi_equals_psi_tower"][n] for n in dd["n_check"]}
        deep.append({"lambda_K": lamK, "alpha_a": alv, "prof": dd["prof"], "nhigh": dd["nhigh"],
                     "phi_equals_psi_tower": ident, "all_pass": bool(all(ident.values())),
                     "coef_etaN_e2_is_zero": gi16["coef_etaN_e2_is_zero"]})
        print(f"  identidad profunda ({lamK}, {alv:g}): Φ_N,n = Ψ_N,n → {ident}", flush=True)

    etaN_clean = bool(all(s["coef_etaN_e2_is_zero"] and s["etaN0_is_1"] for s in ladders))
    closed_ok = bool(all(s["closed_form_pass"] for s in ladders if s["closed_form_applies"]))
    at_bound = next(s for s in ladders if [s["lambda_K"], s["alpha_a"]] == a5["expected_at_bound"]["point"])
    bound_ok = bool(abs(at_bound["muD_minus_one_at_k002"]) <= ra["muD_at_bound_k002_leq"])
    if not etaN_clean:
        outcome_a = "C"
    elif closed_ok and bound_ok:
        outcome_a = "A"
    elif not closed_ok or not bound_ok:
        outcome_a = "B"
    else:
        outcome_a = "INDETERMINADO"
    print(f"  E5a: η_N limpio = {etaN_clean}, forma cerrada = {closed_ok}, |µ_Δ − 1|(k = 0.02, cota) = "
          f"{at_bound['muD_minus_one_at_k002']:.2e} → DESENLACE {outcome_a}", flush=True)

    # ------------------------------------------------------------ E5b
    b5 = prereg["E5b_harness"]
    pt = b5["point"]
    key = (pt["lambda_K"], pt["alpha_a"])
    r_h = ladder_cache.get(key) or ladder_point(S, R(pt["lambda_K"]), R(pt["alpha_a"]))
    gi_h = gauge_invariant_coefficients(r_h)
    cmD = float(sp.N(gi_h["coef_muD_e2"], 30))
    ceU = gi_h["unitary_coef_eta_e2"]
    cmU = gi_h["unitary_coef_mu_local_e2"]
    sv = b5["solver"]
    ad = run_adiabatic_gauge_invariant(S, r_h, b5["k_over_H0"], a_start=b5["a_start"],
                                       a_fit_min=b5["a_fit_min"], e_fit_max=b5["e_fit_max"],
                                       rtol=sv["rtol"], atol=sv["atol"], n_eval=sv["n_eval"])
    rb = b5["rules"]
    for row in ad["rows"]:
        row["ratio_muD_to_ladder"] = row["slope_muD_e2"] / cmD
        row["ratio_etaN_to_unitary"] = row["slope_etaN_e2"] / ceU
        cu = row["control_unitary"]
        cu["ratio_eta_to_ladder"] = cu["slope_eta_e2"] / ceU
        cu["ratio_mu_local_to_ladder"] = cu["slope_mu_local_e2"] / cmU
        row["etaN_ok"] = bool(row["etaN_minus_one_max_abs"] <= rb["etaN_abs_max"])
        row["muD_ok"] = bool(abs(row["ratio_muD_to_ladder"] - 1.0) <= rb["muD_slope_rel_tol"])
        row["purity_ok"] = bool(row["p_spread_std"] <= rb["purity_std_max"])
        row["pass"] = bool(row["etaN_ok"] and row["muD_ok"] and row["purity_ok"] and row["solver_success"])
        row["etaN_matches_unitary"] = bool(abs(row["ratio_etaN_to_unitary"] - 1.0) <= rb["unitary_match_rel_tol"])
        print(f"    k = {row['k_over_H0']:.1f}: max|η_N − 1| = {row['etaN_minus_one_max_abs']:.2e}, "
              f"s_ηN = {row['slope_etaN_e2']:.3e}; s_µΔ = {row['slope_muD_e2']:.4f} "
              f"(×{row['ratio_muD_to_ladder']:.3f}); control unitario η ×{cu['ratio_eta_to_ladder']:.3f}, "
              f"µ_loc ×{cu['ratio_mu_local_to_ladder']:.3f} → {'PASS' if row['pass'] else 'FAIL'}", flush=True)
    if all(r_["pass"] for r_ in ad["rows"]):
        outcome_b = "A"
    elif all(r_["etaN_matches_unitary"] for r_ in ad["rows"]):
        outcome_b = "B"
    else:
        outcome_b = "C"
    print(f"  E5b: DESENLACE {outcome_b}", flush=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome_E5a": outcome_a, "outcome_E5b": outcome_b,
           "preregistration_sha256": psha,
           "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "runtime_seconds": round(time.time() - T0, 1),
           "gauge_transformation": GAUGE_TRANSFORMATION,
           "E5a": {"ladders": ladders, "deep_identity": deep, "etaN_clean_all_points": etaN_clean,
                   "closed_form_pass_small_alpha": closed_ok, "at_bound": at_bound,
                   "bound_pass": bound_ok, "rules": ra},
           "E5b": {"point": pt, "ladder_coefficients": {"coef_muD_e2": cmD, "unitary_coef_eta_e2": ceU,
                                                        "unitary_coef_mu_local_e2": cmU},
                   "rows": ad["rows"], "a_fit_min": ad["a_fit_min"], "rules": rb}}
    (OUTDIR / "mu_eta_atlas_gauge.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    word_a = {"A": "CIERRE gauge-invariante: η_N ≡ 1 en el sector Atlas (sin estrés anisótropo "
                   "lineal) y la única cola observable, la de µ_Δ, es O(α_a²): inobservable en la "
                   "cota PPN. La lectura física de E4a = B (cola de η de gauge unitario) queda SUPERADA",
              "B": "η_N limpio, pero sobrevive una cola gauge-invariante en µ_Δ fuera de la forma "
                   "cerrada: se publica su magnitud",
              "C": "el teorema del estrés anisótropo falla en algún punto: sospecha de error en la "
                   "derivación primero; E4a = B se mantiene",
              "INDETERMINADO": "ningún desenlace preinscrito se cumple"}[outcome_a]
    word_b = {"A": "invariantes CONFIRMADOS numéricamente (η_N ≡ 1 sobre la trayectoria; s_µΔ "
                   "coincide con la escalera)",
              "B": "la combinación de gauge no elimina la cola unitaria: error en la transformación "
                   "o en el teorema",
              "C": "sigue abierta"}[outcome_b]
    md = [f"# E5_Atlas — E5a: desenlace **{outcome_a}** · E5b: desenlace **{outcome_b}**\n",
          f"Preinscripción `{psha[:12]}` (congelada antes de ejecutar); commit `{sha[:9]}`; "
          f"{doc['runtime_seconds']} s.", "",
          f"**Lectura obligatoria E5a**: {word_a}.", "",
          f"**Lectura obligatoria E5b**: {word_b}.", "",
          "## E5a — escalera exacta en invariantes de gauge (Ψ_N = ψ + ḃ, Φ_N = φ − Hb, Δ = δ − 3H l1)", "",
          "| λ_K | α_a | c_s² | η_N,0 | coef η_N−1 (e²) | coef µ_Δ−1 (e²) | forma cerrada −α_a/c_s² | cociente | coef η unitario (E3/E4) | µ_Δ−1 @ k=0.02 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s_ in ladders:
        eta_cell = "0 (exacto)" if s_["coef_etaN_e2_is_zero"] else f"{s_['coef_etaN_e2']:.3e}"
        md.append(f"| {s_['lambda_K']} | {s_['alpha_a']:g} | {s_['cs2']:.3g} | {s_['etaN0']:g} | "
                  f"{eta_cell} | "
                  f"{s_['coef_muD_e2']:.4g} | {s_['coef_muD_e2_closed_leading']:.4g} | "
                  f"{s_['ratio_muD_to_closed']:.4f} | {s_['unitary_coef_eta_e2']:.4f} | "
                  f"{s_['muD_minus_one_at_k002']:.2e} |")
    md += ["", "Identidad profunda de torre (prof = 16): " + "; ".join(
        f"({d_['lambda_K']}, {d_['alpha_a']:g}) → Φ_N,n = Ψ_N,n en n = "
        f"{', '.join(n for n, v in d_['phi_equals_psi_tower'].items() if v)}"
        f"{'' if d_['all_pass'] else ' (FALLA en ' + ', '.join(n for n, v in d_['phi_equals_psi_tower'].items() if not v) + ')'}"
        for d_ in deep) + ".", "",
        f"En la cota PPN ((1.012, 1e-6), k = 0.02 h/Mpc, z = 0): µ_Δ − 1 = "
        f"{at_bound['muD_minus_one_at_k002']:.2e} (forma cerrada {at_bound['muD_minus_one_at_k002_closed']:.2e}); "
        f"η − 1 de gauge unitario publicado en E4: {a5['expected_at_bound']['etaN_minus_one_unitary_E4']:.2e}.", "",
        "## E5b — arnés adiabático en invariantes (ventana a ≥ 0.3, e ≤ 0.02, ajuste e² + e⁴)", "",
        f"Escalera en el punto: coef µ_Δ = {cmD:.4f}; unitarios η = {ceU:.4f}, µ_loc = {cmU:.4f}.", "",
        "| k/H0 | max\\|η_N − 1\\| | s_ηN | s_µΔ | s_µΔ/escalera | q_µΔ (e⁴) | control η unit. ×escalera | control µ_loc unit. ×escalera | std p | PASS |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r_ in ad["rows"]:
        cu = r_["control_unitary"]
        md.append(f"| {r_['k_over_H0']:.1f} | {r_['etaN_minus_one_max_abs']:.1e} | {r_['slope_etaN_e2']:.2e} | "
                  f"{r_['slope_muD_e2']:.4f} | {r_['ratio_muD_to_ladder']:.3f} | {r_['coef_muD_e4']:.0f} | "
                  f"{cu['ratio_eta_to_ladder']:.3f} | {cu['ratio_mu_local_to_ladder']:.3f} | "
                  f"{r_['p_spread_std']:.1e} | {'sí' if r_['pass'] else 'no'} |")
    md += ["", "Estatuto: derivación interna (E8) sin datos observacionales; control externo "
           "(sesión de verificación 15-sep) declarado en la preinscripción; fronteras en el JSON."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nE5a = {outcome_a}, E5b = {outcome_b}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
