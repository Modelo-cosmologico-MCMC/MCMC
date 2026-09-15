#!/usr/bin/env python
"""E4_Atlas — ejecuta bajo la preinscripción congelada: E4a (identidades
PPN del límite khronométrico, cota sobre α_a desde el dataset ppn_bounds,
colas de sonido en la cota con la escalera exacta, limpieza de los
Residuos) y E4b (cierre de E3b con ajuste e² + e⁴). Publica el artefacto.

Uso: python scripts/run_atlas_ppn.py
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
    alpha_a_max_from_ppn,
    eta_tail_coefficient,
    eta_tail_physical,
    khronon_cs2,
)
from mcmc_ontology.data_registry import require_available  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-15_mu_eta_atlas_ppn"
PREREG = OUTDIR / "preregistration.json"
H0_OVER_C = 3.336e-4


def load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción E4_Atlas")
    return (json.loads(PREREG.read_text(encoding="utf-8")),
            hashlib.sha256(PREREG.read_bytes()).hexdigest())


def R(x) -> sp.Rational:
    return sp.Rational(str(x))


def within(x, ref, tol) -> bool:
    return bool(np.isfinite(x) and abs(x / ref - 1.0) <= tol)


def main() -> int:
    T0 = time.time()
    prereg, psha = load_prereg()
    print(f"[prereg] {psha[:12]}…", flush=True)
    from validation.atlas_derivation import build_quadratic_action
    from validation.atlas_ppn_derivation import run_all
    from validation.atlas_tail_derivation import (
        ladder_point,
        ladder_summary,
        run_adiabatic,
    )

    # ------------------------------------------------------------ E4a
    a4 = prereg["E4a_ppn"]
    rules = a4["rules"]
    ident = run_all()
    flags = {k: (ident["khronometric_limit"].get(k, ident["mapping"].get(
        k, ident["aether_parametrization"].get(k)))) for k in a4["identities_to_lock"]}
    identities_pass = bool(all(flags.values()))
    print(f"  identidades PPN: {sum(bool(v) for v in flags.values())}/{len(flags)}", flush=True)

    raw = require_available("ppn_bounds")
    bdoc = json.loads((raw / "ppn_bounds_2026.json").read_text(encoding="utf-8"))
    bv = bdoc["values"]
    msha = hashlib.sha256((OUTDIR.parent.parent / "data" / "manifests" /
                           "ppn_bounds.json").read_bytes()).hexdigest()
    b1, b2 = bv["alpha1_llr"]["bound_abs"], bv["alpha2_solar_spin"]["bound_abs"]
    lam_nom = a4["lambda_nominal"]
    bound_nom = alpha_a_max_from_ppn(b1, b2, 1.0 + lam_nom)
    scan = [{"lambda": lv, **alpha_a_max_from_ppn(b1, b2, 1.0 + lv)} for lv in a4["lambda_scan"]]
    strong = {"alpha1_pulsars": bv["alpha1_pulsars"]["bound_abs"] / 4.0,
              "alpha2_pulsars": alpha_a_max_from_ppn(1.0, bv["alpha2_pulsars"]["bound_abs"],
                                                     1.0 + lam_nom)["alpha_a_max_from_alpha2"]}
    amax = bound_nom["alpha_a_max"]
    print(f"  α_a^max = {amax:.3e} (gobierna {bound_nom['governing']}; de α₁: "
          f"{bound_nom['alpha_a_max_from_alpha1']:.2e}, de α₂: "
          f"{bound_nom['alpha_a_max_from_alpha2']:.2e})", flush=True)

    S = build_quadratic_action()
    print(f"  acción cuadrática en {time.time()-T0:.0f}s", flush=True)
    tb = a4["tails_at_bound"]
    pts = [tb["ladder_point"]["lambda_K"], tb["ladder_point"]["alpha_a"]]
    ladders = []
    for lamK, alv in [pts] + tb["trend_points"]:
        r = ladder_point(S, R(lamK), R(alv))
        s = ladder_summary(r)
        s["tails"] = []
        for kk in tb["k_h_over_Mpc"]:
            e = H0_OVER_C / kk
            s["tails"].append({"k_h_over_Mpc": kk, "e": e,
                               "eta_minus_one_full": s["coef_eta_e2"] * e ** 2,
                               "mu_local_minus_one_full": s["coef_mu_local_e2"] * e ** 2,
                               "eta_minus_one_pole_only": float(eta_tail_physical(e, lamK, alv)),
                               "eta_minus_one_qs_truncated": eta_tail_coefficient(lamK) * e ** 2})
        s["cs2"] = khronon_cs2(lamK, 1.0, alv)
        ladders.append(s)
        print(f"  escalera ({lamK}, {alv:g}): coef_η = {s['coef_eta_e2']:.4f}, coef_µ_loc = "
              f"{s['coef_mu_local_e2']:.4f}, c_s² = {s['cs2']:.3g}, η−1(k=0.02) = "
              f"{s['tails'][0]['eta_minus_one_full']:.2e} (solo polo "
              f"{s['tails'][0]['eta_minus_one_pole_only']:.2e})", flush=True)
    tail_bound = ladders[0]["tails"][0]["eta_minus_one_full"]
    cleanliness = (amax / 2.0) / (1.5 * lam_nom)
    if not identities_pass:
        outcome_a = "C"
    elif amax <= rules["alpha_a_max_leq"] and tail_bound <= rules["tail_unobservable_leq"] \
            and cleanliness <= rules["residues_cleanliness_leq"]:
        outcome_a = "A"
    elif amax <= rules["alpha_a_max_leq"] and tail_bound >= rules["tail_observable_geq"]:
        outcome_a = "B"
    else:
        outcome_a = "INDETERMINADO"
    print(f"  E4a: η−1(k=0.02) en la cota = {tail_bound:.2e}, limpieza de los Residuos = "
          f"{cleanliness:.2e} → DESENLACE {outcome_a}", flush=True)

    # ------------------------------------------------------------ E4b
    b4 = prereg["E4b_e3b_closure"]
    pt = b4["point"]
    r_h = ladder_point(S, R(pt["lambda_K"]), R(pt["alpha_a"]))
    ce = float(sp.N(r_h["coef_eta"], 30))
    cm = float(sp.N(r_h["coef_mu_relloc"], 30))
    sv = b4["solver"]
    ad = run_adiabatic(S, r_h, b4["k_over_H0"], a_start=b4["a_start"],
                       e_fit_max=b4["e_fit_max"], rtol=sv["rtol"], atol=sv["atol"],
                       n_eval=sv["n_eval"])
    rb = b4["rules"]
    ce_qs = b4["expected_numbers"]["eta_qs_truncated_coefficient"]
    cm_qs = b4["expected_numbers"]["mu_local_qs_truncated_coefficient"]
    for row in ad["rows"]:
        row["ratio_eta_quartic_to_ladder"] = row["slope_eta_e2_quartic"] / ce
        row["ratio_mu_quartic_to_ladder"] = row["slope_mu_local_e2_quartic"] / cm
        row["match_ladder"] = bool(within(row["slope_eta_e2_quartic"], ce, rb["slope_rel_tol"])
                                   and within(row["slope_mu_local_e2_quartic"], cm, rb["slope_rel_tol"]))
        row["match_qs"] = bool(within(row["slope_eta_e2_quartic"], ce_qs, rb["slope_rel_tol"])
                               and within(row["slope_mu_local_e2_quartic"], cm_qs, rb["slope_rel_tol"]))
        row["purity_ok"] = bool(row["p_spread_std"] <= rb["purity_std_max"])
        row["pass"] = bool(row["match_ladder"] and row["purity_ok"] and row["solver_success"])
        print(f"    k = {row['k_over_H0']:.1f}: s_η = {row['slope_eta_e2_quartic']:.3f} "
              f"(×{row['ratio_eta_quartic_to_ladder']:.3f}), q_η = {row['coef_eta_e4']:.1f}; "
              f"s_µ = {row['slope_mu_local_e2_quartic']:.3f} (×{row['ratio_mu_quartic_to_ladder']:.3f})"
              f" → {'PASS' if row['pass'] else 'FAIL'}", flush=True)
    if all(r["pass"] for r in ad["rows"]):
        outcome_b = "A"
    elif all(r["match_qs"] for r in ad["rows"]):
        outcome_b = "B"
    else:
        outcome_b = "C"
    print(f"  E4b: DESENLACE {outcome_b}", flush=True)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {"outcome_E4a": outcome_a, "outcome_E4b": outcome_b,
           "preregistration_sha256": psha, "ppn_bounds_manifest_sha256": msha,
           "official_bytes_verified": bool(bdoc.get("official_bytes_verified", False)),
           "executed_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
           "runtime_seconds": round(time.time() - T0, 1),
           "E4a": {"identities": flags, "identities_pass": identities_pass,
                   "derivation": ident, "bounds_used": {"alpha1": b1, "alpha2": b2},
                   "alpha_a_bound_nominal": bound_nom, "alpha_a_bound_scan": scan,
                   "strong_field_indicative": strong, "ladders": ladders,
                   "eta_minus_one_at_bound_k002": tail_bound,
                   "residues_cleanliness": cleanliness, "rules": rules},
           "E4b": {"point": pt, "ladder_coefficients": {"coef_eta_e2": ce, "coef_mu_local_e2": cm},
                   "qs_truncated": {"coef_eta_e2": ce_qs, "coef_mu_local_e2": cm_qs},
                   "rows": ad["rows"], "rules": rb}}
    (OUTDIR / "mu_eta_atlas_ppn.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    word_a = {"A": "las colas de sonido del canal Atlas son inobservables y "
                   "G_cosmo/G_N − 1 → −(3/2)ε_K en forma limpia: el canal queda ENFOCADO "
                   "en el Contraste de los Residuos",
              "B": "HALLAZGO ESTRUCTURAL: la cola O(e²) de η no se apaga con α_a → 0; "
                   "se publica sin retocar",
              "C": "el mapeo khronométrico falla: las cotas PPN no se aplican",
              "INDETERMINADO": "ningún desenlace preinscrito se cumple"}[outcome_a]
    caveat = "" if doc["official_bytes_verified"] else \
        " — cotas TRANSCRITAS: verificación de bytes oficiales pendiente del autor"
    word_b = {"A": "cola física CONFIRMADA numéricamente con el ajuste e² + e⁴",
              "B": "ganan los coeficientes QS truncados: error en la escalera",
              "C": "sigue abierta"}[outcome_b]
    md = [f"# E4_Atlas — E4a: desenlace **{outcome_a}** · E4b: desenlace **{outcome_b}**\n",
          f"Preinscripción `{psha[:12]}` (congelada antes de ejecutar); manifest de cotas "
          f"`{msha[:12]}`; commit `{sha[:9]}`; {doc['runtime_seconds']} s.", "",
          f"**Lectura obligatoria E4a**: {word_a}{caveat}.", "",
          f"**Lectura obligatoria E4b**: {word_b}.", "",
          "## E4a — PPN del sector Atlas", "",
          f"Identidades: {sum(bool(v) for v in flags.values())}/{len(flags)} en PASS. "
          f"α₁ = −4α_a; α₂ = −α_a/2 + O(α_a²). Cotas (campo débil): |α₁| < {b1:g} (LLR), "
          f"|α₂| < {b2:g} (giro solar) ⟹ **α_a ≤ {amax:.2e}** (gobierna {bound_nom['governing']}; "
          f"de α₁: {bound_nom['alpha_a_max_from_alpha1']:.2e}). Campo fuerte (indicativo): "
          f"α_a < {strong['alpha1_pulsars']:.1e} (α̂₁), < {strong['alpha2_pulsars']:.1e} (α̂₂).", "",
          "| λ = λ_K − 1 | α_a^max (α₂) | α_a^max (α₁) |", "|---|---|---|"]
    for s_ in scan:
        md.append(f"| {s_['lambda']} | {s_['alpha_a_max_from_alpha2']:.3e} | "
                  f"{s_['alpha_a_max_from_alpha1']:.3e} |")
    md += ["", "### Colas en la cota (escalera exacta, z = 0, e = (H0/c)/k)", "",
           "| λ_K | α_a | c_s² | coef η−1 (e²) | coef µ_loc−1 (e²) | k | η−1 completo | η−1 solo polo | η−1 QS truncada |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s_ in ladders:
        for t_ in s_["tails"]:
            md.append(f"| {s_['lambda_K']} | {s_['alpha_a']:g} | {s_['cs2']:.3g} | "
                      f"{s_['coef_eta_e2']:.4f} | {s_['coef_mu_local_e2']:.4f} | {t_['k_h_over_Mpc']} | "
                      f"{t_['eta_minus_one_full']:.2e} | {t_['eta_minus_one_pole_only']:.2e} | "
                      f"{t_['eta_minus_one_qs_truncated']:.2e} |")
    md += ["", f"Limpieza de los Residuos: (α_a^max/2)/((3/2)ε_K) = {cleanliness:.2e}.", "",
           "## E4b — cierre de E3b (ajuste e² + e⁴)", "",
           f"Escalera: coef η = {ce:.4f}, coef µ_loc = {cm:.4f}; QS truncada {ce_qs:.3f}, {cm_qs:.3f}.", "",
           "| k/H0 | s_η | s_η/escalera | q_η (e⁴) | s_µ | s_µ/escalera | q_µ (e⁴) | std p | PASS |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r_ in ad["rows"]:
        md.append(f"| {r_['k_over_H0']:.1f} | {r_['slope_eta_e2_quartic']:.3f} | "
                  f"{r_['ratio_eta_quartic_to_ladder']:.3f} | {r_['coef_eta_e4']:.0f} | "
                  f"{r_['slope_mu_local_e2_quartic']:.3f} | {r_['ratio_mu_quartic_to_ladder']:.3f} | "
                  f"{r_['coef_mu_local_e4']:.0f} | {r_['p_spread_std']:.1e} | "
                  f"{'sí' if r_['pass'] else 'no'} |")
    md += ["", "Estatuto: derivación interna (E8) sin datos observacionales nuevos; cotas "
           "PPN transcritas con aviso de procedencia; fronteras declaradas en la "
           "preinscripción."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nE4a = {outcome_a}, E4b = {outcome_b}. Artefacto: {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
