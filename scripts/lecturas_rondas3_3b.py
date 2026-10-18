#!/usr/bin/env python
"""Lecturas a posteriori de las rondas 3 y 3b — SIN LETRA y SIN CORRIDAS (decisiones del autor del 18-oct sobre #51–#53).

    python scripts/lecturas_rondas3_3b.py      # escribe results/2026-10-18_lecturas_rondas3_3b/lecturas.{json,md}

Lee artefactos ya publicados y evalúa fórmulas cerradas; no corre nada de las rondas. Porta a este repositorio los guiones
del apéndice de la decisión del autor (comprobaciones_decisiones_51_53.py, q_equilibrio_saturante.py y
escaneo_control_saturante.py), que dieron las mismas cifras sobre estos artefactos. Dos bloques:

  * LÁMINAS (rondas 3 y 3b): la banda inestable bajo k_c, el suelo implícito de la banda, la regla de banda nueva
    (validation.qualification.band_rule_cell, no retroactiva) aplicada a todas las celdas, la regla UV con el cierre
    de la ventana, el k_c admisible por celda, por qué otra siembra no sirve, y las nueve celdas de la ronda 3 con la
    banda limpia frente a los criterios de A.
  * HALO (ronda 3): L1 el control saturante responde a la fuerza añadida; L2 ningún punto saturante viable es inestable
    en la zona resuelta; L3 t_weak no es un hito común entre amplitudes; L4 la puerta global de energía ordena los brazos
    al revés; L5 el newtoniano respira; L6 la firma ultravioleta a A_Sculptor.

Nada de esto da letra a las rondas 3 o 3b ni toca una regla congelada.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.qualification import (  # noqa: E402
    _check_no_letters,
    admissible_k_cut,
    band_growth_rate,
    band_rule_cell,
    sheet_cell_excluded_at_close,
    window_close_time,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-18_lecturas_rondas3_3b"
R3 = ROOT / "results" / "2026-09-28_cj_criterion_round3"
R3B = ROOT / "results" / "2026-10-18_cj_criterion_round3b_q1p5"
QUAL_SHEETS = ROOT / "results" / "2026-09-28_qualification_sheets" / "qualification.json"
QUAL_Q15 = ROOT / "results" / "2026-10-18_qualification_sheets_q1p5" / "qualification.json"
HALO = ROOT / "results" / "2026-09-28_halo_round3"

NG, N_C, A_NL, MARGIN = 512, 128, 1e-2, 10.0
C_KMS, G_KPC, H0 = 299792.458, 4.30091e-6, 67.86705532886631
SLOW_CELLS = ((1.2, 4), (2.0, 8))                      # las dos celdas de la ronda 3 sin la banda limpia


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


# ================================================================ láminas
def sheets() -> dict:
    diag = _load(R3B / "diagnostic_band_below_kc.json")
    band = {}
    for q in (1.2, 1.5, 2.0):
        b = band_growth_rate(q, N_C, NG)
        g4 = math.log(30.0) / window_close_time(q, 4, NG)
        band[str(q)] = {**b, "ratio_to_n4": b["gamma_band"] / g4}
    # suelo implícito: la banda (todos los modos ≤ n_c) cruza A_nl en t_cruce ⟹ A_suelo = A_nl·exp(−γ_banda·t_cruce)
    implied = []
    for r in diag["rows"]:
        if r["q"] > 1 and r["t_band_below_kc_over_A_nl"] is not None:
            g = band_growth_rate(r["q"], N_C, NG)["gamma_band"]
            implied.append({"round": r["round"], "run": r["run"], "t_cross": r["t_band_below_kc_over_A_nl"],
                            "a_floor_implied": A_NL * math.exp(-g * r["t_band_below_kc_over_A_nl"])})
    floor = float(np.exp(np.median(np.log([x["a_floor_implied"] for x in implied]))))
    rule = []
    for r in diag["rows"]:
        if r["q"] <= 1 or r["run"].endswith("beams512"):
            continue
        with_margin = band_rule_cell(r["q"], r["mode"], NG, floor, margin=MARGIN)
        no_margin = band_rule_cell(r["q"], r["mode"], NG, floor, margin=1.0)
        rule.append({"round": r["round"], "cell": [r["q"], r["mode"]], "t_close": with_margin["t_close"],
                     "efolds_to_close": with_margin["efolds_to_close"], "predicted_band_at_close": with_margin["predicted_band_at_close"],
                     "measured_band_at_close": r["band_below_kc_at_close"], "excluded_margin10": with_margin["excluded"],
                     "excluded_no_margin": no_margin["excluded"], "diagnostic_closed_before_band": r["window_closed_before_band_nonlinear"]})
    q2n4 = band_rule_cell(2.0, 4, NG, floor, margin=MARGIN)
    rule.append({"round": "ronda 3 (excluida a priori, sin corrida)", "cell": [2.0, 4], "t_close": q2n4["t_close"],
                 "efolds_to_close": q2n4["efolds_to_close"], "predicted_band_at_close": q2n4["predicted_band_at_close"],
                 "measured_band_at_close": None, "excluded_margin10": q2n4["excluded"],
                 "excluded_no_margin": band_rule_cell(2.0, 4, NG, floor, margin=1.0)["excluded"], "diagnostic_closed_before_band": None})
    # regla UV de #46 con el cierre de la ventana (γ_UV medido con k_c = ½k_Nyq en la cualificación de la misma q)
    qs, q15 = _load(QUAL_SHEETS), _load(QUAL_Q15)
    pre3 = _load(R3 / "preregistration.json")
    uv = []
    for q, n in ((1.2, 4), (2.0, 8), (1.5, 4), (1.5, 8)):
        src = q15 if q == 1.5 else qs
        g_uv = src["limits"][f"gamma_uv_max_ng512_nb1024_kc0.5_q{q}"]["value"]
        noise_meas = src["limits"]["uv_noise_max_ng512_nb1024_kc0.5"]["value"]
        seed = pre3["instrument"]["seed_amp_by_mode"][str(n)]
        noise = max(noise_meas, seed ** 2)
        t_win = math.log(10.0) * window_close_time(q, n, NG) / math.log(30.0)
        r = sheet_cell_excluded_at_close(g_uv, q, n, NG, A_NL, noise)
        uv.append({"cell": [q, n], "gamma_uv": g_uv, "gamma_uv_times_t_window": g_uv * t_win,
                   "gamma_uv_times_t_close": r["gamma_uv_times_t_close"], "ln_budget": r["ln_budget"], "excluded_at_close": r["excluded"]})
    kcut = {f"{q}_{n}": admissible_k_cut(q, n, NG, floor, margin=MARGIN)["n_c"] for q in (1.2, 1.5, 2.0) for n in (4, 8, 16, 32)}
    slow = band_rule_cell(1.5, 4, NG, floor, margin=1.0)
    need_floor = A_NL * math.exp(-slow["efolds_to_close"])
    # ronda 3: criterios de A sobre las once celdas publicadas y sobre las nueve con la banda limpia
    t3 = _load(R3 / "cj_criterion_round3.json")["table"]
    ratios = {(r["q"], r["mode"]): r["ratio_measured_over_instrument"] for r in t3 if r.get("ratio_measured_over_instrument") is not None}
    robust = {}
    for label, drop in (("once", ()), ("nueve", SLOW_CELLS)):
        per_q = {}
        for q in (1.2, 2.0):
            vals = [c for (qq, n), c in sorted(ratios.items()) if qq == q and (qq, n) not in drop]
            per_q[str(q)] = {"ratios": vals, "max_abs_dev": max(abs(1.0 - v) for v in vals),
                             "rel_spread": (max(vals) - min(vals)) / float(np.mean(vals))}
        robust[label] = per_q
    growth08 = [r["growth_factor"] for r in t3 if r["q"] == 0.8]
    return {"band": band, "implied_floor": {"runs": implied, "median": floor},
            "band_rule": {"a_floor": floor, "margin": MARGIN, "ln_budget_margin10": math.log(A_NL / (MARGIN * floor)),
                          "ln_budget_no_margin": math.log(A_NL / floor), "cells": rule,
                          "excluded_margin10": sorted([tuple(x["cell"]) for x in rule if x["excluded_margin10"]])},
            "uv_rule_at_close": uv, "admissible_n_c": kcut,
            "seed_argument": {"cell": [1.5, 4], "t_close": slow["t_close"], "efolds_to_close": slow["efolds_to_close"],
                              "floor_needed_no_margin": need_floor},
            "round3_robustness": {**robust, "q0.8_growth_factors": growth08, "dropped": [list(c) for c in SLOW_CELLS]}}


# ================================================================ halo
def _dlog(a: float, b: float) -> float:
    return math.log10(a / b)


def _nfw():
    rho_crit = 3.0 * (H0 / 1000.0) ** 2 / (8.0 * math.pi * G_KPC)
    r200 = (3.0 * 1e11 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    rs = r200 / 10.0
    rho_s = 1e11 / (4.0 * math.pi * rs ** 3 * (math.log(11.0) - 10.0 / 11.0))

    def rho(r):                                        # M☉/kpc³
        return rho_s / ((r / rs) * (1.0 + r / rs) ** 2)

    def mass(r):
        return 4.0 * math.pi * rho_s * rs ** 3 * (math.log(1.0 + r / rs) - (r / rs) / (1.0 + r / rs))

    def sigma2(r):                                     # Jeans isótropo newtoniano
        return quad(lambda x: rho(x) * G_KPC * mass(x) / x ** 2, r, 50.0 * r200, limit=400)[0] / rho(r)
    return rho, sigma2


def _q_equilibrium(rho_pc: float, sigma_n2: float, eps_max: float, rho_star: float) -> float:
    """q con σ de equilibrio Newton + Cronos: σ² = σ_N² + P_C/ρ, P_C = c²(ρε − G(ρ)), G' = ε."""
    from dynamics.epsilon_c_saturating import deps_c_drho, eps_c
    g_int = quad(lambda x: float(eps_c(x, eps_max, rho_star)), 0.0, rho_pc)[0]
    p_c = C_KMS ** 2 * (float(eps_c(rho_pc, eps_max, rho_star)) - g_int / rho_pc)
    return C_KMS ** 2 * rho_pc * float(deps_c_drho(rho_pc, eps_max, rho_star)) / (sigma_n2 + p_c)


def halo() -> dict:
    from dynamics.epsilon_c_saturating import (
        deps_c_drho,
        oort_bound_on_deps,
        q_local,
        stability_systems,
    )
    pre = _load(HALO / "preregistration.json")
    runs = {p.stem: _load(p) for p in sorted((HALO / "runs").glob("*.json"))}
    newton, sub, above = runs["newton_N1e6"], runs["sat_sub_ds0.04"], runs["sat_above_ds0.04"]
    # ---- L1: respuesta a la fuerza añadida
    series, early = [], []
    for sn, ss, sa in zip(newton["snapshots"], sub["snapshots"], above["snapshots"]):
        t = sn["t_gyr"] * 1e3
        row = {"t_myr": t, "sub_01": _dlog(ss["M_within"]["0.1"], sn["M_within"]["0.1"]),
               "above_01": _dlog(sa["M_within"]["0.1"], sn["M_within"]["0.1"]),
               "sub_04": _dlog(ss["M_within"]["0.4"], sn["M_within"]["0.4"]),
               "above_04": _dlog(sa["M_within"]["0.4"], sn["M_within"]["0.4"])}
        row["ratio_01"] = row["above_01"] / row["sub_01"] if row["sub_01"] != 0.0 else None
        (early if t < 0.5 else series).append(row)
    sat = pre["saturating_control"]
    rho, sig2 = _nfw()
    qp = sat["points"]["above"]["q_profile"]
    r_q, q_q = np.array([x["r_kpc"] for x in qp]), np.array([x["q"] for x in qp])
    i = int(np.where(q_q < 1.0)[0][0])
    r_q1 = math.exp(np.interp(0.0, [math.log(q_q[i]), math.log(q_q[i - 1])], [math.log(r_q[i]), math.log(r_q[i - 1])]))
    q_eq = {}
    for name in ("sub", "above"):
        p = sat["points"][name]
        prof = []
        for row in p["q_profile"]:
            rho_pc = p["rho_star"] * rho(row["r_kpc"]) / rho(0.1)                   # ρ_NFW con ρ(0.1 kpc) = ρ*
            sig_n2 = C_KMS ** 2 * rho_pc * float(deps_c_drho(rho_pc, p["eps_max"], p["rho_star"])) / row["q"]
            prof.append({"r_kpc": row["r_kpc"], "q_newton": row["q"], "q_eq": _q_equilibrium(rho_pc, sig_n2, p["eps_max"], p["rho_star"])})
        q_eq[name] = {"eps_max": p["eps_max"], "q_eq_max": max(x["q_eq"] for x in prof),
                      "profile_inner": [x for x in prof if x["r_kpc"] <= 0.5]}
    l1 = {"series": series, "early_below_0p5_myr": early, "eps_max_ratio": above["eps_max_declared"] / sub["eps_max_declared"],
          "D_F_at_soft": {"sub": sub["events"][0]["D_F_at_soft"], "above": above["events"][0]["D_F_at_soft"]},
          "above_q_newton_gt1_below_kpc": r_q1, "q_equilibrium": q_eq}
    # ---- L2: ¿algún punto saturante viable inestable en la zona resuelta?
    rr = np.geomspace(0.02, 2.0, 60)
    rho_pc = np.array([rho(r) * 1e-9 for r in rr])
    s2 = np.array([sig2(r) for r in rr])
    oort = oort_bound_on_deps()
    ext = [s for s in stability_systems() if not s["system"].startswith("halo NFW")]

    def viable(em, rst):
        return bool(deps_c_drho(oort["rho0_msun_pc3"], em, rst) <= oort["deps_max_at_rho0"]) and \
            all(bool(q_local(s["rho"], s["sigma2"], em, rst) < 1.0) for s in ext)
    scan = {"rho_nfw_0p1": rho(0.1) * 1e-9, "sigma_n_0p1": math.sqrt(sig2(0.1)), "rows": []}
    for r_u in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        rst = rho(r_u) * 1e-9
        ok = [em for em in np.geomspace(1e-10, 1e-5, 101) if viable(em, rst)]
        if not ok:
            scan["rows"].append({"r_u": r_u, "rho_star": rst, "eps_max_viable": None})
            continue
        em = float(max(ok))
        q = np.array([_q_equilibrium(rp, sn2, em, rst) for rp, sn2 in zip(rho_pc, s2)])
        scan["rows"].append({"r_u": r_u, "rho_star": rst, "eps_max_viable": em,
                             **{f"q_eq_max_r_ge_3eps_{e}": float(q[rr >= 3.0 * e].max()) for e in (0.1, 0.05)},
                             **{f"unstable_resolved_{e}": bool(np.any((q > 1.0) & (rr >= 3.0 * e))) for e in (0.1, 0.05)}})
    # ---- L3: t_weak entre amplitudes
    rcj = pre["prediction_from_criterion"]
    l3 = {"r_CJ_A_S": rcj["r_CJ_kpc_A_sculptor"], "r_CJ_0p05": rcj["r_CJ_kpc_0p05"], "density_factor": (1.0 / 0.05) ** (2.0 / 3.0),
          "free_fall_ratio": 0.05 ** -0.5, "by_eps": {}}
    for e in ("0.2", "0.1", "0.05", "0.025"):
        a, b = runs[f"sweep_AS_eps{e}"], runs[f"sweep_b005_eps{e}"]
        b_exit = "régimen" in (b["stop_reason"] or "")
        ta, tb = a["t_final_gyr"] * 1e3, b["t_final_gyr"] * 1e3
        l3["by_eps"][e] = {"t_weak_A_S": ta, "dlogM01_A_S": _dlog(a["snapshots"][-1]["M_within"]["0.1"], a["snapshots"][0]["M_within"]["0.1"]),
                           "b005_exits": b_exit, "t_b005": tb, "eps_c_max_b005": b["snapshots"][-1]["eps_c_max"],
                           "dlogM01_b005": _dlog(b["snapshots"][-1]["M_within"]["0.1"], b["snapshots"][0]["M_within"]["0.1"]),
                           "ratio_t_weak": tb / ta if b_exit else None}
    # ---- L4: |ΔE_self|/|ΔU_C| en la parada (y en el primer tramo de los brazos a 0.05·A_S)
    l4 = {}
    for name in ("sweep_AS_eps0.2", "sweep_AS_eps0.1", "sweep_AS_eps0.05", "sweep_AS_eps0.025", "series_AS_ds0.02", "series_AS_ds0.08",
                 "ctrlN_AS_N1e5", "sweep_b005_eps0.2", "sweep_b005_eps0.1", "sweep_b005_eps0.05", "sweep_b005_eps0.025", "ctrlN_b005_N1e5"):
        d = runs[name]
        e0, e1, e_1 = d["snapshots"][0]["energy"], d["snapshots"][-1]["energy"], d["snapshots"][1]["energy"]
        d_e, d_u = abs(e1["E_self"] - e0["E_self"]), abs(e1["U_self"] - e0["U_self"])
        l4[name] = {"N": d["N"], "dt_max_myr": d["dt_max_myr"], "n_steps": d["n_steps"], "dE_rel": d_e / abs(e0["E_self"]),
                    "eta_E": d_e / d_u, "eta_E_first_segment": abs(e_1["E_self"] - e0["E_self"]) / abs(e_1["U_self"] - e0["U_self"]),
                    "t_first_segment_myr": d["snapshots"][1]["t_gyr"] * 1e3}
    # ---- L5: el newtoniano respira
    m01 = [_dlog(s["M_within"]["0.1"], newton["snapshots"][0]["M_within"]["0.1"]) for s in newton["snapshots"]]
    i_min = int(np.argmin(m01))
    l5 = {"dlogM01_min": min(m01), "t_min_myr": newton["snapshots"][i_min]["t_gyr"] * 1e3,
          "dlogM04_final": _dlog(newton["snapshots"][-1]["M_within"]["0.4"], newton["snapshots"][0]["M_within"]["0.4"]),
          "newtonian_arms": [k for k in runs if k.startswith("newton")]}
    # ---- L6: firma ultravioleta a A_Sculptor
    eps = ["0.2", "0.1", "0.05", "0.025"]
    tw = {e: runs[f"sweep_AS_eps{e}"]["t_final_gyr"] * 1e3 for e in eps}
    td = {"0.08": runs["series_AS_ds0.08"]["t_final_gyr"] * 1e3, "0.04": tw["0.1"], "0.02": runs["series_AS_ds0.02"]["t_final_gyr"] * 1e3}
    l6 = {"p_eps_local": {f"{eps[i]}→{eps[i + 1]}": math.log(tw[eps[i]] / tw[eps[i + 1]]) / math.log(2.0) for i in range(3)},
          "eps_c_max_t0": {e: runs[f"sweep_AS_eps{e}"]["events"][0]["eps_c_max"] for e in eps},
          "factor_to_threshold": {e: 1e-3 / runs[f"sweep_AS_eps{e}"]["events"][0]["eps_c_max"] for e in eps},
          "t_weak_by_ds": td, "p_ds_local": {"0.08→0.04": math.log(td["0.08"] / td["0.04"]) / math.log(2.0),
                                             "0.04→0.02": math.log(td["0.04"] / td["0.02"]) / math.log(2.0)}}
    return {"L1_saturating_responds_to_added_force": l1, "L2_no_viable_unstable_point": scan, "L3_t_weak_not_common_milestone": l3,
            "L4_global_energy_gate_inverts": l4, "L5_newtonian_breathes": l5, "L6_uv_signature_A_S": l6}


def _md(doc: dict) -> str:
    s, h = doc["sheets"], doc["halo"]
    out = ["# Lecturas a posteriori de las rondas 3 y 3b (sin letra, sin corridas)", "",
           f"{doc['status']} Commit `{doc['code_commit'][:9]}`. Porta los guiones del apéndice de la decisión del autor del 18-oct.", "",
           "## Láminas", "", "| q | modos inestables | más rápido | tasa / tasa(n = 4) |", "|---|---|---|---|"]
    out += [f"| {q} | n ≤ {b['n_unstable_max']} | n = {b['n_fastest']} | {b['ratio_to_n4']:.2f} |" for q, b in s["band"].items()]
    br = s["band_rule"]
    out += ["", f"Suelo implícito de la banda (mediana de {len(s['implied_floor']['runs'])} corridas): A_suelo ≈ {s['implied_floor']['median']:.1e}. "
            f"Presupuesto con margen ×10: {br['ln_budget_margin10']:.1f} e-pliegues; sin margen: {br['ln_budget_no_margin']:.1f}.", "",
            "| celda | t_cierre | e-pliegues | banda prevista | banda medida | excluida (×10) |", "|---|---|---|---|---|---|"]
    for c in br["cells"]:
        meas = "—" if c["measured_band_at_close"] is None else f"{c['measured_band_at_close']:.1e}"
        out.append(f"| ({c['cell'][0]}, {c['cell'][1]}) | {c['t_close']:.3f} | {c['efolds_to_close']:.1f} | {c['predicted_band_at_close']:.1e} | "
                   f"{meas} | {'sí' if c['excluded_margin10'] else 'no'} |")
    out += ["", "Regla UV con el cierre de la ventana: " + "; ".join(
        f"({u['cell'][0]}, {u['cell'][1]}) γ_UV·t_ventana = {u['gamma_uv_times_t_window']:.2f}, γ_UV·t_cierre = {u['gamma_uv_times_t_close']:.2f} "
        f"frente a {u['ln_budget']:.2f}" for u in s["uv_rule_at_close"]) + ".", "",
        "n_c admisible por celda (margen ×10; 128 = el instrumento de las rondas 3 y 3b): "
        + "; ".join(f"({k.split('_')[0]}, {k.split('_')[1]}) {v}" for k, v in s["admissible_n_c"].items()) + ".", "",
        f"Siembra: la ventana de (1.5, 4) cierra en t = {s['seed_argument']['t_close']:.3f} con cualquier amplitud; la banda hace "
        f"{s['seed_argument']['efolds_to_close']:.1f} e-pliegues y haría falta A_suelo < {s['seed_argument']['floor_needed_no_margin']:.1e}.", ""]
    rb = s["round3_robustness"]
    for label in ("once", "nueve"):
        out.append(f"Ronda 3, {label} celdas: " + "; ".join(
            f"q = {q}: máx |1 − c| = {v['max_abs_dev']:.3f}, dispersión {v['rel_spread']:.3f}" for q, v in rb[label].items())
            + f"; q = 0.8 crecimiento {max(rb['q0.8_growth_factors']):.1f} (≤ 3).")
    l1, l2, l3, l4, l5, l6 = (h[k] for k in ("L1_saturating_responds_to_added_force", "L2_no_viable_unstable_point", "L3_t_weak_not_common_milestone",
                                             "L4_global_energy_gate_inverts", "L5_newtonian_breathes", "L6_uv_signature_A_S"))
    out += ["", "## Halo, ronda 3", "",
            "**L1.** Δ_C(<0.1) (dex) de «sub» y «above» frente al newtoniano: " + "; ".join(
                f"{r['t_myr']:.2f} Myr {r['sub_01']:+.3f} / {r['above_01']:+.3f}" for r in l1["series"][:4])
            + f"; cociente de ε_max {l1['eps_max_ratio']:.2f}; D_F(ε_soft, 0) = {l1['D_F_at_soft']['sub']:.2f} / {l1['D_F_at_soft']['above']:.2f}; "
              f"«above» con q_N > 1 solo en r ≲ {l1['above_q_newton_gt1_below_kpc']:.3f} kpc; q_eq máx = "
              f"{l1['q_equilibrium']['sub']['q_eq_max']:.2f} / {l1['q_equilibrium']['above']['q_eq_max']:.2f}.",
            "", "**L2.** " + "; ".join(
                f"ρ* = ρ_NFW({r['r_u']}): ε_max viable {r['eps_max_viable']:.2e}, q_eq máx {r['q_eq_max_r_ge_3eps_0.1']:.2f} (ε_soft 0.1) / "
                f"{r['q_eq_max_r_ge_3eps_0.05']:.2f} (0.05)" for r in l2["rows"] if r["eps_max_viable"] is not None) + ".",
            "", f"**L3.** r_CJ = {l3['r_CJ_A_S']:.3f} / {l3['r_CJ_0p05']:.3f} kpc; densidad extra ×{l3['density_factor']:.2f}; " + "; ".join(
                f"ε_soft {e}: A_S {v['t_weak_A_S']:.4f} Myr, 0.05·A_S " + (f"{v['t_b005']:.3f} Myr (cociente {v['ratio_t_weak']:.0f})" if v["b005_exits"]
                                                                          else f"sin salida en {v['t_b005']:.1f} Myr")
                for e, v in l3["by_eps"].items()) + f" (caída libre {l3['free_fall_ratio']:.2f}).",
            "", "**L4.** η_E = |ΔE_self|/|ΔU_C|: " + "; ".join(f"{k} {v['eta_E']:.3f}" for k, v in l4.items()) + ".",
            "", f"**L5.** Newtoniano: Δlog M(<0.1) mínimo {l5['dlogM01_min']:+.3f} dex (t = {l5['t_min_myr']:.1f} Myr); M(<0.4) final "
                f"{l5['dlogM04_final']:+.3f} dex; brazos newtonianos: {l5['newtonian_arms']}.",
            "", "**L6.** p_ε local " + ", ".join(f"{k}: {v:.2f}" for k, v in l6["p_eps_local"].items()) + "; p_ds " +
            ", ".join(f"{k}: {v:.2f}" for k, v in l6["p_ds_local"].items()) + ".", ""]
    return "\n".join(out)


def main() -> int:
    doc = {"artifact": "lecturas a posteriori sin letra y sin corridas (decisiones del autor del 18-oct sobre #51–#53)",
           "status": "Lectura a posteriori, SIN LETRA: no da letra a las rondas 3 o 3b ni toca reglas congeladas.",
           "code_commit": _git_head(), "written_utc": datetime.now(timezone.utc).isoformat(),
           "sources": [str(p.relative_to(ROOT)) for p in (R3, R3B, QUAL_SHEETS, QUAL_Q15, HALO)],
           "sheets": sheets(), "halo": halo()}
    _check_no_letters(doc)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "lecturas.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTDIR / "lecturas.md").write_text(_md(doc) + "\n", encoding="utf-8")
    print(f"lecturas escritas en {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
