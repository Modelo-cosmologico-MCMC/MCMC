#!/usr/bin/env python
"""Intercambio de masa entre la burbuja de MCV y la ECV en el borde (documento vivo del 24-oct, PR-64): tasas por halo y
corridas del código gravotérmico con cada ley de borde, sin letra.

    python scripts/run_gravothermal_exchange.py    # escribe results/2026-10-24_gravothermal_exchange/

  - Tasas por unidad de masa, en unidades de H, para c = 5, 10, 15 y 20, con las tres afueras de la decisión 6: el NFW
    prolongado (las cifras del documento), la capa del borde del propio código con la fuerza de la ECV y la media del
    entorno.
  - El halo de referencia del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g; σ̂ = 0.249), con la fuerza de la ECV, la pared en
    R₀ y 600 capas hasta 100 r_s, con cada ley de borde:
      sin intercambio;
      brazo A (solo atrapamiento: barrido y condensación, ξ_s = ξ) con ξ = 0.2, 0.5 y 1;
      dos sentidos (decisión del autor del 24-oct: todas las posibilidades en el borde):
        contacto reversible con ξ = 0.2, ξ_eq del código, 0.5 y 1;
        barrido y contacto (ξ_s = ξ) con ξ = 0.5 y 1;
        evaporación sola (E2);
        barrido y evaporación (ξ_s = 0.5, sin condensación);
        contacto en su forma de Onsager lineal con ξ = 1.
  - Convergencia: cuatro de esas corridas con el doble de capas.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.ecv_exchange import (  # noqa: E402
    border_rates,
    environment_outskirts,
    nfw_outskirts,
)
from dynamics.gravothermal import (  # noqa: E402
    GravothermalConfig,
    GravothermalHalo,
    code_outskirts,
    ecv_config,
    run_collapse,
    sigma_hat,
    time_unit_gyr,
)
from dynamics.mcv_equilibrium import nfw_halo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-24_gravothermal_exchange"
CONCENTRATIONS = (5.0, 10.0, 15.0, 20.0)
HALO = nfw_halo(1e12, 10.0)
SIGMA_HAT = sigma_hat(10.0, HALO["rho_s"], HALO["r_s"])
BASE = ecv_config(SIGMA_HAT, 10.0, base=GravothermalConfig(sigma_hat=SIGMA_HAT, n_shells=600))
XI_EQ = float(GravothermalHalo(BASE).rho[-1]) / BASE.q_E           # equilibrio del contacto con la capa del borde del código
ARMS = {"sin": {},
        "A_0.2": {"xi": 0.2, "xi_s": 0.2}, "A_0.5": {"xi": 0.5, "xi_s": 0.5}, "A_1": {"xi": 1.0, "xi_s": 1.0},
        "contacto_0.2": {"xi": 0.2, "evaporation": True}, "contacto_eq": {"xi": XI_EQ, "evaporation": True},
        "contacto_0.5": {"xi": 0.5, "evaporation": True}, "contacto_1": {"xi": 1.0, "evaporation": True},
        "barrido_contacto_0.5": {"xi": 0.5, "xi_s": 0.5, "evaporation": True},
        "barrido_contacto_1": {"xi": 1.0, "xi_s": 1.0, "evaporation": True},
        "E2": {"evaporation": True}, "barrido_E2": {"xi_s": 0.5, "evaporation": True},
        "onsager_1": {"xi": 1.0, "evaporation": True, "contact_form": "onsager-lineal"}}
CONVERGENCE = ("sin", "contacto_1", "E2", "A_0.5")


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _job(job):
    name, cfg = job
    r = run_collapse(cfg)
    keep = ("collapsed", "steps", "t_c", "C_sigma_t_c", "energy_rel_change", "border_flux_rel", "outside_thermal_influence",
            "wall_net_force", "wall_net_force_end", "exchange")
    return name, {"config": r["config"], **{k: r[k] for k in keep}}


def _rates_table() -> list[dict]:
    rows = []
    for c in CONCENTRATIONS:
        for model, o in (("nfw_prolongado", nfw_outskirts(c)), ("capa_del_borde", code_outskirts(c)),
                         ("media_del_entorno", environment_outskirts(c))):
            rows.append({"c": c, "outskirts": model, "rho_b_over_rho_E": o["rho_b"] / o["q_E"],
                         "evaporation": border_rates(o)["evaporation"], "xi_eq": border_rates(o)["xi_eq"],
                         "net_xi_1": border_rates(o, xi=1.0)["net"], "net_xi_05": border_rates(o, xi=0.5)["net"],
                         "A_xi_03_accretion_067": border_rates(o, xi=0.3, xi_s=0.3, evaporation=False, accretion=0.67)["net"]})
    return rows


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(k, replace(BASE, **v)) for k, v in ARMS.items()]
    jobs += [(f"fino_{k}", replace(BASE, n_shells=2 * BASE.n_shells - 1, **ARMS[k])) for k in CONVERGENCE]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        runs = dict(ex.map(_job, jobs))
    rates = _rates_table()
    tu = time_unit_gyr(HALO["rho_s"])
    t_ref, t_fine = runs["sin"]["t_c"], runs["fino_sin"]["t_c"]
    table = {}
    for k in ARMS:
        r, ex = runs[k], runs[k]["exchange"]
        table[k] = {"t_c_gyr": r["t_c"] * tu, "shift": r["t_c"] / t_ref - 1.0, "mass_ratio_end": ex["mass_ratio_end"] if ex else 1.0,
                    "rate_per_mass_H0": (ex["initial_rate_per_mass"] / nfw_outskirts(10.0)["H"]) if ex else 0.0,
                    "energy_budget_residual": ex["energy_budget_residual"] if ex else r["energy_rel_change"],
                    "splits": ex["splits"] if ex else 0, "merges": ex["merges"] if ex else 0}
    convergence = {k: {"shift_ref": runs[k]["t_c"] / t_ref - 1.0, "shift_fine": runs[f"fino_{k}"]["t_c"] / t_fine - 1.0}
                   for k in CONVERGENCE if k != "sin"}
    doc = {"title": "Intercambio de masa en el borde de ECV: tasas por halo y corridas del código gravotérmico, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_gravothermal_exchange.py", "modules": ["dynamics/ecv_exchange.py", "dynamics/gravothermal.py"],
           "halo": {"M200": 1e12, "c": 10.0, "sigma_over_m": 10.0, "sigma_hat": SIGMA_HAT, "time_unit_gyr": tu,
                    "n_shells": BASE.n_shells, "xi_eq_code": XI_EQ},
           "rates": rates, "runs": runs, "table": table, "convergence": convergence,
           "summary": {"xi_eq_code": XI_EQ, "xi_eq_nfw": border_rates(nfw_outskirts(10.0))["xi_eq"],
                       "evaporation_code_c10": next(r["evaporation"] for r in rates if r["c"] == 10.0 and r["outskirts"] == "capa_del_borde"),
                       "evaporation_nfw_c10": border_rates(nfw_outskirts(10.0))["evaporation"],
                       "max_energy_budget_residual": max(abs(v["energy_budget_residual"]) for v in table.values()),
                       "convergence_max_abs_diff": max(abs(v["shift_fine"] - v["shift_ref"]) for v in convergence.values())}}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = doc["summary"]
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. Tasas por unidad de la masa dentro de R₀, en unidades "
          "de H, sin acreción salvo la columna del brazo A (acreción 0.67·H0, la media de un halo de 10¹² M☉ hoy).", "",
          "| c | afueras | ρ_b/ρ_E = ξ_eq | evaporación (E2) | contacto ξ = 1 | contacto ξ = 0.5 | A, ξ = ξ_s = 0.3 |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['c']:g} | {r['outskirts'].replace('_', ' ')} | {r['xi_eq']:.3f} | {r['evaporation']:.3f} | {r['net_xi_1']:+.3f} | "
           f"{r['net_xi_05']:+.3f} | {r['A_xi_03_accretion_067']:+.3f} |" for r in rates]
    md += ["", f"Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g; σ̂ = {SIGMA_HAT:.3f}), con la fuerza de la ECV y la pared en "
           f"R₀ ({BASE.n_shells} capas); ξ_eq del código = {XI_EQ:.3f}:", "",
           "| ley de borde | t_c (Gyr) | cambio de t_c | masa final / inicial | tasa inicial (H0) | cierre de la energía |",
           "|---|---|---|---|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v['t_c_gyr']:.2f} | {v['shift']:+.2%} | {v['mass_ratio_end']:.3f} | "
           f"{v['rate_per_mass_H0']:+.3f} | {v['energy_budget_residual']:+.1e} |" for k, v in table.items()]
    md += ["", "Convergencia (el doble de capas): " + "; ".join(
        f"{k.replace('_', ' ')}: {v['shift_ref']:+.2%} frente a {v['shift_fine']:+.2%}" for k, v in convergence.items()) + ".",
        "", f"- Con la fuerza de la ECV, la capa del borde del código es más densa y más caliente que el NFW prolongado: "
        f"ξ_eq = {s['xi_eq_code']:.3f} frente a {s['xi_eq_nfw']:.3f}, y la evaporación sale {s['evaporation_code_c10']:.3f}·H "
        f"frente a {s['evaporation_nfw_c10']:.3f}·H (c = 10). Las cifras del documento son las del NFW prolongado.",
        f"- La energía cierra: lo que traen el intercambio y el trabajo sobre la pared explica el cambio de la energía salvo "
        f"la deriva del propio código (≤ {s['max_energy_budget_residual']:.1e} de |E|).", "",
        "**Lectura**: sin letra (E8). Comprobación del instrumento con cada ley de borde; ninguna cifra elige la ley."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
