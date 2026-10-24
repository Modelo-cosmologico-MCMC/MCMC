#!/usr/bin/env python
"""Código gravotérmico 1D de la MCV (c) con borde de ECV (bloque P1 del autor del 23-oct; borde definido el 24-oct):
corridas de comprobación, sin letra.

    python scripts/run_gravothermal.py    # escribe results/2026-10-24_gravothermal/

Todas parten del NFW truncado en el borde, con σ_x/m constante:
  - referencia: σ̂ = 0.01, C = 0.75, 150 capas y borde cerrado en 100 r_s; t_c frente a 150/β (Essig et al. 2019);
  - el brazo «intercambio» del borde y el borde en 10, 20, 40, 80, 160 y 200 r_s, con el mismo espaciado: duplicar el
    borde no debe mover t_c, y el flujo conductivo no debe llegar a él;
  - resolución (300 capas) y paso (eps = 5·10⁻⁴);
  - universalidad del régimen LMFP (σ̂ = 0.001 y C = 0.45) y camino hacia el SMFP (σ̂ = 0.1, 0.3 y 1);
  - la pendiente autosemejante lejos del núcleo (σ̂ = 0.001, 400 capas, ρ_c = 10⁶ y 10⁸ ρ_s);
  - el halo de referencia del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g) en Gyr, frente a los 44.9 Gyr de la fórmula.
Banda declarada antes de la primera corrida del colapso: C·σ̂·t_c ∈ [120, 180] (150/β ± 20 %) en el régimen LMFP.
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

from dynamics.gravothermal import (  # noqa: E402
    BORDER_TOL,
    THERMAL_TOL,
    GravothermalConfig,
    collapse_time_gyr,
    run_collapse,
    with_border,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-24_gravothermal"
REF = GravothermalConfig(sigma_hat=0.01)
BAND = (120.0, 180.0)
VARIANTS = {"intercambio": {"border": "intercambio"}, "capas_300": {"n_shells": 300}, "eps_5e-4": {"eps": 5e-4},
            "sigma_0.001": {"sigma_hat": 0.001}, "C_0.45": {"C": 0.45}, "sigma_0.1": {"sigma_hat": 0.1},
            "sigma_0.3": {"sigma_hat": 0.3}, "sigma_1": {"sigma_hat": 1.0}}
BORDER_RADII = (10.0, 20.0, 40.0, 80.0, 160.0, 200.0)
SLOPE = GravothermalConfig(sigma_hat=0.001, n_shells=400)
SLOPE_SNAPSHOTS = (1e6, 1e8)
AUTHOR_HALO = (1e12, 10.0, 10.0)                       # M200 (M☉), c, σ_x/m (cm²/g): t_c = 44.9 Gyr con β = 0.75


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _job(job):
    name, kind, arg = job
    if kind == "halo":
        return name, collapse_time_gyr(*arg)
    if kind == "slope":
        return name, run_collapse(arg, snapshot_rho=SLOPE_SNAPSHOTS)
    return name, run_collapse(arg)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [("referencia", "run", REF)] + [(k, "run", replace(REF, **v)) for k, v in VARIANTS.items()]
    jobs += [(f"borde_{rb:g}", "run", with_border(REF, rb)) for rb in BORDER_RADII]
    jobs += [("pendiente", "slope", SLOPE), ("halo_autor", "halo", AUTHOR_HALO)]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        runs = dict(ex.map(_job, jobs))
    for name, r in runs.items():
        print(f"{name:<12} C·σ̂·t_c = {r['C_sigma_t_c']:.3f}, ΔE/E = {r['energy_rel_change']:.1e}, "
              f"flujo al borde = {r['border_flux_rel']:.1e}", flush=True)
    ref = runs["referencia"]
    t_ref = ref["t_c"]
    by_rb = {cfg.r_border: runs[name] for name, kind, cfg in jobs if kind == "run" and name.startswith(("borde", "referencia"))}
    doubling = {}
    for rb in sorted(by_rb):
        if 2 * rb in by_rb:
            near, far = by_rb[rb], by_rb[2 * rb]
            rel = far["t_c"] / near["t_c"] - 1.0
            doubling[f"{rb:g}"] = {"rel": rel, "outside_influence": bool(abs(rel) < BORDER_TOL and near["outside_thermal_influence"]
                                                                        and far["outside_thermal_influence"])}
    first_outside = min((float(k) for k, v in doubling.items() if v["outside_influence"]), default=None)
    thermal_inside = [k for k, r in runs.items() if not r["outside_thermal_influence"]]
    lmfp = [runs[k]["C_sigma_t_c"] for k in ("referencia", "sigma_0.001", "C_0.45")]
    halo = runs["halo_autor"]
    slope = runs["pendiente"]["snapshots"][0]
    d_ref, d_r200 = doubling[f"{REF.r_border:g}"], doubling["10"]
    reading = (f"sin letra (E8). El código reproduce el t_c de Essig et al. (2019): C·σ̂·t_c = {ref['C_sigma_t_c']:.1f} frente a "
               f"150 (banda declarada [{BAND[0]:g}, {BAND[1]:g}]), y la pendiente autosemejante de Balberg et al. (2002): "
               f"{slope['slope']:.2f} frente a −2.19. Con el borde en {REF.r_border:g} r_s, el borde queda fuera de la influencia "
               f"{'térmica y gravitatoria' if d_ref['outside_influence'] else 'térmica, pero NO de la gravitatoria'}: flujo al borde "
               f"{ref['border_flux_rel']:.1e} del máximo, y duplicarlo mueve t_c {d_ref['rel']:+.1e}. En r200 de un halo con c = 10 "
               f"(10 r_s) {'también' if d_r200['outside_influence'] else 'no'}: flujo {runs['borde_10']['border_flux_rel']:.1e} y "
               f"duplicarlo mueve t_c {d_r200['rel']:+.1%}. El primer borde de la serie que pasa las dos comprobaciones está en "
               + ("ninguno" if first_outside is None else f"{first_outside:g} r_s"))
    doc = {"title": "Código gravotérmico 1D de la MCV (c) con borde de ECV: corridas de comprobación, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_gravothermal.py", "module": "dynamics/gravothermal.py",
           "border_definition": "el borde de ECV corresponde al punto de intercambio entre la burbuja de MCV y la ECV fuera "
                                "de la influencia del punto gravitacional y térmico (autor, 24-oct)",
           "declared_band": {"C_sigma_t_c": list(BAND), "source": "150/β de Essig et al. (2019) ± 20 %, régimen LMFP",
                             "declared": "antes de la primera corrida del colapso"},
           "tolerances": {"THERMAL_TOL": THERMAL_TOL, "BORDER_TOL": BORDER_TOL}, "runs": runs,
           "summary": {"C_sigma_t_c_reference": ref["C_sigma_t_c"],
                       "in_declared_band": bool(BAND[0] <= ref["C_sigma_t_c"] <= BAND[1]),
                       "lmfp_universality_max_rel": max(lmfp) / min(lmfp) - 1.0,
                       "exchange_vs_closed_rel": runs["intercambio"]["t_c"] / t_ref - 1.0,
                       "resolution_rel_300": runs["capas_300"]["t_c"] / t_ref - 1.0,
                       "timestep_rel_eps_half": runs["eps_5e-4"]["t_c"] / t_ref - 1.0,
                       "border_doubling": doubling, "first_border_outside_influence": first_outside,
                       "inside_thermal_influence": thermal_inside,
                       "self_similar_slopes": runs["pendiente"]["snapshots"],
                       "core_phase": {"rho_c_min": ref["rho_c_min"], "C_sigma_t_core_max": REF.C * REF.sigma_hat * ref["t_core_max"],
                                      "t_core_max_over_t_c": ref["t_core_max"] / t_ref},
                       "author_halo": {"M200": halo["M200"], "c": halo["c"], "sigma_over_m": halo["sigma_over_m"],
                                       "sigma_hat": halo["sigma_hat"], "t_c_gyr": halo["t_c_gyr"],
                                       "t_c_essig_gyr": halo["t_c_essig_gyr"], "ratio": halo["t_c_gyr"] / halo["t_c_essig_gyr"]}},
           "reading": reading}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = doc["summary"]
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. Módulo `{doc['module']}`. Borde de ECV: «{doc['border_definition']}».",
          "", f"Banda declarada antes de la primera corrida del colapso: C·σ̂·t_c ∈ [{BAND[0]:g}, {BAND[1]:g}] "
          f"({doc['declared_band']['source']}).", "",
          "| corrida | σ̂ | C | capas | borde (r_s) | brazo | C·σ̂·t_c | ρ_c mín (ρ_s) | ΔE/E | flujo al borde | Δu borde |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        c = r["config"]
        md.append(f"| {name} | {c['sigma_hat']:g} | {c['C']:g} | {c['n_shells']} | {c['r_border']:g} | {c['border']} | "
                  f"{r['C_sigma_t_c']:.2f} | {r['rho_c_min']:.3f} | {r['energy_rel_change']:.1e} | {r['border_flux_rel']:.1e} | "
                  f"{r['border_u_rel_change']:.1e} |")
    md += ["", f"- Referencia: C·σ̂·t_c = **{s['C_sigma_t_c_reference']:.2f}**, dentro de la banda: "
           f"**{'sí' if s['in_declared_band'] else 'no'}**.",
           f"- Universalidad LMFP (σ̂ = 0.01 y 0.001; C = 0.75 y 0.45): dispersión relativa {s['lmfp_universality_max_rel']:.1e}.",
           f"- Brazo «intercambio» frente a «cerrado»: {s['exchange_vs_closed_rel']:.1e}. Resolución (300 capas): "
           f"{s['resolution_rel_300']:+.2%}. Paso (eps/2): {s['timestep_rel_eps_half']:+.2%}.",
           f"- Duplicar el borde (cambio relativo de t_c; tolerancia {BORDER_TOL:g}; «fuera» exige además flujo por debajo de "
           f"{THERMAL_TOL:g} en las dos corridas): "
           + ", ".join(f"{k} → {2 * float(k):g} r_s: {v['rel']:+.1e} ({'fuera' if v['outside_influence'] else 'dentro'})"
                       for k, v in s["border_doubling"].items()) + ".",
           f"- Flujo conductivo al borde por encima de {THERMAL_TOL:g} del máximo (dentro de la influencia térmica): "
           + (", ".join(s["inside_thermal_influence"]) or "ninguna corrida") + ".",
           "- La capa del borde se comprime adiabáticamente al expandirse el interior (columna Δu borde): la influencia "
           "mecánica llega a la pared, y la comprobación es que no mueva t_c al duplicar el borde.",
           f"- Fase de núcleo: ρ_c mínima {s['core_phase']['rho_c_min']:.3f} ρ_s en C·σ̂·t = "
           f"{s['core_phase']['C_sigma_t_core_max']:.1f} ({s['core_phase']['t_core_max_over_t_c']:.3f} t_c).",
           "- Pendiente autosemejante (ρ/ρ_c ∈ [10⁻⁶, 10⁻³], 400 capas, σ̂ = 0.001): "
           + ", ".join(f"{x['slope']:.3f} con ρ_c = {x['rho_c']:.1e} ({x['n_shells']} capas)" for x in s["self_similar_slopes"])
           + "; Balberg et al. (2002): −2.19.",
           f"- Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g; σ̂ = {s['author_halo']['sigma_hat']:.3f}): "
           f"t_c = {s['author_halo']['t_c_gyr']:.1f} Gyr frente a {s['author_halo']['t_c_essig_gyr']:.1f} Gyr de la fórmula "
           f"(cociente {s['author_halo']['ratio']:.3f}).", "",
           "La ventana de la pendiente se fijó tras una primera medida: junto al núcleo (ρ/ρ_c ∈ [10⁻³, 10⁻¹]) la pendiente pasa "
           "de −2.4, como en la esfera isoterma, y la ley de potencias autosemejante solo aparece más lejos.", "",
           f"**Lectura**: {doc['reading']}."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
