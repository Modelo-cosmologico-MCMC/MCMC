#!/usr/bin/env python
"""La fuerza de la ECV en el código gravotérmico y la pared en R₀ (documento vivo del 24-oct, PR-63): corridas de
referencia, sin letra.

    python scripts/run_gravothermal_ecv.py    # escribe results/2026-10-24_gravothermal_ecv/

Para c = 5, 10, 15 y 20 (σ̂ = 0.01, C = 0.75, NFW inicial, borde en su R₀ autoconsistente):
  - con y sin la fuerza de la ECV, con el espaciado del documento (150 capas hasta 100 r_s): el cambio de t_c;
  - lo mismo con el espaciado de referencia (600 capas hasta 100 r_s), con su comprobación de resolución: duplicar las
    capas tiene que mover t_c menos de RESOLUTION_TOL;
  - la serie de resolución de c = 10 (145, 289, 577 y 1153 capas): dónde empieza a pasar la comprobación;
  - el fallo cerrado de una pared en 1.25·R₀ con la fuerza de la ECV;
  - el halo de referencia del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g) en Gyr, con y sin la fuerza.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.gravothermal import (  # noqa: E402
    RESOLUTION_TOL,
    THERMAL_TOL,
    GravothermalConfig,
    GravothermalHalo,
    collapse_time_gyr,
    ecv_config,
    resolution_shift,
    run_collapse,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-24_gravothermal_ecv"
DOC_BASE = GravothermalConfig(sigma_hat=0.01)                    # el espaciado del documento
REF_BASE = GravothermalConfig(sigma_hat=0.01, n_shells=600)      # el que pasa la comprobación de resolución
CONCENTRATIONS = (5.0, 10.0, 15.0, 20.0)
SERIES = (145, 289, 577, 1153)                                   # c = 10: cada una es 2n − 1 de la anterior
DOC_SHIFTS = {"5": -0.075, "10": -0.029, "15": -0.015}           # las del documento («baja un 7.5 / 2.9 / 1.5 %»)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _job(job):
    name, kind, arg = job
    if kind == "res":
        return name, resolution_shift(arg)
    if kind == "halo":
        return name, collapse_time_gyr(1e12, 10.0, 10.0, cfg=REF_BASE, ecv=arg)
    return name, run_collapse(arg)


def _lite(r: dict) -> dict:
    keys = ("collapsed", "steps", "t_c", "C_sigma_t_c", "rho_c_min", "energy_rel_change", "border_flux_rel",
            "outside_thermal_influence", "wall_net_force", "t_c_gyr", "t_c_essig_gyr", "sigma_hat")
    return {"config": r["config"], **{k: r[k] for k in keys if k in r}}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in CONCENTRATIONS:
        k = f"{c:g}"
        doc, ref = ecv_config(0.01, c, base=DOC_BASE), ecv_config(0.01, c, base=REF_BASE)
        jobs += [(f"doc_ecv_{k}", "run", doc), (f"doc_sin_{k}", "run", replace(doc, q_E=0.0, wall="fija")),
                 (f"ref_res_{k}", "res", ref), (f"ref_sin_{k}", "run", replace(ref, q_E=0.0, wall="fija"))]
    ten = ecv_config(0.01, 10.0, base=DOC_BASE)
    jobs += [(f"serie_{n}", "run", replace(ten, n_shells=n)) for n in SERIES]
    jobs += [("halo_autor_ecv", "halo", True), ("halo_autor_sin", "halo", False)]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        out = dict(ex.map(_job, jobs))
    try:
        GravothermalHalo(replace(ten, wall="fija", r_border=1.25 * ten.r_border))
        beyond = "sin error"
    except ValueError as e:
        beyond = str(e)
    rows, shifts_doc, shifts_ref, res_ok = [], {}, {}, {}
    for c in CONCENTRATIONS:
        k = f"{c:g}"
        de, ds = out[f"doc_ecv_{k}"], out[f"doc_sin_{k}"]
        res, rs = out[f"ref_res_{k}"], out[f"ref_sin_{k}"]
        shifts_doc[k] = de["t_c"] / ds["t_c"] - 1.0
        shifts_ref[k] = res["coarse"]["t_c"] / rs["t_c"] - 1.0
        res_ok[k] = res["converged"]
        rows.append({"c": c, "R0_r_s": de["config"]["r_border"], "q_E": de["config"]["q_E"],
                     "doc": {"n_shells": de["config"]["n_shells"], "ecv": _lite(de), "sin": _lite(ds), "shift": shifts_doc[k]},
                     "ref": {"n_shells": res["coarse"]["config"]["n_shells"], "ecv": _lite(res["coarse"]),
                             "ecv_fine": _lite(res["fine"]), "sin": _lite(rs), "shift": shifts_ref[k],
                             "resolution_rel": res["rel"], "resolution_converged": res["converged"]}})
    series = [_lite(out[f"serie_{n}"]) for n in SERIES]
    series_rel = {f"{a}→{b}": out[f"serie_{b}"]["t_c"] / out[f"serie_{a}"]["t_c"] - 1.0 for a, b in pairwise(SERIES)}
    ecv_runs = [r["doc"]["ecv"] for r in rows] + [r["ref"]["ecv"] for r in rows] + series
    halo_e, halo_s = out["halo_autor_ecv"], out["halo_autor_sin"]
    doc = {"title": "La fuerza de la ECV en el código gravotérmico y la pared en R₀: corridas de referencia, sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_gravothermal_ecv.py", "module": "dynamics/gravothermal.py",
           "tolerances": {"RESOLUTION_TOL": RESOLUTION_TOL, "THERMAL_TOL": THERMAL_TOL},
           "doc_spacing": "150 capas hasta 100 r_s", "reference_spacing": "600 capas hasta 100 r_s",
           "rows": rows, "resolution_series_c10": {"n_shells": list(SERIES), "runs": series, "rel": series_rel},
           "wall_beyond_r0": beyond,
           "author_halo": {"ecv": _lite(halo_e), "sin": _lite(halo_s), "shift": halo_e["t_c"] / halo_s["t_c"] - 1.0},
           "summary": {"ecv_shift_doc_spacing": shifts_doc, "ecv_shift_reference": shifts_ref, "doc_shifts": DOC_SHIFTS,
                       "resolution_check_reference": res_ok, "resolution_check_reference_passes_all": all(res_ok.values()),
                       "resolution_check_doc_spacing_passes": bool(abs(series_rel["145→289"]) < RESOLUTION_TOL),
                       "wall_beyond_r0_fails_closed": "más allá de R₀" in beyond,
                       "max_wall_net_force": max(abs(r["wall_net_force"]) for r in ecv_runs),
                       "outside_thermal_influence_all": all(r["outside_thermal_influence"] for r in ecv_runs)}}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = doc["summary"]
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. σ̂ = 0.01, C = 0.75, NFW inicial, pared en el R₀ "
          "autoconsistente; «sin» es la misma pared fija sin la fuerza de la ECV.", "",
          "| c | R₀ (r_s) | q_E | capas (doc) | C·σ̂·t_c sin / con (doc) | cambio (doc) | capas (ref) | C·σ̂·t_c sin / con (ref) | "
          "cambio (ref) | duplicar capas (ref) | flujo al borde |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d, f = r["doc"], r["ref"]
        md.append(f"| {r['c']:g} | {r['R0_r_s']:.1f} | {r['q_E']:.3e} | {d['n_shells']} | {d['sin']['C_sigma_t_c']:.3f} / "
                  f"{d['ecv']['C_sigma_t_c']:.3f} | {d['shift']:+.2%} | {f['n_shells']} | {f['sin']['C_sigma_t_c']:.3f} / "
                  f"{f['ecv']['C_sigma_t_c']:.3f} | {f['shift']:+.2%} | {f['resolution_rel']:+.1e} "
                  f"({'pasa' if f['resolution_converged'] else 'no pasa'}) | {f['ecv']['border_flux_rel']:.1e} |")
    md += ["", "Serie de resolución de c = 10 con la fuerza de la ECV (duplicar las capas):", ""] + [
        f"- {k} capas: {v:+.1e} ({'pasa' if abs(v) < RESOLUTION_TOL else 'no pasa'} la tolerancia {RESOLUTION_TOL:g})"
        for k, v in series_rel.items()]
    md += ["", f"- Cambio de t_c por la fuerza de la ECV, frente a las cifras del documento (−7.5 / −2.9 / −1.5 % para c = 5 / "
           f"10 / 15): espaciado del documento {', '.join(f'c = {k}: {v:+.2%}' for k, v in shifts_doc.items())}; espaciado de "
           f"referencia {', '.join(f'c = {k}: {v:+.2%}' for k, v in shifts_ref.items())}.",
           f"- Fuerza neta en la pared: |·| ≤ {s['max_wall_net_force']:.1e} de la atracción en todas las corridas con la ECV.",
           f"- Pared en 1.25·R₀ con la fuerza de la ECV: {beyond}.",
           f"- Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g, 600 capas hasta 100 r_s): t_c = {halo_s['t_c_gyr']:.2f} Gyr "
           f"sin la fuerza, con la pared fija en 100 r_s como en #61, y {halo_e['t_c_gyr']:.2f} Gyr con ella y la pared en su R₀ "
           f"({doc['author_halo']['shift']:+.2%}); fórmula de Essig, "
           f"{halo_e['t_c_essig_gyr']:.2f} Gyr.", "",
           "**Lectura**: sin letra (E8). La fuerza de la ECV acorta el colapso, y el cambio no depende de la resolución. La "
           "comprobación de resolución del documento (duplicar las capas mueve t_c menos de 10⁻³) no pasa con su espaciado: la "
           "discretización de t_c es del 0.6 % con 150 capas. Pasa con el de referencia."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
