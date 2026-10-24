#!/usr/bin/env python
"""Todas las posibilidades del brazo de dos sentidos, con sus consecuencias para la ontología y su respuesta en el fondo
(documento vivo del 24-oct, PR-65; decisión del autor del 24-oct). Sin datos y sin letra.

    python scripts/run_halo_exchange.py    # escribe results/2026-10-24_halo_exchange/

  1. Comprobación cruzada: el modelo de las cifras del documento (c = 10 para todos los halos, tasas de hoy fijas en
     tiempo físico) reproduce su tabla de Δ.
  2. El catálogo: cada atrapamiento (T1, T2, T3, T4, Tc y el barrido con condensación) con cada escape (E1, E2, E3, E4),
     24 combinaciones, con el modelo completo (c(M, z) de Dutton y Macciò 2014, tasas de cada época) y las afueras del
     NFW prolongado y de la capa del borde del código. Para cada una:
       - las consecuencias para la ontología: dónde actúa, mecanismo, constantes, si se apaga cuando las burbujas llenan el
         espacio y su encaje;
       - Δ en una rejilla de constantes;
       - la constante de equilibrio (Δ = 0);
       - los signos de Δ que alcanza con constantes no negativas;
       - cuándo actúa.
  3. Sensibilidad de las combinaciones principales a las alternativas del documento: Press–Schechter, M_min, k = 2,
     cierre lineal, afueras del entorno, c = 10, tasas de hoy y z_ini = 20.
Ninguna cifra elige la ley ni lleva letra: la letra es de la preinscripción del ajuste del fondo, con todos los brazos
congelados antes de los datos.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.halo_exchange import (  # noqa: E402
    COSMO,
    DOC_SETUP,
    LAW_ONTOLOGY,
    Setup,
    code_border_table,
    combine,
    delta_today,
    law_profiles,
    timing,
    two_way_catalogue,
)

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "2026-10-24_halo_exchange"
GRID = (0.1, 0.3, 1.0)                                     # constantes de atrapamiento y de escape
DOC_TABLE = {"T1": 0.1020, "T2": 0.0624, "T3": 0.0186, "T1+E2": 0.0924, "T1+E3": 0.0243, "T1+E1": -0.1151,
             "T2+E2": 0.0518, "T2+E3": -0.0227, "T2+E1": -0.1769, "T3+E2": 0.0071, "T3+E3": -0.0750, "T3+E1": -0.2466,
             "C1": 0.0455, "C0.5": 0.0176, "C0.2": 0.0001, "T3+C1": 0.0625, "T3+C0.5": 0.0356}
KEY = {"T1+E1 (ec. 7 tal cual)": {"T1": 0.3, "E1": 0.3}, "contacto ξ = 1": {"Tc": 1.0, "E2": 1.0},
       "contacto ξ = 0.5": {"Tc": 0.5, "E2": 1.0}, "barrido y contacto ξ_s = ξ = 0.5": {"T3": 0.5, "Tc": 0.5, "E2": 1.0},
       "A: barrido y condensación ξ_s = ξ = 0.5": {"T3": 0.5, "Tc": 0.5}, "T2+E3": {"T2": 0.3, "E3": 0.3},
       "E2 sola": {"E2": 1.0}}


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def _constants(laws: tuple[str, ...], t: float, e: float) -> dict:
    """Constantes de una combinación: atrapamiento t (Tc y T3 comparten ξ_s = ξ, la decisión 3 del documento), escape e;
    E2 no tiene constante y E4 lleva su cota con ξ_4 = 1."""
    out = {}
    for law in laws:
        if law in ("T1", "T2", "T3", "T4", "Tc"):
            out[law] = t
        elif law in ("E1", "E3"):
            out[law] = e
        else:
            out[law] = 1.0
    return out


def _doc_check() -> dict:
    p = law_profiles(DOC_SETUP)
    z = p["z"]

    def d(c, feeds=False):
        return delta_today(combine(p, c, sweep_feeds=feeds), z)
    got = {"T1": d({"T1": 0.3}), "T2": d({"T2": 0.3}), "T3": d({"T3": 0.3})}
    for t in ("T1", "T2", "T3"):
        got[f"{t}+E2"] = d({t: 0.3, "E2": 1.0})
        got[f"{t}+E3"] = d({t: 0.3, "E3": 0.3})
        got[f"{t}+E1"] = d({t: 0.3, "E1": 0.3})
    for xi in (1.0, 0.5, 0.2):
        got[f"C{xi:g}"] = d({"Tc": xi, "E2": 1.0})
    got["T3+C1"], got["T3+C0.5"] = d({"T3": 0.3, "Tc": 1.0, "E2": 1.0}), d({"T3": 0.3, "Tc": 0.5, "E2": 1.0})
    fed = {"T3+C1": d({"T3": 0.3, "Tc": 1.0, "E2": 1.0}, True), "T3+C0.5": d({"T3": 0.3, "Tc": 0.5, "E2": 1.0}, True)}
    return {"delta": got, "doc": DOC_TABLE, "max_abs_diff": max(abs(got[k] - v) for k, v in DOC_TABLE.items()),
            "sweep_fed_by_contact": fed, "f_h_0": float(p["f_h"][0]), "P_ocean_z1": float(np.interp(1.0, z, p["P"]))}


def _catalogue(p: dict, p20: dict) -> list[dict]:
    z = p["z"]
    rows = []
    for combo in two_way_catalogue():
        laws = combo["laws"]
        trap, esc = laws[:-1], laws[-1]

        def dl(t, e):
            return delta_today(combine(p, _constants(laws, t, e)), z)
        grid = {f"{t:g}|{e:g}": dl(t, e) for t in GRID for e in (GRID if esc in ("E1", "E3") else (1.0,))}
        e_ref = 0.3 if esc in ("E1", "E3") else 1.0
        try:                                                    # constante de atrapamiento del equilibrio, con e_ref
            balance = brentq(lambda t: dl(t, e_ref), 0.0, 1.99 if "T3" in trap else 50.0)
        except ValueError:
            balance = None
        signs = sorted({"+" if v > 1e-4 else "−" if v < -1e-4 else "0" for v in grid.values()} | ({"0"} if balance else set()))
        q = combine(p, _constants(laws, 0.3, e_ref))
        ont = [LAW_ONTOLOGY[x] for x in laws]
        rows.append({"name": combo["name"], "label": combo["label"], "laws": list(laws),
                     "where": " / ".join(o["donde"] for o in ont), "mechanism": all(o["mecanismo"] for o in ont),
                     "constants": [o["constante"] for o in ont if o["constante"]],
                     "switches_off_with_bubbles": [x for x, o in zip(laws, ont) if not o["se_apaga_con_las_burbujas"]],
                     "fit": {x: LAW_ONTOLOGY[x]["encaje"] for x in laws},
                     "consequences": {x: LAW_ONTOLOGY[x]["consecuencia"] for x in laws},
                     "delta_grid": grid, "balance_trapping_constant": balance, "escape_constant_for_balance": e_ref,
                     "reachable_signs": signs, "timing": timing(q, z),
                     "delta_ref_zini20": delta_today(combine(p20, _constants(laws, 0.3, e_ref)), p20["z"])})
    return rows


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc_check = _doc_check()
    table = code_border_table()
    setups = {"nfw": Setup(), "codigo": Setup(outskirts="codigo")}
    profiles = {k: law_profiles(s, code_table=table) for k, s in setups.items()}
    early = {k: law_profiles(Setup(outskirts=s.outskirts, z_ini=20.0, n_z=801), code_table=table) for k, s in setups.items()}
    catalogue = {k: _catalogue(profiles[k], early[k]) for k in setups}
    env = law_profiles(Setup(outskirts="entorno"))
    env_xi_eq = {f"{zz:g}": float(np.interp(zz, env["z"], env["xi_eq"])) for zz in (0.0, 0.5, 1.0, 2.0)}
    z_cross = (COSMO["Omega_L"] / COSMO["Omega_m"]) ** (1.0 / 3.0) - 1.0       # ξ_eq = ρ̄_m(z)/ρ_E = 1
    alternatives = {"recomendado (ST, 10⁸, k = 1, porosidad, NFW, DM14)": Setup(),
                    "Press–Schechter": Setup(mass_function="PS"), "M_min = 10⁶": Setup(M_min=1e6),
                    "M_min = 10¹⁰": Setup(M_min=1e10), "k = 2": Setup(k=2.0), "cierre lineal": Setup(closure="lineal"),
                    "afueras: capa del borde": Setup(outskirts="codigo"), "afueras: entorno": Setup(outskirts="entorno"),
                    "c = 10": Setup(concentration="c10"), "tasas de hoy": Setup(rate_epoch="z0"), "z_ini = 20": Setup(z_ini=20.0, n_z=801)}
    sensitivity = {}
    for name, s in alternatives.items():
        p = law_profiles(s, code_table=table)
        sensitivity[name] = {k: delta_today(combine(p, c), p["z"]) for k, c in KEY.items()}
        sensitivity[name]["ξ_eq hoy"] = float(p["xi_eq"][0])
        sensitivity[name]["P_océano hoy"] = float(p["P"][0])
    doc = {"title": "Todas las posibilidades del brazo de dos sentidos: consecuencias para la ontología y respuesta del fondo, "
                    "sin letra",
           "written_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
           "generator": "scripts/run_halo_exchange.py", "module": "cosmology/halo_exchange.py",
           "author_decision": "«respecto al brazo en dos sentidos, prefiero probar todas las posibilidades con las consecuencias "
                              "a la ontología y a partir de ahí hacer las pruebas para ver qué resultado nos dan» (24-oct)",
           "doc_check": doc_check, "code_border_table": table.table, "law_ontology": LAW_ONTOLOGY,
           "catalogue": catalogue, "sensitivity": sensitivity, "environment_xi_eq_by_z": env_xi_eq,
           "e3_dark_age_survival_from_z1100_to_z20": {f"{x:g}": (21.0 / 1101.0) ** x for x in (0.03, 0.1, 0.3)},
           "profiles_recommended": {k: [float(x) for x in v] for k, v in profiles["nfw"].items()
                                    if isinstance(v, np.ndarray)}}
    (OUTDIR / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# {doc['title']}", "", f"Commit `{doc['code_commit'][:9]}`. Decisión del autor: {doc['author_decision']}.", "",
          "## 1. Las cifras del documento", "",
          f"Con su modelo (c = 10 para todos los halos, tasas de hoy fijas en tiempo físico, ST con M_min = 10⁸ M☉, porosidad "
          f"con k = 1): f_h(0) = {doc_check['f_h_0']:.3f}, P_océano(z = 1) = {doc_check['P_ocean_z1']:.3f}, y la tabla de Δ "
          f"del documento se reproduce con una diferencia máxima de {doc_check['max_abs_diff']:.1e}. El documento suma el "
          f"barrido y el contacto; su propia ley de borde hace que el barrido se alimente también del contacto: barrido y "
          f"contacto (ξ_s = 0.3) dan {doc_check['sweep_fed_by_contact']['T3+C1']:+.4f} con ξ = 1 y "
          f"{doc_check['sweep_fed_by_contact']['T3+C0.5']:+.4f} con ξ = 0.5, frente a {DOC_TABLE['T3+C1']:+.4f} y "
          f"{DOC_TABLE['T3+C0.5']:+.4f} de la suma.", "",
          "## 2. Las leyes y sus consecuencias para la ontología", "",
          "| ley | dónde actúa | mecanismo | constante | se apaga con las burbujas | encaje (documento) | consecuencia |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {k} {v['nombre']} | {v['donde']} | {'sí' if v['mecanismo'] else 'no'} | {v['constante'] or '—'} | "
           f"{'sí' if v['se_apaga_con_las_burbujas'] else 'no'} | {v['encaje']} | {v['consecuencia']} |" for k, v in LAW_ONTOLOGY.items()]
    for n, (k, title) in enumerate((("nfw", "afueras del NFW prolongado"), ("codigo", "afueras de la capa del borde del código"))):
        md += ["", f"## {3 + n}. Las 24 combinaciones de dos sentidos ({title}; DM14, tasas de cada época)", "",
               "Δ con la constante de atrapamiento 0.1 / 0.3 / 1 (con ξ_s = ξ cuando van juntos) y, para E1 y E3, la de escape "
               "0.3, integrando desde z = 6; equilibrio: la constante de atrapamiento con Δ = 0; signos de Δ alcanzables en la "
               "rejilla; la última columna integra desde z = 20 con la constante 0.3.", "",
               "| combinación | Δ (0.1 / 0.3 / 1) | equilibrio | signos | z del máximo | fracción en z > 1 | sin apagarse | "
               "Δ (0.3), desde z = 20 |", "|---|---|---|---|---|---|---|---|"]
        for r in catalogue[k]:
            e = "0.3" if r["laws"][-1] in ("E1", "E3") else "1"
            ds = " / ".join(f"{r['delta_grid'][f'{t:g}|{e}']:+.3f}" for t in GRID)
            bal = "—" if r["balance_trapping_constant"] is None else f"{r['balance_trapping_constant']:.3f}"
            lab = f" ({r['label']})" if r["label"] else ""
            md.append(f"| {r['name']}{lab} | {ds} | {bal} | {' '.join(r['reachable_signs'])} | {r['timing']['z_peak']:.1f} | "
                      f"{r['timing']['z>1']:.2f} | {', '.join(r['switches_off_with_bubbles']) or '—'} | "
                      f"{r['delta_ref_zini20']:+.3f} |")
    md += ["", "## 5. Sensibilidad de las combinaciones principales", "",
           "| alternativa | " + " | ".join(KEY) + " | ξ_eq hoy | P_océano hoy |", "|---|" + "---|" * (len(KEY) + 2)]
    md += [f"| {name} | " + " | ".join(f"{v[k]:+.4f}" for k in KEY) + f" | {v['ξ_eq hoy']:.3f} | {v['P_océano hoy']:.3f} |"
           for name, v in sensitivity.items()]
    surv = doc["e3_dark_age_survival_from_z1100_to_z20"]
    md += ["", "## 6. Dos consecuencias que la integración desde z = 6 oculta", "",
           "- **E3 en la edad oscura.** Antes de los halos todo es océano, y la reversión literal actúa sobre toda la materia "
           "a ritmo ξ_e·H. Desde z = 1100 hasta z = 20, la fracción que sobrevive es (21/1101)^ξ_e: "
           + ", ".join(f"{v:.2f} con ξ_e = {x}" for x, v in surv.items())
           + ". Por eso las combinaciones con E3 cambian al integrar desde z = 20 (última columna). E3 solo es compatible "
           "con el CMB si actúa en vacíos de verdad, tras formarse las estructuras, o con ξ_e ≪ 0.1.",
           "- **Las afueras del entorno.** Con ρ_b = ρ̄_m(z), el equilibrio del contacto se mueve con la época: ξ_eq = "
           + ", ".join(f"{v:.2f} en z = {zz}" for zz, v in env_xi_eq.items())
           + f". Con ξ = 1, el contacto escapa a partir de z = (Ω_Λ/Ω_m)^(1/3) − 1 = {z_cross:.2f}, donde ξ_eq = 1, y con "
           "ξ < 1 desde antes; por eso esas afueras dan Δ negativos."]
    md += ["", "**Lectura**: sin letra (E8). Cada combinación queda descrita por sus consecuencias y su respuesta en el fondo; "
           "ninguna cifra elige la ley. La letra es de la preinscripción del ajuste del fondo, con todos los brazos congelados "
           "antes de los datos."]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
