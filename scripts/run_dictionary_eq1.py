#!/usr/bin/env python
"""Diccionario, ecuación 1: el mapa de ligaduras de la forma saturante
ε_c(ρ) = ε_max·ρ^{3/2}/(ρ^{3/2} + ρ*^{3/2}) y la predicción del escalón en
la RAR (orden del autor del 22-sep, §10). Publica
results/2026-09-22_dictionary_eps_c/.

    python scripts/run_dictionary_eq1.py

No es un experimento con desenlaces preinscritos: es un mapa de
consistencia (E8) sobre una forma declarada, con tres ligaduras ya
derivadas (Oort–K_z, estabilidad de Cronos–Jeans, perfil de Sculptor —
esta última en fallo cerrado) y la condición de calibración del 5E en
Sculptor. La predicción de la RAR se CONGELA (sha256) antes de que haya
bytes de SPARC: su contraste será una preinscripción propia.
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

from dynamics.epsilon_c_saturating import (  # noqa: E402
    A_SCULPTOR,
    FORM,
    constraint_map,
    rar_step_prediction,
    weak_law_amplitude,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-22_dictionary_eps_c"
# puntos DECLARADOS de la predicción RAR: (ε_max, ρ*) sobre la curva de calibración de Sculptor con Υ⋆ = 1 en tres ρ*,
# y un punto del borde de Oort con ρ* = 0.03 (la ley débil equivalente A = ε_max/ρ*^{3/2} se publica junto a cada uno)
RAR_RHO_STAR = (1e-3, 1e-2, 3e-2)
RAR_DISCS = ({"Sigma0_msun_pc2": 100.0, "h_kpc": 3.0, "h_z_kpc": 0.3}, {"Sigma0_msun_pc2": 300.0, "h_kpc": 3.0, "h_z_kpc": 0.3},
             {"Sigma0_msun_pc2": 1000.0, "h_kpc": 2.0, "h_z_kpc": 0.3})


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    m = constraint_map()
    cal1 = m["sculptor_calibration"]["1.0"]
    # predicción RAR: tres puntos sobre la curva de calibración de Sculptor (Υ⋆ = 1) + el borde de Oort
    lr = np.array(cal1["log10_rho_star"])
    le = np.array(cal1["log10_eps_max"])
    points = []
    for rs in RAR_RHO_STAR:
        i = int(np.argmin(abs(lr - np.log10(rs))))
        points.append({"label": f"calibración Sculptor Υ⋆ = 1, ρ* = {rs:g}", "eps_max": float(10.0 ** le[i]), "rho_star": float(10.0 ** lr[i]),
                       "oort_ok": bool(cal1["oort_ok"][i]), "others_ok": bool(cal1["stability_ok_without_sculptor_self"][i])})
    D_max = m["oort"]["deps_max_at_rho0"]
    rs = 0.03
    # ε_max del borde de Oort en ρ* = 0.03: ε_c'(ρ₀) = D_max ⟹ ε_max = D_max·(ρ₀^{3/2} + ρ*^{3/2})²/(1.5·ρ₀^{1/2}·ρ*^{3/2})
    rho0 = m["oort"]["rho0_msun_pc3"]
    e_edge = D_max * (rho0 ** 1.5 + rs ** 1.5) ** 2 / (1.5 * np.sqrt(rho0) * rs ** 1.5)
    points.append({"label": "borde de Oort–K_z (2σ), ρ* = 0.03", "eps_max": float(e_edge), "rho_star": rs, "oort_ok": True, "others_ok": None})
    rar = []
    for pt in points:
        for disc in RAR_DISCS:
            r = rar_step_prediction(pt["eps_max"], pt["rho_star"], **disc)
            r["point"] = pt
            r["weak_law_A_over_A_sculptor"] = weak_law_amplitude(pt["eps_max"], pt["rho_star"]) / A_SCULPTOR
            rar.append(r)
    pred = {"kind": "predicción CONGELADA de la forma saturante para la RAR (escalón donde ρ(R) cruza ρ*), antes de cualquier byte de SPARC",
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha, "form": FORM, "points": points, "discs": list(RAR_DISCS),
            "curves": rar,
            "how_to_falsify": ("con SPARC ingerido (Rotmod_LTG, MassModels_Lelli2016c): la desviación g_obs − g_bar debe concentrarse en el "
                               "intervalo de g_bar donde ρ_disco ≈ ρ* (un escalón), no seguir una función continua de g_bar; la "
                               "preinscripción del contraste (letras y tolerancias) se congelará cuando el autor aporte los bytes"),
            "status": "predicción de la forma declarada, condicional a (ε_max, ρ*): E13 (no es señal hasta el contraste)"}
    (OUTDIR / "rar_step_prediction.json").write_text(json.dumps(pred, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    psha = hashlib.sha256((OUTDIR / "rar_step_prediction.json").read_bytes()).hexdigest()
    doc = {"kind": "mapa de consistencia (E8) del diccionario, ecuación 1: forma saturante y sus ligaduras", "executed_utc": datetime.now(timezone.utc).isoformat(),
           "code_commit": sha, "form": FORM, "A_sculptor": A_SCULPTOR, "map": {k: v for k, v in m.items() if k != "allowed_mask"},
           "allowed_mask_shape": [len(m["grid"]["log10_eps_max"]), len(m["grid"]["log10_rho_star"])],
           "rar_prediction_sha256": psha,
           "what_is_not_claimed": ["la forma (E1) es DECLARADA (mínima compatible con Def. 6.4 y §11.4), no derivada de la ontología",
                                   "ningún parámetro se ajusta: el mapa dice qué (ε_max, ρ*) sobreviven a las ligaduras ya derivadas",
                                   "la ligadura (iii) (perfil de Sculptor) no se evalúa: walker2009 sigue en fallo cerrado",
                                   "la predicción RAR no se contrasta: SPARC sin ingerir; el contraste será una preinscripción propia"]}
    (OUTDIR / "eps_c_saturating.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    np.save(OUTDIR / "allowed_mask.npy", np.array(m["allowed_mask"], dtype=np.int8))
    o = m["oort"]
    md = ["# Diccionario, ecuación 1 — la forma saturante ε_c(ρ) y sus ligaduras (22-sep-2026)", "",
          f"Commit `{sha[:9]}`. Forma DECLARADA: {FORM}. A_Sculptor = {A_SCULPTOR:.4e} (congelada, 5E).", "",
          "## Ligaduras", "",
          f"- **(i) Oort–K_z**: Δρ_eff(0) = −c²ε_c'(ρ₀)ρ''(0)/(4πG) ≤ margen + 2σ = {o['room']['rho_room_0'] + 2*o['room']['rho_room_0_err']:.4f} M☉/pc³ "
          f"(ρ₀ = {o['rho0_msun_pc3']:.3f}, losa declarada) ⟹ ε_c'(ρ₀) ≤ {o['deps_max_at_rho0']:.3e}; en la ley débil equivale a "
          f"A ≤ {o['power_law_equivalent_A_max']/A_SCULPTOR:.3f}·A_Sculptor (el 0.060 de §3.26).",
          "- **(ii) estabilidad** q = c²ρε_c'(ρ)/σ² < 1 en la tabla declarada: " + "; ".join(f"{s['system']} (ρ = {s['rho']:.3g}, σ² = {s['sigma2']:.3g})" for s in m["systems"]) + ".",
          f"- **(iii) perfil de Sculptor**: {m['sculptor_profile_constraint']}.", "",
          f"Región permitida por (i) ∧ (ii) sobre la malla log ε_max ∈ [−14, −2] × log ρ* ∈ [−5, 2]: {m['allowed_fraction']*100:.1f} % "
          f"(solo Oort {m['allowed_by_oort_fraction']*100:.1f} %, solo estabilidad {m['allowed_by_stability_fraction']*100:.1f} %). "
          "Máscara en `allowed_mask.npy`.", "",
          "## La calibración de Sculptor frente a las ligaduras", "",
          "Reproducir la FUERZA del 5E en el centro de Sculptor fija ε_c'(ρ_S) = (3/2)A_S ρ_S^{1/2} y define una curva ε_max(ρ*). "
          "Sobre ella:", "",
          "| Υ⋆ | ρ_S [M☉/pc³] | q_S bajo su propia calibración (independiente de la forma) | ρ* con Oort ∧ resto de sistemas | ρ* con Oort ∧ TODOS (incl. Sculptor) | mínimo exceso sobre Oort (ρ*) |",
          "|---|---|---|---|---|---|"]
    for u, c in m["sculptor_calibration"].items():
        md.append(f"| {u} | {c['rho_S_msun_pc3']:.4f} | {c['q_sculptor_self_form_independent']:.2f} | {c['n_rho_star_oort_and_others_ok']} de {len(c['log10_rho_star'])} "
                  f"| {c['n_rho_star_both_ok']} | ×{c['min_oort_excess_ratio']:.2f} (ρ* = {c['rho_star_at_min_ratio']:.1e}) |")
    md += ["", m["form_independent_note"] + ".", "",
           "**Lectura**: (a) la saturación SÍ puede reconciliar la fuerza de Sculptor con la vecindad solar si la densidad estelar central de "
           "Sculptor es baja (Υ⋆ = 1: ρ_S = 0.016 ≪ ρ₀ = 0.097; con Υ⋆ = 2 también hay ρ* que pasan Oort; con Υ⋆ = 3 ninguno) — la ley "
           "débil pura no podía (A_S excluida ×17); (b) pero q_S > 1 para los tres Υ⋆: Sculptor es inestable bajo su propia calibración "
           "sea cual sea la forma, porque la calibración fija ρ_S·ε_c'(ρ_S). Ese es el hecho que ninguna forma ε_c(ρ) cambia: o la "
           "amplitud del 5E no es la fuerza de Sculptor, o el criterio de Cronos–Jeans no se aplica a Sculptor (σ_los ≠ σ del medio), o "
           "la inestabilidad es real y el perfil σ_los(R) lo dirá (ligadura iii, pendiente de bytes).", "",
           "## Predicción congelada: el escalón en la RAR", "",
           f"`rar_step_prediction.json` (sha256 `{psha[:12]}…`), {len(rar)} curvas ({len(points)} puntos × {len(RAR_DISCS)} discos exponenciales declarados). "
           "Con saturación g_C = c²ε_c'(ρ)|∇ρ| solo actúa donde ρ ≈ ρ*: la desviación g_obs − g_bar se concentra en el intervalo de g_bar donde "
           "ρ_disco(R) cruza ρ*, no sigue una función continua de g_bar.", "",
           "| punto | disco Σ₀ [M☉/pc²] | A_débil/A_S | R(ρ = ρ*) [kpc] | máx g_C/g_bar | R del máximo [kpc] |", "|---|---|---|---|---|---|"]
    for r in rar:
        r_star = "—" if r["R_where_rho_equals_rho_star_kpc"] is None else "%.2f" % r["R_where_rho_equals_rho_star_kpc"]
        md.append(f"| {r['point']['label']} | {r['disc']['Sigma0_msun_pc2']:g} | {r['weak_law_A_over_A_sculptor']:.3g} | {r_star} | "
                  f"{r['max_boost']:.3g} | {r['R_of_max_boost_kpc']:.2f} |")
    md += ["", "## Lo que NO afirma", ""] + [f"- {s}" for s in doc["what_is_not_claimed"]]
    (OUTDIR / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"mapa: permitido {m['allowed_fraction']*100:.1f} %; calibración de Sculptor: " +
          ", ".join(f"Υ={u}: Oort∧resto {c['n_rho_star_oort_and_others_ok']}, todos {c['n_rho_star_both_ok']}, q_S {c['q_sculptor_self_form_independent']:.2f}"
                    for u, c in m["sculptor_calibration"].items()) + f"; predicción RAR congelada sha {psha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
