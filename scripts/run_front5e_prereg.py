#!/usr/bin/env python
"""Preinscripción computacional del Frente 5E (OBLIGATORIA antes de
ingerir o inspeccionar SPARC).

Genera results/2026-08-16_front5e_sparc/preregistration.{json,md} con:
la amplitud transferida (calculada por la función canónica, no
copiada), las convenciones congeladas, la hipótesis primaria, la
comparación primaria, los estadísticos predeclarados, los cortes y
subconjuntos predeclarados, y el commit exacto del código.

Uso: python scripts/run_front5e_prereg.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynamics.sculptor_transfer import frozen_config  # noqa: E402

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-16_front5e_sparc")

PREDECLARED = {
    "hypothesis_H5E": (
        "A_req^Sculptor es compatible con curvas de rotación SPARC "
        "reales: añadir Cronos(A congelada) a los bariones no empeora "
        "sistemáticamente el ajuste."),
    "primary_comparison": (
        "Por galaxia: M0 = bariones vs M1 = bariones + "
        "Cronos(A_req^Sculptor). v_model² = v_bar² + v_cronos² "
        "(composición en cuadratura — consecuencia de la dinámica: "
        "las aceleraciones se suman, v² = R·g). Cronos no añade "
        "ningún parámetro ajustado a SPARC."),
    "statistics": {
        "chi2": "χ² = Σ_i [(v_model(R_i) − v_obs,i)/e_v,i]² por galaxia",
        "delta_chi2": "Δχ² = χ²(bar+Cr) − χ²(bar); Δχ²<0 ⟹ Cronos "
                      "mejora; Δχ²>0 ⟹ empeora",
        "aggregates": [
            "N_bins por galaxia", "χ²_ν por galaxia (ν = N_bins)",
            "mediana de Δχ² por galaxia",
            "fracción de galaxias con Δχ² < 0 (mejoradas)",
            "fracción con Δχ² > 0 (empeoradas)",
            "suma global Σ Δχ² (NUNCA sola: puede dominarla una "
            "galaxia con muchos puntos)",
            "bootstrap entre galaxias (semilla y n en la config "
            "congelada) de la mediana y de la suma",
        ],
        "structural_metrics": {
            "B_inner": "v_cronos(2R_d/3) / v_obs(2R_d/3)",
            "F_outer": "v_cronos / sqrt(max(v_obs² − v_bar², 0)) en "
                       "x = 3-4, donde esté definido",
            "shape_x5": "v_cronos(2R_d/3)/v_cronos(4R_d) ≈ 5 — "
                        "identidad de la clase 5C, medida por galaxia",
        },
    },
    "model_scopes": {
        "5E-A_exponential": (
            "test PRINCIPAL: exactamente la realización de disco "
            "exponencial del 5C (ρ del plano medio desde Σ0 = "
            "Υ_d·L_disk/(2π·R_d²) y ζ congelado), A = A_req^Sculptor."),
        "5E-B_extended": (
            "solo si el catálogo trae fotometría suficiente "
            "(disco+bulbo+gas): extensión bariónica realista, "
            "declarada como HIPÓTESIS NUEVA — nunca presentada como "
            "la demostración del 5C."),
    },
    "predeclared_cuts_and_subsets": {
        "full_sample": "toda galaxia ingerible con N_bins ≥ 5 y "
                       "R_d > 0 en el catálogo",
        "quality_sample": "corte de calidad del PROPIO catálogo "
                          "(bandera Q ≤ 2 si existe; inclinación "
                          "i ≥ 30° si existe) — metadatos SPARC, "
                          "nunca el residuo de Cronos",
        "HSB_LSB_split": "por la mediana de la densidad superficial "
                         "característica observable del catálogo "
                         "(columna concreta fijada en schema_report "
                         "ANTES de calcular ningún Δχ²)",
        "gas_dominated": "v_gas² > Υ_d·v_disk² en el último punto "
                         "medido (observables, sin Cronos)",
        "no_removal_rule": "ninguna galaxia se excluye por Δχ² ni "
                           "por incomodidad del resultado",
    },
    "verdict_scope_precommitment": {
        "if_systematic_worsening": (
            "la realización débil de Cronos probada NO constituye un "
            "mecanismo galáctico común capaz de explicar "
            "simultáneamente Sculptor y discos rotantes bajo las "
            "hipótesis evaluadas — NO 'MCMC falsado', NO 'ρ_id "
            "falsada', NO 'toda dinámica Cronos falsada'"),
        "if_partial": "compatibilidad parcial que exige validación "
                      "fuera de muestra adicional (LITTLE THINGS) — "
                      "sin claim de validación universal",
        "if_5EB_works_and_5EA_fails": (
            "la extensión bariónica es una hipótesis NUEVA a validar "
            "independientemente — no se reescribe el 5C"),
        "if_data_unavailable": (
            "5E observacional ABIERTO por ausencia de datos "
            "ingeridos; el resultado estructural 5C permanece y no "
            "se eleva a veredicto observacional"),
    },
    "rho_id_untouched": (
        "ρ_id no se usa para reparar el 5E: M_id(<347 pc) = 1.78e7 "
        "M⊙ y la curva ρ0(r_c) (marcador exacto r_c = 300 pc, "
        "ρ0 = 0.176 M⊙/pc³) quedan como estaban."),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = frozen_config()
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "front": "5E — falsación cruzada Sculptor ↔ SPARC",
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "generating_function":
            "dynamics.sculptor_transfer.frozen_config "
            "(A: dynamics.sculptor_transfer.sculptor_A_req)",
        "A_units": "(M_sol/pc^3)^(-3/2); epsilon_c = A * rho^(3/2)",
        "eq_11_5_reading": (
            "la transferencia de A es independiente de la lectura "
            "L/P de la ec. (11.5) — A_req es el problema inverso de "
            "Sculptor, no la cota; ambas lecturas declaradas en "
            "dynamics/weak_field.py"),
        "frozen_config": asdict(cfg),
        **PREDECLARED,
    }
    (OUT / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    md = ["# Preinscripción del Frente 5E (antes de tocar SPARC)\n",
          f"- **Commit**: `{sha}`",
          f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
          f"- **A transferida (Υ⋆ = {cfg.upsilon_sculptor:.0f})**: "
          f"{cfg.A_sculptor:.6e} (M⊙/pc³)^(−3/2) — calculada por "
          "`dynamics.sculptor_transfer.sculptor_A_req`, no copiada",
          f"- **Sensibilidad declarada**: A(Υ⋆=1) = "
          f"{cfg.A_sculptor_sensitivity[0]:.3e}, A(Υ⋆=3) = "
          f"{cfg.A_sculptor_sensitivity[1]:.3e}",
          f"- **β (Sculptor)**: {cfg.beta_sculptor}; **R_half**: "
          f"{cfg.R_half_pc} pc; **σ_obs**: {cfg.sigma_obs_kms} km/s",
          f"- **Convención SPARC congelada**: Υ_disk = "
          f"{cfg.upsilon_disk_sparc}, Υ_bul = "
          f"{cfg.upsilon_bulge_sparc} (3.6 μm, estándar Lelli+16); "
          f"sensibilidad Υ_disk ∈ {cfg.upsilon_disk_sensitivity}",
          f"- **ζ = h/R_d**: {cfg.zeta_disc} (sensibilidad "
          f"{cfg.zeta_sensitivity})",
          f"- **Bootstrap**: n = {cfg.bootstrap_n}, semilla = "
          f"{cfg.bootstrap_seed}",
          "",
          f"**Hipótesis primaria H_5E**: {PREDECLARED['hypothesis_H5E']}",
          "",
          f"**Comparación primaria**: {PREDECLARED['primary_comparison']}",
          "",
          "**Regla central**: no se pregunta qué parámetros hacen "
          "funcionar a Cronos en SPARC; se pregunta si la predicción "
          "que Sculptor ya fijó sobrevive cuando SPARC no puede "
          "modificarla. A queda congelada; el candado es ejecutable "
          "(tests/test_front5e_lock.py). El detalle completo de "
          "estadísticos, cortes, subconjuntos y alcance del veredicto "
          "está en preregistration.json (mismo generador).",
          ]
    (OUT / "preregistration.md").write_text("\n".join(md) + "\n",
                                            encoding="utf-8")
    print(f"A congelada = {cfg.A_sculptor:.6e}  (commit {sha[:9]})")
    print(f"Preinscripción: {OUT}/preregistration.{{json,md}}")


if __name__ == "__main__":
    main()
