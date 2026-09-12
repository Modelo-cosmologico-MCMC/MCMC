#!/usr/bin/env python
"""Preinscripción de la predicción perturbativa fσ8 — OBLIGATORIA
ANTES de computar la banda (rama theory/perturbations-linear-growth).

El resultado ABURRIDO va preinscrito como desenlace esperado: dado que
el fondo normalizado es casi indistinguible de ΛCDM (#12, ε dominada
por el prior), la expectativa previa es fσ8^MCMC ≈ fσ8^ΛCDM dentro de
la banda del prior de ε_Λ. Ese desenlace NO es fracaso: es el control
de consistencia — el sector lineal no debe inventar una desviación
grande donde el fondo casi no cambia.

Uso: python scripts/run_perturbations_prereg.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.growth_prediction import (  # noqa: E402
    CHAINS_V1,
    DRAW_SEED,
    N_DRAWS,
    OUTDIR,
    SIGMA8_EXTERNAL,
    Z_BAND,
)

PREREG = {
    "front": ("sector perturbativo — crecimiento lineal exacto y "
              "predicción out-of-sample de fσ8(z)"),
    "expected_outcome_A": {
        "statement": (
            "DESENLACE ESPERADO (y NO fracaso — control de "
            "consistencia): fσ8^MCMC ≈ fσ8^ΛCDM dentro de la banda "
            "del prior de ε_Λ. El fondo normalizado es casi "
            "indistinguible de ΛCDM (#12: ε dominada por el prior), "
            "así que el sector lineal NO debe inventar una desviación "
            "grande donde el fondo casi no cambia."),
        "classification_rule": (
            "A ⟺ max_z |R_p50(z) − 1| ≤ envolvente(z) del prior "
            "(|R − 1| con ε = ±1σ_prior = ±0.05 al θ de referencia) "
            "en TODA la malla Y |Δχ²_out-of-sample| ≤ 2.0"),
    },
    "contrary_outcome_B": {
        "statement": (
            "DESENLACE CONTRARIO: desviación mayor que la banda — "
            "señal de error de implementación O de física nueva, A "
            "DISCRIMINAR (no se publica como señal sin discriminar)"),
        "discrimination_procedure": [
            "(i) identidad exacta R(z; ε = 0) ≡ 1 (test de la suite)",
            "(ii) crosscheck del integrador de crecimiento con una "
            "segunda implementación independiente (patrón #14) — "
            "enumerado como paso siguiente, no ejecutado aquí",
            "(iii) solo si (i)+(ii) sobreviven: la desviación se "
            "publica como propiedad del modelo, con su tamaño y su "
            "dependencia de ε",
        ],
    },
    "method_frozen": {
        "growth": ("ODE exacta D'' + (2 + dlnH/dlna)·D' − "
                   "(3/2)·Ω_m(a)·D = 0 integrada (RK4) sobre el H(z) "
                   "del modelo — cosmology/extended_likelihoods."
                   "growth_D_f, YA en el árbol (#14-adjacent); NO la "
                   "aproximación γ de Linder"),
        "posterior": ("SOLO CC+BAO+SNe, sin crecimiento: ajuste v1 "
                      "corregido results/2026-08-10_production_fit/"
                      "chains_mcmc.npz (H0, Ω_m, ε, z_trans; 16000 "
                      "muestras)"),
        "chains_sha256": None,      # se computa al congelar
        "zero_new_parameters": (
            "el número con cero parámetros nuevos es la RAZÓN "
            "R(z) = [f·D](ε)/[f·D](ε = 0) al mismo (H0, Ω_m): σ8 se "
            "cancela exactamente"),
        "sigma8_external": {
            "value": SIGMA8_EXTERNAL,
            "role": ("SOLO para el contraste absoluto con la "
                     "compilación RSD; calibración externa declarada "
                     "(constante del corpus, estatuto Ap. F), "
                     "idéntica en ambos brazos, NO ajustada")},
        "rsd_data": ("compilación embebida data/boss_eboss/"
                     "fsigma8.txt (referencias por punto en la "
                     "cabecera; checksums en download_data) — "
                     "out-of-sample: NO entró en el ajuste v1"),
        "z_band": [float(Z_BAND[0]), float(Z_BAND[-1]),
                   int(len(Z_BAND))],
        "n_draws": N_DRAWS,
        "draw_seed": DRAW_SEED,
        "theta_reference": ("mediana por parámetro del posterior de "
                            "fondo"),
        "no_fitting": ("aquí NO se ajusta nada: sin optimizadores, "
                       "sin samplers, sin σ8 libre (vigilado por "
                       "test)"),
    },
    "epistemic_order_declared_not_executed": {
        "order": ("ε_c/Atlas → µ(k,z), η(k,z) → fσ8, C_L^φφ, P(k) → "
                  "datos (frentes 3 y 10)"),
        "this_work": ("usa µ = η = 1 (límite GR): la predicción de "
                      "esta rama es el sector de fondo propagado al "
                      "crecimiento lineal, con cero parámetros "
                      "nuevos"),
        "prohibition": (
            "PROHIBICIÓN EXPLÍCITA: nunca elegir µ(k,z), η(k,z) "
            "desde los datos de lensing — µ y η deben DERIVARSE de "
            "Atlas/ε_c; el artefacto que justificará el correo único "
            "a Silvestri/Huterer es exactamente ese (ecuaciones con "
            "límite GR y amplitud prevista), no «estoy desarrollando "
            "un sector perturbativo»"),
    },
    "publication_commitment": (
        "la banda predicha se publica como artefacto versionado "
        "(prediction_band.json + report.md) SEA CUAL SEA el "
        "desenlace, citando esta preinscripción por sha256; ni la "
        "regla de clasificación ni las semillas se retocan tras ver "
        "el resultado"),
}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    chains_sha = hashlib.sha256(CHAINS_V1.read_bytes()).hexdigest()
    doc = {
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        **PREREG,
    }
    doc["method_frozen"]["chains_sha256"] = chains_sha
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    md = [
        "# Preinscripción — predicción perturbativa fσ8 "
        "(out-of-sample)\n",
        f"- **Commit**: `{sha}`",
        f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
        f"- **Cadenas de fondo (CC+BAO+SNe, sin crecimiento)**: "
        f"`{CHAINS_V1.name}` sha256 `{chains_sha[:12]}…`",
        f"- **σ8 externa (solo contraste absoluto)**: "
        f"{SIGMA8_EXTERNAL}",
        f"- **Banda**: z ∈ [0, 2] (41 puntos), {N_DRAWS} draws, "
        f"semilla {DRAW_SEED}",
        "",
        "**Desenlace esperado (A, y NO fracaso)**: fσ8^MCMC ≈ "
        "fσ8^ΛCDM dentro de la banda del prior de ε_Λ — el control "
        "de consistencia del sector lineal. Regla: "
        "max_z |R_p50 − 1| ≤ envolvente(ε = ±0.05) y "
        "|Δχ²_oos| ≤ 2.0.",
        "",
        "**Desenlace contrario (B)**: desviación mayor — error de "
        "implementación o física nueva, a discriminar por el "
        "procedimiento enumerado (identidad ε = 0; crosscheck "
        "independiente del integrador; solo entonces publicación "
        "como propiedad del modelo).",
        "",
        "**Orden epistemológico declarado, no ejecutado**: ε_c/Atlas "
        "→ µ, η → fσ8, C_L^φφ, P(k) → datos. **Prohibición**: nunca "
        "elegir µ, η desde los datos de lensing.",
    ]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                               encoding="utf-8")
    print(f"Preinscripción congelada en {OUTDIR} (commit {sha[:9]}; "
          f"cadenas {chains_sha[:12]})")


if __name__ == "__main__":
    main()
