#!/usr/bin/env python
"""Preinscripción del frente 2 (β de Fokker-Planck) — ANTES del espectro.

Genera results/2026-08-19_front2_fp_beta/preregistration.{json,md}
desde el módulo canónico (core/fokker_planck_beta.py): coeficientes
derivados, cierres declarados, punto de evaluación, los CUATRO
desenlaces con reglas cuantitativas y el compromiso de no-ajuste.

PROHIBIDO en este script: computar autovalores, s0 o cualquier
espectro. El candado (tests/test_front2_fp_lock.py) vigila que la
preinscripción no cambie después.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fokker_planck_beta import (  # noqa: E402
    BETA_COEFFICIENTS,
    canonical_closure,
    spinodal_canonical_point,
)

OUT = (Path(__file__).resolve().parent.parent / "results"
       / "2026-08-19_front2_fp_beta")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cl = canonical_closure()
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    doc = {
        "front": "2 (exponente de Victoria desde las β reales)",
        "stage": "preinscripción — congelada ANTES de computar ningún "
                 "autovalor/espectro/s0",
        "code_commit": sha,
        "derivation": {
            "object": "β_{M0²}, β_B, β_{C0} de la reducción de "
                      "Fokker-Planck del Flujo del Camino (Def. 4.4)",
            "equation": "dV/dt = a·ΔV − b·(∇V)² (Polchinski d=0, "
                        "kernel isótropo de ritmo unidad)",
            "coefficients": BETA_COEFFICIENTS,
            "declared_approximations": [
                "truncamiento cúbico de la jerarquía (variante "
                "cuártica como sistemático medido)",
                "sector η·χ despreciado: O(δ0⁶) por el escalado (3.2)",
                "diccionario t ↔ lnS: τ = dt/dlnS NO derivable del "
                "corpus recogido — reescala s0 linealmente; el TIPO "
                "espectral y |Im μ|/|Re μ| son invariantes de τ",
                "G isótropa (anisotropía explorada vía g = a/b)",
            ],
        },
        "closure": {
            "a": cl.a, "b": cl.b, "flow_sign": cl.flow_sign,
            "truncation": cl.truncation, "tau": cl.tau,
            "g_grid": list(cl.g_grid),
            "robustness_window": list(cl.robustness_window),
            "s0_band": cl.s0_band, "s0_target": cl.s0_target,
        },
        "evaluation_point": {
            "where": "espinodal D = 0 (Def. 8.4) en unidades naturales",
            "M0_sq": spinodal_canonical_point()[0],
            "B": spinodal_canonical_point()[1],
            "C0": spinodal_canonical_point()[2],
        },
        "outcomes": {
            "A": "sin par complejo en el punto canónico (g = 1, "
                 "signo +): la reducción NO produce la cascada DSI — "
                 "desfavorable al mecanismo, se publica en portada",
            "B": "par complejo con s0 (τ = 1) FUERA de la banda ±10% "
                 "de π/ln10: cascada genérica pero λ ≠ 10 bajo el "
                 "diccionario canónico; se publica τ* para λ = 10",
            "C": "par complejo con s0 (τ = 1) DENTRO de la banda "
                 "±10%: candidato a derivación (condicional al "
                 "diccionario, se declara)",
            "D": "el TIPO espectral cambia dentro de la ventana de "
                 "robustez g ∈ [0.5, 2]: dependiente del cierre — se "
                 "publica el mapa de fases completo y NINGÚN veredicto "
                 "sobre λ",
            "rule": "clasificar primero en el punto canónico (g = 1, "
                    "signo +, truncamiento cúbico, τ = 1); D "
                    "prevalece sobre B/C si el tipo no es robusto en "
                    "la ventana; el barrido g_grid entero se publica "
                    "SIEMPRE, caiga donde caiga",
        },
        "secondary_published_regardless": [
            "Q = |Im μ|/|Re μ| en el punto canónico (invariante de τ)",
            "τ* que λ = 10 exigiría (si hay par complejo)",
            "signo de dD/dt en la espinodal sobre el barrido "
            "(compatibilidad con Obs. 8.6: el flujo debe hundir D)",
            "sensibilidad al truncamiento (cuártico vs cúbico)",
            "control de Langevin: deriva medida de los acoplos vs β "
            "predichas (validación interna E8, régimen declarado)",
        ],
        "pledge": [
            "los coeficientes de BETA_COEFFICIENTS no se tocan "
            "después de ver ningún autovalor",
            "banda, malla g y ventana de robustez congeladas aquí",
            "sin optimización de ningún cierre hacia s0_target "
            "(candado ejecutable contra scipy.optimize/curve_fit en "
            "los ficheros del frente)",
            "el resultado se publica sea cual sea el desenlace, con "
            "el mismo tono",
        ],
        "verdict_scope": "interno al programa (E8): una reducción con "
                         "cierres declarados de la Def. 4.4 — no una "
                         "derivación única; si el corpus fija otro "
                         "cierre, se recalcula con él",
    }
    (OUT / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    g = doc["closure"]
    (OUT / "preregistration.md").write_text(f"""# Preinscripción — frente 2: β de Fokker-Planck (Def. 4.4)

Commit: `{sha[:12]}`. CONGELADA antes de computar ningún autovalor.

## Derivación (cierres declarados)

dV/dt = a·ΔV − b·(∇V)² (Polchinski d = 0); truncamiento cúbico
(cuártico = sistemático); sector η·χ O(δ0⁶) despreciado con
declaración; diccionario τ = dt/dlnS no derivable del corpus recogido
(reescala s0; el tipo espectral y Q = |Im|/|Re| son invariantes).

Coeficientes derivados y congelados (por unidad de t):

    β_M0² = −8·a·B − 2·b·M0⁴
    β_B   = −24·a·C0 − 8·b·M0²·B
    β_C0  = −6·b·(B² + 2·M0²·C0)

## Punto de evaluación

Espinodal D = 0 (Def. 8.4) en unidades naturales: (M0², B, C0) =
({doc["evaluation_point"]["M0_sq"]}, {doc["evaluation_point"]["B"]},
{doc["evaluation_point"]["C0"]}).

## Cierre canónico y espacio de robustez

a = {g["a"]}, b = {g["b"]}, signo = +{g["flow_sign"]}, τ = {g["tau"]};
malla g = a/b: {len(g["g_grid"])} puntos log-espaciados en
[{g["g_grid"][0]}, {g["g_grid"][-1]}]; ventana de robustez
g ∈ {g["robustness_window"]}; banda de s0: ±{g["s0_band"]:.0%} de
π/ln10 = {g["s0_target"]:.4f}.

## Los cuatro desenlaces (regla cuantitativa, congelada)

- **A** — sin par complejo en el punto canónico: la reducción no
  produce la cascada DSI (desfavorable; portada).
- **B** — par complejo, s0(τ=1) fuera de banda: λ ≠ 10 bajo el
  diccionario canónico; se publica τ*.
- **C** — par complejo, s0(τ=1) en banda: candidato (condicional al
  diccionario).
- **D** — el tipo espectral cambia en g ∈ {g["robustness_window"]}:
  dependiente del cierre; mapa completo, sin veredicto sobre λ.

Se publican SIEMPRE: el barrido entero, Q, τ*, el signo de dD/dt
(Obs. 8.6), la sensibilidad al truncamiento y el control de Langevin.

## Compromiso

Sin ajuste de coeficientes tras conocer s0; banda/malla/ventana
congeladas; publicación íntegra sea cual sea el desenlace. Candado:
`tests/test_front2_fp_lock.py`.
""", encoding="utf-8")
    print(f"Preinscripción: {OUT}/preregistration.{{json,md}}")


if __name__ == "__main__":
    main()
