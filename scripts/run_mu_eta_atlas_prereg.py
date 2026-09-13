#!/usr/bin/env python
"""Preinscripción del canal Atlas de µ/η (frente 3) — OBLIGATORIA antes
de ejecutar la re-derivación simbólica que se publica y el arnés
numérico E2.

Congela: las identidades E1 que la derivación desde la acción debe
reproducir (fórmulas fijadas en cosmology/mu_eta_atlas.py), el punto y
los umbrales del arnés E2 (integración sin aproximación QS), los
desenlaces enumerados, las fronteras declaradas (colas, superhorizonte,
PPN, acoplamiento fuerte) y las prohibiciones (sin datos; sin ajustar
(λ_K, ξ, α_a) a nada).

Uso: python scripts/run_mu_eta_atlas_prereg.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_atlas import (  # noqa: E402
    eta_tail_coefficient,
    growth_index_matter_era,
    mu_atlas_subhorizon,
)
from mcmc_ontology import constants as C  # noqa: E402

OUTDIR = (Path(__file__).resolve().parent.parent / "results"
          / "2026-09-13_mu_eta_atlas")

HARNESS_POINT = {"lambda_K": 1.05, "xi": 1.02, "alpha_a": 0.30}
HARNESS_K_OVER_H0 = [66.667, 200.0, 600.0]
HARNESS_A_START = 1e-3
HARNESS_WINDOW = [0.3, 1.0]
E2_TOL_MU = 0.01          # |mediana(µ_num)/µ_QS − 1| ≤ 1 %
E2_TOL_ETA = 0.01         # |mediana(η_num) − 1| ≤ 1 %
E2_TOL_P = 0.01           # |p_num/p_QS − 1| ≤ 1 %
E2_PURITY_STD = 0.01      # std(p) en la ventana ≤ 0.01 (modo puro)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    hp = HARNESS_POINT
    doc = {
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "front": ("frente 3 — canal Atlas de µ(k,a), η(k,a) derivado desde "
                  "la Acción de Gea (9.1) + Término de Cronos (9.4)"),
        "statute": (
            "derivación desde la acción del tratado en gauge unitario con "
            "materia = polvo (Sorkin–Schutz), sin objetos ajenos; "
            "verificación por cuatro vías (límite GR, fondo/Sello, cruce "
            "khronométrico, integración numérica completa). Es teoría "
            "pura: no carga datos ni ajusta parámetros."),
        "objects_from_treatise": {
            "action": ("L = (1/16πG_B) N√q [K_ijK^ij − λ_K K² + ξ R³[q] + "
                       "α_a a_i a^i] + L_polvo"),
            "parameters": {"lambda_K": "1 + ε_K (constants.EPSILON_K = "
                                       f"{C.EPSILON_K})",
                           "xi": "c_T² = ξ ⟹ GW170817 fija ξ = 1 a 1e-15",
                           "alpha_a": "ventana (9.4): 0 < α_a < 2ξ"},
            "normalization": "G_N ≡ G_B/ξ (Sello de Newton 9.3)",
        },
        "E1_identities_to_lock": {
            "statement": ("la re-derivación simbólica desde la acción debe "
                          "reproducir EXACTAMENTE las formas cerradas "
                          "fijadas en cosmology/mu_eta_atlas.py"),
            "identities": [
                "fondo: G_cosmo = 2G_B/(3λ_K − 1) (Sello 9.3 reproducido)",
                "δ = j0 + 3φ (contraste del polvo)",
                "µ_Atlas(e) = 2ξ/[(2ξ − α_a) + 3(3λ_K − 1)e²]",
                "µ_Atlas(e → 0)·(1 − α_a/2ξ) = 1",
                "η_Atlas(e → 0) = 1 (para λ_K ≠ 1; en λ_K = 1 la forma QS "
                "es 0/0 y el límite e → 0 se toma primero)",
                "G_growth = G_local = G_B/(ξ − α_a/2) (cancelación)",
                "c_s²·α_a(3λ_K − 1) = ξ(2ξ − α_a)(λ_K − 1)",
                "coef. cinético del khronon ∝ (3λ_K − 1)/(λ_K − 1)",
                "c_T² = ξ",
                "límite GR (1, 1, 0) ⟹ µ = η = 1",
            ],
            "expected_numbers": {
                "mu_subhorizon_harness": mu_atlas_subhorizon(
                    hp["lambda_K"], hp["xi"], hp["alpha_a"]),
                "p_qs_harness": growth_index_matter_era(
                    hp["lambda_K"], hp["xi"], hp["alpha_a"]),
                "eta_tail_coefficient_harness": eta_tail_coefficient(
                    hp["lambda_K"], hp["xi"]),
                "eta_tail_coefficient_treatise": eta_tail_coefficient(
                    1.0 + C.EPSILON_K, 1.0),
            },
        },
        "E2_numerical_integration": {
            "statement": ("integración del sistema lineal COMPLETO (sin "
                          "aproximación QS) en materia dominante, ICs "
                          "QS-consistentes en a_start y RELAJACIÓN al modo "
                          "creciente; medición en la ventana tardía"),
            "harness_point": hp,
            "k_over_H0": HARNESS_K_OVER_H0,
            "a_start": HARNESS_A_START,
            "window_a": HARNESS_WINDOW,
            "solver": "Radau, rtol 1e-9, atol 1e-15",
            "rules": {
                "mu": f"|mediana(µ_num)/µ_QS(e) − 1| ≤ {E2_TOL_MU}",
                "eta": f"|mediana(η_num) − 1| ≤ {E2_TOL_ETA}",
                "growth_index": (f"|p_num/p_QS − 1| ≤ {E2_TOL_P} con p_QS "
                                 "analítico (independiente de la "
                                 "normalización de µ)"),
                "mode_purity": (f"std(p) en la ventana ≤ {E2_PURITY_STD} y "
                                "ningún cambio de signo de δ en la ventana"),
            },
            "declared_non_test": (
                "la cola QS de η (coef. ~45 en el punto del arnés) NO se "
                "espera reproducida por η_num: la aproximación QS descarta "
                "∂_t y velocidades del mismo orden e²; se publica la "
                "diferencia como diagnóstico, sin puerta"),
        },
        "outcomes_enumerated": {
            "A_expected": ("E1 todas las identidades reproducidas Y E2 "
                           "dentro de umbrales ⟹ ATLAS_STATUS pasa a "
                           "DERIVADO-NULO al orden dominante: la firma "
                           "sub-horizonte de (µ, η) es SOLO la del canal "
                           "Cronos"),
            "B_contrary": ("alguna identidad no reproducida o E2 fuera de "
                           "umbral ⟹ error de implementación (arnés/ICs) o "
                           "de derivación (formas cerradas), a DISCRIMINAR: "
                           "(i) rehacer E1 con la derivación independiente; "
                           "(ii) variar a_start y ventana; (iii) solo "
                           "entonces revisar las formas cerradas. Nunca se "
                           "ajustan los umbrales"),
        },
        "declared_frontiers_not_derived": [
            "coeficientes completos de las colas O(e²/(λ_K−1)) — exigen el "
            "sector de velocidades (la cola QS de η tiene el polo "
            "1/(λ_K−1): parámetro pequeño efectivo e/√(λ_K−1) ~ aH/(c_s k); "
            "con ε_K = 0.012 la ventana sub-horizonte se estrecha ×~9)",
            "régimen superhorizonte",
            "cotas PPN de marco preferido de la clase khronométrica sobre "
            "(ε_K, α_a) — contraste externo pendiente, análogo al de BBN",
            "acoplamiento fuerte a Λ_sc ~ M_P√α_a",
        ],
        "erratum_candidate_v36_H22": (
            "H.2.2 escribe «c_s² = α/(2−α) → 0 cuando α → 0»; la derivación "
            "da c_s² = (2−α_a)(λ_K−1)/(α_a(3λ_K−1)) (ξ = 1): → 0 cuando "
            "λ_K → 1, diverge a λ_K fijo cuando α_a → 0. Λ_sc ~ M_P√α_a → 0 "
            "se mantiene. Precisión, no retractación — propuesta para la "
            "fe de erratas de v36, decisión del autor"),
        "residues_refinement": (
            "G_cosmo/G_local = (2ξ − α_a)/(3λ_K − 1) ≈ 1 − (3/2)ε_K − α_a/2: "
            "BBN acota la combinación; el −1.8 % de (9.5) es α_a ≪ ε_K; el "
            "crecimiento sub-horizonte no mide ninguna de las dos "
            "(cancelación)"),
        "prohibitions": {
            "no_data": "ningún dataset entra; vigilado por test",
            "no_fitting": "(λ_K, ξ, α_a) no se ajustan a nada aquí",
            "no_threshold_tuning": ("umbrales E2, punto del arnés, a_start y "
                                    "ventana congelados; un cambio exige "
                                    "nueva preinscripción"),
        },
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Preinscripción — canal Atlas de µ/η (frente 3)\n",
        f"- **Commit**: `{sha}`",
        f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
        f"- **Punto del arnés E2**: λ_K = {hp['lambda_K']}, ξ = {hp['xi']}, "
        f"α_a = {hp['alpha_a']}; k/H0 ∈ {HARNESS_K_OVER_H0}; a_start = "
        f"{HARNESS_A_START}; ventana a ∈ {HARNESS_WINDOW}",
        f"- **Umbrales E2**: µ {E2_TOL_MU}, η {E2_TOL_ETA}, p {E2_TOL_P}, "
        f"pureza std(p) ≤ {E2_PURITY_STD}",
        f"- **Esperados**: µ_sub = "
        f"{doc['E1_identities_to_lock']['expected_numbers']['mu_subhorizon_harness']:.6f}, "
        f"p_QS = "
        f"{doc['E1_identities_to_lock']['expected_numbers']['p_qs_harness']:.6f}",
        "",
        "**E1**: la re-derivación desde la acción reproduce las diez "
        "identidades fijadas. **E2**: µ_num → µ_QS, η_num → 1, p_num → "
        "p_QS al 1 %, modo puro. **Desenlace A esperado** ⟹ Atlas "
        "DERIVADO-NULO al orden dominante. **B** ⟹ error a discriminar; "
        "los umbrales no se tocan.",
        "",
        "**Fronteras**: colas O(e²/(λ_K−1)) (sector de velocidades), "
        "superhorizonte, PPN, acoplamiento fuerte. **Erratum candidata** "
        "H.2.2 (decisión del autor). Sin datos; sin ajuste.",
    ]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                               encoding="utf-8")
    print(f"Preinscripción congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
