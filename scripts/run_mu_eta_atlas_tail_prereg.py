#!/usr/bin/env python
"""Congela la preinscripción de E3_Atlas — la cola física O(e²) del canal
Atlas CON el sector de velocidades — ANTES de ejecutar la escalera de
producción y el arnés adiabático. No importa sympy ni integra nada: solo
fija reglas, umbrales, malla y desenlaces enumerados.

Uso: python scripts/run_mu_eta_atlas_tail_prereg.py
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
    tail_pole_residues,
)

OUTDIR = Path(__file__).resolve().parent.parent / "results" / \
    "2026-09-13_mu_eta_atlas_tail"

HARNESS = {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
LADDER_GRID = [[1.05, 0.30], [1.0125, 0.30], [1.10, 0.30], [1.05, 0.10],
               [1.05, 0.60], [1.012, 0.012]]
RESIDUE_ALPHAS = [0.10, 0.30, 0.60]
H_STEPS = [1 / 200, 1 / 100, 1 / 50, 1 / 25]
K_VALUES = [66.667, 133.333, 200.0]
RULES = {"E3a_residue_rel_tol": 1e-4, "E3b_slope_rel_tol": 0.10,
         "E3b_loglog_exponent_range": [1.9, 2.1], "E3b_purity_std_max": 0.01}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    lam, al = HARNESS["lambda_K"], HARNESS["alpha_a"]
    expected_residues = {str(a): tail_pole_residues(a) for a in RESIDUE_ALPHAS}
    doc = {
        "title": "E3_Atlas — cola O(e²) del canal Atlas con el sector de "
                 "velocidades: escalera exacta del modo creciente y arnés "
                 "numérico con condiciones iniciales adiabáticas",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "generator": "scripts/run_mu_eta_atlas_tail_prereg.py",
        "context": {
            "predecessor": "results/2026-09-13_mu_eta_atlas (PR #18): desenlace "
                           "A, colas O(e²) DECLARADAS con coeficientes "
                           "pendientes; el arnés E2 con ICs QS-consistentes "
                           "midió η − 1 = +2.97e-4 en k/H0 = 66.667 "
                           "(e ≈ 0.01) y residuos que no escalan como e²",
            "author_independent_derivation": "adenda C.2 (13-sep, noche 2): "
                "escalera exacta t^(−n/3) sobre el sistema completo, ξ = 1; "
                "residuos del polo P_η = 3α_a/(2−α_a), P_µ = −P_η·p(2p−1)/3; "
                "predicción en el punto del arnés: η − 1 = 17.07·e² "
                "(1.71e-3 en e = 0.01); partes regulares Q_η ≈ 5.0/6.1/8.4, "
                "Q_µ ≈ −5.0/−6.2/−9.4 en α_a = 0.1/0.3/0.6",
            "hypothesis_for_E2_discrepancy": "las ICs QS-consistentes excitan "
                "los modos oscilatorios del khronon (c_s² = 0.132 en el punto "
                "del arnés), que contaminan η con un suelo que no escala como "
                "e²; las ICs adiabáticas de la escalera aíslan el modo "
                "creciente puro. Es una hipótesis A CONTRASTAR, no un "
                "resultado",
        },
        "assumptions_frozen": {
            "xi": 1.0, "background": "materia dominante, a = t^{2/3}, κ = m = 1, "
            "J̄⁰ = (4/3)(3λ_K − 1)", "gauge": "unitario (foliación entrópica "
            "física)", "matter": "polvo (Sorkin–Schutz)",
            "normalization": "c_{j0,0} = 1; los cocientes µ, η no dependen de "
            "la dirección libre de renormalización del modo",
            "mu_reference": "µ_loc = µ·(1 − α_a/2) (respecto de G_local = "
            "G_growth, cancelación de #18)",
        },
        "E3a_ladder": {
            "tower": {"nlow": -4, "nhigh": 16, "prof": 12},
            "grid_lambda_alpha": LADDER_GRID,
            "leading_order_rules": [
                "mu0 == 1/(1 − α_a/2) EXACTO (simbólico) en toda la malla",
                "eta0 == 1 EXACTO (simbólico) en toda la malla",
                "cabeceras: j0 sin términos super-dominantes; ψ y φ empiezan "
                "en n = 2; términos impares n = 3 nulos (paridad física)",
            ],
            "residues": {
                "alphas": RESIDUE_ALPHAS, "h_steps": H_STEPS,
                "scheme": "f(h) = h·coef(1 + h); ajuste polinómico cúbico "
                          "exacto por los cuatro pasos; P = f(0), Q = f'(0); "
                          "error O(h⁴)",
                "closed_forms_to_lock": {
                    "P_eta": "3α_a/(2 − α_a)",
                    "P_mu_local": "−P_η · p(2p−1)/3 con p(p+½) = 3/(2−α_a)",
                    "identity": "P_µ/P_η = −p(2p−1)/3 (GR: −1/3)",
                    "qs_over_physical": "2(2−α_a)/(3α_a) — el residuo QS "
                                        "truncado es 2 y no depende de α_a"},
                "rule": f"|P/P_cerrado − 1| ≤ {RULES['E3a_residue_rel_tol']:g} "
                        "para P_η y P_µ en los tres α_a",
                "expected_numbers": expected_residues,
            },
            "expected_numbers_harness_point": {
                "mu0": mu_atlas_subhorizon(lam, 1.0, al),
                "p": growth_index_matter_era(lam, 1.0, al),
                "eta_qs_truncated_coefficient": eta_tail_coefficient(lam, 1.0),
                "mu_local_qs_truncated_coefficient": -3 * (3 * lam - 1) / (2 - al),
                "author_prediction_coef_eta": 17.07,
            },
        },
        "E3b_adiabatic_harness": {
            "point": HARNESS, "k_over_H0": K_VALUES, "a_start": 0.1,
            "ic_tower_nmax": 8, "e_fit_max": 0.02,
            "solver": {"method": "Radau", "rtol": 1e-10, "atol": 1e-18,
                       "n_eval": 600},
            "rules": {
                "slope_eta": f"|s_η/coef_η(escalera) − 1| ≤ "
                             f"{RULES['E3b_slope_rel_tol']:g} en CADA k",
                "slope_mu_local": f"|s_µ/coef_µ(escalera) − 1| ≤ "
                                  f"{RULES['E3b_slope_rel_tol']:g} en CADA k",
                "loglog_exponent": "exponente de |η − 1| frente a e ∈ "
                                   f"{RULES['E3b_loglog_exponent_range']} en "
                                   "cada k",
                "mode_purity": f"std(p_loc) ≤ {RULES['E3b_purity_std_max']:g}",
                "solver": "success en cada k",
            },
            "qs_reference_for_B": "las pendientes coinciden (≤ 10 %) con los "
                                  "coeficientes QS truncados en vez de con la "
                                  "escalera",
            "control_non_gating": "misma integración con las ICs QS-"
                "consistentes del arnés de #18 (a_start = 1e-3, ventana "
                "a ≥ 0.3): se publican pendiente, exponente log-log y fracción "
                "de signos negativos de η − 1 como diagnóstico del suelo de "
                "ruido. NO gobierna el desenlace",
        },
        "outcomes_enumerated": {
            "rules": RULES,
            "E3a": {"A": "todas las reglas de orden dominante y de residuos en "
                         "PASS → residuos del polo DERIVADOS y fijados en "
                         "cosmology.mu_eta_atlas.tail_pole_residues",
                    "B": "alguna falla → las formas cerradas NO se fijan; se "
                         "publica el número obtenido y la escalera queda como "
                         "cálculo abierto"},
            "E3b": {"A_confirmed": "todas las reglas en PASS → la cola física "
                        "queda CONFIRMADA por integración completa; la "
                        "discrepancia del arnés E2 se atribuye a las ICs (si el "
                        "control lo muestra) y ATLAS_STATUS pasa a «residuo del "
                        "polo derivado y confirmado numéricamente»",
                    "B_qs_wins": "las pendientes coinciden con los coeficientes "
                        "QS truncados y no con la escalera → la escalera es "
                        "errónea; SOSPECHA DE ERROR EN LA DERIVACIÓN PRIMERO; "
                        "las formas cerradas de E3a se retiran del módulo",
                    "C_open": "ni A ni B → discrepancia ABIERTA; el estatuto "
                        "permanece «derivación exacta de la escalera, pendiente "
                        "de confirmación numérica independiente»; se publican "
                        "todos los números"},
            "rule": "Nunca se ajustan los umbrales tras ver los números. E3a y "
                    "E3b se clasifican por separado y se publican ambos con el "
                    "mismo peso. El desenlace aburrido no es un fracaso.",
        },
        "declared_frontiers_not_derived": [
            "partes regulares Q_η(α_a), Q_µ(α_a): solo numéricas",
            "ξ ≠ 1 (GW170817 fija ξ = 1; sin derivación fuera)",
            "régimen superhorizonte y O(e⁴)",
            "cotas PPN de marco preferido sobre (ε_K, α_a)",
            "acoplamiento fuerte a Λ_sc ~ M_P√α_a",
        ],
        "prohibitions": {"no_data": True, "no_lensing_choice_of_mu_eta": True,
                         "no_threshold_tuning": True,
                         "no_change_to_E1_E2_of_PR18": True},
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción E3_Atlas — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene "
          "el generador). ANTES de ejecutar la escalera de producción y el "
          "arnés adiabático.", "",
          "## E3a — escalera exacta (ξ = 1)", "",
          f"Malla (λ_K, α_a): {LADDER_GRID}. Orden dominante: µ₀ = 1/(1−α_a/2) "
          "y η₀ = 1 exactos. Residuos del polo en α_a = "
          f"{RESIDUE_ALPHAS} con pasos h = {H_STEPS} (ajuste cúbico): "
          f"|P/P_cerrado − 1| ≤ {RULES['E3a_residue_rel_tol']:g} para P_η = "
          "3α/(2−α) y P_µ = −P_η·p(2p−1)/3.", "",
          "## E3b — arnés con ICs adiabáticas", "",
          f"Punto {HARNESS}, k/H0 = {K_VALUES}, a_start = 0.1, ajuste en "
          f"e ≤ 0.02. Reglas: pendientes de η−1 y µ_loc−1 frente a e² dentro "
          f"del {RULES['E3b_slope_rel_tol']:.0%} de la escalera en cada k; "
          "exponente log-log ∈ [1.9, 2.1]; pureza de modo ≤ 0.01. Predicción "
          "independiente del autor (adenda C.2): 17.07·e² en η.", "",
          "## Desenlaces", "",
          "E3a: A (residuos derivados) / B (no fijados). E3b: A (cola "
          "confirmada) / B (gana la QS truncada: error en la escalera) / C "
          "(abierto). Nunca se ajustan los umbrales; todo se publica.", "",
          "Control no vinculante: ICs QS-consistentes de #18 (suelo de ruido).",
          "", "Sin datos; ξ = 1; fronteras declaradas en el JSON."]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                                encoding="utf-8")
    print(f"Preinscripción E3_Atlas congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
