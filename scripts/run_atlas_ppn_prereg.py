#!/usr/bin/env python
"""Congela la preinscripción de E4_Atlas ANTES de ejecutar:
E4a — PPN de marco preferido del sector Atlas (límite khronométrico de
Einstein-aether), cota sobre α_a desde las cotas transcritas y colas de
sonido en la cota (escalera exacta); E4b — cierre de E3b con una
preinscripción NUEVA (ajuste e² + e⁴ en vez de la puerta de exponente).

Uso: python scripts/run_atlas_ppn_prereg.py
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

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-15_mu_eta_atlas_ppn"
EPS_K = 0.012
RULES_A = {"alpha_a_max_leq": 1e-5, "tail_unobservable_leq": 1e-5,
           "tail_observable_geq": 1e-3, "residues_cleanliness_leq": 1e-3,
           "tail_k_h_over_Mpc": 0.02}
RULES_B = {"slope_rel_tol": 0.10, "purity_std_max": 0.01}
HARNESS = {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
K_VALUES = [66.667, 133.333, 200.0]
IDENTITIES = ["c14_is_alpha", "c13_is_beta", "alpha1_matches_closed",
              "alpha2_matches_closed", "alpha1_beta0_is_minus_4alpha",
              "alpha2_beta0_matches_closed", "alpha2_beta0_leading_is_minus_alpha_half",
              "cs2_matches_at_beta0", "Gcosmo_matches_at_beta0", "cT2_xi1_is_beta0"]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    lam, al = HARNESS["lambda_K"], HARNESS["alpha_a"]
    doc = {
        "title": "E4_Atlas — PPN de marco preferido del sector Atlas (α₁, α₂ como "
                 "límite khronométrico de Einstein-aether), cota sobre α_a y colas "
                 "de sonido en la cota; cierre de E3b con ajuste e² + e⁴",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "generator": "scripts/run_atlas_ppn_prereg.py",
        "E4a_ppn": {
            "derivation": "α₁, α₂ de Einstein-aether (Foster & Jacobson 2006) en el "
                          "límite c_ω → ∞ con c₁₄ = α, c₁₃ = β, c₂ = λ; mapeo al "
                          "sector del tratado α_a = α, λ_K = 1 + λ, ξ = 1 ⇔ β = 0",
            "identities_to_lock": IDENTITIES,
            "closed_forms": {"alpha1": "4(α−2β)/(β−1) → −4α_a (β = 0)",
                             "alpha2": "(α−2β)(αβ+2αλ+α−β²−3βλ−3β−λ)/((α−2)(β−1)(β+λ)) "
                                       "→ α_a(2α_aλ+α_a−λ)/(λ(2−α_a)) = −α_a/2 + O(α_a²)"},
            "bounds_dataset": "ppn_bounds",
            "provenance_caveat": "cuatro cotas publicadas TRANSCRITAS (nota del "
                "autor + literatura); bytes oficiales no verificables desde el "
                "entorno: el artefacto lleva el aviso hasta que el autor verifique",
            "governing_rule": "la cota de campo débil más restrictiva gobierna "
                              "(LLR sobre α₁, giro solar sobre α₂); las de campo "
                              "fuerte (α̂) se publican como indicativas",
            "lambda_nominal": EPS_K,
            "lambda_scan": [0.001, 0.003, 0.012, 0.03, 0.1],
            "tails_at_bound": {
                "ladder_point": {"lambda_K": 1.0 + EPS_K, "alpha_a": 1e-6},
                "trend_points": [[1.0 + EPS_K, 1e-4], [1.0 + EPS_K, 1e-3],
                                 [1.001, 1e-6], [1.05, 1e-6]],
                "trend_note": "informativos, NO gobiernan el desenlace: tendencia en α_a a λ_K fijo y en λ_K a α_a fijo (α_a = 1e-6)",
                "k_h_over_Mpc": [0.02, 0.05, 0.1],
                "convention": "z = 0, e = (H0/c)/k, H0/c = 3.336e-4 h/Mpc; "
                              "coeficiente COMPLETO de e² de la escalera exacta "
                              "(no el residuo del polo): en α_a ≪ λ_K − 1 la "
                              "expansión P/(λ−1) + Q de E3 no aplica",
                "pole_only_reference": "(3/2)(aH/(c_s k))² con c_s² = "
                                       "(2−α_a)(λ_K−1)/(α_a(3λ_K−1))",
            },
            "rules": RULES_A,
            "outcomes": {
                "A_focused": "identidades en PASS, α_a^max ≤ 1e-5, η − 1 (k = 0.02, "
                             "escalera completa) ≤ 1e-5 y (α_a^max/2)/((3/2)ε_K) ≤ "
                             "1e-3 → las colas de sonido son inobservables y "
                             "G_cosmo/G_N − 1 → −(3/2)ε_K en forma limpia: el canal "
                             "Atlas queda ENFOCADO en el Contraste de los Residuos",
                "B_tails_alive": "identidades en PASS y α_a^max ≤ 1e-5, pero η − 1 "
                                 "(k = 0.02) ≥ 1e-3 → HALLAZGO ESTRUCTURAL: la cola "
                                 "O(e²) no se apaga con α_a → 0 (orden de límites; "
                                 "khronon con c_s → ∞); se publica como tal, sin "
                                 "retocar, y la magnitud observacional se declara",
                "C_mapping_broken": "alguna identidad falla → el sector del tratado "
                                    "NO es la clase khronométrica con β = 0 en la forma "
                                    "supuesta; las cotas PPN no se aplican y se publica "
                                    "la identidad que falla",
                "INDETERMINADO": "cualquier otro caso (p. ej. 1e-5 < η − 1 < 1e-3)",
                "rule": "Nunca se ajustan los umbrales tras ver los números. Los "
                        "umbrales de observabilidad (1e-5 / 1e-3 en η − 1 a k = 0.02) "
                        "se fijan por la precisión alcanzable de la clase Euclid/DESI "
                        "en η a gran escala, no por el resultado.",
            },
        },
        "E4b_e3b_closure": {
            "note": "preinscripción NUEVA y distinta de E3 (results/2026-09-13_"
                    "mu_eta_atlas_tail, cuyo E3b = C permanece publicado): sustituye "
                    "la puerta de exponente log-log, fijada sin calibrar la curvatura "
                    "O(e⁴), por un ajuste con término e⁴ explícito",
            "point": HARNESS, "k_over_H0": K_VALUES, "a_start": 0.1,
            "ic_tower_nmax": 8, "e_fit_max": 0.02,
            "solver": {"method": "Radau", "rtol": 1e-10, "atol": 1e-18, "n_eval": 600},
            "fit": "X − 1 = s·e² + q·e⁴ por k, mínimos cuadrados, X ∈ {η, µ_loc}",
            "rules": RULES_B,
            "expected_numbers": {"mu0": mu_atlas_subhorizon(lam, 1.0, al),
                                 "p": growth_index_matter_era(lam, 1.0, al),
                                 "eta_qs_truncated_coefficient": eta_tail_coefficient(lam, 1.0),
                                 "mu_local_qs_truncated_coefficient": -3 * (3 * lam - 1) / (2 - al),
                                 "ladder_coef_eta_from_E3": 17.0735,
                                 "ladder_coef_mu_local_from_E3": -11.3612},
            "outcomes": {
                "A_confirmed": "|s/coef(escalera) − 1| ≤ 0.10 para η y µ_loc en CADA k, "
                               "std(p_loc) ≤ 0.01, solver OK → cola física CONFIRMADA "
                               "numéricamente; el estatuto pasa de «pendiente de "
                               "confirmación numérica» a «confirmada»",
                "B_qs_wins": "las pendientes coinciden (≤ 10 %) con los coeficientes QS "
                             "truncados y no con la escalera → error en la escalera; "
                             "sospecha de error en la derivación primero",
                "C_open": "ni A ni B → sigue abierta",
                "rule": "Nunca se ajustan los umbrales tras ver los números.",
            },
        },
        "prohibitions": {"no_data": True, "no_threshold_tuning": True,
                         "no_change_to_E1_E2_E3_artifacts": True,
                         "no_lensing_choice_of_mu_eta": True},
        "declared_frontiers": [
            "acoplamiento fuerte: Λ_sc ~ M_P√α_a ~ 1e-3 M_P en la cota (H.2.2); "
            "α_a → 0 estricto es singular (c_s² → ∞)",
            "las cotas de campo fuerte llevan parámetros con sombrero (α̂); solo las "
            "de campo débil gobiernan",
            "PPN de campo débil supone el mapeo khronométrico con β = 0, verificado "
            "en las identidades; ξ ≠ 1 no se deriva",
            "Ġ/G y κ_D de púlsares no se ingieren en esta ronda (cotas genéricas del "
            "sector, sin fórmula derivada aquí)",
            "transcripción de las cotas pendiente de verificación por el autor",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción E4_Atlas — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene el "
          "generador), ANTES de ejecutar.", "",
          "## E4a — PPN", "",
          f"Diez identidades a fijar (límite c_ω → ∞ y mapeo β = 0). Cotas del "
          f"dataset `ppn_bounds` (TRANSCRITAS). Reglas: {json.dumps(RULES_A)}. "
          "Desenlaces: A (canal enfocado en los Residuos) / B (colas vivas: hallazgo "
          "estructural) / C (mapeo roto) / INDETERMINADO.", "",
          "## E4b — cierre de E3b (preinscripción nueva)", "",
          f"Punto {HARNESS}, k/H0 = {K_VALUES}, ajuste X − 1 = s·e² + q·e⁴ en e ≤ 0.02; "
          f"reglas {json.dumps(RULES_B)}; desenlaces A / B / C. E3b = C permanece "
          "publicado.", "", "Sin datos; umbrales congelados; fronteras en el JSON."]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preinscripción E4_Atlas congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
