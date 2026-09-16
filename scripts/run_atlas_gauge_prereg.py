#!/usr/bin/env python
"""Congela la preinscripción de E5_Atlas ANTES de ejecutar: cierre
GAUGE-INVARIANTE de las colas O(e²) del sector Atlas.

E5a — escalera exacta en invariantes de gauge: Ψ_N = ψ + ḃ, Φ_N = φ − Hb,
Δ = δ − 3H l1 (reparametrización temporal t → t + T sobre el ansatz de
gauge unitario de validation/atlas_derivation.py). Se contrasta el
teorema «sin estrés anisótropo lineal ⟹ η_N ≡ 1» y la forma cerrada
µ_Δ − 1 = −α_a (aH/(c_s k))² [1 + O(α_a)].

E5b — arnés numérico con ICs adiabáticas que mide η_N y µ_Δ sobre la
trayectoria (E3b/E4b medían η y µ_loc de gauge unitario).

CONTROL EXTERNO DECLARADO: la sesión de verificación del autor
(15-sep-2026, noche) calculó ANTES de esta congelación, fuera del repo y
desde la misma escalera, η_N − 1 = 0 a O(e²) en cinco puntos y µ_Δ − 1
(respecto de G_B) = −1.0e-4/≈0/−0.0127/−1.396/−3.833 en (λ_K, α_a) =
(1.0001, 1e-4)/(1.012, 1e-6)/(1.012, 0.012)/(1.05, 0.3)/(1.0125, 0.3).
Los umbrales de abajo se fijan por el teorema y por la forma cerrada,
no por esos valores; los puntos se incluyen para que el artefacto los
reproduzca o los contradiga. El desarrollo de la maquinaria
(validation/atlas_tail_derivation.py) se ejercitó sobre esos mismos
puntos de control y la ventana del arnés se calibró sobre las
pendientes UNITARIAS ya publicadas en E4b (17.0735, −11.3612), nunca
sobre η_N ni µ_Δ.

Uso: python scripts/run_atlas_gauge_prereg.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.mu_eta_atlas import khronon_cs2, mu_delta_tail_leading  # noqa: E402

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "2026-09-15_mu_eta_atlas_gauge"
EPS_K = 0.012
H0_OVER_C = 3.336e-4
RULES_A = {"etaN_e2_coefficient_exact_zero": True,
           "muD_closed_form_rel_tol": 0.05, "muD_closed_form_alpha_max": 0.012,
           "muD_at_bound_k002_leq": 1e-5, "tail_k_h_over_Mpc": 0.02}
RULES_B = {"etaN_abs_max": 1e-6, "muD_slope_rel_tol": 0.10,
           "unitary_match_rel_tol": 0.10, "purity_std_max": 0.01}
LADDER_POINTS = [[1.0 + EPS_K, 1e-6], [1.0 + EPS_K, 1e-4], [1.0 + EPS_K, 1e-3],
                 [1.001, 1e-6], [1.05, 1e-6], [1.0001, 1e-4], [1.0 + EPS_K, EPS_K],
                 [1.05, 0.30], [1.0125, 0.30]]
IDENTITY_DEEP_POINTS = [[1.05, 0.30], [1.0 + EPS_K, 1e-6]]
HARNESS = {"lambda_K": 1.05, "alpha_a": 0.30, "xi": 1.0}
K_VALUES = [133.333, 200.0, 300.0, 400.0]


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    e02 = H0_OVER_C / RULES_A["tail_k_h_over_Mpc"]
    doc = {
        "title": "E5_Atlas — cierre gauge-invariante de las colas O(e²) del sector "
                 "Atlas: η_N = Φ_N/Ψ_N y µ_Δ (Bardeen + Δ comóvil) desde la escalera "
                 "exacta y en el arnés adiabático",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "generator": "scripts/run_atlas_gauge_prereg.py",
        "gauge_transformation": {
            "reparametrization": "t → t + T, T = ε T(t) cos(kx), sobre N = 1 + εψ cos, "
                                 "N_x = ∂_x(εb cos), q_ij = a²(1 − 2εφ cos)δ_ij, polvo con ℓ̄̇ = −m",
            "rules": {"psi": "ψ − Ṫ", "b": "b + T", "phi": "φ + HT",
                      "delta": "δ + 3HT", "l1": "l1 + mT"},
            "newtonian_gauge": "b = 0 con T = −b",
            "invariants": {"Psi_N": "ψ + ḃ", "Phi_N": "φ − Hb", "Delta": "δ − 3H l1 (m = 1)"},
            "order": "b arranca en n = 1 de la torre; ḃ y Hb caen en n = 4 — el MISMO "
                     "orden e² de las colas de E3/E4, que son por tanto de gauge unitario",
        },
        "theorem_to_lock": "a orden lineal el sector no tiene fuente de estrés anisótropo "
                           "(a_i a^i es cuadrático en ∂N; (1−λ_K)K² es pura traza) ⟹ la "
                           "ecuación ij sin traza es la de GR ⟹ Φ_N = Ψ_N (η_N ≡ 1) a todo "
                           "orden en e",
        "external_control": {
            "who": "sesión de verificación del autor, 15-sep-2026 (noche), fuera del repo",
            "when": "ANTES de esta congelación",
            "etaN_minus_one_e2": {"all_points": 0.0},
            "muD_minus_one_e2_rel_GB": {"(1.0001, 1e-4)": -1.0e-4, "(1.012, 1e-6)": "≈ 0",
                                        "(1.012, 0.012)": -0.0127, "(1.05, 0.3)": -1.396,
                                        "(1.0125, 0.3)": -3.833},
            "normalization_note": "los valores externos son respecto de G_B; el repo mide "
                                  "µ_Δ respecto de la G local (factor 1 − α_a/2), como µ_loc "
                                  "en E3/E4",
            "role": "control, no experimento del repo: no fija umbrales",
        },
        "E5a_ladder": {
            "points": LADDER_POINTS,
            "tower": {"nlow": -4, "nhigh": 16, "prof": 12, "reliable_n_max": 4},
            "deep_identity": {"points": IDENTITY_DEEP_POINTS, "nhigh": 20, "prof": 16,
                              "n_check": [0, 2, 4, 6, 8],
                              "note": "informativo: Φ_N,n = Ψ_N,n para n ≤ 8 (teorema a "
                                      "todo orden); no gobierna el desenlace"},
            "closed_form": "µ_Δ − 1 = −α_a e²/c_s² = −α_a²(3λ_K−1)/((2−α_a)(λ_K−1)) e² "
                           "(orden dominante en α_a; cosmology.mu_eta_atlas.mu_delta_tail_leading)",
            "expected_at_bound": {
                "point": [1.0 + EPS_K, 1e-6], "k_h_over_Mpc": 0.02, "e": e02,
                "cs2": khronon_cs2(1.0 + EPS_K, 1.0, 1e-6),
                "muD_minus_one_closed": float(mu_delta_tail_leading(e02, 1.0 + EPS_K, 1e-6)),
                "etaN_minus_one_unitary_E4": 1.27e-3,
            },
            "rules": RULES_A,
            "outcomes": {
                "A_closed": "coef η_N(e²) = 0 EXACTO (aritmética racional) y η_N,0 = 1 en "
                            "TODOS los puntos; |coef µ_Δ/forma cerrada − 1| ≤ 0.05 en los "
                            "puntos con α_a ≤ 0.012; |µ_Δ − 1|(k = 0.02, cota PPN) ≤ 1e-5 "
                            "→ cierre gauge-invariante: η_N ≡ 1 en el sector Atlas, la cola "
                            "de µ_Δ es O(α_a²) e inobservable en la cota; la lectura física "
                            "de E4a = B queda SUPERADA (era de gauge unitario)",
                "B_tail_survives": "η_N limpio en todos los puntos pero µ_Δ NO sigue la forma "
                                   "cerrada (cola de orden α_a⁰ o α_a¹) o |µ_Δ − 1|(k = 0.02) "
                                   "> 1e-5 → sobrevive una cola gauge-invariante en µ: se "
                                   "publica su magnitud y E4a = B se reformula en µ_Δ",
                "C_theorem_fails": "coef η_N(e²) ≠ 0 o η_N,0 ≠ 1 en ALGÚN punto → el teorema "
                                   "del estrés anisótropo falla o la transformación de gauge "
                                   "está mal: sospecha de error en la derivación primero; "
                                   "E4a = B se mantiene como está",
                "INDETERMINADO": "cualquier otro caso",
                "rule": "Nunca se ajustan los umbrales tras ver los números. El 0.05 de la "
                        "forma cerrada acota las correcciones O(α_a) (≤ 4α_a en α_a ≤ "
                        "0.012); el 1e-5 es el umbral de inobservabilidad de E4.",
            },
        },
        "E5b_harness": {
            "note": "preinscripción NUEVA (E3b = C y E4b = C permanecen publicados): el "
                    "arnés mide por primera vez los invariantes η_N y µ_Δ; brazo k/H0 = "
                    "66.7 retirado (lección de E4b) y ventana a ≥ 0.3 (descarta el "
                    "transitorio de las ICs truncadas), calibrados sobre las pendientes "
                    "UNITARIAS publicadas (E4b), que quedan como control en el artefacto",
            "point": HARNESS, "k_over_H0": K_VALUES, "a_start": 0.1, "a_fit_min": 0.3,
            "ic_tower_nmax": 8, "e_fit_max": 0.02,
            "solver": {"method": "Radau", "rtol": 1e-10, "atol": 1e-18, "n_eval": 600},
            "fit": "X − 1 = s·e² + q·e⁴ por k, mínimos cuadrados, X ∈ {η_N, µ_Δ}; control "
                   "X ∈ {η, µ_loc} (unitario)",
            "targets": {"etaN": "η_N ≡ 1 sobre TODA la trayectoria (teorema, no solo O(e²))",
                        "muD": "s_µΔ = coef µ_Δ(e²) de la escalera E5a en el mismo punto"},
            "rules": RULES_B,
            "outcomes": {
                "A_confirmed": "max|η_N − 1| ≤ 1e-6 en la ventana, |s_µΔ/escalera − 1| ≤ 0.10, "
                               "std(p_loc) ≤ 0.01 y solver OK en CADA k → invariantes "
                               "confirmados numéricamente",
                "B_gauge_wrong": "|s_ηN/coef η unitario − 1| ≤ 0.10 en todos los k (la "
                                 "combinación de gauge no elimina la cola de gauge unitario) "
                                 "→ error en la transformación o en el teorema; sospecha de "
                                 "error en la derivación primero",
                "C_open": "ni A ni B → sigue abierta; se publica qué regla falla",
                "rule": "Nunca se ajustan los umbrales tras ver los números. El 1e-6 es "
                        "~1e4 × rtol sobre cocientes O(1); el 0.10 es el de E3b/E4b.",
            },
        },
        "prohibitions": {"no_data": True, "no_threshold_tuning": True,
                         "no_change_to_E1_E2_E3_E4_artifacts": True,
                         "no_lensing_choice_of_mu_eta": True},
        "declared_frontiers": [
            "materia = polvo sin estrés anisótropo propio; radiación y neutrinos no incluidos",
            "ξ = 1 (GW170817); la cola de µ_Δ a α_a ~ O(1) lleva correcciones O(α_a) no "
            "cerradas (el artefacto publica el cociente)",
            "n = 6, 8 de la torre requieren prof ≥ 16 y sus valores individuales no son "
            "estables frente al corte: solo la identidad Φ_N,n − Ψ_N,n se lee ahí",
            "la escalera sigue siendo el objeto: E5b es su confirmación numérica en "
            "invariantes, no una derivación independiente del teorema",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción E5_Atlas — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene el "
          "generador), ANTES de ejecutar. Control externo (sesión de verificación "
          "15-sep) declarado en el JSON: no fija umbrales.", "",
          "## E5a — escalera en invariantes de gauge", "",
          f"Puntos {LADDER_POINTS}; reglas {json.dumps(RULES_A)}. Desenlaces: A (cierre: "
          "η_N ≡ 1, µ_Δ = O(α_a²)) / B (cola gauge-invariante en µ_Δ) / C (teorema falla) "
          "/ INDETERMINADO.", "",
          "## E5b — arnés adiabático en invariantes (preinscripción nueva)", "",
          f"Punto {HARNESS}, k/H0 = {K_VALUES}, ventana a ≥ 0.3 y e ≤ 0.02, ajuste "
          f"X − 1 = s·e² + q·e⁴; reglas {json.dumps(RULES_B)}; desenlaces A / B / C. "
          "E3b = C y E4b = C permanecen publicados.", "",
          "Sin datos; umbrales congelados; fronteras en el JSON."]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preinscripción E5_Atlas congelada en {OUTDIR} (commit {sha[:9]})")


if __name__ == "__main__":
    main()
