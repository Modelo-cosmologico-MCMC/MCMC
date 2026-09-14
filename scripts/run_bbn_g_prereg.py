#!/usr/bin/env python
"""Congela la preinscripción del Contraste de los Residuos vía BBN
(G_cosmo/G_N libre sobre Y_P de EMPRESS XV + D/H) ANTES de que el
ejecutor lea los valores observacionales. Cita por sha256 la tabla de
respuesta derivada con PRyMordial (teoría; sin datos) y fija reglas,
priors, brazos, degeneraciones declaradas y desenlaces enumerados.

Uso: python scripts/run_bbn_g_prereg.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmology.bbn_g import (  # noqa: E402
    OUTDIR,
    RESPONSE_TABLE,
    RULES,
    delta_G_model,
    response_constants,
)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=OUTDIR.parent.parent).stdout.strip()
    if not RESPONSE_TABLE.exists():
        raise SystemExit("FALLO CERRADO: falta la tabla de respuesta PRyMordial")
    rsha = hashlib.sha256(RESPONSE_TABLE.read_bytes()).hexdigest()
    resp = response_constants()
    dG_model = delta_G_model()
    doc = {
        "title": "Contraste de los Residuos vía BBN: G_cosmo/G_N libre sobre "
                 "Y_P (EMPRESS XV) + D/H (Cooke+2018) con prior Planck en ω_b",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        "generator": "scripts/run_bbn_g_prereg.py",
        "model_prediction": {
            "formula": "G_cosmo/G_N = (2ξ − α_a)/(3λ_K − 1), ξ = 1 (GW170817), "
                       "α_a = 0 declarado (α_a ≪ ε_K; ver frente 3)",
            "epsilon_K": 0.012, "alpha_a": 0.0,
            "delta_G_model": dG_model,
            "first_order": -1.5 * 0.012,
            "note": "la predicción es de FONDO (Friedmann); BBN solo ve G en H — "
                    "la física nuclear y débil no ve G. El modelo predice el "
                    "SIGNO (G_cosmo < G_N) y la magnitud ≈ −1.8 % si α_a ≪ ε_K",
        },
        "response_model": {
            "source": "PRyMordial @ 725d8a8 (Burns, Tait & Valli 2023, "
                      "arXiv:2307.07061), red pequeña (12 reacciones), tasas "
                      "PRIMAT; NACRE II como sistemático nuclear",
            "response_table_sha256": rsha,
            "implementation_of_G": "reescalado de la masa de Planck en H con η_b "
                                   "fijo (G_N de laboratorio en la densidad "
                                   "crítica)",
            "form": "ln X = ln X₀ + a_G ln(1+δ_G) + a_N ΔN + b_N ΔN² + "
                    "a_ω ln(ω_b/ω_b⁰) + a_τ ln(τ_n/τ_n⁰), X ∈ {Y_P, D/H}",
            "constants": resp,
        },
        "data_declared": {
            "dataset": "bbn_abundances",
            "provenance_caveat": "cinco valores publicados TRANSCRITOS; bytes "
                "oficiales no verificables desde el entorno (egreso denegado): "
                "el estatuto del artefacto llevará este aviso hasta que el "
                "autor verifique la transcripción",
            "principal": {"Y_P": "Y_P_empress_xv", "DH": "DH_cooke_2018"},
            "control": {"Y_P": "Y_P_aver_2021"},
            "priors": {"omega_b": "omega_b_planck_2018 (gaussiano)",
                       "tau_n": "tau_n_pdg_2023 (gaussiano)"},
            "theory_errors_added_in_quadrature": {
                "Y_P": RULES["sigma_th_YP"],
                "DH": f"max({RULES['sigma_th_DH_min']:g}, |NACRE II − PRIMAT|) "
                      "en D/H (sistemático de tasas nucleares)",
            },
        },
        "arms": {
            "0_sm_check": "δ_G = 0, N_eff = SM: χ² y pulls de Y_P y D/H (control)",
            "1_principal": "δ_G libre (prior plano en [−0.30, +0.30]), N_eff = "
                           "3.044 fijo, ω_b y τ_n marginalizados con sus priors; "
                           "posterior en malla → CI68/CI95, mediana, pull de "
                           "δ_G^model y de 0",
            "2_degeneracy": "δ_G y ΔN_eff libres (prior plano ΔN ∈ [−1.5, +1.5]); "
                            "se publica la banda 2D, la dirección degenerada y la "
                            "marginal de δ_G — DECLARADO: G y N_eff son casi "
                            "degenerados en BBN; el veredicto NO se toma de este brazo",
            "3_control_aver": "brazo 1 con Y_P = Aver+2021 en vez de EMPRESS XV",
            "4_single_probe": "brazo 1 solo con Y_P y solo con D/H (quién manda)",
            "xi_e": "asimetría leptónica ξ_e NO modelada: degeneración declarada "
                    "(Y_P baja también con ξ_e > 0); se publica como frontera",
        },
        "outcomes_enumerated": {
            "rules": RULES,
            "order": "C → B → A (exhaustivo sobre el brazo principal)",
            "C_exclusion": "δ_G^model ∉ CI95(δ_G) → el −1.8 % queda EXCLUIDO por "
                           "BBN al 95 %: resultado negativo de primera clase; "
                           "PROHIBIDO reparametrizar ε_K o invocar α_a a posteriori",
            "B_identification": "0 ∉ CI95(δ_G) y δ_G^model ∈ CI95 → "
                                "IDENTIFICACIÓN de G_cosmo ≠ G_N: SOSPECHA DE ERROR "
                                "PRIMERO (tasas nucleares, ξ_e, N_eff, transcripción "
                                "de los datos); jamás se publica como detección sin "
                                "el brazo 2 y los bytes oficiales verificados",
            "A_compatible": "0 ∈ CI95 y δ_G^model ∈ CI95 → banda compatible sin "
                            "identificación: el desenlace aburrido esperado; se "
                            "publica el pull de −1.8 % y el signo del central SIN "
                            "leerlo como señal (E13: el signo del central no es señal)",
            "rule": "Nunca se ajustan umbrales, priors ni errores teóricos tras ver "
                    "los números; el veredicto se toma del brazo 1; los brazos 2–4 "
                    "se publican íntegros como contexto.",
        },
        "prohibitions": {"no_threshold_tuning": True,
                         "no_epsilon_K_refit": True,
                         "no_Neff_from_empress_as_data": True,
                         "no_sign_as_signal": True},
        "declared_frontiers": [
            "ξ_e (asimetría leptónica) no modelada",
            "degeneración G ↔ N_eff casi exacta en BBN (brazo 2 solo informa)",
            "sistemático de tasas nucleares en D/H cubierto por |NACRE II − PRIMAT| "
            "y un mínimo declarado, no por una MC completa de tasas",
            "α_a = 0 en la predicción (el frente 3 acota α_a; PPN pendiente)",
            "transcripción de los datos pendiente de verificación por el autor",
        ],
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene el "
          f"generador). Tabla de respuesta PRyMordial `{rsha[:12]}`.", "",
          f"Predicción del modelo: δ_G = G_cosmo/G_N − 1 = {dG_model:+.5f} "
          "(ε_K = 0.012, α_a = 0, ξ = 1).", "",
          "Brazo principal: δ_G libre (plano en [−0.30, 0.30]), N_eff = 3.044, "
          "ω_b y τ_n marginalizados. Errores teóricos en cuadratura: "
          f"σ_th(Y_P) = {RULES['sigma_th_YP']}, σ_th(D/H) = max("
          f"{RULES['sigma_th_DH_min']:g}, |NACRE II − PRIMAT|).", "",
          "Desenlaces (orden C → B → A): C = δ_G^model ∉ CI95 (exclusión); "
          "B = 0 ∉ CI95 (identificación → sospecha primero); A = ambos dentro "
          "(banda compatible sin identificación). Nunca se ajustan umbrales.", "",
          "Datos: cinco valores publicados TRANSCRITOS (bytes oficiales no "
          "verificables desde el entorno) — el artefacto llevará el aviso.",
          "", "Degeneraciones declaradas: N_eff (brazo 2, solo informa), ξ_e (no "
          "modelada), tasas nucleares (sistemático declarado)."]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preinscripción BBN-G congelada en {OUTDIR} (commit {sha[:9]}); "
          f"δ_G^model = {dG_model:+.5f}; respuesta {rsha[:12]}")


if __name__ == "__main__":
    main()
