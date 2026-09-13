#!/usr/bin/env python
"""Preinscripción del PR #15 (validación por mocks del pipeline SN
Dovekie) — OBLIGATORIA antes de ejecutar ninguna puerta y antes de que
ningún script de inferencia toque el HD real.

Genera results/2026-09-12_dovekie_mocks/preregistration.{json,md} con
las cuatro puertas, tolerancias, enteros binomiales EXACTOS para
N = 25 (computados aquí con scipy.stats.binom y congelados como
enteros), semillas, cosmología inyectada, alcance declarado y
desenlaces enumerados con compromiso de no-ajuste (patrón #10/#13).

Uso: python scripts/run_dovekie_mocks_prereg.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.dovekie_mocks import (  # noqa: E402
    COV_KIND_PRODUCTION,
    INJECTED,
    K_COSMOLOGIES,
    MOCK_SEED,
    N_MOCKS,
    OUTDIR,
    PRIOR_EPS,
    PRIOR_EPS_GAUSS,
    PRIOR_OMEGA_M,
    PRIOR_ZTRANS,
)

N_SN = 1820   # conteo verificado por bytes (npz nsn = 1820; el HD
              # completo pasa el corte oficial zHD > 0); la cifra
              # «1828» del informe de traspaso del 12-sep era errónea

# Región de aceptación binomial central ≥ 95 % para el conteo de
# aciertos de cobertura (enteros exactos, congelados):
COV68_K = (int(binom.ppf(0.025, N_MOCKS, 0.68)),
           int(binom.ppf(0.975, N_MOCKS, 0.68)))
COV95_K = (int(binom.ppf(0.025, N_MOCKS, 0.95)),
           int(binom.ppf(0.975, N_MOCKS, 0.95)))
# Falsa preferencia: bajo el nulo ΛCDM, ΔAIC < 0 ⟺ Δχ²_ganancia > 4;
# cota conservadora P(χ²₂ > 4) = e⁻² (los 2 dof extra están además
# parcialmente dominados por el prior, lo que solo REDUCE la tasa):
P_FALSE_PREF = math.exp(-2.0)
FALSEPREF_K_MAX = int(binom.ppf(0.975, N_MOCKS, P_FALSE_PREF))
SIGMA_PHAT_68 = math.sqrt(0.68 * 0.32 / N_MOCKS)

PREREG = {
    "front": "PR #15 — validación por mocks del pipeline SN Dovekie",
    "scope_declaration": {
        "pipeline_entry_point": (
            "nivel Hubble diagram + covarianza (4_DISTANCES_COVMAT del "
            "release DES-SN5YR @ c9a4fcaf). Las etapas DES aguas "
            "arriba (ajuste SALT3 de curvas de luz, BBC/bias "
            "corrections) NO se reimplementan ni se validan aquí: las "
            "validó DES con sus 25 mocks fotométricos dentro de su "
            "pipeline."),
        "des_photometric_mocks": (
            "1_SIMULATIONS @ c9a4fcaf contiene fotometría SNANA cruda "
            "(HEAD/PHOT.FITS, ~2900 curvas de luz por realización), "
            "no diagramas de Hubble; consumirla exigiría la pila "
            "SNANA/Pippin (SALT3 + BBC), que este repositorio no "
            "contiene. Registrada en data/sources.yaml como "
            "des_dovekie_photmocks (no ingerida, fuera del alcance). "
            "README de los mocks: sha256 "
            "f9a0f2135a6c84ca65aee1320d8e47aba15c648aa981dc18a3576cf436d4ed25."),
        "own_mock_design": (
            "25 realizaciones sintéticas a NIVEL HD: μ_mock = "
            "μ_model(z; cosmología inyectada de los mocks DES) + ΔM_m "
            "+ η_m con η_m ~ N(0, C) (C = inversa de la W oficial "
            "STAT+SYS, vía Cholesky), ΔM_m ~ N(0, 1 mag) para "
            "ejercitar la marginalización de M, sobre los z del HD "
            "real (el diseño del experimento) con N = 1820. La "
            "columna MU real NO se lee: el generador consume "
            "load_dovekie_design(), que no la devuelve."),
        "generator_shares_mu_model_with_fit": (
            "LIMITACIÓN DECLARADA: los mocks se generan con el mismo "
            "μ_model de producción que luego ajusta — un sesgo común a "
            "ambos no sería detectable por las puertas 2-4; lo cubre "
            "la puerta 1b (integrador de producción vs cuadratura "
            "adaptativa independiente, patrón #14)."),
    },
    "injected_cosmology": {
        "source": ("README de 1_SIMULATIONS @ c9a4fcaf: H0 = 70.0, "
                   "OMEGA_MATTER = 0.315, OMEGA_LAMBDA = 0.685, "
                   "w0 = −1, wa = 0"),
        "our_parametrization": {"Omega_m": INJECTED["Omega_m"],
                                "eps_Lambda": INJECTED["eps"],
                                "H0_fixed": INJECTED["H0"],
                                "note": ("con ε = 0 el fondo MCMC es "
                                         "ΛCDM exacto (Prop. A.1); "
                                         "z_trans es irrelevante en "
                                         "ese límite")},
    },
    "frozen_numbers": {
        "n_sn": N_SN,
        "n_sn_note": ("verificado por bytes: nsn = 1820 en ambos npz "
                      "y 1820 filas del HD pasan el corte oficial "
                      "zHD > 0; la cifra 1828 del traspaso era "
                      "errónea y queda corregida aquí"),
        "n_mocks": N_MOCKS,
        "mock_seed": MOCK_SEED,
        "cov_kind_production": COV_KIND_PRODUCTION,
        "emcee": {"nwalkers": 32, "nsteps": 800, "burn": "nsteps//2",
                  "thin": 4, "seed_per_mock": "42 + índice del mock"},
        "priors": {"Omega_m": list(PRIOR_OMEGA_M),
                   "eps_gauss": list(PRIOR_EPS_GAUSS),
                   "eps_support": list(PRIOR_EPS),
                   "z_trans": list(PRIOR_ZTRANS),
                   "note": "los del Apéndice F, idénticos a #12"},
        "K_cosmologies": [list(k) for k in K_COSMOLOGIES],
        "coverage_k68_range": list(COV68_K),
        "coverage_k95_range": list(COV95_K),
        "false_preference_k_max": FALSEPREF_K_MAX,
    },
    "gates": {
        "1a_formula_equivalence": {
            "statement": (
                "χ̃² de la implementación A (producción) y de la B "
                "(independiente, einsum) frente a la función oficial "
                "cov_log_likelihood ejecutada desde los bytes del "
                "release (extraída por AST), con el MISMO vector μ y "
                "la MISMA W, sobre las K = 8 cosmologías × 25 vectores "
                "mock × {STATONLY, STAT+SYS}"),
            "tolerance": "max |Δχ²| ≤ 1e-6 (absoluto)",
            "count_gate": ("conteo de SNe idéntico: parser A == "
                           "parser B == nsn(npz) == 1820, ambos npz"),
            "declared_limit": (
                "el harness cosmosis del script oficial no es "
                "ejecutable aquí; la parte que define la likelihood "
                "(cov_log_likelihood) se ejecuta desde los bytes "
                "oficiales; el desempaquetado de W se implementa dos "
                "veces (A y B) y se compara elemento a elemento "
                "(max |ΔW| = 0 exigido: misma fuente, mismos bytes)"),
        },
        "1b_distance_equivalence": {
            "statement": ("μ_model de producción (cumulativa O(h⁴) + "
                          "interpolación) vs cuadratura adaptativa "
                          "independiente (quad, epsrel 1e-12) sobre "
                          "los 1820 z × K = 8 cosmologías"),
            "tolerance": ("max |Δμ| ≤ 1e-5 mag y |Δχ̃²| ≤ 0.01 al "
                          "evaluar ambos μ en la fórmula A con la W "
                          "de producción y un vector mock fijo "
                          "(el primero)"),
        },
        "2_parameter_recovery": {
            "statement": (
                "brazo ΛCDM (posterior 1D determinista en malla, "
                "prior plano): pull_m = (p50_m(Ω_m) − 0.315)/σ_m con "
                "σ_m = (p84 − p16)/2, sobre los 25 mocks"),
            "gate": ("|media de pulls| ≤ 0.6 = 3·SEM, con SEM = "
                     "1/√25 = 0.2 (cota en múltiplos de la SEM, no "
                     "ad hoc)"),
            "diagnostics_published_not_gated": (
                "desviación típica de los pulls (esperada ~1; su "
                "calibración la vigila la puerta 3, no una segunda "
                "puerta redundante) y sesgo agregado en unidades "
                "físicas"),
        },
        "3_coverage_binomial_exact": {
            "statement": ("conteo k de mocks cuyo intervalo creíble "
                          "central contiene Ω_m = 0.315"),
            "accept_68": {"k_range": list(COV68_K),
                          "definition": ("región central ≥ 95 % de "
                                         "Binomial(25, 0.68): "
                                         "binom.ppf(0.025/0.975)")},
            "accept_95": {"k_range": list(COV95_K),
                          "definition": ("región central ≥ 95 % de "
                                         "Binomial(25, 0.95)")},
            "declared_power": (
                f"σ_p̂ = √(0.68·0.32/25) = {SIGMA_PHAT_68:.3f} — "
                "±9 puntos porcentuales a 1σ: un 64 % o un 72 % "
                "observado NO es fallo"),
            "mandatory_wording_on_pass": (
                "«no se detecta una descalibración incompatible con "
                "el tamaño de la muestra de mocks» — NUNCA «coverage "
                "validado al X %»"),
        },
        "4_no_false_preference": {
            "statement": (
                "los mocks NO contienen la extensión MCMC (ε = 0 "
                "inyectado); se ajustan ambos brazos y se exige que "
                "la comparación no la favorezca sistemáticamente"),
            "definitions": {
                "chi2_min_lcdm": ("mínimo de malla 1D + refinado "
                                  "parabólico local (k = 1 parámetro "
                                  "ajustado; M marginalizada en ambos "
                                  "brazos por igual)"),
                "chi2_min_mcmc": ("mínimo sobre el CIERRE del soporte "
                                  "del prior: malla predeclarada "
                                  "(ε × z_trans: esquinas + centro) "
                                  "con refinado 1D acotado de Ω_m "
                                  "(k = 3)"),
                "AIC": "χ²_min + 2k", "BIC": "χ²_min + k·ln(1820)",
                "delta": "Δ = (MCMC) − (ΛCDM); Δ > 0 favorece ΛCDM"},
            "gate_aic": (
                f"mediana(ΔAIC) > 0 Y n(ΔAIC < 0) ≤ {FALSEPREF_K_MAX} "
                "de 25 [región ≥ 97.5 % de Binomial(25, e⁻²): bajo el "
                "nulo, ΔAIC < 0 ⟺ Δχ²_ganancia > 4, y "
                "P(χ²₂ > 4) = e⁻² ≈ 0.135 es cota conservadora — el "
                "prior gaussiano de ε solo reduce la tasa]"),
            "gate_bic": ("mediana(ΔBIC) > 0 Y n(ΔBIC < 0) ≤ "
                         f"{FALSEPREF_K_MAX} de 25 (misma cota, "
                         "conservadora: ΔBIC = ΔAIC + 2·(ln1820 − 2) "
                         "> ΔAIC siempre)"),
            "gate_eps": (
                "ε_Λ consistente con cero: 0 ∈ CI95 central de ε en "
                f"≥ {COV95_K[0]} de 25 mocks (entero binomial de "
                "p = 0.95) Y |mediana sobre mocks de p50(ε)| ≤ 0.05 "
                "(una σ del prior). NOTA PREINSCRITA: el prior de ε "
                "está centrado en 0.012, no en 0 — un p50 medio ≈ "
                "0.012 con SNe poco informativas es el comportamiento "
                "esperado del prior y NO constituye preferencia; por "
                "eso la puerta es de intervalos, no de medias a 3σ"),
        },
    },
    "outcomes_enumerated": {
        "PASS": ("las cuatro puertas superadas ⟹ mock_validation.json "
                 "con status PASS y el sha256 de esta preinscripción; "
                 "SOLO entonces el HD real entra en scripts de "
                 "inferencia (barrera ejecutable "
                 "cosmology.dovekie_sn.require_mock_validation_pass)"),
        "FAIL_any_gate": (
            "status FAIL con la(s) puerta(s) fallida(s) publicadas "
            "íntegras; se diagnostica y corrige el PIPELINE (nunca "
            "las tolerancias ni los enteros congelados) y se re-corre "
            "TODO el bloque de puertas; el HD real sigue vedado"),
        "no_adjustment_commitment": (
            "compromiso de no-ajuste: tolerancias, enteros binomiales, "
            "semillas, N_MOCKS y priors quedan congelados aquí ANTES "
            "de ejecutar puerta alguna; cambiarlos tras ver un "
            "resultado exige una preinscripción NUEVA con changelog y "
            "el fallo previo publicado"),
    },
    "declared_roles": {
        "dovekie_mocks": "validación del pipeline (este PR)",
        "dovekie_real": "primera aplicación (posterior al PASS)",
        "unite": ("benchmark armonizado de robustez, NO replicación "
                  "independiente (contiene DES-SN5YR/Dovekie y "
                  "Pantheon+)"),
        "union3": "comprobación más genuinamente externa",
        "pantheonplus": ("disección de qué componente de Unite mueve "
                         "el resultado"),
    },
    "prohibitions": {
        "no_real_mu_before_pass": (
            "ningún script de inferencia consume la columna MU real "
            "antes del PASS; el candado ejecutable vive en "
            "cosmology.dovekie_sn (load_dovekie_hd) y el candado de "
            "protocolo en tests/test_dovekie_lock.py"),
        "no_gate_tuning": ("prohibido re-tocar tolerancias/enteros "
                           "tras ver resultados (ver "
                           "no_adjustment_commitment)"),
        "independent_b_isolation": (
            "validation/dovekie_independent.py no importa "
            "cosmology.dovekie_sn ni cosmology.background (vigilado "
            "por el candado)"),
    },
}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"],
                         capture_output=True, text=True,
                         cwd=OUTDIR.parent.parent).stdout.strip()
    doc = {
        "preregistered_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": sha,
        **PREREG,
    }
    (OUTDIR / "preregistration.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    md = [
        "# Preinscripción PR #15 — validación por mocks del pipeline "
        "SN Dovekie\n",
        f"- **Commit**: `{sha}`",
        f"- **Fecha (UTC)**: {doc['preregistered_utc']}",
        f"- **N SNe (verificado por bytes)**: {N_SN} — la cifra 1828 "
        "del traspaso era errónea",
        f"- **Mocks**: {N_MOCKS} realizaciones a nivel HD, semilla "
        f"{MOCK_SEED}, covarianza {COV_KIND_PRODUCTION}, cosmología "
        f"inyectada Ω_m = {INJECTED['Omega_m']}, ε = 0 (la de los "
        "mocks DES @ c9a4fcaf)",
        "",
        "**Puertas** (tolerancias y enteros CONGELADOS aquí, antes de "
        "ejecutar nada):",
        "1a. |Δχ²| ≤ 1e-6 entre A, B y la fórmula oficial ejecutada "
        "desde los bytes del release; conteo 1820 idéntico.",
        "1b. |Δμ| ≤ 1e-5 mag (integrador producción vs quad "
        "independiente); |Δχ̃²| ≤ 0.01.",
        "2. |media de pulls de Ω_m| ≤ 0.6 (3·SEM, SEM = 1/√25).",
        f"3. cobertura: k₆₈ ∈ {list(COV68_K)}, k₉₅ ∈ {list(COV95_K)} "
        f"(binomial exacta, N = 25; σ_p̂ = {SIGMA_PHAT_68:.3f}).",
        f"4. mediana(ΔAIC) > 0, mediana(ΔBIC) > 0, n(Δ<0) ≤ "
        f"{FALSEPREF_K_MAX}; 0 ∈ CI95(ε) en ≥ {COV95_K[0]} de 25 y "
        "|mediana p50(ε)| ≤ 0.05.",
        "",
        "**Redacción obligatoria si la puerta 3 pasa**: «no se "
        "detecta una descalibración incompatible con el tamaño de la "
        "muestra de mocks» — nunca «coverage validado al X %».",
        "",
        "**Alcance**: la validación cubre la etapa cosmológica "
        "(HD + covarianza), el punto de entrada de nuestro pipeline; "
        "los mocks fotométricos DES validan las etapas DES aguas "
        "arriba y quedan fuera (registrados, no ingeridos).",
        "",
        "**Barrera**: el HD real (columna MU) no entra en ningún "
        "script de inferencia sin mock_validation.json en PASS "
        "citando el sha256 de ESTE fichero (candado ejecutable + "
        "candado de protocolo en la suite).",
    ]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n",
                                               encoding="utf-8")
    print(f"Preinscripción congelada en {OUTDIR} (commit {sha[:9]})")
    print(f"k68 ∈ {COV68_K}, k95 ∈ {COV95_K}, "
          f"falsa preferencia ≤ {FALSEPREF_K_MAX}/25")


if __name__ == "__main__":
    main()
