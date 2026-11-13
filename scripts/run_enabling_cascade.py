#!/usr/bin/env python
"""Cribado preinscrito de la cascada de habilitación (R6 del plan del 11-nov-2026; decisiones 4, 5 y 7 del autor del
13-nov): ¿da el Basal con tres direcciones latentes, y pesos fijados por una sola ley de escala, tres colapsos hacia
fuera con cocientes de umbrales cerca de 10, en el reloj S completo?

    python scripts/run_enabling_cascade.py prereg    # congela reglas y parámetros (commit posterior al generador), antes de correr
    python scripts/run_enabling_cascade.py run       # celdas con letra, controles y sensibilidad
    python scripts/run_enabling_cascade.py analyze   # letra por δ₀, global la peor; falla cerrado sin preinscripción

Con `--round 2` los tres subcomandos trabajan sobre la ronda 2 (directorio y preinscripción propios): las mismas reglas,
puertas, controles y brazos, con las amplitudes latentes integradas en logaritmo. La ronda 1 quedó INDETERMINADO porque
el flotante agotaba la semilla de q_2 en la representación lineal.

Letra por δ₀ sobre los gatillos de los tres colapsos (regla del plan, la del candidato 3): A, los dos cocientes
S_{k+1}/S_k en [7, 14]; B, uno; C, ninguno o menos de tres colapsos completos; INDETERMINADO, puertas numéricas o
controles fallidos. La global es la peor. Ninguna letra deriva λ ni dice que una dirección latente sea una dimensión.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.enabling_cascade import (  # noqa: E402
    cascade_letter,
    run_cascade,
    scale_law_shares,
)
from mcmc_ontology import constants as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROUND_DIRS = {1: "2026-11-13_enabling_cascade", 2: "2026-11-13_enabling_cascade_round2"}
ROUND = 1
OUTDIR = ROOT / "results" / ROUND_DIRS[1]
PREREG = OUTDIR / "preregistration.json"
RUNS = OUTDIR / "runs"
DECISIONS = ROOT / "docs" / "decisiones_autor_2026-11-13.md"
DECISION_ROUND2 = ROOT / "docs" / "decision_autor_2026-11-13_r6_ronda2.md"

DELTA0_GRID = [0.003, 0.01, 0.0554]
SEED = 1e-6
BASE = {"lam_scale": C.LAMBDA_DECADE, "gate_theta": 0.9, "gate_Delta": 0.05, "seeds": [SEED, SEED, SEED],
        "dS_step": 2e-4, "noise_floor": 0.0, "tension_reference": "measured", "gated": True}
BASE_CLOCK = {"d_sigma_hat": 5e-4}
# lo único que cambia en la ronda 2: se integra ln q̂_k en todos los brazos
ROUND2_BASE = {**BASE, "amplitude_representation": "log"}
FLOOR = 1e-18
# brazo → (entradas del modelo que cambian, entradas del reloj que cambian). Todo lo demás es BASE / BASE_CLOCK
ARMS = {
    "base": ({}, {}),
    "convergencia": ({"dS_step": 1e-4}, {"d_sigma_hat": 2.5e-4}),
    "semilla_1e-3": ({"seeds": [1e-3, 1e-3, 1e-3]}, {}),
    "semilla_1e-9": ({"seeds": [1e-9, 1e-9, 1e-9]}, {}),
    "sin_semilla": ({"seeds": [0.0, 0.0, 0.0]}, {}),
    "suelo_1e-18": ({"noise_floor": FLOOR}, {}),
    "sin_compuerta": ({"gated": False}, {}),
    "jerarquia_rota": ({"seeds": [0.0, SEED, SEED]}, {}),
    "pesos_iguales": ({"shares": [1.0, 1.0, 1.0, 1.0]}, {}),
    "lambda_5": ({"lam_scale": 5.0, "noise_floor": FLOOR}, {}),
    "lambda_7": ({"lam_scale": 7.0, "noise_floor": FLOOR}, {}),
    "lambda_14": ({"lam_scale": 14.0, "noise_floor": FLOOR}, {}),
    "lambda_20": ({"lam_scale": 20.0, "noise_floor": FLOOR}, {}),
    "referencia_ley": ({"tension_reference": "law"}, {}),
    "theta_0.8": ({"gate_theta": 0.8}, {}),
    "theta_0.95": ({"gate_theta": 0.95}, {}),
    "Delta_0.02": ({"gate_Delta": 0.02}, {}),
    "Delta_0.1": ({"gate_Delta": 0.1}, {}),
}
LETTER_ARMS = ["base"]
GATE_ARMS = ["convergencia"]
CONTROL_ARMS = ["sin_semilla", "semilla_1e-3", "semilla_1e-9", "suelo_1e-18", "sin_compuerta", "jerarquia_rota", "pesos_iguales"]
SENSITIVITY_ARMS = ["lambda_5", "lambda_7", "lambda_14", "lambda_20", "referencia_ley", "theta_0.8", "theta_0.95", "Delta_0.02",
                    "Delta_0.1"]
RULES = {
    "evaluated_on": "los S de los gatillos de los tres colapsos completos (gatillo y habilitación) del brazo base, por δ₀",
    "A_interval": [7.0, 14.0], "n_collapses_required": 3,
    "letters": {"A": "los dos cocientes S_{k+1}/S_k en [7, 14]", "B": "uno de los dos en [7, 14]",
                "C": "ninguno en [7, 14], o menos de tres colapsos completos",
                "INDETERMINADO": "alguna puerta numérica o algún control obligatorio falla"},
    "order": "puertas → controles → C → B → A",
    "per_cell": "letra por δ₀; la global es la peor con el orden A < B < INDETERMINADO < C",
    "gates": {
        "finished": "el descenso del brazo base termina (velocidad proyectada ≈ 0, ninguna dirección sembrada inestable), sin "
                    "divergencia y sin subdesbordamiento de una dirección sembrada",
        "monotonia_max_dV_over_T_total": 1e-12,
        "identity_max_abs_diff": 1e-6,
        "trigger_is_min_eigenvalue": "en cada colapso el autovalor que cruza cero es el mínimo del hessiano reducido",
        "hierarchy": "los colapsos salen en el orden V1D, V2D, V3D",
        "final_state": "hessiano reducido definido positivo al final y |V_final − V_mín| ≤ final_V_tol·T_total",
        "final_V_tol": 1e-9,
        "convergence_rel": 1e-4,
        "convergence": "con el paso a la mitad (d_sigma_hat y dS_step) hay el mismo número de colapsos y cada S de gatillo "
                       "cambia ≤ convergence_rel en valor relativo",
    },
    "controls": {
        "sin_semilla": "con las tres semillas a cero no hay ningún colapso completo y f final coincide con la parte del Basal, "
                       "T₀/T_total, a no_seed_rel",
        "no_seed_rel": 1e-6,
        "semilla": "con semilla 1e-3 y 1e-9 hay los mismos colapsos y cada S de gatillo cambia ≤ seed_rel respecto de 1e-6",
        "seed_rel": 1e-3,
        "suelo": "con un suelo de ruido de 1e-18 (fuente continua en lugar de semilla inicial) cada S de gatillo cambia ≤ seed_rel",
        "sin_compuerta": "sin compuerta (m_k² ≡ +1) no hay ningún gatillo ni colapso",
        "jerarquia_rota": "con q_1 sin sembrar y q_2, q_3 sembradas no hay ningún colapso completo",
        "pesos_iguales": "con el reparto 1:1:1:1 la regla de letras da C (la A no sale por construcción)",
    },
    "sensitivity_without_letter": "barrido de λ (5, 7, 14, 20; con el suelo de 1e-18 para que el flotante no apague q_2), "
                                  "referencia de los pesos c̄·δ₀³ en lugar de la T₀ medida, y compuerta (θ_g = 0.8 y 0.95; "
                                  "Δ_g = 0.02 y 0.1): se publican con la lectura de la regla, sin letra",
    "published_without_letter": ["desviación de cada gatillo respecto de 0.009 / 0.099 / 0.999",
                                 "recuento de números elegidos: con la ley de escala, λ, θ_g y Δ_g; con pesos independientes, "
                                 "tres pesos, θ_g y Δ_g",
                                 "el juguete radial del oráculo (sin inclinación), como referencia ya pública"],
    "no_letter_on_lambda": True,
}
E13 = ("A en los tres δ₀, con cocientes próximos a 11.3 y 10.1. Es lo que da el juguete radial, ya público, y aquí los pesos se "
       "fijan respecto de la Tensión Primordial MEDIDA del Basal, de modo que la inclinación del reloj casi no mueve el reparto: "
       "la A es casi segura y no prueba nada sobre λ, que entra a mano. Lo que puede fallar, y es lo que el cribado añade, son las "
       "puertas y los controles en el reloj completo (Plano Dual, inclinación, frontera φ_E = 0): que la Monotonía se cumpla paso "
       "a paso, que cada gatillo sea el autovalor mínimo, que sin semilla o sin compuerta no haya cascada y que los umbrales en S "
       "no dependan de la semilla. Una B, una C o un INDETERMINADO serían el resultado con contenido. En el barrido de λ se "
       "espera que los cocientes sigan a 1 + λ y a λ + 1/(1 + λ): la A solo para λ entre 7 y 13, que es decir que la Década entra "
       "por λ y no sale de la dinámica")


E13_ROUND2 = ("A en los tres δ₀, con cocientes entre 11.24 y 11.30 y 10.12. No es una expectativa ciega: son los valores de la "
              "ronda 1 (brazo base en δ₀ = 0.003 y 0.01; brazo con suelo de ruido en los tres). Lo que la ronda 2 puede añadir es "
              "que la semilla inicial, integrada sin agotarse, complete la cascada en δ₀ = 0.0554 y supere los controles de "
              "semilla en los tres δ₀. Si un control falla ahora, ya no será por el flotante. La A sigue sin decir nada sobre λ, "
              "que entra a mano")
ROUND2_ORDERED_BY = ("decisión del autor del 13-nov-2026, «R6 — Apertura de ronda 2 mediante integración logarítmica» "
                     "(docs/decision_autor_2026-11-13_r6_ronda2.md), tras el INDETERMINADO de la ronda 1; R6 del plan del 11-nov, "
                     "decisiones 4, 5 y 7")


def select_round(n: int) -> None:
    """Apunta el guion a una ronda: directorio, preinscripción y corridas propios."""
    global ROUND, OUTDIR, PREREG, RUNS
    if n not in ROUND_DIRS:
        raise SystemExit(f"FALLO CERRADO: ronda desconocida ({n})")
    ROUND = n
    OUTDIR = ROOT / "results" / ROUND_DIRS[n]
    PREREG = OUTDIR / "preregistration.json"
    RUNS = OUTDIR / "runs"


def _round2_context() -> dict:
    """Lo que la ronda 2 hereda y declara: la ronda 1 con su letra, lo único que cambia y lo que ya se conoce."""
    out1 = ROOT / "results" / ROUND_DIRS[1]
    an1 = out1 / "analysis.json"
    if not an1.exists():
        raise SystemExit("FALLO CERRADO: la ronda 2 se congela después del análisis de la ronda 1")
    if not DECISION_ROUND2.exists():
        raise SystemExit("FALLO CERRADO: falta la decisión del autor que abre la ronda 2")
    # el texto literal de la decisión: los párrafos citados del acta
    literal = [ln[2:] for ln in DECISION_ROUND2.read_text(encoding="utf-8").splitlines() if ln.startswith("> ")]
    a = json.loads(an1.read_text(encoding="utf-8"))
    known = {str(c["delta0"]): {"letter": c["letter"], "S_triggers": c.get("S_triggers", []), "ratios": c.get("ratios", [])}
             for c in a["cells"]}
    floor = {}
    for d0 in DELTA0_GRID:
        f = out1 / "runs" / f"d{d0}_suelo_1e-18.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            floor[str(d0)] = {"S_triggers": r["S_triggers"], "ratios": r["ratios"]}
    return {
        "round2_opening_decision": {"file": "docs/decision_autor_2026-11-13_r6_ronda2.md", "sha256": _sha(DECISION_ROUND2),
                                    "literal": literal},
        "author_conditions": {
            "same_model": "mismas ecuaciones físicas, condiciones iniciales, pesos, compuertas, controles, brazos, valores de δ₀ "
                          "y criterios de aceptación que la ronda 1",
            "only_change": "integrar y_k = ln q_k en lugar de q_k, con el tratamiento logarítmico de los acoplamientos y eventos "
                           "que lo requieran",
            "verified_before_run": {
                "equivalencia de las ecuaciones": "tests/test_enabling_cascade.py: test_latent_gradient_is_proportional_to_the_"
                                                 "amplitude, test_log_velocity_is_the_linear_velocity_divided_by_the_amplitude y "
                                                 "test_log_representation_matches_the_linear_one_on_the_toy (gatillos a 1e-6)",
                "conservación de los estados físicamente nulos": "test_log_representation_keeps_null_states_null (una dirección "
                                                                "sin sembrar sigue en cero exacto; sin q_1 no se abren las superiores)",
                "convergencia numérica": "test_log_representation_converges_under_step_halving (paso a la mitad, gatillos a 1e-5); "
                                        "en la ronda, la puerta de convergencia congelada, celda a celda"},
            "not_blind": "esta segunda ronda no es ciega, puesto que ya se conocen los resultados y cocientes de la primera; no se "
                         "interpretará como una confirmación estadísticamente independiente",
            "noise_source_arm": "el brazo con fuente continua de ruido permanece como hipótesis física alternativa y no sustituye "
                                "al brazo principal: sigue siendo un control, sin letra",
            "objective": "determinar si los casos antes no integrables admiten una trayectoria numérica estable y si, bajo las "
                         "reglas originales, se activan los modos de estructuración de V1D, V2D y V3D",
            "letter": "la letra se asigna solo por los criterios ya congelados; la ronda 1 queda como salió"},
        "previous_round": {"dir": f"results/{ROUND_DIRS[1]}", "preregistration_sha256": _sha(out1 / "preregistration.json"),
                           "analysis_sha256": _sha(an1), "letter_global": a["letter_global"],
                           "cause": "en la representación lineal el flotante agota la semilla de q_2 mientras su compuerta está "
                                    "cerrada (por debajo de 4.9e-324) en δ₀ = 0.01 con semilla 1e-9 y en δ₀ = 0.0554; el segundo "
                                    "gatillo se dispara y la dirección no se habilita (lectura_a_posteriori.md)"},
        "what_changes": "solo la representación de las amplitudes latentes: se integra ln q̂_k (amplitude_representation = "
                        "'log'; d ln q̂_k/dσ̂ = −r_k/G con ∂V̂/∂q̂_k = r_k·q̂_k) en todos los brazos. Reglas, puertas, controles, "
                        "brazos, rejilla de δ₀ y tolerancias son los de la ronda 1, sin tocar",
        "known_before_freeze": {
            "note": "la ronda 2 NO es ciega: estas cifras de la ronda 1 se conocen al congelar. En el juguete radial la "
                    "representación logarítmica reproduce los gatillos de la lineal a 1e-6 (tests). Ninguna celda con inclinación "
                    "se ha corrido con la representación logarítmica antes de congelar",
            "round1_base_cells": known, "round1_noise_floor_arm": floor},
    }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def prereg(_args) -> int:
    if PREREG.exists():
        raise SystemExit(f"FALLO CERRADO: la preinscripción ya está congelada (sha256 {_sha(PREREG)}); no se sobrescribe")
    if RUNS.exists() and any(RUNS.iterdir()):
        raise SystemExit("FALLO CERRADO: hay corridas; la preinscripción se congela antes de correr")
    if not DECISIONS.exists():
        raise SystemExit("FALLO CERRADO: falta el acta de las decisiones del autor del 13-nov-2026")
    ctx2 = _round2_context() if ROUND == 2 else None
    base = ROUND2_BASE if ROUND == 2 else BASE
    OUTDIR.mkdir(parents=True, exist_ok=True)
    doc = {
        "title": "Cribado de la cascada de habilitación: tres direcciones latentes con pesos por una ley de escala de un número, "
                 "en el reloj S completo" + (" — ronda 2 (amplitudes en logaritmo)" if ROUND == 2 else ""),
        "round": ROUND,
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": _git_head(),
        "generator": "scripts/run_enabling_cascade.py prereg", "generator_commit_must_precede_freeze": True,
        "ordered_by": "R6 del «Plan de acción para el repositorio (11-nov-2026)», abierto por las decisiones 4, 5 y 7 del autor "
                      "del 13-nov-2026",
        "author_decisions": {
            "file": "docs/decisiones_autor_2026-11-13.md", "sha256": _sha(DECISIONS),
            "decision_4": "Latente primero",
            "decision_5": "Los tres q_k no representan tres nuevas esencias fundamentales independientes. Los interpreto como "
                          "parámetros de orden asociados a modos de estructuración emergente de la Unidad Dual. […] El modelo R6 "
                          "puede utilizar tres amplitudes q_k como descripción efectiva de estas transiciones, pero su "
                          "interpretación como modos emergentes deberá justificarse mediante una dinámica común derivada de la "
                          "Unidad Dual. Su mera introducción no constituye una demostración de emergencia dimensional.",
            "decision_7": "Adopto provisionalmente una progresión geométrica de razón λ para los pesos del Basal, q_1 y q_2, con "
                          "un peso residual para q_3, de forma que los cuatro permanezcan normalizados. Para λ = 10, la "
                          "distribución propuesta es 0,9 %, 9 %, 90 % y 0,1 %, respectivamente. […] El resultado deberá "
                          "contrastarse mediante simulaciones del sistema completo, pruebas de estabilidad y controles negativos.",
            "note": "citas literales abreviadas con […]; el texto completo está en el fichero, cuyo sha256 se congela aquí",
        },
        "question": "Con el Basal del reloj y tres direcciones latentes cuyos pesos salen de la ley de escala con λ = 10, ¿hay "
                    "tres colapsos completos, en orden, con los dos cocientes de umbrales en [7, 14], y se cumplen la Monotonía y "
                    "los controles?",
        "declared": {
            "delta0_grid": DELTA0_GRID,
            "clock": "reloj por defecto (core/s_clock.py): rayo θ_nuc = 0, punto de escape, acoplos fijos, con la inclinación "
                     "(ē = 1), G = 1, d_sigma_hat = 5e-4",
            "model": "V̂ = V̂0 + Σ_k W_k·[½·m_k²·q̂_k² + ¼·q̂_k⁴], m_k² = −tanh((Y_{k−1} − θ_g)/Δ_g), Y_0 = ρ̂²/ρ̂₊², Y_k = q̂_k² "
                     "(core/enabling_cascade.py); flujo de gradiente puro en las cinco variables con la rigidez G del reloj",
            "base": base, "base_clock": BASE_CLOCK,
            "scale_law": "fracciones (λ−1)/λ³, (λ−1)/λ², (λ−1)/λ, 1/λ³ de la tensión nominal; con λ = 10: "
                         f"{list(scale_law_shares(C.LAMBDA_DECADE))}; W_k = 4·(profundidad nominal), respecto de la T₀ MEDIDA",
            "lambda": "λ = 10 del canon (mcmc_ontology.constants.LAMBDA_DECADE): CALIBRADO, no derivado",
            "S": "S ≡ f = (V_inicial − V)/T_total, con T_total del potencial ampliado (mínimo en el dominio físico)",
            "events": "gatillo_k: el autovalor del hessiano reducido (dirección radial del Plano Dual y direcciones latentes) "
                      "asociado a q_k pasa de positivo a no positivo; habilitacion_k: Y_k cruza ½; colapso_kD: los dos, con el S "
                      "del gatillo",
            "integrator": "RK4 con paso h = min(d_sigma_hat, dS_step·T_total/Σ̇), proyectado sobre φ̂ ≥ 0 como el reloj"
                          + ("; las amplitudes latentes se integran en logaritmo" if ROUND == 2 else ""),
            "arms": {k: {"model": v[0], "clock": v[1]} for k, v in ARMS.items()},
            "letter_arms": LETTER_ARMS, "gate_arms": GATE_ARMS, "control_arms": CONTROL_ARMS, "sensitivity_arms": SENSITIVITY_ARMS,
            "reference_without_letter": "juguete radial del oráculo: δ₀ = 0.01 con ē = 0 (cifras ya públicas en la revisión del 11-nov)",
        },
        "rules": RULES,
        "pre_freeze_checks": "el instrumento se comprobó con el juguete radial sin inclinación (cifras ya públicas: gatillos, "
                             "tensión total, retraso por semilla, control sin semilla, sin compuerta, jerarquía rota y pesos "
                             "iguales) y con el reloj completo SIN semilla en los tres δ₀ (sin cascada: reproduce el descenso del "
                             "reloj por defecto a 1e-13). Ninguna celda con inclinación y semilla se corrió antes de congelar",
        "oracle_finding_before_freeze": "en el juguete, el tercer gatillo del oráculo (σ̂ = 1.675) depende de un suelo numérico de "
                                        "su integrador (≈ 1e-18): con la dinámica exacta de una semilla inicial, q_2 decae e^−517 "
                                        "con la compuerta cerrada y el gatillo cae en σ̂ = 3.18; en S el umbral es el mismo "
                                        "(0.99006). Por eso el brazo base usa la dinámica exacta y el suelo es un control",
        "expectation_E13": E13_ROUND2 if ROUND == 2 else E13,
        "what_this_cannot_decide": [
            "λ = 10: entra como número calibrado de la ley de escala; la coincidencia con la Ley de la Década es una hipótesis "
            "pendiente de derivación (decisión 7)",
            "que un parámetro de orden q_k sea una dimensión espacial: son tres amplitudes añadidas; su interpretación como modos "
            "emergentes exige una dinámica común derivada de la Unidad Dual (decisión 5)",
            "la compuerta: es un acople postulado, con dos números elegidos",
            "si el colapso es nucleado o espinodal (decisión 6, pendiente; R7)",
            "nada observable: recorrido interno del reloj (E8)"],
        "prohibitions": {"no_threshold_tuning": True, "no_runs_before_freeze": True, "default_clock_untouched": True},
    }
    if ROUND == 2:
        doc["ordered_by"] = ROUND2_ORDERED_BY
        doc["pre_freeze_checks"] = ("las tres verificaciones que el autor pide antes de ejecutar se hicieron con juguetes sin "
                                    "inclinación y están en la suite: equivalencia de las ecuaciones (la velocidad en logaritmo es "
                                    "la lineal dividida por la amplitud, y los gatillos coinciden a 1e-6), conservación de los "
                                    "estados nulos (una dirección sin sembrar sigue en cero exacto) y convergencia (paso a la "
                                    "mitad, gatillos a 1e-5). Además, aguanta semillas de 1e-30 y abre la dirección que la "
                                    "representación lineal deja en un subnormal. Ninguna celda con inclinación se corrió con la "
                                    "representación logarítmica antes de congelar")
        doc["oracle_finding_before_freeze"] = ("el fallo de la ronda 1 no se vio en el juguete porque allí W_2 = 336 y el redondeo "
                                               "de los subnormales deja crecer la amplitud; con W_2 ≥ 361 no")
        doc.update(ctx2)
    PREREG.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sha = _sha(PREREG)
    D = doc["declared"]
    md = [f"# Preinscripción — {doc['title']}", "",
          f"Congelada {doc['frozen_utc']} en el commit `{doc['code_commit'][:9]}` (contiene el generador), ANTES de ninguna corrida "
          f"con letra. sha256 de `preregistration.json`: `{sha}`.", "",
          f"**Pregunta**: {doc['question']}", "", "## Decisiones del autor (13-nov-2026)", "",
          f"Acta: `{doc['author_decisions']['file']}` (sha256 `{doc['author_decisions']['sha256']}`).", "",
          f"- **Decisión 4**: «{doc['author_decisions']['decision_4']}».",
          f"- **Decisión 5**: «{doc['author_decisions']['decision_5']}»",
          f"- **Decisión 7**: «{doc['author_decisions']['decision_7']}»", "", "## Declarado", ""]
    md += [f"- **{k}**: {v}" for k, v in D.items() if k != "arms"]
    md += ["", "### Brazos", "", "| Brazo | Cambia en el modelo | Cambia en el reloj |", "|---|---|---|"]
    md += [f"| {k} | {json.dumps(v['model'], ensure_ascii=False) if v['model'] else '—'} | "
           f"{json.dumps(v['clock'], ensure_ascii=False) if v['clock'] else '—'} |" for k, v in D["arms"].items()]
    md += ["", "## Reglas (congeladas)", ""]
    for k, v in RULES.items():
        if isinstance(v, dict):
            md += [f"- **{k}**:"] + [f"    - {kk}: {vv}" for kk, vv in v.items()]
        elif isinstance(v, list) and k != "A_interval":
            md += [f"- **{k}**:"] + [f"    - {x}" for x in v]
        else:
            md.append(f"- **{k}**: {v}")
    if ROUND == 2:
        pr, kb, od = doc["previous_round"], doc["known_before_freeze"], doc["round2_opening_decision"]
        md += ["", "## Decisión del autor que abre la ronda 2", "", f"Acta: `{od['file']}` (sha256 `{od['sha256']}`). Texto literal:", ""]
        for i, ln in enumerate(od["literal"]):
            md += [f"> {ln}"] + ([">"] if i < len(od["literal"]) - 1 else [])
        md += ["", "### Cómo se cumple cada condición", ""]
        for kk, vv in doc["author_conditions"].items():
            if isinstance(vv, dict):
                md += [f"- **{kk}**:"] + [f"    - {a}: {b}" for a, b in vv.items()]
            else:
                md.append(f"- **{kk}**: {vv}")
        md += ["", "## Ronda anterior y lo que cambia", "",
               f"- **Ronda 1**: `{pr['dir']}`, preinscripción `{pr['preregistration_sha256']}`, letra global {pr['letter_global']}. "
               f"Causa: {pr['cause']}.", f"- **Lo que cambia**: {doc['what_changes']}.", f"- **Lo que ya se conoce**: {kb['note']}."]
        md += [f"    - ronda 1, brazo base, δ₀ = {k}: letra {v['letter']}; gatillos {v['S_triggers']}; cocientes {v['ratios']}"
               for k, v in kb["round1_base_cells"].items()]
        md += [f"    - ronda 1, suelo de ruido 1e-18, δ₀ = {k}: gatillos {v['S_triggers']}; cocientes {v['ratios']}"
               for k, v in kb["round1_noise_floor_arm"].items()]
    md += ["", "## Comprobaciones antes de congelar", "", doc["pre_freeze_checks"] + ".", "", doc["oracle_finding_before_freeze"] + ".",
           "", "## Expectativa (E13, escrita antes de congelar)", "", doc["expectation_E13"] + ".", "", "## Lo que no puede decidir", ""]
    md += [f"- {s}" for s in doc["what_this_cannot_decide"]]
    (OUTDIR / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"preinscripción congelada: {PREREG} sha256 {sha}")
    return 0


def _load_prereg() -> tuple[dict, str]:
    if not PREREG.exists():
        raise SystemExit("FALLO CERRADO: falta la preinscripción de la cascada de habilitación")
    return json.loads(PREREG.read_text(encoding="utf-8")), _sha(PREREG)


def _compact(r: dict, wall: float, with_trajectory: bool) -> dict:
    keep = ("delta0", "T0_basal_hat", "T_ref_hat", "T_total_hat", "T_total_nominal_hat", "shares", "nominal_thresholds",
            "decade_thresholds", "W", "x_esc_hat", "x_plus_hat", "y_min", "events", "collapses", "S_triggers", "S_half_enabling",
            "ratios", "n_triggers", "n_collapses", "checks", "steps", "finished", "stop_reason", "h_max", "h_min_used",
            "underflow_modes", "sigma_hat_final", "S_final", "f_final")
    keep = keep + (("amplitude_representation",) if "amplitude_representation" in r else ())
    out = {k: r[k] for k in keep}
    cfg = dict(r["config"])
    clock = cfg.pop("clock")
    out["config"] = {**cfg, "clock": {k: clock[k] for k in ("delta0", "m_bar", "b_bar", "e_bar", "C0", "G", "theta_nuc", "nucleation",
                                                             "d_sigma_hat", "thresholds", "couplings_flow")}}
    out["wall_s"] = round(wall, 2)
    if with_trajectory:
        # trayectoria remuestreada en S (401 puntos): suficiente para el informe, sin guardar ~30 000 muestras
        tr = {k: np.asarray(v) for k, v in r["trajectory"].items()}
        S_grid = np.linspace(0.0, float(tr["S"][-1]), 401)
        out["trajectory_resampled_in_S"] = {"S": S_grid.tolist(), **{k: np.interp(S_grid, tr["S"], tr[k]).tolist()
                                                                     for k in ("sigma_hat", "Y0", "Y1", "Y2", "Y3", "theta")}}
    return out


def _run_cell(delta0: float, model: dict, clock: dict, base: dict, base_clock: dict, with_trajectory: bool = False) -> dict:
    kw = {**base, **model}
    kw["seeds"] = tuple(kw["seeds"])
    if "shares" in kw:
        kw["shares"] = tuple(kw["shares"])
    t0 = time.time()
    r = run_cascade(delta0, clock_kwargs={**base_clock, **clock}, **kw)
    return _compact(r, time.time() - t0, with_trajectory)


def cmd_run(_args) -> int:
    pre, psha = _load_prereg()
    D = pre["declared"]
    RUNS.mkdir(parents=True, exist_ok=True)
    for d0 in D["delta0_grid"]:
        for name, arm in D["arms"].items():
            path = RUNS / f"d{d0}_{name}.json"
            if path.exists():
                continue
            rec = _run_cell(d0, arm["model"], arm["clock"], D["base"], D["base_clock"], with_trajectory=name in D["letter_arms"])
            rec.update({"arm": name, "preregistration_sha256": psha, "code_commit": _git_head()})
            path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {path.name}: {rec['n_collapses']} colapsos, S = {[round(x, 6) for x in rec['S_triggers']]}, "
                  f"cocientes {[round(x, 3) for x in rec['ratios']]} ({rec['stop_reason']}; {rec['wall_s']} s)", flush=True)
    path = RUNS / "oracle_toy_d0.01_e0.json"
    if not path.exists():
        rec = _run_cell(0.01, {}, {"e_bar": 0.0}, D["base"], D["base_clock"], with_trajectory=True)
        rec.update({"arm": "juguete radial del oráculo (sin letra)", "preregistration_sha256": psha, "code_commit": _git_head()})
        path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  {path.name}: S = {[round(x, 6) for x in rec['S_triggers']]}", flush=True)
    return 0


def _rel_change(a: list, b: list) -> float | None:
    """Mayor cambio relativo entre dos listas de gatillos; None si no tienen la misma longitud."""
    if len(a) != len(b):
        return None
    return max((abs(x - y) / abs(y) for x, y in zip(a, b)), default=0.0)


def _worst(letters: list) -> str:
    order = ["A", "B", "INDETERMINADO", "C"]
    return max(letters, key=order.index)


def _cell(d0: float, R: dict, load) -> dict:
    """Letra de un δ₀: puertas → controles → regla."""
    G, K = R["gates"], R["controls"]
    base = load(d0, "base")
    if base is None:
        return {"delta0": d0, "letter": "INDETERMINADO", "reason": "falta la corrida base", "gates": {}, "controls": {}}
    ck = base["checks"]
    gates = {
        "finished": bool(base["finished"]) and not base["underflow_modes"],
        "monotonia": bool(ck["monotonia_4_5"]["max_dV_over_T_total"] <= G["monotonia_max_dV_over_T_total"]),
        "identity": bool(ck["S_equals_f_identity"]["max_abs_diff"] < G["identity_max_abs_diff"]),
        "trigger_is_min_eigenvalue": ck["gatillo_es_autovalor_minimo"]["pass"] is not False,
        "hierarchy": bool(ck["secuencia_jerarquica"]["pass"]),
        "final_state": bool(ck["estado_final"]["stable"]
                            and abs(ck["estado_final"]["V_final_minus_V_min_over_T_total"]) <= G["final_V_tol"]),
    }
    conv = load(d0, "convergencia")
    dconv = None if conv is None else _rel_change(conv["S_triggers"], base["S_triggers"])
    gates["convergence"] = dconv is not None and dconv <= G["convergence_rel"]
    controls, detail = {}, {"convergence_rel_change": dconv}
    ns = load(d0, "sin_semilla")
    if ns is not None:
        share = ns["T0_basal_hat"] / ns["T_total_hat"]
        detail["no_seed_f_final_over_basal_share"] = ns["f_final"] / share
        controls["sin_semilla"] = ns["n_collapses"] == 0 and abs(ns["f_final"] / share - 1.0) <= K["no_seed_rel"]
    else:
        controls["sin_semilla"] = False
    for name, key in (("semilla_1e-3", "semilla_1e-3"), ("semilla_1e-9", "semilla_1e-9"), ("suelo_1e-18", "suelo")):
        r = load(d0, name)
        d = None if r is None else _rel_change(r["S_triggers"], base["S_triggers"])
        detail[f"{name}_rel_change"] = d
        controls[name] = d is not None and d <= K["seed_rel"]
    ng = load(d0, "sin_compuerta")
    controls["sin_compuerta"] = ng is not None and ng["n_collapses"] == 0 and ng["n_triggers"] == 0
    jr = load(d0, "jerarquia_rota")
    controls["jerarquia_rota"] = jr is not None and jr["n_collapses"] == 0
    pe = load(d0, "pesos_iguales")
    pe_letter = None if pe is None else cascade_letter(pe["S_triggers"], R)
    detail["pesos_iguales"] = None if pe is None else {"S_triggers": pe["S_triggers"], **pe_letter}
    controls["pesos_iguales"] = pe_letter is not None and pe_letter["letter"] == "C"
    L = cascade_letter(base["S_triggers"], R)
    failed = [f"puerta {k}" for k, v in gates.items() if not v] + [f"control {k}" for k, v in controls.items() if not v]
    letter = "INDETERMINADO" if failed else L["letter"]
    thr = base["decade_thresholds"]
    return {"delta0": d0, "letter": letter, "reason": "; ".join(failed) if failed else L["reason"], "rule_reading": L,
            "S_triggers": base["S_triggers"], "ratios": base["ratios"], "S_half_enabling": base["S_half_enabling"],
            "sigma_hat_triggers": [c["sigma_hat"] for c in base["collapses"]],
            "deviation_from_decade": [s / t - 1.0 for s, t in zip(base["S_triggers"], thr)],
            "T0_basal_hat": base["T0_basal_hat"], "T_total_hat": base["T_total_hat"], "T_total_nominal_hat": base["T_total_nominal_hat"],
            "max_dV_over_T_total": ck["monotonia_4_5"]["max_dV_over_T_total"],
            "identity_max_abs_diff": ck["S_equals_f_identity"]["max_abs_diff"],
            "seals": ck["sellos_como_asentamientos"], "gates": gates, "controls": controls, "detail": detail,
            "steps": base["steps"], "sigma_hat_final": base["sigma_hat_final"]}


def cmd_analyze(_args) -> int:
    pre, psha = _load_prereg()
    R, D = pre["rules"], pre["declared"]

    def load(d0, name):
        p = RUNS / f"d{d0}_{name}.json"
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("preregistration_sha256") != psha:
            raise SystemExit(f"FALLO CERRADO: {p.name} se corrió con otra preinscripción")
        return rec

    cells = [_cell(d0, R, load) for d0 in D["delta0_grid"]]
    sens = []
    for d0 in D["delta0_grid"]:
        for name in D["sensitivity_arms"]:
            r = load(d0, name)
            if r is None:
                continue
            L = cascade_letter(r["S_triggers"], R)
            sens.append({"delta0": d0, "arm": name, "n_collapses": r["n_collapses"], "S_triggers": r["S_triggers"],
                         "ratios": r["ratios"], "rule_reading_without_letter": L["letter"], "finished": r["finished"],
                         "underflow_modes": r["underflow_modes"], "nominal_thresholds": r["nominal_thresholds"]})
    toy_path = RUNS / "oracle_toy_d0.01_e0.json"
    toy = json.loads(toy_path.read_text(encoding="utf-8")) if toy_path.exists() else None
    glob = _worst([c["letter"] for c in cells])
    doc = {"round": ROUND, "prereg_sha256": psha, "code_commit": _git_head(), "analyzed_utc": datetime.now(timezone.utc).isoformat(),
           "letter_global": glob, "cells": cells, "sensitivity_without_letter": sens,
           "oracle_toy_without_letter": None if toy is None else {k: toy[k] for k in ("S_triggers", "ratios", "S_half_enabling",
                                                                                      "T_total_hat", "T_total_nominal_hat")},
           "free_numbers": {"scale_law": ["λ", "θ_g", "Δ_g"], "independent_weights": ["w_1", "w_2", "w_3", "θ_g", "Δ_g"]},
           "reading": "comprobación interna (E8): la letra dice si el modelo efectivo, con λ puesto a mano, produce tres colapsos "
                      "ordenados con cocientes de Década en el reloj completo y supera sus controles; no deriva λ ni la emergencia "
                      "dimensional (decisiones 5 y 7 del autor)"}
    (OUTDIR / "analysis.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Cascada de habilitación{' (ronda 2)' if ROUND == 2 else ''} — letra global: **{glob}**", "",
          f"Preinscripción `{psha[:12]}…`; commit `{doc['code_commit'][:9]}`. Letra por δ₀; la global es la peor.", "",
          "| δ₀ | gatillos S (V1D, V2D, V3D) | cocientes | frente a 0.009 / 0.099 / 0.999 | letra | motivo |", "|---|---|---|---|---|---|"]
    for c in cells:
        md.append(f"| {c['delta0']:g} | {', '.join(f'{x:.6f}' for x in c.get('S_triggers', []))} | "
                  f"{', '.join(f'{x:.3f}' for x in c.get('ratios', []))} | "
                  f"{', '.join(f'{100 * x:+.2f} %' for x in c.get('deviation_from_decade', []))} | {c['letter']} | {c['reason']} |")
    md += ["", "## Puertas y controles", "", "| δ₀ | " + " | ".join(list(cells[0]["gates"]) + list(cells[0]["controls"])) + " |",
           "|---|" + "---|" * (len(cells[0]["gates"]) + len(cells[0]["controls"]))]
    for c in cells:
        md.append(f"| {c['delta0']:g} | " + " | ".join("sí" if v else "NO" for v in list(c["gates"].values()) + list(c["controls"].values())) + " |")
    md += ["", "| δ₀ | mayor ΔV/T_total | abs(S − f) máx. | cambio con el paso a la mitad | con semilla 1e-3 | con semilla 1e-9 | con suelo 1e-18 |",
           "|---|---|---|---|---|---|---|"]
    for c in cells:
        d = c.get("detail", {})

        def fmt(x):
            return "—" if x is None else f"{x:.1e}"
        md.append(f"| {c['delta0']:g} | {fmt(c.get('max_dV_over_T_total'))} | {fmt(c.get('identity_max_abs_diff'))} | "
                  f"{fmt(d.get('convergence_rel_change'))} | {fmt(d.get('semilla_1e-3_rel_change'))} | "
                  f"{fmt(d.get('semilla_1e-9_rel_change'))} | {fmt(d.get('suelo_1e-18_rel_change'))} |")
    if ROUND == 1:
        md += ["", "## Sensibilidad (sin letra)", "", "| δ₀ | brazo | colapsos | gatillos S | cocientes | lectura de la regla |",
               "|---|---|---|---|---|---|"]
        for s in sens:
            md.append(f"| {s['delta0']:g} | {s['arm']} | {s['n_collapses']} | {', '.join(f'{x:.5f}' for x in s['S_triggers'])} | "
                      f"{', '.join(f'{x:.2f}' for x in s['ratios'])} | {s['rule_reading_without_letter']} |")
    else:
        # desde la ronda 2 la tabla dice si la corrida terminó: una celda detenida no es una lectura de la regla
        md += ["", "## Sensibilidad (sin letra)", "",
               "| δ₀ | brazo | termina | colapsos | gatillos S | cocientes | lectura de la regla |", "|---|---|---|---|---|---|---|"]
        for s in sens:
            md.append(f"| {s['delta0']:g} | {s['arm']} | {'sí' if s['finished'] else 'NO'} | {s['n_collapses']} | "
                      f"{', '.join(f'{x:.5f}' for x in s['S_triggers'])} | {', '.join(f'{x:.2f}' for x in s['ratios'])} | "
                      f"{s['rule_reading_without_letter'] if s['finished'] else '— (no termina)'} |")
    if toy is not None:
        md += ["", "## Juguete radial del oráculo (referencia ya pública, sin letra)", "",
               f"Gatillos S = {', '.join(f'{x:.6f}' for x in toy['S_triggers'])}; cocientes {', '.join(f'{x:.3f}' for x in toy['ratios'])}; "
               f"tensión total {toy['T_total_hat']:.2f} frente a la nominal {toy['T_total_nominal_hat']:.2f}."]
    md += ["", f"**Lectura**: {doc['reading']}.", ""]
    (OUTDIR / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"letra global: {glob}; por δ₀: " + ", ".join(f"{c['delta0']:g}: {c['letter']}" for c in cells))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=sorted(ROUND_DIRS))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "run", "analyze"):
        sub.add_parser(c)
    args = ap.parse_args()
    select_round(args.round)
    return {"prereg": prereg, "run": cmd_run, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
