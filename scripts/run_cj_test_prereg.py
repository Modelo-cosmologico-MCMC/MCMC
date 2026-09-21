#!/usr/bin/env python
"""Preinscripción del TEST DEL CRITERIO de Cronos–Jeans con predicción
convergida (frente 5 refundado, pieza 1; propuesta del autor del 21-sep).

Se congela ANTES de ejecutar ninguna corrida de producción: sistema,
instrumento, predicción cinética exacta (cronos.cronos_jeans_kinetic),
reglas, tolerancias y desenlaces. Los pilotos de desarrollo se declaran
en el JSON y NO fijan ningún umbral: fijan solo el instrumento (N, ng,
dt, amplitud de siembra, ventana lineal).

Uso: python scripts/run_cj_test_prereg.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from cronos.cronos_jeans_kinetic import (  # noqa: E402
    gamma_over_k_instrument,
    prediction_table,
    transfer_function_cic,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "2026-09-21_cj_criterion_test"

Q_GRID = [0.5, 0.8, 1.2, 2.0]
MODES = [4, 8, 16]
INSTRUMENT = {"N": 2_000_000, "ng": 512, "dt": 1e-3, "sample_every": 2,      # muestreo cada 0.002 L/σ
              "quiet_start": True, "n_beams": 1024,                            # per_beam = 1953 ≥ 2·ng (sin aliasing retículo/malla)
              "per_beam_rule": "N/n_beams ≥ 2·ng: el retículo de cada haz debe ser más fino que la malla CIC",
              "seed_amp": 2e-5, "seed": 1,                                       # |δ_k|(0) = seed_amp/2 = 1e-5
              "T_rule": "q > 1: T = 1.5·ln(1e-3/|δ_k|(0))/γ_cin(k) + 0.1 (llegar a 1e-3, treinta veces por encima de la ventana); q < 1: T = 0.6 L/σ",
              "T_stable": 0.6, "T_safety": 1.5, "delta_end": 1e-3,
              "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512}}
RULES = {"linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)] = [3e-5, 3e-4]: una década de amplitud, "
                               "≥ 30 veces por debajo de la escala donde los pilotos vieron no linealidad (~1e-2) y "
                               "tras el transitorio de mezcla de fases (los primeros ~3·(kσ)⁻¹)",
         "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0,
         "min_points_linear": 8, "min_r2": 0.98,
         "rate_rel_tol": 0.25,                        # |γ_med/γ_inst − 1| ≤ 0.25 (γ_inst = predicción cinética con W(k))
         "k_independence_rel_spread": 0.25,            # (max − min)/media del cociente γ_med/γ_inst entre modos
         "stable_max_growth_factor": 3.0,              # q < 1: max|δ_k|/|δ_k|(0) ≤ 3 (transitorio incluido)
         "beam_convergence_rel_tol": 0.10,             # |γ/k(512 haces) − γ/k(1024 haces)|/γ/k(1024) ≤ 0.10
         "fluid_discriminates_if": "|γ_cin − γ_fluido|/γ_cin ≥ 0.25 en q = 1.2 y 2.0 (aquí: 2.0 y 0.63)"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=OUT.parent.parent).stdout.strip()
    pred = prediction_table(Q_GRID)
    amp0 = 0.5 * INSTRUMENT["seed_amp"]
    h = 1.0 / INSTRUMENT["ng"]
    windows = {}
    for r in pred:
        for mode in MODES:
            k = 2.0 * np.pi * mode
            W = transfer_function_cic(k, h)
            g_inst = gamma_over_k_instrument(r["q"], k, h)
            if r["q"] > 1.0:
                T = INSTRUMENT["T_safety"] * np.log(INSTRUMENT["delta_end"] / amp0) / (g_inst * k) + 0.1
                windows[f"q{r['q']}_n{mode}"] = {"window": [RULES["linear_window_lo_factor"] * amp0,
                                                            RULES["linear_window_hi_factor"] * amp0], "T": float(T),
                                                 "W_k": W, "q_eff": r["q"] * W, "gamma_over_k_instrument": g_inst,
                                                 "gamma_over_k_continuum": r["gamma_over_k_kinetic"]}
            else:
                windows[f"q{r['q']}_n{mode}"] = {"window": None, "T": INSTRUMENT["T_stable"], "W_k": W, "q_eff": r["q"] * W,
                                                 "gamma_over_k_instrument": 0.0, "gamma_over_k_continuum": 0.0}
    doc = {
        "title": "Test del criterio de Cronos–Jeans con predicción convergida (teoría cinética lineal) en un medio homogéneo de láminas sin gravedad",
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "code_commit": sha,
        "generator": "scripts/run_cj_test_prereg.py",
        "question": "¿Reproduce el instrumento de láminas el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q), proporcional a k, de la ley local ε_c = A·ρ^(3/2)? Si no, ningún halo con esta ley dice nada.",
        "system": {"sigma": 1.0, "rho0": 1.0, "L": 1.0, "gravity": False,
                   "force": "+c²∂_x ε_c(ρ), ε_c = A·ρ^(3/2), con (3/2)c²A·ρ0^(3/2) ≡ q·σ²"},
        "instrument": INSTRUMENT, "q_grid": Q_GRID, "modes": MODES,
        "prediction_kinetic": pred, "windows_and_T": windows, "delta_k_initial": amp0,
        "prediction_text": ("umbral exacto en q = 1 (F(0) = 1); para q > 1, √π·y·e^{y²}·erfc(y) = 1 − 1/q y γ = √2·k·σ·y(q): "
                            "γ/k independiente de k; para q < 1 ningún modo crece (amortiguamiento de Landau). "
                            "El límite fluido γ = kσ√(q−1) sobreestima la tasa cerca del umbral (×3 en q = 1.2). "
                            "PREDICCIÓN PARA EL INSTRUMENTO (la que se contrasta): el depósito/recogida CIC y el gradiente "
                            "centrado multiplican el acoplo por W(k) = sinc⁴(kh/2)·sin(kh)/(kh) (derivado, no ajustado), así "
                            "que γ/k = √2·σ·y(q·W(k)); cerca del umbral una corrección del 1–2 % en q es del 5–10 % en γ. "
                            "La independencia de k se contrasta sobre el cociente medido/predicho."),
        "rules": RULES,
        "gates": {"linear_phase_resolved": "≥ min_points_linear puntos en la ventana con r² ≥ min_r2 para CADA modo con q > 1",
                  "seed_noise": "con arranque silencioso, |δ_k|(0) de los modos NO sembrados < 0.1·|δ_k|(0) sembrado",
                  "beam_convergence": "γ/k en (q = 1.2, n = 8) con 512 haces difiere del de 1024 haces en ≤ 10 %: el instrumento "
                                      "no depende de la discretización en velocidad; si falla → INDETERMINADO"},
        "outcomes": {
            "order": "puertas → C → B → A → INDETERMINADO; nunca se ajustan umbrales tras ver los números",
            "C_criterion_or_instrument_broken": "crecimiento (factor > 2) en q < 1, o ausencia de fase lineal creciente en q > 1 en todos los modos: sospecha de error primero (instrumento), después el criterio",
            "B_threshold_ok_rate_off": "umbral correcto (estable en q < 1, crece en q > 1) pero γ/k fuera de la tolerancia cinética en algún q, o dependencia de k más allá de la dispersión permitida: se publica γ/k medido frente a cinético y fluido; el instrumento no reproduce la tasa",
            "A_kinetic_reproduced": "umbral correcto, γ/k dentro de ±25 % de la predicción cinética en q = 1.2 y 2.0 y dispersión entre modos ≤ 25 %: el instrumento reproduce el criterio con su predicción convergida",
            "INDETERMINADO": "fase lineal no resuelta (puntos o r² insuficientes) en algún modo con q > 1, o corridas ausentes"},
        "expectations_E13": "A; la tasa cinética, no la fluida (el test 1D del autor con ruido de Poisson dio tasa/k ≈ 0.2–0.6 en q = 1.2, compatible con ambas)",
        "development_declaration": {
            "pilots": "tres rondas de pilotos con N = 5e5 y ng = 256, declaradas: (i) siembra 0.005 con arranque silencioso por "
                      "permutación — la amplitud se estancaba en ~1e-2 y el ruido de fondo era ~1e-3: no había ventana lineal; "
                      "(ii) arranque multihaz con 128–4096 haces y siembra 1e-4 — con pocos haces aparecían modos de haz y "
                      "recurrencias, con demasiados (retículo por haz más grueso que la malla) aliasing, y para q = 2 la "
                      "fase puramente creciente solo se separa de los modos amortiguados a amplitudes ≲ 1e-3; (iii) siembra "
                      "1e-5, ventana de una década [3e-5, 3e-4], modos 8 y 16, 1024 haces con retículo más fino que la malla: "
                      "fase exponencial limpia. Los pilotos SÍ mostraron tasas; se declaran como lo que son (calibración del "
                      "instrumento) y ninguna tolerancia ni desenlace se ajustó a ellas: las tolerancias (25 %, 10 %) y los "
                      "desenlaces se fijaron antes de la ronda (iii)",
            "what_they_fixed": "solo el instrumento (siembra, arranque multihaz, ventana, modos, muestreo y T); NO se tocó ninguna "
                               "tolerancia ni desenlace tras ver tasas medidas",
            "budget": "13 corridas (4 q × 3 modos + control de haces) con N = 2e6: ~1–4 min cada una"},
        "prohibitions": {"no_threshold_tuning": True, "no_data": True, "no_change_to_A_sculptor": True},
    }
    (OUT / "preregistration.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [f"# Preinscripción — {doc['title']}\n",
          f"Congelada {doc['frozen_utc']} en el commit `{sha[:9]}` (contiene el generador), ANTES de ninguna corrida de producción.", "",
          f"Pregunta: {doc['question']}", "",
          "| q | y(q) | γ/k cinético | γ/k fluido | inestable |", "|---|---|---|---|---|"]
    for r in pred:
        md.append(f"| {r['q']} | {r['y']:.4f} | {r['gamma_over_k_kinetic']:.4f} | {r['gamma_over_k_fluid']:.4f} | {'sí' if r['unstable'] else 'no'} |")
    md += ["", f"Instrumento: {json.dumps(INSTRUMENT)}", "", f"Reglas: {json.dumps(RULES, ensure_ascii=False)}", "",
           "Desenlaces: C (crecimiento con q < 1 o sin crecimiento con q > 1: error primero) / B (umbral bien, tasa mal) / "
           "A (cinética reproducida ±25 %) / INDETERMINADO (fase lineal no resuelta). Sin datos; umbrales congelados."]
    (OUT / "preregistration.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Preinscripción del test del criterio congelada en {OUT} (commit {sha[:9]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
