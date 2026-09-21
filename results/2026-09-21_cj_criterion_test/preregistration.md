# Preinscripción — Test del criterio de Cronos–Jeans con predicción convergida (teoría cinética lineal) en un medio homogéneo de láminas sin gravedad

Congelada 2026-09-21T12:03:48.219023+00:00 en el commit `18bccf550` (contiene el generador), ANTES de ninguna corrida de producción.

Pregunta: ¿Reproduce el instrumento de láminas el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q), proporcional a k, de la ley local ε_c = A·ρ^(3/2)? Si no, ningún halo con esta ley dice nada.

| q | y(q) | γ/k cinético | γ/k fluido | inestable |
|---|---|---|---|---|
| 0.5 | 0.0000 | 0.0000 | 0.0000 | no |
| 0.8 | 0.0000 | 0.0000 | 0.0000 | no |
| 1.2 | 0.1055 | 0.1492 | 0.4472 | sí |
| 2.0 | 0.4328 | 0.6120 | 1.0000 | sí |

Instrumento: {"N": 2000000, "ng": 512, "dt": 0.001, "sample_every": 2, "quiet_start": true, "n_beams": 1024, "per_beam_rule": "N/n_beams \u2265 2\u00b7ng: el ret\u00edculo de cada haz debe ser m\u00e1s fino que la malla CIC", "seed_amp": 2e-05, "seed": 1, "T_rule": "q > 1: T = 1.5\u00b7ln(1e-3/|\u03b4_k|(0))/\u03b3_cin(k) + 0.1 (llegar a 1e-3, treinta veces por encima de la ventana); q < 1: T = 0.6 L/\u03c3", "T_stable": 0.6, "T_safety": 1.5, "delta_end": 0.001, "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512}}

Reglas: {"linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)] = [3e-5, 3e-4]: una década de amplitud, ≥ 30 veces por debajo de la escala donde los pilotos vieron no linealidad (~1e-2) y tras el transitorio de mezcla de fases (los primeros ~3·(kσ)⁻¹)", "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0, "min_points_linear": 8, "min_r2": 0.98, "rate_rel_tol": 0.25, "k_independence_rel_spread": 0.25, "stable_max_growth_factor": 3.0, "beam_convergence_rel_tol": 0.1, "fluid_discriminates_if": "|γ_cin − γ_fluido|/γ_cin ≥ 0.25 en q = 1.2 y 2.0 (aquí: 2.0 y 0.63)"}

Desenlaces: C (crecimiento con q < 1 o sin crecimiento con q > 1: error primero) / B (umbral bien, tasa mal) / A (cinética reproducida ±25 %) / INDETERMINADO (fase lineal no resuelta). Sin datos; umbrales congelados.
