# Preinscripción — Test del criterio de Cronos–Jeans, ronda 2: siembra del modo propio exacto, retículo exacto, modos 4/8/16/32

Congelada 2026-09-21T23:41:56.191911+00:00 en el commit `1d7f199d7` (contiene el generador), ANTES de ninguna corrida de producción. sha256 de `preregistration.json`: `3fd12bf248887ac173c2211c6d5c506461716374502413a82ac5fb3a58087538`.

Pregunta: ¿Reproduce el instrumento de láminas, con fase lineal resuelta en TODAS las celdas, el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q·W(k)) proporcional a k de la ley local ε_c = A·ρ^(3/2)?

| q | y(q) | γ/k cinético | γ/k fluido | inestable |
|---|---|---|---|---|
| 0.8 | 0.0000 | 0.0000 | 0.0000 | no |
| 1.2 | 0.1055 | 0.1492 | 0.4472 | sí |
| 2.0 | 0.4328 | 0.6120 | 1.0000 | sí |

| celda | W(k) | γ/k instrumento | ventana | T | puntos esperados en la ventana |
|---|---|---|---|---|---|
| q0.8_n4 | 0.9992 | 0.0000 | None | 0.600 | — |
| q0.8_n8 | 0.9968 | 0.0000 | None | 0.600 | — |
| q0.8_n16 | 0.9872 | 0.0000 | None | 0.600 | — |
| q0.8_n32 | 0.9497 | 0.0000 | None | 0.600 | — |
| q1.2_n4 | 0.9992 | 0.1485 | [3.0000000000000004e-05, 0.00030000000000000003] | 1.950 | 617 |
| q1.2_n8 | 0.9968 | 0.1465 | [3.0000000000000004e-05, 0.00030000000000000003] | 1.038 | 313 |
| q1.2_n16 | 0.9872 | 0.1384 | [3e-06, 2.9999999999999997e-05] | 0.845 | 165 |
| q1.2_n32 | 0.9497 | 0.1062 | [3e-06, 2.9999999999999997e-05] | 0.585 | 108 |
| q2.0_n4 | 0.9992 | 0.6112 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.550 | 150 |
| q2.0_n8 | 0.9968 | 0.6089 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.326 | 75 |
| q2.0_n16 | 0.9872 | 0.5994 | [3e-06, 2.9999999999999997e-05] | 0.272 | 38 |
| q2.0_n32 | 0.9497 | 0.5619 | [3e-06, 2.9999999999999997e-05] | 0.192 | 20 |

Instrumento: {"N": 2097152, "ng": 512, "dt": 0.001, "sample_every": 1, "full_modes_every": 50, "quiet_start": true, "n_beams": 1024, "cells_per_beam": 4, "lattice_rule": "N/n_beams = p·ng con p ENTERO (p = 4): depósito CIC de cada haz exactamente uniforme; sin batido retículo/malla (la regla de la ronda 1, N/n_beams ≥ 2·ng, no bastaba)", "seed_eigenmode": true, "seed_eigenmode_rule": "cada haz j se desplaza ξ_j = −Im[c_j e^{ikx}]/k con c_j = κ v_j (v_j + iy')/(σ²(v_j² + y'²))·δρ̂ e y' = γ/k de la predicción cinética PARA EL INSTRUMENTO (q·W(k)); q ≤ 1: siembra en densidad (no hay modo creciente)", "seed_amp_by_mode": {"4": 2e-05, "8": 2e-05, "16": 2e-06, "32": 2e-06}, "seed_amp_rule": "|δ_k|(0) = seed_amp/2: 1e-5 en los modos bajos y 1e-6 en los altos (los armónicos de la siembra escalan como δ²: una siembra menor baja el suelo de los modos altos, que crecen más deprisa)", "seed": 1, "T_rule": "q > 1: T = 1.5·ln(1e-3/|δ_k|(0))/(γ_inst/k · k) + 0.1; q < 1: T = 0.6 L/σ", "T_stable": 0.6, "T_safety": 1.5, "delta_end": 0.001, "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8}}

Reglas: {"linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)]: una década de amplitud, tras el transitorio", "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0, "min_points_linear": 8, "min_r2": 0.98, "rate_rel_tol": 0.25, "k_independence_rel_spread": 0.25, "stable_max_growth_factor": 3.0, "beam_convergence_rel_tol": 0.02}

Desenlaces: {"order": "puertas → C → B → A; INDETERMINADO si una puerta falla; ningún umbral se ajusta tras ver los números", "C_criterion_fails": "crecimiento (factor > 3) en q = 0.8, o ausencia de crecimiento en alguna q > 1, o tasa NO proporcional a k (dispersión del cociente medido/predicho entre modos > 25 % en alguna q > 1)", "B_finite_size": "umbral y proporcionalidad correctos pero γ/k dentro del 25 % solo para n ≥ 8 (el modo n = 4 fuera de tolerancia en alguna q > 1): efecto de tamaño finito real, a entender, no un fallo del criterio", "A_kinetic_reproduced": "todas las celdas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k), dispersión entre modos ≤ 25 % y estabilidad en q = 0.8: el instrumento reproduce el criterio con su predicción convergida", "INDETERMINADO": "puerta violada de nuevo (fase lineal, ruido de siembra, retículo, convergencia en haces) o corridas ausentes: entonces se cambia de instrumento (Vlasov euleriano 1D), no de siembra"}

Pilotos declarados: 22-sep, N = 5e5–2e6, ng = 256, q = 2, n ∈ {4, 8}: (i) siembra del modo propio frente a siembra en densidad con N/n_beams = 488 y 976 (no múltiplo de ng): r² 0.39 → 0.67 en n = 4, ambas con el mismo crecimiento explosivo de modos altos a t ≈ 0.15–0.18 dominado al final por el modo 8 ó 4 y con los modos altos partiendo de ~1e-10; la siembra conjugada (−y') decae al principio, confirmando el signo del modo; (ii) N/n_beams = 4·ng: los modos no sembrados arrancan en 1.5e-10 y solo se hacen relevantes tras cerrar la ventana; r² = 1.0000 y γ/k = 0.6086 frente a 0.6089. Los pilotos mostraron tasas y se declaran como calibración del instrumento.
