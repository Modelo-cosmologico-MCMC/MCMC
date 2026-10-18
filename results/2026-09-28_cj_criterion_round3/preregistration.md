# Preinscripción — Test del criterio de Cronos–Jeans, ronda 3: banda ultravioleta controlada (filtro k_c declarado, celdas excluidas a priori por la cualificación E8-Q)

Congelada 2026-09-30T14:25:07.048005+00:00 en el commit `e0bac6c7b` (contiene el generador), ANTES de ninguna corrida de producción. sha256 de `preregistration.json`: `ae9abba01c26834464249edb2bcdadf9627aec2d8704bf2d6519998309d8cdbe`. Cualificación citada: `8a35a6a4380b…` (results/2026-09-28_qualification_sheets).

Pregunta: ¿Reproduce el instrumento de láminas, con la banda ultravioleta controlada y la fase lineal resuelta en todas las celdas no excluidas, el umbral q = 1 y la tasa cinética exacta γ/k = √2·σ·y(q·W(k)) proporcional a k de la ley local ε_c = A·ρ^(3/2)?

Celdas excluidas a priori: ['q2.0_n4']

| celda | W(k) | γ/k instrumento | ventana | T | t_ventana cierra | t_uv_nonlinear | excluida | puntos esperados |
|---|---|---|---|---|---|---|---|---|
| q0.8_n4 | 0.9992 | 0.0000 | None | 0.600 | nan | nan | no | — |
| q0.8_n8 | 0.9968 | 0.0000 | None | 0.600 | nan | nan | no | — |
| q0.8_n16 | 0.9872 | 0.0000 | None | 0.600 | nan | nan | no | — |
| q0.8_n32 | 0.9497 | 0.0000 | None | 0.600 | nan | nan | no | — |
| q1.2_n4 | 0.9992 | 0.1485 | [3.0000000000000004e-05, 0.00030000000000000003] | 1.950 | 0.911 | 5.387 | no | 617 |
| q1.2_n8 | 0.9968 | 0.1465 | [3.0000000000000004e-05, 0.00030000000000000003] | 1.038 | 0.462 | 5.387 | no | 313 |
| q1.2_n16 | 0.9872 | 0.1384 | [3e-06, 2.9999999999999997e-05] | 0.845 | 0.244 | 5.387 | no | 165 |
| q1.2_n32 | 0.9497 | 0.1062 | [3e-06, 2.9999999999999997e-05] | 0.585 | 0.159 | 5.387 | no | 108 |
| q2.0_n4 | 0.9992 | 0.6112 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.244 | 0.221 | 0.244 | sí | 150 |
| q2.0_n8 | 0.9968 | 0.6089 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.244 | 0.111 | 0.244 | no | 75 |
| q2.0_n16 | 0.9872 | 0.5994 | [3e-06, 2.9999999999999997e-05] | 0.244 | 0.056 | 0.244 | no | 38 |
| q2.0_n32 | 0.9497 | 0.5619 | [3e-06, 2.9999999999999997e-05] | 0.192 | 0.030 | 0.244 | no | 20 |

Instrumento: {"N": 2097152, "ng": 512, "dt": 0.001, "sample_every": 1, "full_modes_every": 50, "quiet_start": true, "n_beams": 1024, "cells_per_beam": 4, "lattice_rule": "N/n_beams = p·ng con p ENTERO (p = 4): depósito CIC de cada haz exactamente uniforme", "k_cut_frac": 0.5, "uv_band_frac": 0.5, "filter_rule": "paso bajo de Fourier sobre la densidad depositada antes de la no linealidad: k ≤ k_cut_frac·k_Nyquist; corrección de W(k): γ_inst(q, k) = γ_cin(q·W(k)) si k ≤ k_c, 0 si k > k_c", "seed_eigenmode": true, "seed_eigenmode_rule": "cada haz j se desplaza ξ_j = −Im[c_j e^{ikx}]/k con c_j = κ v_j (v_j + iy')/(σ²(v_j² + y'²))·δρ̂ e y' = γ/k de la predicción cinética PARA EL INSTRUMENTO (q·W(k)); q ≤ 1: siembra en densidad (no hay modo creciente)", "seed_amp_by_mode": {"4": 2e-05, "8": 2e-05, "16": 2e-06, "32": 2e-06}, "seed": 1, "T_rule": "q > 1: T = mín(1.5·ln(1e-3/|δ_k|(0))/(γ_inst/k · k) + 0.1, t_uv_nonlinear(ng 512, 1024 haces, k_c, q) de la cualificación); q < 1: T = 0.6 L/σ", "T_stable": 0.6, "T_safety": 1.5, "delta_end": 0.001, "beam_convergence_run": {"q": 1.2, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8}}

Reglas: {"linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)]: una década de amplitud, tras el transitorio", "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0, "min_points_linear": 8, "min_r2": 0.98, "rate_rel_tol": 0.25, "k_independence_rel_spread": 0.25, "stable_max_growth_factor": 3.0, "beam_convergence_rel_tol": 0.02, "uv_amp_nonlinear": 0.01, "exclusion_source": "results/2026-09-28_qualification_sheets (E8-Q): celda excluida si γ_UV·t_ventana > ln(A_nl/A_ruido_ef) con ng 512, 1024 haces, k_c = ½ k_Nyq; las celdas excluidas no se corren ni cuentan"}

Desenlaces: {"order": "puertas → C → B → A; INDETERMINADO si una puerta falla; ningún umbral se ajusta tras ver los números; las celdas excluidas a priori no cuentan", "C_criterion_fails": "crecimiento (factor > 3) en q = 0.8, o ausencia de crecimiento en alguna q > 1, o tasa NO proporcional a k (dispersión del cociente medido/predicho entre los modos no excluidos > 25 % en alguna q > 1)", "B_finite_size": "umbral y proporcionalidad correctos pero γ/k dentro del 25 % solo para n ≥ 8 (el modo n = 4 fuera de tolerancia en alguna q > 1 no excluida): efecto de tamaño finito real, a entender, no un fallo del criterio", "A_kinetic_reproduced": "todas las celdas no excluidas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k), dispersión entre modos ≤ 25 % y estabilidad en q = 0.8: el instrumento reproduce el criterio con su predicción convergida", "INDETERMINADO": "puerta violada (fase lineal, ruido de siembra, retículo, filtro, UV, convergencia en haces) o corridas ausentes"}

Expectativa (E13): A: la ronda 2 dio 11/12 celdas al 1–2 % de la cinética y la celda que falló es la que la cualificación excluye a priori
