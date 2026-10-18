# Preinscripción — Test del criterio de Cronos–Jeans, ronda 3b: ampliación a q = 1.5 con el instrumento, las tolerancias y la regla de exclusión de la ronda 3

Congelada 2026-10-18T09:36:50.072405+00:00 en el commit `fb9e52b9d` (contiene el generador y la adenda de cualificación), ANTES de ninguna corrida de producción. sha256 de `preregistration.json`: `d0c16affe6501f2a4156223f911b1b95ac841a0179652d497be213e65c1b80ac`. Cualificación citada: `1a3283f3d200…` (results/2026-10-18_qualification_sheets_q1p5). Ronda 3: desenlace A, intacta.

Pregunta: ¿Reproduce el instrumento de láminas, en q = 1.5 y con la fase lineal resuelta en todas las celdas no excluidas, la tasa cinética exacta γ/k = √2·σ·y(q·W(k)) proporcional a k de la ley local ε_c = A·ρ^(3/2)?

Celdas excluidas a priori: ninguna

| celda | W(k) | γ/k instrumento | ventana | T | t_ventana cierra | t_uv_nonlinear | excluida | puntos esperados |
|---|---|---|---|---|---|---|---|---|
| q1.5_n4 | 0.9992 | 0.3420 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.723 | 0.396 | 0.723 | no | 268 |
| q1.5_n8 | 0.9968 | 0.3398 | [3.0000000000000004e-05, 0.00030000000000000003] | 0.504 | 0.199 | 0.723 | no | 135 |
| q1.5_n16 | 0.9872 | 0.3312 | [3e-06, 2.9999999999999997e-05] | 0.411 | 0.102 | 0.723 | no | 69 |
| q1.5_n32 | 0.9497 | 0.2969 | [3e-06, 2.9999999999999997e-05] | 0.274 | 0.057 | 0.723 | no | 39 |

Instrumento: {"N": 2097152, "ng": 512, "dt": 0.001, "sample_every": 1, "full_modes_every": 50, "quiet_start": true, "n_beams": 1024, "cells_per_beam": 4, "lattice_rule": "N/n_beams = p·ng con p ENTERO (p = 4): depósito CIC de cada haz exactamente uniforme", "k_cut_frac": 0.5, "uv_band_frac": 0.5, "filter_rule": "paso bajo de Fourier sobre la densidad depositada antes de la no linealidad: k ≤ k_cut_frac·k_Nyquist; corrección de W(k): γ_inst(q, k) = γ_cin(q·W(k)) si k ≤ k_c, 0 si k > k_c", "seed_eigenmode": true, "seed_eigenmode_rule": "cada haz j se desplaza ξ_j = −Im[c_j e^{ikx}]/k con c_j = κ v_j (v_j + iy')/(σ²(v_j² + y'²))·δρ̂ e y' = γ/k de la predicción cinética PARA EL INSTRUMENTO (q·W(k))", "seed_amp_by_mode": {"4": 2e-05, "8": 2e-05, "16": 2e-06, "32": 2e-06}, "seed": 1, "T_rule": "T = mín(1.5·ln(1e-3/|δ_k|(0))/(γ_inst/k · k) + 0.1, t_uv_nonlinear(ng 512, 1024 haces, k_c, q = 1.5) de la adenda de cualificación)", "T_safety": 1.5, "delta_end": 0.001, "beam_convergence_run": {"q": 1.5, "mode": 8, "n_beams_alt": 512, "cells_per_beam_alt": 8}}

Reglas: {"linear_window_rule": "fase lineal = |δ_k| ∈ [3·|δ_k|(0), 30·|δ_k|(0)]: una década de amplitud, tras el transitorio", "linear_window_lo_factor": 3.0, "linear_window_hi_factor": 30.0, "min_points_linear": 8, "min_r2": 0.98, "rate_rel_tol": 0.25, "k_independence_rel_spread": 0.25, "stable_max_growth_factor": 3.0, "beam_convergence_rel_tol": 0.02, "uv_amp_nonlinear": 0.01, "exclusion_source": "results/2026-10-18_qualification_sheets_q1p5 (E8-Q, adenda en q = 1.5): celda excluida si γ_UV·t_ventana > ln(A_nl/A_ruido_ef) con ng 512, 1024 haces, k_c = ½ k_Nyq; las celdas excluidas no se corren ni cuentan", "same_as_round3": "tolerancias, puertas, instrumento y regla de exclusión idénticos a results/2026-09-28_cj_criterion_round3"}

Desenlaces: {"order": "puertas → C → B → A; INDETERMINADO si una puerta falla; ningún umbral se ajusta tras ver los números; las celdas excluidas a priori no cuentan", "C_criterion_fails": "ausencia de crecimiento (factor ≤ 3) en q = 1.5, o tasa NO proporcional a k (dispersión del cociente medido/predicho entre los modos no excluidos > 25 %), o algún modo n ≥ 8 fuera del 25 %", "B_finite_size": "proporcionalidad correcta pero γ/k dentro del 25 % solo para n ≥ 8 (el modo n = 4 fuera de tolerancia)", "A_kinetic_reproduced": "todas las celdas no excluidas dentro del 25 % de γ_cin(1.5, k)·W(k) y dispersión entre modos ≤ 25 %", "INDETERMINADO": "puerta violada (fase lineal, ruido de siembra, retículo, filtro, UV, convergencia en haces) o corridas ausentes"}

Expectativa (E13): A: la ronda 3 reprodujo la cinética exacta en q = 1.2 y q = 2.0 dentro de 0.4–3 %; q = 1.5 está entre ambas
