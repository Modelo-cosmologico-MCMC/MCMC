# Preinscripción Nivel A — Nivel A del frente 5 — halo aislado NFW (1e11 M☉, c = 10) con y sin Cronos v3 a la amplitud única A_Sculptor; controles de exclusión (cierre cosmológico) y de respuesta (10·A_Sculptor)

Congelada 2026-09-17T17:24:20.926746+00:00 en el commit `ec2fed05c` (contiene el generador), ANTES de generar ninguna condición inicial de las corridas preinscritas. Pilotos de desarrollo declarados en el JSON.

Sistema 1e+11 M☉, c = 10.0, N = 400000, ε_soft = 0.1 kpc; D_F(ε_soft) = 229, D_F = 1 en 0.89 kpc.

Brazos: a (newtoniano (referencia); semillas [1, 2, 3]; 2.0 Gyr), b (Cronos v3 completo (fuerza + fricción con compuerta + lapse) a A_Sculptor; semillas [1, 2, 3]; 2.0 Gyr), b_static (puerta numérica: Cronos a A_Sculptor con el campo CONGELADO en t = 0 (K + W + U_Cronos se conserva); semillas [1]; 0.25 Gyr), b_res (control de convergencia del campo: como b con k_inner = 128 (radio interior del campo medio ×2 en partículas); semillas [1]; 2.0 Gyr), c (control de exclusión: cierre cosmológico ρ_c = 200·ρ̄_m (A ≈ 41.5) con parada al violar el régimen débil; semillas [1]; 2.0 Gyr), cprime (control de respuesta: 10·A_Sculptor (régimen débil válido; D_F = 1 en ≈ 2 kpc); semillas [1]; 0.5 Gyr).

Puertas: {"weak_regime_eps_max": 0.001, "energy_rel_tol_newton": 0.005, "energy_rel_tol_frozen": 0.005, "equilibrium_dlog10_max": 0.1}. Reglas: {"bands_kpc": {"inner_0p1_0p4": [0.1, 0.4], "inner_0p4_1": [0.4, 1.0], "core_1_2p3": [1.0, 2.3], "fit_2p3_5": [2.3, 5.0], "outer_5_20": [5.0, 20.0]}, "band_min_log10_shift": 0.1, "band_sigma_factor": 3.0, "fit_window_kpc": [0.5, 5.0], "cored_delta_rmse_min": 0.05, "cored_rc_min_kpc": 0.5, "runaway_final_ratio_min": 3.0, "interior_resolved_N_min": 2000, "control_min_log10_shift": 0.1, "resolution_control_max_log10": 0.15}.

Desenlaces: C (núcleo por Cronos: sospecha de error) / B (interior modificado, signo publicado; subcaso runaway) / A (aburrido) / INDETERMINADO. Expectativa E13: contracción del interior, sin núcleo kpc.

Sin datos; umbrales congelados; fronteras en el JSON.
