# Frente 5 (b), ronda 3 — capas esféricas: desenlace **INDETERMINADO**

Preinscripción `6fba6b53d929`; cualificación `64b9693e6d9a`; análisis en `2d65c8b1a`. Puertas: {'energy': False, 'inner_shells': True, 'newton_stationarity': True, 'N_control': True, 'runs_complete': True}. Fallos de energía: ['sweep_b005_eps0.2', 'sweep_b005_eps0.1', 'sweep_b005_eps0.05', 'sweep_b005_eps0.025', 'ctrlN_b005_N1e5'] (anunciados por la cualificación: ninguno). Corridas ausentes: ninguna. Estacionariedad newtoniana: 0.0286 dex.

| brazo | forma | t_weak [Myr] | r_exit [kpc] | M(<0.1) salida | M(<0.4) fin | ε_máx fin | |ΔE/E| | N(<0.4) ini | pasos | parada |
|---|---|---|---|---|---|---|---|---|---|---|
| newton_N1e6 | weak | — | — | 3.272e+06 | 5.693e+07 | 0.00e+00 | 1.97e-07 | 17953 | 334 | — |
| sweep_AS_eps0.2 | weak | 2.932 | 0.000 | 2.609e+07 | 5.535e+07 | 1.01e-03 | 2.37e-02 | 17953 | 981 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| sweep_AS_eps0.1 | weak | 0.599 | 0.000 | 4.107e+06 | 5.357e+07 | 1.02e-03 | 4.03e-03 | 17953 | 564 | ε_c máx = 1.02e-03 > 0.001 (fuera del régimen débil) |
| sweep_AS_eps0.05 | weak | 0.076 | 0.000 | 3.513e+06 | 5.332e+07 | 1.01e-03 | 3.98e-04 | 17953 | 218 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| sweep_AS_eps0.025 | weak | 0.011 | 0.000 | 3.471e+06 | 5.330e+07 | 1.02e-03 | 7.65e-04 | 17953 | 7 | ε_c máx = 1.02e-03 > 0.001 (fuera del régimen débil) |
| sweep_b005_eps0.2 | weak | — | — | 1.684e+07 | 6.188e+07 | 2.24e-05 | 1.41e-03 | 17953 | 989 | presupuesto de tiempo de pared agotado |
| sweep_b005_eps0.1 | weak | — | — | 7.875e+06 | 5.471e+07 | 1.43e-04 | 2.87e-03 | 17953 | 1138 | presupuesto de tiempo de pared agotado |
| sweep_b005_eps0.05 | weak | 9.661 | 0.000 | 9.185e+06 | 5.605e+07 | 1.00e-03 | 1.32e-02 | 17953 | 1304 | ε_c máx = 1.00e-03 > 0.001 (fuera del régimen débil) |
| sweep_b005_eps0.025 | weak | 0.477 | 0.000 | 3.519e+06 | 5.327e+07 | 1.29e-03 | 2.27e-03 | 17953 | 606 | ε_c máx = 1.29e-03 > 0.001 (fuera del régimen débil) |
| series_AS_ds0.02 | weak | 0.136 | 0.000 | 3.636e+06 | 5.330e+07 | 1.02e-03 | 9.31e-04 | 17953 | 268 | ε_c máx = 1.02e-03 > 0.001 (fuera del régimen débil) |
| series_AS_ds0.08 | weak | 1.172 | 0.000 | 9.164e+06 | 5.454e+07 | 1.01e-03 | 3.14e-03 | 17953 | 537 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| sat_sub_ds0.02 | saturating | — | — | 5.000e+06 | 6.495e+07 | 5.39e-09 | 3.37e-05 | 17953 | 334 | — |
| sat_sub_ds0.04 | saturating | — | — | 4.974e+06 | 6.309e+07 | 5.32e-09 | 8.65e-06 | 17953 | 334 | — |
| sat_sub_ds0.08 | saturating | — | — | 5.034e+06 | 6.275e+07 | 5.16e-09 | 1.31e-06 | 17953 | 334 | — |
| sat_above_ds0.04 | saturating | — | — | 7.948e+06 | 7.612e+07 | 1.61e-08 | 2.74e-05 | 17953 | 334 | — |
| ctrlN_AS_N1e5 | weak | 0.558 | 0.000 | 4.430e+06 | 5.385e+07 | 1.00e-03 | 1.01e-02 | 1795 | 533 | ε_c máx = 1.00e-03 > 0.001 (fuera del régimen débil) |
| ctrlN_b005_N1e5 | weak | 30.764 | 0.000 | 3.401e+07 | 7.913e+07 | 1.00e-03 | 7.93e-02 | 1795 | 3180 | ε_c máx = 1.00e-03 > 0.001 (fuera del régimen débil) |

Barrido ε_soft (AS): t_weak por ε_soft {'0.2': 0.0029316981435224985, '0.1': 0.0005993826638942288, '0.05': 7.573079762711772e-05, '0.025': 1.0584294884056518e-05}; ajuste p = 2.7325528113434197, r² = 0.9970474180551231; potencia True; independiente False

Barrido ε_soft (b005): t_weak por ε_soft {'0.2': None, '0.1': None, '0.05': 0.009660895019321412, '0.025': 0.0004765068636445843}; ajuste p = None, r² = None; potencia False; independiente False

Serie ds a A_Sculptor: {'complete': True, 't_weak_by_ds': {'0.02': 0.0001359114892117779, '0.04': 0.0005993826638942288, '0.08': 0.0011717114343039329}, 'all_exit': True, 'UV_no_convergence': True}

Control saturante: {'sub_arms': ['sat_sub_ds0.02', 'sat_sub_ds0.04', 'sat_sub_ds0.08'], 'sub_any_exit': False, 'sub_M04_final_by_arm': {'sat_sub_ds0.02': 64948391.91611506, 'sat_sub_ds0.04': 63089102.29637362, 'sat_sub_ds0.08': 62745029.83898925}, 'sub_converges_with_ds': True, 'above_exits': False, 'above_t_weak_gyr': None, 'complete': True, 'as_predicted': False}

Control de N: {'AS': {'t_weak_1e6': 0.0005993826638942288, 't_weak_1e5': 0.0005576134656356315, 'ratio': 0.93031296903514, 'within_factor': True}, 'b005': {'t_weak_1e6': None, 't_weak_1e5': 0.03076389365300668, 'ratio': None, 'within_factor': None}}

letra bajo la regla congelada; A: t_weak ∝ ε_soft^p con p > 0 en las dos amplitudes, la serie ds a A_Sculptor no converge y el control saturante se comporta como predice la ecuación 1 (sub-umbral estable y convergente, supra-umbral inestable) — primera letra A positiva posible del frente 5 y la ecuación 1 como física; B: p > 0 pero el sub-umbral también sale (la saturante pierde su candidatura); C: t_weak independiente de ε_soft (el integrador manda; la lectura de la adenda se retira). Nada decide la ley ni la amplitud (E8, E13).

## Lo que no decide

- si la Ley de Cronos débil o la saturante es la ley (el brazo saturante es CONTROL: prueba la ecuación 1 como física solo si se cumple la predicción)
- la amplitud (A_Sculptor es hipótesis del 5E)
- los modos no radiales y el interior por debajo de ε_soft
