# Frente 5 (b), ronda 2 — capas esféricas: desenlace **INDETERMINADO**

Preinscripción `1d6682f968fa`; análisis en `11c8d04f0`. Puertas: {'energy': False, 'inner_shells': True, 'newton_stationarity': True, 'N_control': True, 'runs_complete': True}. Corridas ausentes: ninguna. Estacionariedad newtoniana (a_N1e6): 0.0277 dex.

## Salida del régimen débil, energía y puertas por brazo

| brazo | t_weak [Myr] | r_exit [kpc] | M(<0.1) salida | M(<0.4) salida | ε_máx final | |ΔE/E| | |Δ(K+W)/E| | W_fric/|E| | N(<0.4) ini → fin | pasos | parada |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a_N1e6 | — | — | 2.993e+06 | 5.641e+07 | 0.00e+00 | 2.23e-06 | 2.23e-06 | 0.00e+00 | 17953 → 13852 | 667 | — |
| a_N1e5 | — | — | 3.639e+06 | 6.526e+07 | 0.00e+00 | 2.17e-05 | 2.17e-05 | 0.00e+00 | 1795 → 1450 | 667 | — |
| bAS_ds0.02_N1e6 | — | — | 3.750e+06 | 5.331e+07 | 3.53e-04 | 2.05e-03 | 2.75e-03 | 1.13e-07 | 17953 → 17953 | 28 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| bAS_ds0.04_N1e6 | — | — | 5.725e+06 | 5.357e+07 | 8.00e-04 | 8.82e-03 | 7.19e-02 | 1.15e-04 | 17953 → 17524 | 474 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| bAS_ds0.08_N1e6 | 3.830 | 0.000 | 1.472e+07 | 6.187e+07 | 1.03e-03 | 9.64e-03 | 2.56e-01 | 2.51e-03 | 17953 → 16733 | 535 | ε_c máx = 1.03e-03 > 0.001 (fuera del régimen débil) |
| b005_ds0.02_N1e6 | 28.279 | 0.000 | 1.821e+07 | 6.594e+07 | 1.01e-03 | 5.56e-02 | 2.57e-01 | 7.15e-03 | 17953 → 12524 | 1775 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| b005_ds0.04_N1e6 | — | — | 1.800e+07 | 5.513e+07 | 9.13e-04 | 1.78e-01 | 3.93e-01 | 1.54e-01 | 17953 → 12615 | 2144 | — |
| b005_ds0.08_N1e6 | — | — | 1.691e+07 | 5.445e+07 | 1.39e-04 | 1.55e-02 | 3.36e-02 | 1.18e-03 | 17953 → 13331 | 2201 | — |
| bAS_ds0.04_N1e5 | — | — | 4.322e+06 | 5.399e+07 | 6.82e-04 | 9.98e-03 | 3.94e-02 | 3.29e-05 | 1795 → 1760 | 398 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_ds0.04_N1e5 | 31.392 | 0.000 | 3.173e+07 | 7.942e+07 | 1.00e-03 | 1.46e-01 | 2.53e-01 | 3.10e-02 | 1795 → 1265 | 3278 | ε_c máx = 1.00e-03 > 0.001 (fuera del régimen débil) |

## Serie A_Sculptor (r_CJ = 0.709 kpc)

t_weak por brazo [Gyr]: {'bAS_ds0.02_N1e6': None, 'bAS_ds0.04_N1e6': None, 'bAS_ds0.08_N1e6': 0.003830327301829119}; todas salen: False; firma UV (t_weak decrece al refinar): False; r_exit: {'bAS_ds0.02_N1e6': None, 'bAS_ds0.04_N1e6': None, 'bAS_ds0.08_N1e6': 0.0}; salida dentro de r_CJ: False; M(<0.1) en la salida: {'bAS_ds0.02_N1e6': 3749809.832049797, 'bAS_ds0.04_N1e6': 5724758.976542186, 'bAS_ds0.08_N1e6': 14716787.39563651}.

## Serie 0.05·A_Sculptor (r_CJ = 0.180 kpc)

t_weak por brazo [Gyr]: {'b005_ds0.02_N1e6': 0.028278900029957876, 'b005_ds0.04_N1e6': None, 'b005_ds0.08_N1e6': None}; todas salen: False; firma UV (t_weak decrece al refinar): False; r_exit: {'b005_ds0.02_N1e6': 0.0, 'b005_ds0.04_N1e6': None, 'b005_ds0.08_N1e6': None}; salida dentro de r_CJ: False; M(<0.1) en la salida: {'b005_ds0.02_N1e6': 18206310.96781225, 'b005_ds0.04_N1e6': 17998775.437535223, 'b005_ds0.08_N1e6': 16910198.34577697}.

## Cociente log10 b/a por bandas (instante común con a_N1e6)

| brazo | t común [Myr] | [0.1,0.2) | [0.2,0.4) | [0.4,1.0) | [1.0,2.3) | [2.3,5.0) |
|---|---|---|---|---|---|---|
| bAS_ds0.02_N1e6 | 0.156 | -0.012 | -0.000 | -0.000 | -0.000 | +0.000 |
| bAS_ds0.04_N1e6 | 0.702 | -0.148 | +0.011 | -0.001 | +0.001 | -0.000 |
| bAS_ds0.08_N1e6 | 3.830 | -0.049 | -0.287 | -0.016 | -0.002 | +0.000 |
| b005_ds0.02_N1e6 | 28.279 | +0.122 | -0.077 | -0.019 | -0.003 | +0.000 |
| b005_ds0.04_N1e6 | 100.000 | -0.460 | -0.132 | -0.034 | +0.000 | -0.001 |
| b005_ds0.08_N1e6 | 100.000 | -0.513 | -0.115 | -0.035 | -0.000 | +0.000 |
| bAS_ds0.04_N1e5 | 0.552 | -0.151 | -0.005 | -0.004 | +0.002 | -0.001 |
| b005_ds0.04_N1e5 | 31.392 | -0.058 | +0.016 | -0.005 | -0.016 | -0.002 |

Control de N (ds = 0.04): {'bAS': {'t_weak_1e6': None, 't_weak_1e5': None, 'ratio_1e5_over_1e6': None, 'within_factor': None}, 'b005': {'t_weak_1e6': None, 't_weak_1e5': 0.03139155563810685, 'ratio_1e5_over_1e6': None, 'within_factor': None}}

letra bajo la regla congelada; A confirma la predicción del criterio (las dos amplitudes salen del régimen débil dentro de r_CJ, antes al refinar — la firma γ ∝ k — y A_Sculptor antes que 0.05·A_Sculptor); B, saturación no lineal a A_Sculptor; C, salida sin firma UV o fuera de r_CJ, o 0.05 estable con A_Sculptor inestable; ninguna letra es afirmación sobre el tratado: sitúa la Ley de Cronos débil con A_Sculptor frente a su propia inestabilidad ultravioleta (E8); los modos no radiales y el interior bajo ε_soft no están en el instrumento; el número no es señal (E13).

## Lo que no decide

- si la Ley de Cronos débil es la ley correcta (forma ε_c(ρ): diccionario)
- la amplitud (A_Sculptor es hipótesis del 5E; 0.05 es brazo)
- los modos no radiales y el interior por debajo de ε_soft = 0.1 kpc
