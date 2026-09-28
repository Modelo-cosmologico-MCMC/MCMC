# Cualificación del instrumento «capas esféricas (halo_shells)» (E8-Q)

Commit `0dbbd5679`; `qualification.json` sha256 `64b9693e6d9a2d8b490ded802129ba313dd9ee1aca15266957282f12d895e4e0`. E8-Q: cualificación de instrumento — sin letras ni umbrales de desenlace; publica límites computables del instrumento y criterios de exclusión calculables antes de correr; una preinscripción solo puede congelar puertas que la cualificación haya mostrado alcanzables.

## Rejilla

- **N**: [100000]
- **ds**: [0.02, 0.04, 0.08]
- **eps_soft**: [0.2, 0.1, 0.05]
- **rank_update**: [False, True]
- **dt_min_myr**: [0.0001]
- **dt_max_myr**: 0.15
- **eta_dt**: 0.05
- **eta_field**: 0.02
- **k_max**: 10
- **arms**: {newton: {cronos: False, amplitude: 0, t_end_gyr: 0.05}, AS: {cronos: True, amplitude: 1, t_end_gyr: 0.02}, b005: {cronos: True, amplitude: 0.05, t_end_gyr: 0.01}}
- **extra**: [(5 entradas; véase el JSON), (5 entradas; véase el JSON), (5 entradas; véase el JSON), (5 entradas; véase el JSON), (5 entradas; véase el JSON)]
- **max_wall_s_per_run**: 480
- **system**: {M200_msun: 1e+11, c: 10, H0: 67.87, seed: 1, r_decay_factor: 0.3, refine_r_kpc: 2, refine_beta: 1.5}

## Límites medidos

- **energy_floor_newton_rank_update0**: {value: 1.012e-05, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 7.86e-06, best_key: newton_N100000_ds0.02_eps0.2_ru0_dtmin0.0001, worst_key: newton_N100000_ds0.04_eps0.025_ru0_dtmin0.0001}
- **energy_floor_newton_rank_update1**: {value: 2.251e-06, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 6.679e-08, best_key: newton_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001, worst_key: newton_N100000_ds0.02_eps0.05_ru1_dtmin0.0001}
- **energy_floor_AS_rank_update0**: {value: 0.07667, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 0.002486, best_key: AS_N100000_ds0.02_eps0.1_ru0_dtmin0.0001, worst_key: AS_N100000_ds0.04_eps0.2_ru0_dtmin0.0001}
- **energy_floor_AS_rank_update1**: {value: 0.0296, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 0.001237, best_key: AS_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001, worst_key: AS_N100000_ds0.02_eps0.2_ru1_dtmin0.0001}
- **energy_floor_b005_rank_update0**: {value: 0.006794, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 2.365e-05, best_key: b005_N100000_ds0.08_eps0.2_ru0_dtmin0.0001, worst_key: b005_N100000_ds0.02_eps0.1_ru0_dtmin0.0001}
- **energy_floor_b005_rank_update1**: {value: 0.0002781, meaning: máximo de |ΔE/E| sobre la rejilla (suelo conservador), best: 9.039e-07, best_key: b005_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001, worst_key: b005_N100000_ds0.04_eps0.05_ru1_dtmin0.0001}
- **newton_stationarity_dex_max**: {value: 0.09941}
- **AS_t_weak_myr_eps0.05_ru0**: {value: 0.1509, all: [0.1509, 0.344]}
- **AS_t_weak_myr_eps0.05_ru1**: {value: 0.1509, all: [0.1509, 0.3407]}
- **AS_t_weak_myr_eps0.1_ru0**: {value: 0.1798, all: [0.1798, 0.9906]}
- **AS_t_weak_myr_eps0.1_ru1**: {value: 0.1797, all: [0.1797, 0.5513, 0.9965]}
- **AS_t_weak_myr_eps0.2_ru0**: {value: 0.9058, all: [0.9058]}
- **AS_t_weak_myr_eps0.2_ru1**: {value: 2.09, all: [2.09]}
- **AS_t_weak_myr_eps0.025_ru0**: {value: 0.15, all: [0.15]}
- **AS_t_weak_myr_eps0.025_ru1**: {value: 0.15, all: [0.15]}

## Criterios de exclusión (calculables antes de correr)

- **energy_gate_rule**: puerta de energía de una preinscripción = 3 × el suelo medido aquí en el brazo newtoniano equivalente (mismo N, ds, ε_soft, modo de rango); los brazos de Cronos cuyo suelo medido supere esa puerta quedan declarados a priori como no cualificados para ella
- **stationarity_rule**: la puerta de estacionariedad newtoniana no puede ser menor que newton_stationarity_dex_max
- **dt_floor_rule**: una corrida que alcance dt_min antes de salir del régimen débil no tiene t_weak: la preinscripción debe declarar dt_min ≤ el mínimo con el que las corridas de A_Sculptor de esta rejilla salen

## Notas

- El suelo del error de energía en los brazos de Cronos durante el colapso de la cúspide es el que la ronda 2 no pudo bajar con η_dt ni η_field; aquí se mide con y sin rango actualizado en los subpasos.
- Los tiempos de salida t_weak se publican como límites del instrumento (dependencia de ε_soft, ds y N), no como resultado físico: E8-Q.
- Nada de esta cualificación decide una letra.

## Corridas

| clave | |ΔE/E| | estacionariedad [dex] | t_fin [Myr] | t_weak [Myr] | ε_máx fin | pasos | reordenaciones | pared [s] | parada |
|---|---|---|---|---|---|---|---|---|---|
| AS_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001 | 1.2e-03 | 0.000 | 0.17 | — | 2.2e-04 | 59 | 59 | 489.4 | presupuesto de tiempo de pared agotado |
| AS_N100000_ds0.02_eps0.05_ru0_dtmin0.0001 | 4.8e-03 | 0.001 | 0.15 | 0.151 | 1.4e-03 | 2 | 0 | 1.0 | ε_c máx = 1.39e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.02_eps0.05_ru1_dtmin0.0001 | 4.8e-03 | 0.001 | 0.15 | 0.151 | 1.4e-03 | 2 | 2 | 1.9 | ε_c máx = 1.39e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.02_eps0.1_ru0_dtmin0.0001 | 2.5e-03 | 0.002 | 0.18 | 0.180 | 1.0e-03 | 110 | 0 | 20.1 | ε_c máx = 1.02e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.02_eps0.1_ru1_dtmin0.0001 | 2.5e-03 | 0.002 | 0.18 | 0.180 | 1.0e-03 | 109 | 109 | 33.7 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.02_eps0.2_ru0_dtmin0.0001 | 7.1e-03 | 0.024 | 0.91 | 0.906 | 1.0e-03 | 502 | 0 | 172.6 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.02_eps0.2_ru1_dtmin0.0001 | 3.0e-02 | 0.013 | 2.09 | 2.090 | 1.0e-03 | 841 | 841 | 954.0 | ε_c máx = 1.00e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.04_eps0.025_ru0_dtmin0.0001 | 8.5e-03 | 0.001 | 0.15 | 0.150 | 2.5e-03 | 1 | 0 | 1.2 | ε_c máx = 2.46e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.04_eps0.025_ru1_dtmin0.0001 | 8.5e-03 | 0.001 | 0.15 | 0.150 | 2.5e-03 | 1 | 1 | 3.6 | ε_c máx = 2.50e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.04_eps0.05_ru0_dtmin0.0001 | 2.9e-03 | 0.001 | 0.15 | — | 5.1e-04 | 4 | 0 | 1.4 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| AS_N100000_ds0.04_eps0.05_ru1_dtmin0.0001 | 2.9e-03 | 0.001 | 0.15 | — | 5.1e-04 | 4 | 4 | 2.6 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| AS_N100000_ds0.04_eps0.1_ru0_dtmin0.0001 | 5.2e-03 | 0.006 | 0.55 | — | 5.4e-04 | 413 | 0 | 117.5 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| AS_N100000_ds0.04_eps0.1_ru1_dtmin0.0001 | 4.1e-03 | 0.006 | 0.55 | — | 6.8e-04 | 422 | 422 | 250.8 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| AS_N100000_ds0.04_eps0.1_ru1_dtmin1e-05 | 4.3e-03 | 0.006 | 0.55 | 0.551 | 1.0e-03 | 442 | 442 | 267.9 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.04_eps0.2_ru0_dtmin0.0001 | 7.7e-02 | 0.015 | 4.42 | — | 7.2e-04 | 1018 | 0 | 480.3 | presupuesto de tiempo de pared agotado |
| AS_N100000_ds0.04_eps0.2_ru1_dtmin0.0001 | 2.7e-03 | 0.029 | 1.69 | — | 1.6e-04 | 407 | 407 | 482.7 | presupuesto de tiempo de pared agotado |
| AS_N100000_ds0.08_eps0.05_ru0_dtmin0.0001 | 5.0e-03 | 0.007 | 0.34 | 0.344 | 1.0e-03 | 323 | 0 | 75.5 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.08_eps0.05_ru1_dtmin0.0001 | 5.2e-03 | 0.006 | 0.34 | 0.341 | 1.1e-03 | 359 | 359 | 201.5 | ε_c máx = 1.08e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.08_eps0.1_ru0_dtmin0.0001 | 4.5e-03 | 0.019 | 0.99 | 0.991 | 1.0e-03 | 455 | 0 | 171.2 | ε_c máx = 1.02e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.08_eps0.1_ru1_dtmin0.0001 | 5.1e-03 | 0.017 | 1.00 | 0.996 | 1.0e-03 | 403 | 403 | 451.2 | ε_c máx = 1.01e-03 > 0.001 (fuera del régimen débil) |
| AS_N100000_ds0.08_eps0.2_ru0_dtmin0.0001 | 3.7e-02 | 0.141 | 7.30 | — | 3.1e-04 | 834 | 0 | 480.7 | presupuesto de tiempo de pared agotado |
| AS_N100000_ds0.08_eps0.2_ru1_dtmin0.0001 | 2.1e-03 | 0.035 | 2.69 | — | 8.1e-05 | 377 | 377 | 481.6 | presupuesto de tiempo de pared agotado |
| AS_N300000_ds0.04_eps0.1_ru1_dtmin0.0001 | 1.9e-03 | 0.001 | 0.41 | — | 3.7e-04 | 193 | 193 | 482.9 | presupuesto de tiempo de pared agotado |
| b005_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001 | 9.0e-07 | 0.001 | 0.23 | — | 1.8e-06 | 51 | 51 | 482.9 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.02_eps0.05_ru0_dtmin0.0001 | 5.1e-04 | 0.007 | 0.63 | — | 4.6e-04 | 799 | 0 | 219.8 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.02_eps0.05_ru1_dtmin0.0001 | 2.6e-04 | 0.002 | 0.54 | — | 1.9e-04 | 669 | 669 | 481.2 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.02_eps0.1_ru0_dtmin0.0001 | 6.8e-03 | 0.031 | 5.15 | — | 4.4e-04 | 1511 | 0 | 688.3 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.02_eps0.1_ru1_dtmin0.0001 | 1.4e-04 | 0.004 | 1.21 | — | 4.9e-05 | 445 | 445 | 481.1 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.02_eps0.2_ru0_dtmin0.0001 | 9.5e-04 | 0.050 | 10.00 | — | 3.2e-05 | 1001 | 0 | 519.2 | — |
| b005_N100000_ds0.02_eps0.2_ru1_dtmin0.0001 | 1.0e-04 | 0.011 | 3.45 | — | 1.1e-05 | 422 | 422 | 480.2 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.04_eps0.025_ru0_dtmin0.0001 | 7.1e-05 | 0.001 | 0.15 | — | 7.2e-05 | 16 | 0 | 3.0 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.04_eps0.025_ru1_dtmin0.0001 | 7.1e-05 | 0.001 | 0.15 | — | 7.2e-05 | 16 | 16 | 4.1 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.04_eps0.05_ru0_dtmin0.0001 | 5.2e-04 | 0.006 | 0.84 | — | 1.4e-04 | 512 | 0 | 142.2 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.04_eps0.05_ru1_dtmin0.0001 | 2.8e-04 | 0.008 | 0.82 | — | 1.7e-04 | 576 | 576 | 450.6 | paso global por debajo del suelo dt_min = 1.0e-04 Myr |
| b005_N100000_ds0.04_eps0.1_ru0_dtmin0.0001 | 3.4e-03 | 0.020 | 5.36 | — | 1.4e-04 | 1066 | 0 | 480.5 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.04_eps0.1_ru1_dtmin0.0001 | 1.2e-04 | 0.008 | 2.29 | — | 1.4e-05 | 417 | 417 | 480.2 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.04_eps0.1_ru1_dtmin1e-05 | 2.1e-04 | 0.009 | 2.33 | — | 7.4e-05 | 518 | 518 | 480.7 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.04_eps0.2_ru0_dtmin0.0001 | 9.4e-05 | 0.060 | 10.00 | — | 5.9e-06 | 506 | 0 | 286.0 | — |
| b005_N100000_ds0.04_eps0.2_ru1_dtmin0.0001 | 2.6e-05 | 0.023 | 4.56 | — | 5.0e-06 | 296 | 296 | 482.4 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.08_eps0.05_ru0_dtmin0.0001 | 4.0e-03 | 0.020 | 4.20 | — | 2.2e-04 | 929 | 0 | 480.5 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.08_eps0.05_ru1_dtmin0.0001 | 2.2e-04 | 0.005 | 1.55 | — | 5.0e-05 | 430 | 430 | 481.7 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.08_eps0.1_ru0_dtmin0.0001 | 4.9e-04 | 0.071 | 10.00 | — | 2.1e-05 | 680 | 0 | 397.7 | — |
| b005_N100000_ds0.08_eps0.1_ru1_dtmin0.0001 | 6.5e-05 | 0.028 | 4.10 | — | 1.4e-05 | 324 | 324 | 481.2 | presupuesto de tiempo de pared agotado |
| b005_N100000_ds0.08_eps0.2_ru0_dtmin0.0001 | 2.4e-05 | 0.074 | 10.00 | — | 2.4e-06 | 308 | 0 | 194.2 | — |
| b005_N100000_ds0.08_eps0.2_ru1_dtmin0.0001 | 1.6e-05 | 0.038 | 8.38 | — | 2.4e-06 | 271 | 271 | 481.6 | presupuesto de tiempo de pared agotado |
| b005_N300000_ds0.04_eps0.1_ru1_dtmin0.0001 | 2.6e-06 | 0.002 | 0.89 | — | 4.9e-06 | 124 | 124 | 484.2 | presupuesto de tiempo de pared agotado |
| newton_N1000000_ds0.04_eps0.1_ru1_dtmin0.0001 | 6.7e-08 | 0.016 | 14.85 | — | 0.0e+00 | 99 | 99 | 1502.1 | presupuesto de tiempo de pared agotado |
| newton_N100000_ds0.02_eps0.05_ru0_dtmin0.0001 | 9.6e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 0 | 31.0 | — |
| newton_N100000_ds0.02_eps0.05_ru1_dtmin0.0001 | 2.3e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 334 | 110.0 | — |
| newton_N100000_ds0.02_eps0.1_ru0_dtmin0.0001 | 8.9e-06 | 0.057 | 50.00 | — | 0.0e+00 | 334 | 0 | 34.0 | — |
| newton_N100000_ds0.02_eps0.1_ru1_dtmin0.0001 | 2.2e-06 | 0.056 | 50.00 | — | 0.0e+00 | 334 | 334 | 126.7 | — |
| newton_N100000_ds0.02_eps0.2_ru0_dtmin0.0001 | 7.9e-06 | 0.008 | 50.00 | — | 0.0e+00 | 334 | 0 | 36.5 | — |
| newton_N100000_ds0.02_eps0.2_ru1_dtmin0.0001 | 2.0e-06 | 0.009 | 50.00 | — | 0.0e+00 | 334 | 334 | 143.5 | — |
| newton_N100000_ds0.04_eps0.025_ru0_dtmin0.0001 | 1.0e-05 | 0.099 | 50.00 | — | 0.0e+00 | 334 | 0 | 39.5 | — |
| newton_N100000_ds0.04_eps0.025_ru1_dtmin0.0001 | 2.2e-06 | 0.099 | 50.00 | — | 0.0e+00 | 334 | 334 | 120.3 | — |
| newton_N100000_ds0.04_eps0.05_ru0_dtmin0.0001 | 9.6e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 0 | 34.7 | — |
| newton_N100000_ds0.04_eps0.05_ru1_dtmin0.0001 | 2.3e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 334 | 122.5 | — |
| newton_N100000_ds0.04_eps0.1_ru0_dtmin0.0001 | 8.9e-06 | 0.057 | 50.00 | — | 0.0e+00 | 334 | 0 | 34.4 | — |
| newton_N100000_ds0.04_eps0.1_ru1_dtmin0.0001 | 2.2e-06 | 0.056 | 50.00 | — | 0.0e+00 | 334 | 334 | 149.9 | — |
| newton_N100000_ds0.04_eps0.1_ru1_dtmin1e-05 | 2.2e-06 | 0.056 | 50.00 | — | 0.0e+00 | 334 | 334 | 127.2 | — |
| newton_N100000_ds0.04_eps0.2_ru0_dtmin0.0001 | 7.9e-06 | 0.008 | 50.00 | — | 0.0e+00 | 334 | 0 | 36.4 | — |
| newton_N100000_ds0.04_eps0.2_ru1_dtmin0.0001 | 2.0e-06 | 0.009 | 50.00 | — | 0.0e+00 | 334 | 334 | 153.0 | — |
| newton_N100000_ds0.08_eps0.05_ru0_dtmin0.0001 | 9.6e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 0 | 34.0 | — |
| newton_N100000_ds0.08_eps0.05_ru1_dtmin0.0001 | 2.3e-06 | 0.096 | 50.00 | — | 0.0e+00 | 334 | 334 | 111.1 | — |
| newton_N100000_ds0.08_eps0.1_ru0_dtmin0.0001 | 8.9e-06 | 0.057 | 50.00 | — | 0.0e+00 | 334 | 0 | 40.6 | — |
| newton_N100000_ds0.08_eps0.1_ru1_dtmin0.0001 | 2.2e-06 | 0.056 | 50.00 | — | 0.0e+00 | 334 | 334 | 131.1 | — |
| newton_N100000_ds0.08_eps0.2_ru0_dtmin0.0001 | 7.9e-06 | 0.008 | 50.00 | — | 0.0e+00 | 334 | 0 | 35.1 | — |
| newton_N100000_ds0.08_eps0.2_ru1_dtmin0.0001 | 2.0e-06 | 0.009 | 50.00 | — | 0.0e+00 | 334 | 334 | 150.6 | — |
| newton_N300000_ds0.04_eps0.1_ru1_dtmin0.0001 | 4.9e-07 | 0.021 | 32.10 | — | 0.0e+00 | 214 | 214 | 480.9 | presupuesto de tiempo de pared agotado |
