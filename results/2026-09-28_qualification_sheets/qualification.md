# Cualificación del instrumento «láminas (cronos_jeans_1d)» (E8-Q)

Commit `4a456cd83`; `qualification.json` sha256 `8a35a6a4380b064612de16b533c24e413c0311cf6829223dbf9e3e29e9133f1a`. E8-Q: cualificación de instrumento — sin letras ni umbrales de desenlace; publica límites computables del instrumento y criterios de exclusión calculables antes de correr; una preinscripción solo puede congelar puertas que la cualificación haya mostrado alcanzables.

## Rejilla

- **ng**: [256, 512]
- **cells_per_beam**: [2, 4]
- **q**: [0.8, 1.2, 2]
- **k_cut_frac**: [None, 0.5]
- **n_beams**: [256, 1024]
- **T**: 0.4
- **dt**: 0.001
- **sample_every**: 5
- **seed**: 1
- **uv_band_frac**: 0.5
- **amp_nonlinear_declared**: 0.01
- **round3_cells**: {q: [1.2, 2], modes: [4, 8, 16, 32]}
- **linear_window_factors**: [3, 30]
- **seed_amp_by_mode**: {4: 2e-05, 8: 2e-05, 16: 2e-06, 32: 2e-06}

## Límites medidos

- **uv_noise_max_ng256_nb256_kcNone**: {value: 1.43e-16, runs: [ng256_nb256_p2_q0.8_kcNone, ng256_nb256_p2_q1.2_kcNone, ng256_nb256_p2_q2.0_kcNone, ng256_nb256_p4_q0.8_kcNone, ng256_nb256_p4_q1.2_kcNone, ng256_nb256_p4_q2.0_kcNone]}
- **gamma_uv_max_ng256_nb256_kcNone_q0.8**: {value: 10.2, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q0.8_kcNone: 10.2, ng256_nb256_p4_q0.8_kcNone: 7.599}, uv_final_max: 4.844e-14}
- **t_uv_nonlinear_ng256_nb256_kcNone_q0.8**: {value: 3.126, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng256_nb256_kcNone_q1.2**: {value: 11.43, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q1.2_kcNone: 11.43, ng256_nb256_p4_q1.2_kcNone: 8.572}, uv_final_max: 1.112e-13}
- **t_uv_nonlinear_ng256_nb256_kcNone_q1.2**: {value: 2.788, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng256_nb256_kcNone_q2.0**: {value: 74.59, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q2.0_kcNone: 67.61, ng256_nb256_p4_q2.0_kcNone: 74.59}, uv_final_max: 0.001992}
- **t_uv_nonlinear_ng256_nb256_kcNone_q2.0**: {value: 0.4274, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **uv_noise_max_ng256_nb256_kc0.5**: {value: 1.296e-16, runs: [ng256_nb256_p2_q0.8_kc0.5, ng256_nb256_p2_q1.2_kc0.5, ng256_nb256_p2_q2.0_kc0.5, ng256_nb256_p4_q0.8_kc0.5, ng256_nb256_p4_q1.2_kc0.5, ng256_nb256_p4_q2.0_kc0.5]}
- **gamma_uv_max_ng256_nb256_kc0.5_q0.8**: {value: 8.369, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q0.8_kc0.5: 8.369, ng256_nb256_p4_q0.8_kc0.5: 7.214}, uv_final_max: 2.542e-14}
- **t_uv_nonlinear_ng256_nb256_kc0.5_q0.8**: {value: 3.821, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng256_nb256_kc0.5_q1.2**: {value: 9.179, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q1.2_kc0.5: 9.179, ng256_nb256_p4_q1.2_kc0.5: 6.301}, uv_final_max: 3.514e-14}
- **t_uv_nonlinear_ng256_nb256_kc0.5_q1.2**: {value: 3.484, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng256_nb256_kc0.5_q2.0**: {value: 73.99, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng256_nb256_p2_q2.0_kc0.5: 68.92, ng256_nb256_p4_q2.0_kc0.5: 73.99}, uv_final_max: 0.001682}
- **t_uv_nonlinear_ng256_nb256_kc0.5_q2.0**: {value: 0.4322, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **filter_gamma_ratio_ng256_nb256_q0.8**: {value: 0.8206, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng256_nb256_q1.2**: {value: 0.8029, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng256_nb256_q2.0**: {value: 0.9919, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **uv_noise_max_ng512_nb256_kcNone**: {value: 1.906e-16, runs: [ng512_nb256_p2_q0.8_kcNone, ng512_nb256_p2_q1.2_kcNone, ng512_nb256_p2_q2.0_kcNone, ng512_nb256_p4_q0.8_kcNone, ng512_nb256_p4_q1.2_kcNone, ng512_nb256_p4_q2.0_kcNone]}
- **gamma_uv_max_ng512_nb256_kcNone_q0.8**: {value: 24.29, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q0.8_kcNone: 22.75, ng512_nb256_p4_q0.8_kcNone: 24.29}, uv_final_max: 1.487e-11}
- **t_uv_nonlinear_ng512_nb256_kcNone_q0.8**: {value: 1.3, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb256_kcNone_q1.2**: {value: 31.71, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q1.2_kcNone: 31.71, ng512_nb256_p4_q1.2_kcNone: 31.02}, uv_final_max: 2.432e-10}
- **t_uv_nonlinear_ng512_nb256_kcNone_q1.2**: {value: 0.9964, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb256_kcNone_q2.0**: {value: 143.1, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q2.0_kcNone: 128, ng512_nb256_p4_q2.0_kcNone: 143.1}, uv_final_max: 0.02275}
- **t_uv_nonlinear_ng512_nb256_kcNone_q2.0**: {value: 0.2208, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **uv_noise_max_ng512_nb256_kc0.5**: {value: 1.688e-16, runs: [ng512_nb256_p2_q0.8_kc0.5, ng512_nb256_p2_q1.2_kc0.5, ng512_nb256_p2_q2.0_kc0.5, ng512_nb256_p4_q0.8_kc0.5, ng512_nb256_p4_q1.2_kc0.5, ng512_nb256_p4_q2.0_kc0.5]}
- **gamma_uv_max_ng512_nb256_kc0.5_q0.8**: {value: 16.94, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q0.8_kc0.5: 16.94, ng512_nb256_p4_q0.8_kc0.5: 12.94}, uv_final_max: 4.078e-12}
- **t_uv_nonlinear_ng512_nb256_kc0.5_q0.8**: {value: 1.872, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb256_kc0.5_q1.2**: {value: 23.86, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q1.2_kc0.5: 23.86, ng512_nb256_p4_q1.2_kc0.5: 21.07}, uv_final_max: 2.735e-11}
- **t_uv_nonlinear_ng512_nb256_kc0.5_q1.2**: {value: 1.329, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb256_kc0.5_q2.0**: {value: 141.4, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb256_p2_q2.0_kc0.5: 115.9, ng512_nb256_p4_q2.0_kc0.5: 141.4}, uv_final_max: 0.02009}
- **t_uv_nonlinear_ng512_nb256_kc0.5_q2.0**: {value: 0.2243, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **filter_gamma_ratio_ng512_nb256_q0.8**: {value: 0.6971, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng512_nb256_q1.2**: {value: 0.7527, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng512_nb256_q2.0**: {value: 0.9882, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **uv_noise_max_ng512_nb1024_kcNone**: {value: 1.737e-16, runs: [ng512_nb1024_p4_q0.8_kcNone, ng512_nb1024_p4_q1.2_kcNone, ng512_nb1024_p4_q2.0_kcNone]}
- **gamma_uv_max_ng512_nb1024_kcNone_q0.8**: {value: 8.35, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q0.8_kcNone: 8.35}, uv_final_max: 6.512e-14}
- **t_uv_nonlinear_ng512_nb1024_kcNone_q0.8**: {value: 3.795, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb1024_kcNone_q1.2**: {value: 10.24, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q1.2_kcNone: 10.24}, uv_final_max: 1.256e-13}
- **t_uv_nonlinear_ng512_nb1024_kcNone_q1.2**: {value: 3.094, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb1024_kcNone_q2.0**: {value: 152.2, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q2.0_kcNone: 152.2}, uv_final_max: 0.01185}
- **t_uv_nonlinear_ng512_nb1024_kcNone_q2.0**: {value: 0.2082, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **uv_noise_max_ng512_nb1024_kc0.5**: {value: 2.251e-16, runs: [ng512_nb1024_p4_q0.8_kc0.5, ng512_nb1024_p4_q1.2_kc0.5, ng512_nb1024_p4_q2.0_kc0.5]}
- **gamma_uv_max_ng512_nb1024_kc0.5_q0.8**: {value: 4.812, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q0.8_kc0.5: 4.812}, uv_final_max: 1.963e-14}
- **t_uv_nonlinear_ng512_nb1024_kc0.5_q0.8**: {value: 6.53, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb1024_kc0.5_q1.2**: {value: 5.833, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q1.2_kc0.5: 5.833}, uv_final_max: 1.916e-14}
- **t_uv_nonlinear_ng512_nb1024_kc0.5_q1.2**: {value: 5.387, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **gamma_uv_max_ng512_nb1024_kc0.5_q2.0**: {value: 128.8, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q2.0_kc0.5: 128.8}, uv_final_max: 0.009414}
- **t_uv_nonlinear_ng512_nb1024_kc0.5_q2.0**: {value: 0.2441, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **filter_gamma_ratio_ng512_nb1024_q0.8**: {value: 0.5763, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng512_nb1024_q1.2**: {value: 0.5696, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}
- **filter_gamma_ratio_ng512_nb1024_q2.0**: {value: 0.8461, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}

## Criterios de exclusión (calculables antes de correr)

- **rule**: celda (q, n) excluida si γ_UV(ng, k_c, q) · t_ventana > ln(A_nl / A_ruido_ef), con t_ventana = ln(hi/lo)/γ_inst(q, k) (hi/lo = factores de la ventana lineal), A_nl declarada antes de correr y A_ruido_ef = máx(ruido UV medido, A_semilla²); γ_UV se toma de la MISMA q (máximo sobre celdas por haz); una corrida sin racha ajustable excluye todas sus celdas; una celda sin ventana lineal (γ_inst ≤ 0) queda excluida por no tener qué medir
- **amp_nonlinear_declared**: 0.01
- **n_cells**: 48
- **n_excluded**: 7
- **cells**: (48 entradas; véase la tabla o el JSON)

## Notas

- La banda UV se mide sobre la densidad DEPOSITADA (antes del filtro): con filtro, γ_UV mide lo que el filtro deja pasar al campo de fuerzas a través de la no linealidad, no lo que queda en el depósito.
- γ_UV = 0 significa que la banda no creció por encima de 10× el ruido inicial en T; NaN, que creció sin racha ajustable de ≥ 4 muestras (fallo cerrado: excluye).
- Todas las corridas arrancan del retículo exacto sin siembra: γ_UV es la tasa a la que el instrumento amplifica su propio ruido de redondeo, y crece con q y con ng (no con las celdas por haz).
- Nada de esta cualificación decide una letra ni una tolerancia: fija qué celdas puede incluir la preinscripción de la ronda 3.

## Exclusión de las celdas de la ronda 3

| celda | W(k) | γ_inst | t_ventana | γ_UV | γ_UV·t | ln(A_nl/A_ruido_ef) | excluida | motivo |
|---|---|---|---|---|---|---|---|---|
| ng256_nb256_kcNone_q1.2_n4 | 0.997 | 3.68 | 0.625 | 11.4 | 7.15 | 17.0 | no | — |
| ng256_nb256_kcNone_q1.2_n8 | 0.987 | 6.96 | 0.331 | 11.4 | 3.78 | 17.0 | no | — |
| ng256_nb256_kcNone_q1.2_n16 | 0.950 | 10.68 | 0.216 | 11.4 | 2.46 | 21.6 | no | — |
| ng256_nb256_kcNone_q1.2_n32 | 0.812 | 0.00 | inf | 11.4 | inf | 21.6 | sí | sin ventana lineal (γ_inst ≤ 0) |
| ng256_nb256_kcNone_q2.0_n4 | 0.997 | 15.30 | 0.150 | 74.6 | 11.22 | 17.0 | no | — |
| ng256_nb256_kcNone_q2.0_n8 | 0.987 | 30.13 | 0.076 | 74.6 | 5.70 | 17.0 | no | — |
| ng256_nb256_kcNone_q2.0_n16 | 0.950 | 56.49 | 0.041 | 74.6 | 3.04 | 21.6 | no | — |
| ng256_nb256_kcNone_q2.0_n32 | 0.812 | 83.36 | 0.028 | 74.6 | 2.06 | 21.6 | no | — |
| ng256_nb256_kc0.5_q1.2_n4 | 0.997 | 3.68 | 0.625 | 9.2 | 5.74 | 17.0 | no | — |
| ng256_nb256_kc0.5_q1.2_n8 | 0.987 | 6.96 | 0.331 | 9.2 | 3.04 | 17.0 | no | — |
| ng256_nb256_kc0.5_q1.2_n16 | 0.950 | 10.68 | 0.216 | 9.2 | 1.98 | 21.6 | no | — |
| ng256_nb256_kc0.5_q1.2_n32 | 0.812 | 0.00 | inf | 9.2 | inf | 21.6 | sí | sin ventana lineal (γ_inst ≤ 0) |
| ng256_nb256_kc0.5_q2.0_n4 | 0.997 | 15.30 | 0.150 | 74.0 | 11.13 | 17.0 | no | — |
| ng256_nb256_kc0.5_q2.0_n8 | 0.987 | 30.13 | 0.076 | 74.0 | 5.65 | 17.0 | no | — |
| ng256_nb256_kc0.5_q2.0_n16 | 0.950 | 56.49 | 0.041 | 74.0 | 3.02 | 21.6 | no | — |
| ng256_nb256_kc0.5_q2.0_n32 | 0.812 | 83.36 | 0.028 | 74.0 | 2.04 | 21.6 | no | — |
| ng512_nb256_kcNone_q1.2_n4 | 0.999 | 3.73 | 0.617 | 31.7 | 19.56 | 17.0 | sí | la banda UV alcanza A_nl dentro de la ventana |
| ng512_nb256_kcNone_q1.2_n8 | 0.997 | 7.36 | 0.313 | 31.7 | 9.91 | 17.0 | no | — |
| ng512_nb256_kcNone_q1.2_n16 | 0.987 | 13.92 | 0.165 | 31.7 | 5.25 | 21.6 | no | — |
| ng512_nb256_kcNone_q1.2_n32 | 0.950 | 21.36 | 0.108 | 31.7 | 3.42 | 21.6 | no | — |
| ng512_nb256_kcNone_q2.0_n4 | 0.999 | 15.36 | 0.150 | 143.1 | 21.44 | 17.0 | sí | la banda UV alcanza A_nl dentro de la ventana |
| ng512_nb256_kcNone_q2.0_n8 | 0.997 | 30.60 | 0.075 | 143.1 | 10.76 | 17.0 | no | — |
| ng512_nb256_kcNone_q2.0_n16 | 0.987 | 60.26 | 0.038 | 143.1 | 5.47 | 21.6 | no | — |
| ng512_nb256_kcNone_q2.0_n32 | 0.950 | 112.99 | 0.020 | 143.1 | 2.92 | 21.6 | no | — |
| ng512_nb256_kc0.5_q1.2_n4 | 0.999 | 3.73 | 0.617 | 23.9 | 14.72 | 17.0 | no | — |
| ng512_nb256_kc0.5_q1.2_n8 | 0.997 | 7.36 | 0.313 | 23.9 | 7.46 | 17.0 | no | — |
| ng512_nb256_kc0.5_q1.2_n16 | 0.987 | 13.92 | 0.165 | 23.9 | 3.95 | 21.6 | no | — |
| ng512_nb256_kc0.5_q1.2_n32 | 0.950 | 21.36 | 0.108 | 23.9 | 2.57 | 21.6 | no | — |
| ng512_nb256_kc0.5_q2.0_n4 | 0.999 | 15.36 | 0.150 | 141.4 | 21.19 | 17.0 | sí | la banda UV alcanza A_nl dentro de la ventana |
| ng512_nb256_kc0.5_q2.0_n8 | 0.997 | 30.60 | 0.075 | 141.4 | 10.64 | 17.0 | no | — |
| ng512_nb256_kc0.5_q2.0_n16 | 0.987 | 60.26 | 0.038 | 141.4 | 5.40 | 21.6 | no | — |
| ng512_nb256_kc0.5_q2.0_n32 | 0.950 | 112.99 | 0.020 | 141.4 | 2.88 | 21.6 | no | — |
| ng512_nb1024_kcNone_q1.2_n4 | 0.999 | 3.73 | 0.617 | 10.2 | 6.32 | 17.0 | no | — |
| ng512_nb1024_kcNone_q1.2_n8 | 0.997 | 7.36 | 0.313 | 10.2 | 3.20 | 17.0 | no | — |
| ng512_nb1024_kcNone_q1.2_n16 | 0.987 | 13.92 | 0.165 | 10.2 | 1.69 | 21.6 | no | — |
| ng512_nb1024_kcNone_q1.2_n32 | 0.950 | 21.36 | 0.108 | 10.2 | 1.10 | 21.6 | no | — |
| ng512_nb1024_kcNone_q2.0_n4 | 0.999 | 15.36 | 0.150 | 152.2 | 22.81 | 17.0 | sí | la banda UV alcanza A_nl dentro de la ventana |
| ng512_nb1024_kcNone_q2.0_n8 | 0.997 | 30.60 | 0.075 | 152.2 | 11.45 | 17.0 | no | — |
| ng512_nb1024_kcNone_q2.0_n16 | 0.987 | 60.26 | 0.038 | 152.2 | 5.81 | 21.6 | no | — |
| ng512_nb1024_kcNone_q2.0_n32 | 0.950 | 112.99 | 0.020 | 152.2 | 3.10 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q1.2_n4 | 0.999 | 3.73 | 0.617 | 5.8 | 3.60 | 17.0 | no | — |
| ng512_nb1024_kc0.5_q1.2_n8 | 0.997 | 7.36 | 0.313 | 5.8 | 1.82 | 17.0 | no | — |
| ng512_nb1024_kc0.5_q1.2_n16 | 0.987 | 13.92 | 0.165 | 5.8 | 0.97 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q1.2_n32 | 0.950 | 21.36 | 0.108 | 5.8 | 0.63 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q2.0_n4 | 0.999 | 15.36 | 0.150 | 128.8 | 19.30 | 17.0 | sí | la banda UV alcanza A_nl dentro de la ventana |
| ng512_nb1024_kc0.5_q2.0_n8 | 0.997 | 30.60 | 0.075 | 128.8 | 9.69 | 17.0 | no | — |
| ng512_nb1024_kc0.5_q2.0_n16 | 0.987 | 60.26 | 0.038 | 128.8 | 4.92 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q2.0_n32 | 0.950 | 112.99 | 0.020 | 128.8 | 2.62 | 21.6 | no | — |

## Corridas

| clave | N | haces | γ_UV | puntos | r² | A_ruido | UV final | pared [s] |
|---|---|---|---|---|---|---|---|---|
| ng256_nb256_p2_q0.8_kc0.5 | 131072 | 256 | 8.37 | 70 | 0.865 | 8.5e-17 | 2.5e-14 | 6.8 |
| ng256_nb256_p2_q0.8_kcNone | 131072 | 256 | 10.20 | 71 | 0.963 | 8.8e-17 | 4.8e-14 | 8.1 |
| ng256_nb256_p2_q1.2_kc0.5 | 131072 | 256 | 9.18 | 71 | 0.880 | 8.9e-17 | 3.5e-14 | 7.0 |
| ng256_nb256_p2_q1.2_kcNone | 131072 | 256 | 11.43 | 71 | 0.971 | 9.2e-17 | 1.1e-13 | 7.1 |
| ng256_nb256_p2_q2.0_kc0.5 | 131072 | 256 | 68.92 | 69 | 0.753 | 8.9e-17 | 1.7e-03 | 7.3 |
| ng256_nb256_p2_q2.0_kcNone | 131072 | 256 | 67.61 | 69 | 0.760 | 9.7e-17 | 2.0e-03 | 7.7 |
| ng256_nb256_p4_q0.8_kc0.5 | 262144 | 256 | 7.21 | 68 | 0.828 | 1.2e-16 | 1.9e-14 | 14.7 |
| ng256_nb256_p4_q0.8_kcNone | 262144 | 256 | 7.60 | 66 | 0.938 | 1.3e-16 | 2.8e-14 | 14.2 |
| ng256_nb256_p4_q1.2_kc0.5 | 262144 | 256 | 6.30 | 69 | 0.774 | 1.3e-16 | 2.8e-14 | 14.3 |
| ng256_nb256_p4_q1.2_kcNone | 262144 | 256 | 8.57 | 71 | 0.971 | 1.3e-16 | 4.7e-14 | 14.8 |
| ng256_nb256_p4_q2.0_kc0.5 | 262144 | 256 | 73.99 | 67 | 0.738 | 1.3e-16 | 6.4e-04 | 14.4 |
| ng256_nb256_p4_q2.0_kcNone | 262144 | 256 | 74.59 | 67 | 0.754 | 1.4e-16 | 7.6e-04 | 13.9 |
| ng512_nb1024_p4_q0.8_kc0.5 | 2097152 | 1024 | 4.81 | 69 | 0.851 | 2.0e-16 | 2.0e-14 | 111.3 |
| ng512_nb1024_p4_q0.8_kcNone | 2097152 | 1024 | 8.35 | 69 | 0.943 | 1.7e-16 | 6.5e-14 | 119.2 |
| ng512_nb1024_p4_q1.2_kc0.5 | 2097152 | 1024 | 5.83 | 72 | 0.699 | 2.2e-16 | 1.9e-14 | 112.2 |
| ng512_nb1024_p4_q1.2_kcNone | 2097152 | 1024 | 10.24 | 71 | 0.971 | 1.7e-16 | 1.3e-13 | 117.1 |
| ng512_nb1024_p4_q2.0_kc0.5 | 2097152 | 1024 | 128.76 | 32 | 0.680 | 2.3e-16 | 9.4e-03 | 112.2 |
| ng512_nb1024_p4_q2.0_kcNone | 2097152 | 1024 | 152.18 | 29 | 0.743 | 1.7e-16 | 1.2e-02 | 110.5 |
| ng512_nb256_p2_q0.8_kc0.5 | 262144 | 256 | 16.94 | 75 | 0.914 | 6.3e-17 | 4.1e-12 | 14.4 |
| ng512_nb256_p2_q0.8_kcNone | 262144 | 256 | 22.75 | 75 | 0.988 | 6.9e-17 | 9.3e-12 | 13.8 |
| ng512_nb256_p2_q1.2_kc0.5 | 262144 | 256 | 23.86 | 76 | 0.955 | 6.7e-17 | 2.7e-11 | 14.5 |
| ng512_nb256_p2_q1.2_kcNone | 262144 | 256 | 31.71 | 76 | 0.985 | 8.6e-17 | 2.4e-10 | 14.7 |
| ng512_nb256_p2_q2.0_kc0.5 | 262144 | 256 | 115.86 | 35 | 0.696 | 8.5e-17 | 2.0e-02 | 14.6 |
| ng512_nb256_p2_q2.0_kcNone | 262144 | 256 | 127.97 | 33 | 0.730 | 1.1e-16 | 2.3e-02 | 14.4 |
| ng512_nb256_p4_q0.8_kc0.5 | 524288 | 256 | 12.94 | 71 | 0.878 | 1.6e-16 | 7.0e-13 | 29.2 |
| ng512_nb256_p4_q0.8_kcNone | 524288 | 256 | 24.29 | 72 | 0.969 | 1.6e-16 | 1.5e-11 | 27.8 |
| ng512_nb256_p4_q1.2_kc0.5 | 524288 | 256 | 21.07 | 73 | 0.883 | 1.6e-16 | 1.5e-11 | 28.0 |
| ng512_nb256_p4_q1.2_kcNone | 524288 | 256 | 31.02 | 73 | 0.971 | 1.8e-16 | 1.4e-10 | 27.5 |
| ng512_nb256_p4_q2.0_kc0.5 | 524288 | 256 | 141.38 | 32 | 0.736 | 1.7e-16 | 1.5e-02 | 27.8 |
| ng512_nb256_p4_q2.0_kcNone | 524288 | 256 | 143.07 | 32 | 0.757 | 1.9e-16 | 1.6e-02 | 29.1 |
