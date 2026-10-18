# Cualificación del instrumento «láminas (cronos_jeans_1d)» (E8-Q)

Commit `04bd812dc`; `qualification.json` sha256 `1a3283f3d200f6a31ca1b8abba93b1ad39db14edb95a7e826525ca0a6e37dc56`. E8-Q: cualificación de instrumento — sin letras ni umbrales de desenlace; publica límites computables del instrumento y criterios de exclusión calculables antes de correr; una preinscripción solo puede congelar puertas que la cualificación haya mostrado alcanzables.

## Rejilla

- **ng**: [512]
- **cells_per_beam**: [4]
- **q**: [1.5]
- **k_cut_frac**: [None, 0.5]
- **n_beams**: [1024]
- **T**: 0.4
- **dt**: 0.001
- **sample_every**: 5
- **seed**: 1
- **uv_band_frac**: 0.5
- **amp_nonlinear_declared**: 0.01
- **round3_cells**: {q: [1.5], modes: [4, 8, 16, 32]}
- **linear_window_factors**: [3, 30]
- **seed_amp_by_mode**: {4: 2e-05, 8: 2e-05, 16: 2e-06, 32: 2e-06}

## Límites medidos

- **uv_noise_max_ng512_nb1024_kcNone**: {value: 1.752e-16, runs: [ng512_nb1024_p4_q1.5_kcNone]}
- **gamma_uv_max_ng512_nb1024_kcNone_q1.5**: {value: 50.12, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q1.5_kcNone: 50.12}, uv_final_max: 3.43e-07}
- **t_uv_nonlinear_ng512_nb1024_kcNone_q1.5**: {value: 0.6319, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **uv_noise_max_ng512_nb1024_kc0.5**: {value: 2.275e-16, runs: [ng512_nb1024_p4_q1.5_kc0.5]}
- **gamma_uv_max_ng512_nb1024_kc0.5_q1.5**: {value: 43.45, unit: 1/(L/σ), unfittable_runs: [], per_run: {ng512_nb1024_p4_q1.5_kc0.5: 43.45}, uv_final_max: 5.308e-07}
- **t_uv_nonlinear_ng512_nb1024_kc0.5_q1.5**: {value: 0.723, meaning: ln(A_nl/A_ruido)/γ_UV: la preinscripción no puede pedir T mayor para esta (ng, n_beams, k_c, q)}
- **filter_gamma_ratio_ng512_nb1024_q1.5**: {value: 0.8669, meaning: γ_UV(k_c = ½ k_Nyq) / γ_UV(sin filtro) sobre la densidad depositada}

## Criterios de exclusión (calculables antes de correr)

- **rule**: celda (q, n) excluida si γ_UV(ng, k_c, q) · t_ventana > ln(A_nl / A_ruido_ef), con t_ventana = ln(hi/lo)/γ_inst(q, k) (hi/lo = factores de la ventana lineal), A_nl declarada antes de correr y A_ruido_ef = máx(ruido UV medido, A_semilla²); γ_UV se toma de la MISMA q (máximo sobre celdas por haz); una corrida sin racha ajustable excluye todas sus celdas; una celda sin ventana lineal (γ_inst ≤ 0) queda excluida por no tener qué medir
- **amp_nonlinear_declared**: 0.01
- **n_cells**: 8
- **n_excluded**: 0
- **cells**: {ng512_nb1024_kcNone_q1.5_n4: (9 entradas; véase el JSON), ng512_nb1024_kcNone_q1.5_n8: (9 entradas; véase el JSON), ng512_nb1024_kcNone_q1.5_n16: (9 entradas; véase el JSON), ng512_nb1024_kcNone_q1.5_n32: (9 entradas; véase el JSON), ng512_nb1024_kc0.5_q1.5_n4: (9 entradas; véase el JSON), ng512_nb1024_kc0.5_q1.5_n8: (9 entradas; véase el JSON), ng512_nb1024_kc0.5_q1.5_n16: (9 entradas; véase el JSON), ng512_nb1024_kc0.5_q1.5_n32: (9 entradas; véase el JSON)}

## Notas

- La banda UV se mide sobre la densidad DEPOSITADA (antes del filtro): con filtro, γ_UV mide lo que el filtro deja pasar al campo de fuerzas a través de la no linealidad, no lo que queda en el depósito.
- γ_UV = 0 significa que la banda no creció por encima de 10× el ruido inicial en T; NaN, que creció sin racha ajustable de ≥ 4 muestras (fallo cerrado: excluye).
- Todas las corridas arrancan del retículo exacto sin siembra: γ_UV es la tasa a la que el instrumento amplifica su propio ruido de redondeo, y crece con q y con ng (no con las celdas por haz).
- Nada de esta cualificación decide una letra ni una tolerancia: fija qué celdas puede incluir la preinscripción de la ronda 3b (q = 1.5).

## Exclusión de las celdas de la ronda 3b (q = 1.5)

| celda | W(k) | γ_inst | t_ventana | γ_UV | γ_UV·t | ln(A_nl/A_ruido_ef) | excluida | motivo |
|---|---|---|---|---|---|---|---|---|
| ng512_nb1024_kcNone_q1.5_n4 | 0.999 | 8.59 | 0.268 | 50.1 | 13.43 | 17.0 | no | — |
| ng512_nb1024_kcNone_q1.5_n8 | 0.997 | 17.08 | 0.135 | 50.1 | 6.76 | 17.0 | no | — |
| ng512_nb1024_kcNone_q1.5_n16 | 0.987 | 33.30 | 0.069 | 50.1 | 3.47 | 21.6 | no | — |
| ng512_nb1024_kcNone_q1.5_n32 | 0.950 | 59.69 | 0.039 | 50.1 | 1.93 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q1.5_n4 | 0.999 | 8.59 | 0.268 | 43.5 | 11.64 | 17.0 | no | — |
| ng512_nb1024_kc0.5_q1.5_n8 | 0.997 | 17.08 | 0.135 | 43.5 | 5.86 | 17.0 | no | — |
| ng512_nb1024_kc0.5_q1.5_n16 | 0.987 | 33.30 | 0.069 | 43.5 | 3.00 | 21.6 | no | — |
| ng512_nb1024_kc0.5_q1.5_n32 | 0.950 | 59.69 | 0.039 | 43.5 | 1.68 | 21.6 | no | — |

## Corridas

| clave | N | haces | γ_UV | puntos | r² | A_ruido | UV final | pared [s] |
|---|---|---|---|---|---|---|---|---|
| ng512_nb1024_p4_q1.5_kcNone | 2097152 | 1024 | 50.12 | 71 | 0.751 | 1.8e-16 | 3.4e-07 | 87.1 |
| ng512_nb1024_p4_q1.5_kc0.5 | 2097152 | 1024 | 43.45 | 73 | 0.628 | 2.3e-16 | 5.3e-07 | 76.9 |
