# Rondas 3 y 3b de Cronos–Jeans: la banda inestable bajo k_c (diagnóstico a posteriori, sin letra)

Escrito el 18-oct **después** de ver el INDETERMINADO de la ronda 3b. La puerta UV de ambas rondas mide solo la banda k ≥ k_c (`uv_rms`); con q > 1 los modos k < k_c con q·W(k) > 1 también crecen, con γ ∝ k, desde el redondeo. Definición: banda bajo k_c = √(rms_δ² − uv_rms² − 2|δ_n|²) de la densidad depositada; umbral = A_nl de la puerta UV de cada preinscripción; cierre medido = primera muestra con |δ_n| ≥ el borde alto de la ventana. **No cambia ninguna regla, puerta ni desenlace**: la ronda 3b sigue INDETERMINADA y la ronda 3 sigue A.

| ronda | corrida | t cierre previsto | t cierre medido | t banda > A_nl | banda en el cierre | banda máx | cierra antes | γ/k medido / instrumento | r² |
|---|---|---|---|---|---|---|---|---|---|
| ronda 3 | q0.8_n16 | — | — | — | — | 4.3e-07 | — (estable) | — | 0.9623 |
| ronda 3 | q0.8_n32 | — | — | — | — | 1.3e-06 | — (estable) | — | 0.9372 |
| ronda 3 | q0.8_n4 | — | — | — | — | 7.1e-07 | — (estable) | — | 0.9989 |
| ronda 3 | q0.8_n8 | — | — | — | — | 2.1e-06 | — (estable) | — | 0.8909 |
| ronda 3 | q1.2_n16 | 0.244 | 0.245 | 0.793 | 0.0e+00 | 1.9e-02 | sí | 0.9942 | 1.0000 |
| ronda 3 | q1.2_n32 | 0.159 | 0.162 | — | 0.0e+00 | 4.2e-03 | sí | 0.9731 | 1.0000 |
| ronda 3 | q1.2_n4 | 0.911 | 0.892 | 0.851 | 2.0e-02 | 8.8e-02 | **no** | 1.0087 | 0.9998 |
| ronda 3 | q1.2_n8 | 0.462 | 0.463 | 0.719 | 6.3e-05 | 7.4e-02 | sí | 0.9981 | 1.0000 |
| ronda 3 | q1.2_n8_beams512 | 0.462 | 0.463 | 0.741 | 3.8e-05 | 7.0e-02 | sí | — | — |
| ronda 3 | q2.0_n16 | 0.056 | 0.057 | 0.138 | 0.0e+00 | 4.1e-01 | sí | 0.9973 | 1.0000 |
| ronda 3 | q2.0_n32 | 0.030 | 0.031 | 0.094 | 0.0e+00 | 2.4e-01 | sí | 0.9895 | 1.0000 |
| ronda 3 | q2.0_n8 | 0.111 | 0.112 | 0.111 | 1.3e-02 | 4.4e-01 | **no** | 0.9986 | 1.0000 |
| ronda 3b | q1.5_n16 | 0.102 | 0.103 | 0.261 | 0.0e+00 | 2.2e-01 | sí | 0.9971 | 1.0000 |
| ronda 3b | q1.5_n32 | 0.057 | 0.058 | 0.185 | 0.0e+00 | 3.5e-02 | sí | 0.9850 | 1.0000 |
| ronda 3b | q1.5_n4 | 0.396 | 0.294 | 0.254 | 1.1e-01 | 2.5e-01 | **no** | 1.2318 | 0.9324 |
| ronda 3b | q1.5_n8 | 0.199 | 0.200 | 0.259 | 2.1e-04 | 2.6e-01 | sí | 0.9991 | 1.0000 |
| ronda 3b | q1.5_n8_beams512 | 0.199 | 0.200 | 0.260 | 2.0e-04 | 2.5e-01 | sí | — | — |

Ventanas que cerraron con la banda bajo k_c ya por encima de A_nl: ['q1.2_n4', 'q2.0_n8', 'q1.5_n4'].
