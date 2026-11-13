# Cascada de habilitación — letra global: **INDETERMINADO**

Preinscripción `9040df79900a…`; commit `646f02363`. Letra por δ₀; la global es la peor.

| δ₀ | gatillos S (V1D, V2D, V3D) | cocientes | frente a 0.009 / 0.099 / 0.999 | letra | motivo |
|---|---|---|---|---|---|
| 0.003 | 0.008662, 0.097837, 0.990064 | 11.295, 10.120 | -3.76 %, -1.18 %, -0.89 % | A | 2 de 2 cocientes en [7, 14] |
| 0.01 | 0.008675, 0.097847, 0.990064 | 11.279, 10.118 | -3.61 %, -1.16 %, -0.89 % | INDETERMINADO | control semilla_1e-9 |
| 0.0554 | 0.008706 |  | -3.27 % | INDETERMINADO | puerta finished; puerta final_state; control suelo_1e-18 |

## Puertas y controles

| δ₀ | finished | monotonia | identity | trigger_is_min_eigenvalue | hierarchy | final_state | convergence | sin_semilla | semilla_1e-3 | semilla_1e-9 | suelo_1e-18 | sin_compuerta | jerarquia_rota | pesos_iguales |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.003 | sí | sí | sí | sí | sí | sí | sí | sí | sí | sí | sí | sí | sí | sí |
| 0.01 | sí | sí | sí | sí | sí | sí | sí | sí | sí | NO | sí | sí | sí | sí |
| 0.0554 | NO | sí | sí | sí | sí | NO | sí | sí | sí | sí | NO | sí | sí | sí |

| δ₀ | mayor ΔV/T_total | abs(S − f) máx. | cambio con el paso a la mitad | con semilla 1e-3 | con semilla 1e-9 | con suelo 1e-18 |
|---|---|---|---|---|---|---|
| 0.003 | 9.9e-16 | 1.3e-07 | 1.5e-06 | 2.3e-04 | 3.4e-07 | 2.2e-07 |
| 0.01 | 9.4e-16 | 1.5e-07 | 8.8e-08 | 2.3e-04 | — | 1.5e-07 |
| 0.0554 | 5.8e-17 | 1.1e-10 | 9.7e-08 | 2.3e-04 | 2.3e-10 | — |

## Sensibilidad (sin letra)

| δ₀ | brazo | colapsos | gatillos S | cocientes | lectura de la regla |
|---|---|---|---|---|---|
| 0.003 | lambda_5 | 3 | 0.03086, 0.18992, 0.98448 | 6.15, 5.18 | C |
| 0.003 | lambda_7 | 3 | 0.01685, 0.13833, 0.98872 | 8.21, 7.15 | A |
| 0.003 | lambda_14 | 3 | 0.00456, 0.07023, 0.99034 | 15.41, 14.10 | C |
| 0.003 | lambda_20 | 3 | 0.00228, 0.04929, 0.99031 | 21.59, 20.09 | C |
| 0.003 | referencia_ley | 3 | 0.00929, 0.09842, 0.99007 | 10.59, 10.06 | A |
| 0.003 | theta_0.8 | 3 | 0.00769, 0.09537, 0.96293 | 12.41, 10.10 | A |
| 0.003 | theta_0.95 | 1 | 0.00895 |  | C |
| 0.003 | Delta_0.02 | 3 | 0.00864, 0.09810, 0.98999 | 11.35, 10.09 | A |
| 0.003 | Delta_0.1 | 3 | 0.00875, 0.09667, 0.99045 | 11.05, 10.25 | A |
| 0.01 | lambda_5 | 3 | 0.03091, 0.18994, 0.98448 | 6.15, 5.18 | C |
| 0.01 | lambda_7 | 3 | 0.01688, 0.13835, 0.98872 | 8.20, 7.15 | A |
| 0.01 | lambda_14 | 3 | 0.00456, 0.07023, 0.99034 | 15.39, 14.10 | C |
| 0.01 | lambda_20 | 3 | 0.00229, 0.04930, 0.99031 | 21.56, 20.09 | C |
| 0.01 | referencia_ley | 3 | 0.00982, 0.09890, 0.99008 | 10.07, 10.01 | A |
| 0.01 | theta_0.8 | 3 | 0.00773, 0.09537, 0.96293 | 12.33, 10.10 | A |
| 0.01 | theta_0.95 | 1 | 0.00895 |  | C |
| 0.01 | Delta_0.02 | 3 | 0.00865, 0.09810, 0.98999 | 11.33, 10.09 | A |
| 0.01 | Delta_0.1 | 1 | 0.00876 |  | C |
| 0.0554 | lambda_5 | 3 | 0.03102, 0.19000, 0.98448 | 6.13, 5.18 | C |
| 0.0554 | lambda_7 | 3 | 0.01694, 0.13839, 0.98872 | 8.17, 7.14 | A |
| 0.0554 | lambda_14 | 3 | 0.00458, 0.07025, 0.99034 | 15.34, 14.10 | C |
| 0.0554 | lambda_20 | 3 | 0.00229, 0.04930, 0.99031 | 21.48, 20.09 | C |
| 0.0554 | referencia_ley | 3 | 0.01132, 0.10027, 0.99009 | 8.85, 9.87 | A |
| 0.0554 | theta_0.8 | 1 | 0.00784 |  | C |
| 0.0554 | theta_0.95 | 1 | 0.00896 |  | C |
| 0.0554 | Delta_0.02 | 1 | 0.00869 |  | C |
| 0.0554 | Delta_0.1 | 1 | 0.00879 |  | C |

## Juguete radial del oráculo (referencia ya pública, sin letra)

Gatillos S = 0.008643, 0.097822, 0.990063; cocientes 11.318, 10.121; tensión total 93.20 frente a la nominal 93.43.

**Lectura**: comprobación interna (E8): la letra dice si el modelo efectivo, con λ puesto a mano, produce tres colapsos ordenados con cocientes de Década en el reloj completo y supera sus controles; no deriva λ ni la emergencia dimensional (decisiones 5 y 7 del autor).

