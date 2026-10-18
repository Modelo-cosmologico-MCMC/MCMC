# Test del criterio de Cronos–Jeans, ronda 3b (q = 1.5): desenlace **INDETERMINADO**

Preinscripción `d0c16affe650`; cualificación `1a3283f3d200`; commit `0eba59c2b`. **Lectura obligatoria**: puerta violada: el resultado se retiene. Motivo: fase lineal no resuelta: ['q1.5_n4']. Celdas excluidas a priori: ninguna. La ronda 3 (desenlace A) no se toca.

| q | n | k | γ/k medido | γ/k instrumento (W(k)) | γ/k continuo | γ/k fluido | más cerca de | r² | puntos | crecimiento | UV máx (ventana) | fila |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 4 | 25.1 | 0.4212 | 0.3420 (W = 0.999) | 0.3427 | 0.7071 | kinetic | 0.9324 | 166 | ×3310.4 | 5.5e-03 | no resuelto |
| 1.5 | 8 | 50.3 | 0.3395 | 0.3398 (W = 0.997) | 0.3427 | 0.7071 | kinetic | 1.0000 | 135 | ×2353.5 | 7.2e-06 | dentro de tol. |
| 1.5 | 16 | 100.5 | 0.3302 | 0.3312 (W = 0.987) | 0.3427 | 0.7071 | kinetic | 1.0000 | 70 | ×81187.7 | 7.1e-08 | dentro de tol. |
| 1.5 | 32 | 201.1 | 0.2924 | 0.2969 (W = 0.950) | 0.3427 | 0.7071 | kinetic | 1.0000 | 40 | ×142030.2 | 7.4e-08 | dentro de tol. |

Independencia de k en q = 1.5: dispersión relativa 0.234 → pasa
Convergencia en haces (q = 1.5, n = 8): γ/k = 0.3395 (1024) frente a 0.3396 (512): diferencia relativa 0.0001 → pasa

Estatuto: test numérico interno (E8) bajo preinscripción nueva, ampliación de la ronda 3 a q = 1.5 con celdas excluidas a priori por una adenda de cualificación E8-Q; predicción de la teoría cinética lineal (solo q entra); sin datos; A_Sculptor, la ronda 3 y las cualificaciones no se tocan.
