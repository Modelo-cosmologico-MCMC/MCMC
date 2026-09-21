# Test del criterio de Cronos–Jeans: desenlace **INDETERMINADO**

Preinscripción `3dcd7417db06`; commit `b24f2b46f`. **Lectura obligatoria**: fase lineal no resuelta o puerta violada: el resultado se retiene. Motivo: fase lineal no resuelta: ['q1.2_n4', 'q2.0_n4', 'q2.0_n8'].

| q | modo n | k | γ/k medido | γ/k cinético (instrumento, W(k)) | γ/k cinético (continuo) | γ/k fluido | más cerca de | r² | puntos | factor de crecimiento | veredicto de fila |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 4 | 25.1 | -0.4794 | 0.0000 (W = 0.999) | 0.0000 | 0.0000 | — | 0.989 | 266 | ×1.0 | estable |
| 0.5 | 8 | 50.3 | -0.1353 | 0.0000 (W = 0.997) | 0.0000 | 0.0000 | — | 0.413 | 301 | ×1.0 | estable |
| 0.5 | 16 | 100.5 | -0.3457 | 0.0000 (W = 0.987) | 0.0000 | 0.0000 | — | 0.795 | 94 | ×1.0 | estable |
| 0.8 | 4 | 25.1 | -0.1902 | 0.0000 (W = 0.999) | 0.0000 | 0.0000 | — | 0.974 | 301 | ×1.0 | estable |
| 0.8 | 8 | 50.3 | -0.1062 | 0.0000 (W = 0.997) | 0.0000 | 0.0000 | — | 0.541 | 301 | ×1.0 | estable |
| 0.8 | 16 | 100.5 | -0.1964 | 0.0000 (W = 0.987) | 0.0000 | 0.0000 | — | 0.884 | 112 | ×1.0 | estable |
| 1.2 | 4 | 25.1 | 0.2212 | 0.1485 (W = 0.999) | 0.1492 | 0.4472 | kinetic | 0.917 | 170 | ×1944.6 | no resuelto |
| 1.2 | 8 | 50.3 | 0.1459 | 0.1465 (W = 0.997) | 0.1492 | 0.4472 | kinetic | 1.000 | 157 | ×824.7 | dentro de tol. |
| 1.2 | 16 | 100.5 | 0.1375 | 0.1384 (W = 0.987) | 0.1492 | 0.4472 | kinetic | 1.000 | 84 | ×2555.6 | dentro de tol. |
| 2.0 | 4 | 25.1 | 0.8971 | 0.6112 (W = 0.999) | 0.6120 | 1.0000 | fluid | 0.276 | 26 | ×1673.0 | no resuelto |
| 2.0 | 8 | 50.3 | 0.7167 | 0.6089 (W = 0.997) | 0.6120 | 1.0000 | kinetic | 0.973 | 27 | ×1339.4 | no resuelto |
| 2.0 | 16 | 100.5 | 0.5976 | 0.5994 (W = 0.987) | 0.6120 | 1.0000 | kinetic | 1.000 | 20 | ×14147.5 | dentro de tol. |

Independencia de k en q = 1.2: dispersión relativa 0.428 → FALLA
Independencia de k en q = 2.0: dispersión relativa 0.388 → FALLA
Convergencia en haces (q = 1.2, n = 8): γ/k = 0.1459 (1024) frente a 0.1449 (512): diferencia relativa 0.007 → pasa

Estatuto: test numérico interno (E8) bajo preinscripción; predicción de la teoría cinética lineal sin parámetros ajustados; sin datos observacionales; A_Sculptor y el criterio (fila criterio-cronos-jeans) no se tocan.
