# Test del criterio de Cronos–Jeans, ronda 2: desenlace **INDETERMINADO**

Preinscripción `3fd12bf24888`; commit `6fb3ebe81`. **Lectura obligatoria**: puerta violada: el resultado se retiene; el paso siguiente es otro instrumento, no otra siembra. Motivo: fase lineal no resuelta: ['q2.0_n4'] — cambiar de instrumento (Vlasov euleriano), no de siembra.

| q | n | k | γ/k medido | γ/k instrumento (W(k)) | γ/k continuo | γ/k fluido | más cerca de | r² | puntos | crecimiento | retículo exacto | fila |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 4 | 25.1 | -0.1738 | 0.0000 (W = 0.999) | 0.0000 | 0.0000 | — | 0.9989 | 601 | ×1.0 | True | estable |
| 0.8 | 8 | 50.3 | -0.1864 | 0.0000 (W = 0.997) | 0.0000 | 0.0000 | — | 0.8909 | 521 | ×1.0 | True | estable |
| 0.8 | 16 | 100.5 | -0.1925 | 0.0000 (W = 0.987) | 0.0000 | 0.0000 | — | 0.9623 | 222 | ×1.0 | True | estable |
| 0.8 | 32 | 201.1 | -0.2266 | 0.0000 (W = 0.950) | 0.0000 | 0.0000 | — | 0.9372 | 123 | ×1.0 | True | estable |
| 1.2 | 4 | 25.1 | 0.1498 | 0.1485 (W = 0.999) | 0.1492 | 0.4472 | kinetic | 0.9998 | 597 | ×919.9 | True | dentro de tol. |
| 1.2 | 8 | 50.3 | 0.1462 | 0.1465 (W = 0.997) | 0.1492 | 0.4472 | kinetic | 1.0000 | 314 | ×822.8 | True | dentro de tol. |
| 1.2 | 16 | 100.5 | 0.1376 | 0.1384 (W = 0.987) | 0.1492 | 0.4472 | kinetic | 1.0000 | 166 | ×56102.8 | True | dentro de tol. |
| 1.2 | 32 | 201.1 | 0.1034 | 0.1062 (W = 0.950) | 0.1492 | 0.4472 | kinetic | 1.0000 | 111 | ×35210.4 | True | dentro de tol. |
| 2.0 | 4 | 25.1 | 1.0079 | 0.6112 (W = 0.999) | 0.6120 | 1.0000 | fluid | 0.8062 | 60 | ×3449.8 | True | no resuelto |
| 2.0 | 8 | 50.3 | 0.6080 | 0.6089 (W = 0.997) | 0.6120 | 1.0000 | kinetic | 1.0000 | 76 | ×6006.6 | True | dentro de tol. |
| 2.0 | 16 | 100.5 | 0.5978 | 0.5994 (W = 0.987) | 0.6120 | 1.0000 | kinetic | 1.0000 | 38 | ×63179.0 | True | dentro de tol. |
| 2.0 | 32 | 201.1 | 0.5560 | 0.5619 (W = 0.950) | 0.6120 | 1.0000 | kinetic | 1.0000 | 21 | ×196714.2 | True | dentro de tol. |

Independencia de k en q = 1.2: dispersión relativa 0.036 → pasa
Independencia de k en q = 2.0: dispersión relativa 0.569 → FALLA
Convergencia en haces (q = 1.2, n = 8): γ/k = 0.1462 (1024) frente a 0.1462 (512): diferencia relativa 0.0004 → pasa

Estatuto: test numérico interno (E8) bajo preinscripción nueva; predicción de la teoría cinética lineal (solo q entra); sin datos; A_Sculptor, la fila criterio-cronos-jeans y la ronda 1 no se tocan.
