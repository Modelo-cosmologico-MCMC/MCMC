# Test del criterio de Cronos–Jeans, ronda 3: desenlace **A**

Preinscripción `ae9abba01c26`; cualificación `8a35a6a4380b`; commit `c1a3387ec`. **Lectura obligatoria**: el instrumento reproduce el criterio con su predicción cinética convergida en todas las celdas no excluidas. Motivo: umbral correcto; todas las celdas no excluidas con q > 1 dentro del 25 % de γ_cin(q, k)·W(k); dispersión entre modos ≤ 25 %. Celdas excluidas a priori: ['q2.0_n4'].

| q | n | k | γ/k medido | γ/k instrumento (W(k)) | γ/k continuo | γ/k fluido | más cerca de | r² | puntos | crecimiento | UV máx (ventana) | fila |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 4 | 25.1 | -0.1738 | 0.0000 (W = 0.999) | 0.0000 | 0.0000 | — | 0.9989 | 601 | ×1.0 | 2.9e-07 | estable |
| 0.8 | 8 | 50.3 | -0.1864 | 0.0000 (W = 0.997) | 0.0000 | 0.0000 | — | 0.8909 | 521 | ×1.0 | 1.4e-06 | estable |
| 0.8 | 16 | 100.5 | -0.1925 | 0.0000 (W = 0.987) | 0.0000 | 0.0000 | — | 0.9623 | 222 | ×1.0 | 1.7e-07 | estable |
| 0.8 | 32 | 201.1 | -0.2266 | 0.0000 (W = 0.950) | 0.0000 | 0.0000 | — | 0.9372 | 123 | ×1.0 | 1.2e-06 | estable |
| 1.2 | 4 | 25.1 | 0.1498 | 0.1485 (W = 0.999) | 0.1492 | 0.4472 | kinetic | 0.9998 | 597 | ×1624.0 | 1.0e-03 | dentro de tol. |
| 1.2 | 8 | 50.3 | 0.1462 | 0.1465 (W = 0.997) | 0.1492 | 0.4472 | kinetic | 1.0000 | 314 | ×813.8 | 1.5e-05 | dentro de tol. |
| 1.2 | 16 | 100.5 | 0.1376 | 0.1384 (W = 0.987) | 0.1492 | 0.4472 | kinetic | 1.0000 | 166 | ×55729.4 | 4.6e-07 | dentro de tol. |
| 1.2 | 32 | 201.1 | 0.1034 | 0.1062 (W = 0.950) | 0.1492 | 0.4472 | kinetic | 1.0000 | 111 | ×35285.9 | 9.7e-07 | dentro de tol. |
| 2.0 | 8 | 50.3 | 0.6080 | 0.6089 (W = 0.997) | 0.6120 | 1.0000 | kinetic | 1.0000 | 76 | ×4747.8 | 1.0e-04 | dentro de tol. |
| 2.0 | 16 | 100.5 | 0.5978 | 0.5994 (W = 0.987) | 0.6120 | 1.0000 | kinetic | 1.0000 | 38 | ×48182.9 | 2.5e-08 | dentro de tol. |
| 2.0 | 32 | 201.1 | 0.5560 | 0.5619 (W = 0.950) | 0.6120 | 1.0000 | kinetic | 1.0000 | 21 | ×202133.5 | 2.5e-08 | dentro de tol. |

Independencia de k en q = 1.2: dispersión relativa 0.036 → pasa
Independencia de k en q = 2.0: dispersión relativa 0.009 → pasa
Convergencia en haces (q = 1.2, n = 8): γ/k = 0.1462 (1024) frente a 0.1462 (512): diferencia relativa 0.0004 → pasa

Estatuto: test numérico interno (E8) bajo preinscripción nueva con celdas excluidas a priori por una cualificación E8-Q; predicción de la teoría cinética lineal (solo q entra); sin datos; A_Sculptor, la ronda 2 y la cualificación no se tocan.
