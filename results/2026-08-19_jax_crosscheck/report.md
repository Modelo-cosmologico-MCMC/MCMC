# Crosscheck JAX ↔ NumPy del fondo (6A, prioridad 3)

Commit: `1e5332cf3300`. Implementación independiente
(`validation/jax_background.py`: fórmulas del Apéndice A + mapa S↔z
operativo, integración Gauss-Legendre espectral de 96 nodos, float64;
únicas piezas compartidas: las fórmulas declaradas y las constantes
físicas de `SHARED_CONSTANTS`) contra las implementaciones NumPy de
producción. Estatuto: validación interna de implementación (E8) —
equivalencia numérica, no validación física.

## Puertas predeclaradas y resultado

max |ΔX/X| < 1e-08 y RMS <
1e-10 por bloque; |Δχ²| <
1e-08. Bloque 5 SIN puerta (medición
publicada).

| bloque | max | RMS | puerta |
|---|---|---|---|
| 1 — E, H, D_H (malla z ∈ [0, 3], 4 θ) | 4.44e-16 | 6.72e-17 | PASS |
| 2 — mapa S↔z (S ∈ [1.001, 95]) | 2.22e-16 | 1.10e-16 | PASS |
| 3 — vector DESI (13 comp., 4 θ) | 1.24e-14 | 4.27e-15 | PASS |
| 4 — χ² (argmin publicados + rincón) | 6.87e-11 | — | PASS |

**TODAS LAS PUERTAS PASS.**

Los θ del bloque 3-4 incluyen los argmin PUBLICADOS de
benchmark.json y contrast.json — el crosscheck ata las dos
implementaciones exactamente donde viven los números publicados:

| θ | χ² NumPy | χ² JAX | \|Δχ²\| |
|---|---|---|---|
| benchmark_argmin_lcdm | 10.284009527 | 10.284009527 | 2.13e-12 |
| contrast_argmin_lcdm | 10.284009527 | 10.284009527 | 2.13e-12 |
| contrast_argmin_mcmc | 10.057790583 | 10.057790583 | 2.10e-12 |
| corner | 3814.469483175 | 3814.469483175 | 6.87e-11 |

## El hallazgo del crosscheck (bloque histórico)

La primera ejecución midió el integrador del vector DESI de
producción (trapecio + interpolación lineal) en ~4.9e-8 relativo —
POR ENCIMA de la puerta — con efecto ≤ 2.7e-6 en los χ² publicados
(veredicto 6A robusto: ningún número con 3 decimales se movía). En
vez de relajar la puerta, se corrigió el integrador (cumulativa
Simpson O(h⁴) + cola Gauss-Legendre por bin; verificado a 5.8e-15
contra scipy quad) y se regeneraron los artefactos DESI. El número de
esta tabla es el del integrador corregido.

## Bloque 5 — distancias SNe de producción (sin puerta)

El integrador SNe de producción (`comoving_distance`, trapecio de
2048 puntos, usado por los ajustes v1/v2 ya publicados) difiere del
espectral en **3.00e-06 relativo**
(máx sobre z ∈ [0.01, 2.26] en el θ de producción), es decir
≤ 6.5e-06 mag en μ — despreciable
frente a σ_μ ~ 0.1 de Pantheon+ y sin efecto en ΔAIC/ΔBIC publicados.
Se publica como medición, no se toca la maquinaria de una ronda ya
fusionada; si una ronda futura necesita μ a < 1e-8, este número dice
exactamente qué cambiar.
