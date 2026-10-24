# Intercambio de masa en el borde de ECV: tasas por halo y corridas del código gravotérmico, sin letra

Commit `cf7cdc30d`. Tasas por unidad de la masa dentro de R₀, en unidades de H, sin acreción salvo la columna del brazo A (acreción 0.67·H0, la media de un halo de 10¹² M☉ hoy).

| c | afueras | ρ_b/ρ_E = ξ_eq | evaporación (E2) | contacto ξ = 1 | contacto ξ = 0.5 | A, ξ = ξ_s = 0.3 |
|---|---|---|---|---|---|---|
| 5 | nfw prolongado | 0.237 | 0.062 | +0.198 | +0.068 | +0.210 |
| 5 | capa del borde | 0.393 | 0.120 | +0.186 | +0.033 | +0.226 |
| 5 | media del entorno | 0.460 | 0.159 | +0.187 | +0.014 | +0.240 |
| 10 | nfw prolongado | 0.199 | 0.051 | +0.206 | +0.078 | +0.209 |
| 10 | capa del borde | 0.330 | 0.100 | +0.203 | +0.052 | +0.225 |
| 10 | media del entorno | 0.460 | 0.159 | +0.187 | +0.014 | +0.240 |
| 15 | nfw prolongado | 0.181 | 0.046 | +0.210 | +0.082 | +0.209 |
| 15 | capa del borde | 0.299 | 0.090 | +0.211 | +0.060 | +0.225 |
| 15 | media del entorno | 0.460 | 0.159 | +0.187 | +0.014 | +0.240 |
| 20 | nfw prolongado | 0.169 | 0.043 | +0.212 | +0.084 | +0.208 |
| 20 | capa del borde | 0.281 | 0.084 | +0.216 | +0.066 | +0.224 |
| 20 | media del entorno | 0.460 | 0.159 | +0.187 | +0.014 | +0.240 |

Halo del autor (10¹² M☉, c = 10, σ_x/m = 10 cm²/g; σ̂ = 0.249), con la fuerza de la ECV y la pared en R₀ (580 capas); ξ_eq del código = 0.330:

| ley de borde | t_c (Gyr) | cambio de t_c | masa final / inicial | tasa inicial (H0) | cierre de la energía |
|---|---|---|---|---|---|
| sin | 45.05 | +0.00% | 1.000 | +0.000 | +1.3e-03 |
| A 0.2 | 45.17 | +0.27% | 1.199 | +0.067 | +1.3e-03 |
| A 0.5 | 45.57 | +1.16% | 1.660 | +0.202 | +1.3e-03 |
| A 1 | 46.40 | +3.02% | 3.662 | +0.606 | +1.4e-03 |
| contacto 0.2 | 44.66 | -0.86% | 0.945 | -0.039 | +1.3e-03 |
| contacto eq | 45.03 | -0.05% | 0.996 | +0.000 | +1.3e-03 |
| contacto 0.5 | 45.09 | +0.10% | 1.061 | +0.052 | +1.3e-03 |
| contacto 1 | 45.49 | +0.99% | 1.314 | +0.203 | +1.3e-03 |
| barrido contacto 0.5 | 45.11 | +0.15% | 1.066 | +0.069 | +1.3e-03 |
| barrido contacto 1 | 45.64 | +1.33% | 1.437 | +0.406 | +1.3e-03 |
| E2 | 44.25 | -1.76% | 0.842 | -0.100 | +1.3e-03 |
| barrido E2 | 44.15 | -1.99% | 0.820 | -0.133 | +1.3e-03 |
| onsager 1 | 45.29 | +0.54% | 1.238 | +0.111 | +1.3e-03 |

Convergencia (el doble de capas): contacto 1: +0.99% frente a +0.98%; E2: -1.76% frente a -1.79%; A 0.5: +1.16% frente a +1.16%.

- Con la fuerza de la ECV, la capa del borde del código es más densa y más caliente que el NFW prolongado: ξ_eq = 0.330 frente a 0.199, y la evaporación sale 0.100·H frente a 0.051·H (c = 10). Las cifras del documento son las del NFW prolongado.
- La energía cierra: lo que traen el intercambio y el trabajo sobre la pared explica el cambio de la energía salvo la deriva del propio código (≤ 1.4e-03 de |E|).

**Lectura**: sin letra (E8). Comprobación del instrumento con cada ley de borde; ninguna cifra elige la ley.
