# Cribado del candidato «conversion_current» — desenlace: **no pasa** (sin letra sobre λ)

Preinscripción `41fd300c006e…`; commit `c121af88b`; 147 corridas.

Ventana κ̂ ∈ [0.5, 5.0] (cierre quadrant): 24/24 celdas, 0 pasan (i)+(ii), 0 sin terminar.

κ̂ mínima que pasa en TODA la malla (hasta 10000): None.

## Δβ/β canónica en el punto inicial (M₀², B, C₀), τ = 0.01

| celda | Δβ_{M₀²}/β | Δβ_B/β | Δβ_{C₀}/β |
|---|---|---|---|
| d0.01_kh0.1 | -2.52e-08 | -1.14e-07 | -0.0206 |
| d0.01_kh0.3162 | -7.97e-08 | -3.62e-07 | -0.0652 |
| d0.01_kh10000 | -0.00252 | -0.0114 | -2.06e+03 |
| d0.01_kh1000 | -0.000252 | -0.00114 | -206 |
| d0.01_kh100 | -2.52e-05 | -0.000114 | -20.6 |
| d0.01_kh10 | -2.52e-06 | -1.14e-05 | -2.06 |
| d0.01_kh1 | -2.52e-07 | -1.14e-06 | -0.206 |
| d0.01_kh3.1623 | -7.97e-07 | -3.62e-06 | -0.652 |
| d0.01_kh31.6228 | -7.97e-06 | -3.62e-05 | -6.52 |
| d0.01_kh316.228 | -7.97e-05 | -0.000362 | -65.2 |
| d0.01_kh3162.28 | -0.000797 | -0.00362 | -652 |
| d0.03_kh0.1 | -6.07e-07 | -2.79e-06 | -0.0186 |
| d0.03_kh0.3162 | -1.92e-06 | -8.83e-06 | -0.0589 |
| d0.03_kh10000 | -0.0607 | -0.279 | -1.86e+03 |
| d0.03_kh1000 | -0.00607 | -0.0279 | -186 |
| d0.03_kh100 | -0.000607 | -0.00279 | -18.6 |
| d0.03_kh10 | -6.07e-05 | -0.000279 | -1.86 |
| d0.03_kh1 | -6.07e-06 | -2.79e-05 | -0.186 |
| d0.03_kh3.1623 | -1.92e-05 | -8.83e-05 | -0.589 |
| d0.03_kh31.6228 | -0.000192 | -0.000883 | -5.89 |
| d0.03_kh316.228 | -0.00192 | -0.00883 | -58.9 |
| d0.03_kh3162.28 | -0.0192 | -0.0883 | -589 |

## Corridas con algún cruce D = 0

- ninguna

## Controles

- canon_d0.01_tau0.01: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.000226
- canon_d0.01_tau0.1: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.000204
- canon_d0.01_tau1.0: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.000246
- canon_d0.03_tau0.01: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.00247
- canon_d0.03_tau0.1: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.00181
- canon_d0.03_tau1.0: descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); cruces 0; inestabilidades de masa 1; D_fin 0.00198
- plane_d0.01_tau0.01: idéntica a la canónica (Δβ ≡ 0): True
- plane_d0.01_tau0.1: idéntica a la canónica (Δβ ≡ 0): True
- plane_d0.01_tau1.0: idéntica a la canónica (Δβ ≡ 0): True
- plane_d0.03_tau0.01: idéntica a la canónica (Δβ ≡ 0): True
- plane_d0.03_tau0.1: idéntica a la canónica (Δβ ≡ 0): True
- plane_d0.03_tau1.0: idéntica a la canónica (Δβ ≡ 0): True

## Brazo con la corriente en la trayectoria

- trajcurrent_d0.01_tau0.1_kh0.5: terminó True, divergió False; descenso detenido en la frontera (velocidad proyectada ≈ 0; S no avanza); f_fin = 0.178
- trajcurrent_d0.01_tau0.1_kh1: terminó False, divergió True; divergencia numérica: |Φ|² > 1e+06·ρ₊ o no finito en el paso 1146 (brazo no integrable con d_sigma declarado; se publica, no se reescala); f_fin = -58.7
- trajcurrent_d0.01_tau0.1_kh2.11: terminó False, divergió True; divergencia numérica: |Φ|² > 1e+06·ρ₊ o no finito en el paso 1077 (brazo no integrable con d_sigma declarado; se publica, no se reescala); f_fin = -2.16

**Lectura**: cribado diagnóstico (E13): el desenlace dice si el candidato puede ser la β del frente 2 bajo los cierres declarados; ninguna letra sobre λ; la fila decada-discriminante no cambia salvo «pasa».
