# Nivel A — halo aislado con y sin Cronos v3: desenlace **INDETERMINADO**

Preinscripción `72c534dc1c09` (congelada antes de generar ninguna condición inicial); commit `3a9296296`.

**Lectura obligatoria**: puerta violada o control sin efecto: el resultado se retiene. Motivo: el control c′ (10·A_Sculptor) no altera el interior: implementación en duda.

## Puertas

Equilibrio del brazo a (|Δlog10 ρ| en [1, 10] kpc, media de semillas): 0.019 dex (tolerancia 0.1) → pasa.

| corrida | ΔE/E | puerta | régimen débil | t_final [Gyr] | parada temprana |
|---|---|---|---|---|---|
| a_seed1 | 1.37e-04 | pasa | sí | 2.000 | — |
| a_seed2 | 1.17e-04 | pasa | sí | 2.000 | — |
| a_seed3 | 3.74e-05 | pasa | sí | 2.000 | — |
| b_res_seed1 | 2.54e-03 | publicada | sí | 2.000 | — |
| b_seed1 | 1.72e-02 | publicada | sí | 2.000 | — |
| b_seed2 | 5.14e-03 | publicada | sí | 2.000 | — |
| b_seed3 | 1.39e-02 | publicada | sí | 2.000 | — |
| b_static_seed1 | 2.56e-04 | pasa | sí | 0.250 | — |
| c_seed1 | 0.00e+00 | publicada | NO | 0.000 | régimen débil violado en t = 0 (ε_c ≥ weak_max en r ≥ ε_soft) |
| cprime_seed1 | 3.64e-01 | publicada | sí | 0.055 | runaway: v_well > 1000 km/s en t = 0.0551 Gyr |

## Perfiles finales (t = 2.0 Gyr): cociente b/a por bandas (media de semillas)

| banda [kpc] | log10(ρ_b/ρ_a) | σ_semillas(a) | N_a en banda | significativo |
|---|---|---|---|---|
| inner_0p1_0p4 [0.1, 0.4] | -0.015 | 0.073 | 78 | no |
| inner_0p4_1 [0.4, 1.0] | -0.282 | 0.042 | 881 | sí |
| core_1_2p3 [1.0, 2.3] | -0.100 | 0.005 | 3075 | sí |
| fit_2p3_5 [2.3, 5.0] | -0.012 | 0.002 | 10329 | no |
| outer_5_20 [5.0, 20.0] | +0.002 | 0.001 | 83236 | no |

Ajuste de forma en [0.5, 5.0] kpc — brazo b: preferida cored, Δrmse = +0.020 dex, r_c(cored) = 1.64 kpc, r_s(NFW) = 26.96 kpc; brazo a: preferida nfw, Δrmse = -0.024 dex.

M_b(<0.4 kpc)/M_a(<0.4 kpc) por instantánea: [1.0, 4.32, 0.68, 0.85, 0.93, 1.08] (t = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0] Gyr); colapso progresivo: False (paradas por runaway: [False, False, False], t_stop = [2.0, 2.0, 2.0] Gyr). N_a(<0.4 kpc) final = 97 (resuelto si ≥ 2000).

Control de resolución del campo (k_inner 128 frente a 64, semilla 1, t = 2.000 Gyr): log10(M_res/M_b)(<0.4 kpc) = +0.833 → NO convergido (t_stop: b 2.000, b_res 2.000 Gyr).

Control c′ (10·A_Sculptor, t = 0.055099999999999996 Gyr): log10(ρ_c′/ρ_a) en [0.4, 1) kpc = +0.050 → NO responde.

Control c (cierre cosmológico): semilla 1: parada = True (régimen débil violado en t = 0 (ε_c ≥ weak_max en r ≥ ε_soft)), ε_c,max(t=0) = 3.48e+00

Estatuto: experimento numérico interno (E8) bajo preinscripción, sin datos observacionales; campo de Cronos en aproximación de campo medio esférico; N y duración limitados por el presupuesto del entorno (declarado en la preinscripción).
