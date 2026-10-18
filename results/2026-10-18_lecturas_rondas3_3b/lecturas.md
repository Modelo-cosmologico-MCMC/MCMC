# Lecturas a posteriori de las rondas 3 y 3b (sin letra, sin corridas)

Lectura a posteriori, SIN LETRA: no da letra a las rondas 3 o 3b ni toca reglas congeladas. Commit `c21e8c815`. Porta los guiones del apéndice de la decisión del autor del 18-oct.

## Láminas

| q | modos inestables | más rápido | tasa / tasa(n = 4) |
|---|---|---|---|
| 1.2 | n ≤ 59 | n = 34 | 5.76 |
| 1.5 | n ≤ 88 | n = 51 | 8.46 |
| 2.0 | n ≤ 114 | n = 66 | 10.86 |

Suelo implícito de la banda (mediana de 12 corridas): A_suelo ≈ 1.0e-10. Presupuesto con margen ×10: 16.1 e-pliegues; sin margen: 18.4.

| celda | t_cierre | e-pliegues | banda prevista | banda medida | excluida (×10) |
|---|---|---|---|---|---|
| (1.2, 16) | 0.244 | 5.3 | 2.0e-08 | 0.0e+00 | no |
| (1.2, 32) | 0.159 | 3.4 | 3.2e-09 | 0.0e+00 | no |
| (1.2, 4) | 0.911 | 19.6 | 3.4e-02 | 2.0e-02 | sí |
| (1.2, 8) | 0.462 | 9.9 | 2.1e-06 | 6.3e-05 | no |
| (2.0, 16) | 0.056 | 9.4 | 1.3e-06 | 0.0e+00 | no |
| (2.0, 32) | 0.030 | 5.0 | 1.6e-08 | 0.0e+00 | no |
| (2.0, 8) | 0.111 | 18.5 | 1.2e-02 | 1.3e-02 | sí |
| (1.5, 16) | 0.102 | 7.4 | 1.7e-07 | 0.0e+00 | no |
| (1.5, 32) | 0.057 | 4.1 | 6.5e-09 | 0.0e+00 | no |
| (1.5, 4) | 0.396 | 28.8 | 3.3e+02 | 1.1e-01 | sí |
| (1.5, 8) | 0.199 | 14.5 | 2.0e-04 | 2.1e-04 | no |
| (2.0, 4) | 0.221 | 36.9 | 1.1e+06 | — | sí |

Regla UV con el cierre de la ventana: (1.2, 4) γ_UV·t_ventana = 3.60, γ_UV·t_cierre = 5.31 frente a 17.03; (2.0, 8) γ_UV·t_ventana = 9.69, γ_UV·t_cierre = 14.31 frente a 17.03; (1.5, 4) γ_UV·t_ventana = 11.64, γ_UV·t_cierre = 17.19 frente a 17.03; (1.5, 8) γ_UV·t_ventana = 5.86, γ_UV·t_cierre = 8.65 frente a 17.03.

n_c admisible por celda (margen ×10; 128 = el instrumento de las rondas 3 y 3b): (1.2, 4) 21; (1.2, 8) 128; (1.2, 16) 128; (1.2, 32) 128; (1.5, 4) 19; (1.5, 8) 128; (1.5, 16) 128; (1.5, 32) 128; (2.0, 4) 19; (2.0, 8) 44; (2.0, 16) 128; (2.0, 32) 128.

Siembra: la ventana de (1.5, 4) cierra en t = 0.396 con cualquier amplitud; la banda hace 28.8 e-pliegues y haría falta A_suelo < 3.1e-15.

Ronda 3, once celdas: q = 1.2: máx |1 − c| = 0.027, dispersión 0.036; q = 2.0: máx |1 − c| = 0.011, dispersión 0.009; q = 0.8 crecimiento 1.0 (≤ 3).
Ronda 3, nueve celdas: q = 1.2: máx |1 − c| = 0.027, dispersión 0.025; q = 2.0: máx |1 − c| = 0.011, dispersión 0.008; q = 0.8 crecimiento 1.0 (≤ 3).

## Halo, ronda 3

**L1.** Δ_C(<0.1) (dex) de «sub» y «above» frente al newtoniano: 0.60 Myr +0.003 / +0.009; 1.05 Myr +0.008 / +0.024; 2.10 Myr +0.035 / +0.083; 5.10 Myr +0.127 / +0.321; cociente de ε_max 3.00; D_F(ε_soft, 0) = 1.25 / 3.75; «above» con q_N > 1 solo en r ≲ 0.118 kpc; q_eq máx = 0.33 / 0.67.

**L2.** ρ* = ρ_NFW(0.1): ε_max viable 1.58e-08, q_eq máx 0.25 (ε_soft 0.1) / 0.55 (0.05); ρ* = ρ_NFW(0.15): ε_max viable 1.00e-08, q_eq máx 0.23 (ε_soft 0.1) / 0.40 (0.05); ρ* = ρ_NFW(0.2): ε_max viable 7.08e-09, q_eq máx 0.20 (ε_soft 0.1) / 0.29 (0.05); ρ* = ρ_NFW(0.3): ε_max viable 5.01e-09, q_eq máx 0.16 (ε_soft 0.1) / 0.18 (0.05); ρ* = ρ_NFW(0.4): ε_max viable 4.47e-09, q_eq máx 0.14 (ε_soft 0.1) / 0.14 (0.05); ρ* = ρ_NFW(0.5): ε_max viable 4.47e-09, q_eq máx 0.12 (ε_soft 0.1) / 0.12 (0.05); ρ* = ρ_NFW(0.7): ε_max viable 3.16e-09, q_eq máx 0.07 (ε_soft 0.1) / 0.07 (0.05); ρ* = ρ_NFW(1.0): ε_max viable 2.51e-09, q_eq máx 0.05 (ε_soft 0.1) / 0.05 (0.05).

**L3.** r_CJ = 0.709 / 0.180 kpc; densidad extra ×7.37; ε_soft 0.2: A_S 2.9317 Myr, 0.05·A_S sin salida en 40.0 Myr; ε_soft 0.1: A_S 0.5994 Myr, 0.05·A_S sin salida en 10.4 Myr; ε_soft 0.05: A_S 0.0757 Myr, 0.05·A_S 9.661 Myr (cociente 128); ε_soft 0.025: A_S 0.0106 Myr, 0.05·A_S 0.477 Myr (cociente 45) (caída libre 4.47).

**L4.** η_E = |ΔE_self|/|ΔU_C|: sweep_AS_eps0.2 0.042; sweep_AS_eps0.1 0.048; sweep_AS_eps0.05 0.045; sweep_AS_eps0.025 0.634; series_AS_ds0.02 0.040; series_AS_ds0.08 0.018; ctrlN_AS_N1e5 0.120; sweep_b005_eps0.2 0.182; sweep_b005_eps0.1 0.129; sweep_b005_eps0.05 0.172; sweep_b005_eps0.025 0.167; ctrlN_b005_N1e5 0.181.

**L5.** Newtoniano: Δlog M(<0.1) mínimo -0.222 dex (t = 30.0 Myr); M(<0.4) final +0.029 dex; brazos newtonianos: ['newton_N1e6'].

**L6.** p_ε local 0.2→0.1: 2.29, 0.1→0.05: 2.98, 0.05→0.025: 2.84; p_ds 0.08→0.04: 0.97, 0.04→0.02: 2.14.

