# Ajuste de producción v2 — cinco bloques (la reconciliación)

Datos: 31 CC + 6 BAO + 1371 SNe + CMB comprimido (3) + 11 fσ8 (n = 1422). Semilla 42, 24 walkers × 8000, descarte 4000, thin 4.
ΛCDM con la misma maquinaria (ε=0 exacto, Prop. A.1). Salvedades declaradas: z*/r_s por Hu–Sugiyama (sesgo ~0.3% en l_A, idéntico en ambos modelos); CMB diagonal sin correlaciones; r_d fiducial en BAO; SNe sin ancla (M_B marginalizada en ambos).

## Posteriores (mediana ± 1σ)

### MCMC (k = 6)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 68.1938 | −0.4045 / +0.4036 |
| Omega_m | 0.3119 | −0.0061 / +0.0065 |
| epsilon | 0.0175 | −0.0407 / +0.0426 |
| z_trans | 9.1355 | −4.4643 / +4.7974 |
| omega_b | 0.0224 | −0.0001 / +0.0001 |
| sigma8 | 0.7994 | −0.0290 / +0.0285 |

Aceptación 0.48; τ: 109, 101, 80, 77, 64, 80; convergencia OK

### ΛCDM (k = 4)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 68.1689 | −0.4149 / +0.4156 |
| Omega_m | 0.3122 | −0.0064 / +0.0066 |
| omega_b | 0.0224 | −0.0001 / +0.0001 |
| sigma8 | 0.7998 | −0.0300 / +0.0299 |

Aceptación 0.59; τ: 53, 53, 47, 46; convergencia OK

## χ² por bloque (máximo a posteriori de cada modelo)

| Bloque | χ² MCMC | χ² ΛCDM |
|---|---|---|
| Hz | 14.50 | 14.50 |
| SNe | 1210.61 | 1210.65 |
| BAO | 1.76 | 1.72 |
| CMB | 0.10 | 0.06 |
| fs8 | 8.13 | 8.08 |
| total | 1235.08 | 1235.00 |

## Criterios de información

AIC = 2k − 2lnL; BIC = k·ln(n) − 2lnL; k=6 vs k=4; n=1422.

| Modelo | k | ln L_max | AIC | BIC |
|---|---|---|---|---|
| MCMC | 6 | -617.54 | 1247.08 | 1278.64 |
| ΛCDM | 4 | -617.50 | 1243.00 | 1264.04 |

**ΔAIC (MCMC − ΛCDM) = +4.08; ΔBIC = +14.60** (negativo favorece al MCMC). El resultado se publica sea cual sea.

Cadenas: `chains_mcmc_v2.npz`, `chains_lcdm_v2.npz`.
