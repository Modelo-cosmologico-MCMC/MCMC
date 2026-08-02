# Ajuste de producción v2 — cinco bloques (la reconciliación)

Datos: 31 CC + 6 BAO + 1371 SNe + CMB comprimido (3) + 11 fσ8 (n = 1422). Semilla 42, 24 walkers × 8000, descarte 4000, thin 4.
ΛCDM con la misma maquinaria (ε=0 exacto, Prop. A.1). Salvedades declaradas: z*/r_s por Hu–Sugiyama (sesgo ~0.3% en l_A, idéntico en ambos modelos); CMB diagonal sin correlaciones; r_d fiducial en BAO; SNe sin ancla (M_B marginalizada en ambos).

## Posteriores (mediana ± 1σ)

### MCMC (k = 6)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 67.0597 | −1.5034 / +1.4236 |
| Omega_m | 0.3229 | −0.0149 / +0.0162 |
| epsilon | 0.0153 | −0.0389 / +0.0428 |
| z_trans | 9.0916 | −4.2346 / +4.7968 |
| omega_b | 0.0224 | −0.0002 / +0.0001 |
| sigma8 | 0.8022 | −0.0290 / +0.0284 |

Aceptación 0.44; τ: 157, 152, 141, 106, 71, 97; convergencia OK

### ΛCDM (k = 4)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 67.5705 | −0.7067 / +0.6970 |
| Omega_m | 0.3178 | −0.0090 / +0.0095 |
| omega_b | 0.0224 | −0.0002 / +0.0001 |
| sigma8 | 0.8004 | −0.0295 / +0.0295 |

Aceptación 0.59; τ: 48, 48, 48, 48; convergencia OK

## χ² por bloque (máximo a posteriori de cada modelo)

| Bloque | χ² MCMC | χ² ΛCDM |
|---|---|---|
| Hz | 14.50 | 14.49 |
| SNe | 1210.77 | 1210.73 |
| BAO | 1.70 | 1.68 |
| CMB | 0.02 | 0.07 |
| fs8 | 8.05 | 8.06 |
| total | 1235.04 | 1235.03 |

## Criterios de información

AIC = 2k − 2lnL; BIC = k·ln(n) − 2lnL; k=6 vs k=4; n=1422.

| Modelo | k | ln L_max | AIC | BIC |
|---|---|---|---|---|
| MCMC | 6 | -617.52 | 1247.04 | 1278.60 |
| ΛCDM | 4 | -617.52 | 1243.03 | 1264.07 |

**ΔAIC (MCMC − ΛCDM) = +4.01; ΔBIC = +14.53** (negativo favorece al MCMC). El resultado se publica sea cual sea.

Cadenas: `chains_mcmc_v2.npz`, `chains_lcdm_v2.npz`.
