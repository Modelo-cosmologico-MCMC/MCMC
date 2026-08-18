# Ajuste de producción MCMC vs ΛCDM

Datos: 31 H(z) de cronómetros + 6 BAO + 1371 SNe (cov completa) (n = 1408). Semilla 42, 32 walkers × 4000 pasos, descarte 2000, thin 4. Priors: apéndice F (H0 ~ N(67.4, 5²)).

ΛCDM ajustado sobre LOS MISMOS datos con la MISMA maquinaria (ε = 0 exacto, Prop. A.1); M_B de SNe marginalizada analíticamente en ambos modelos por igual (no cuenta en k).

## Posteriores (mediana ± 1σ)

### MCMC (k = 4)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 67.8671 | −0.6686 / +0.6806 |
| Omega_m | 0.3263 | −0.0161 / +0.0167 |
| epsilon | 0.0172 | −0.0390 / +0.0416 |
| z_trans | 9.0894 | −4.3052 / +4.5799 |

Aceptación 0.57; τ: 53, 52, 53, 48; convergencia OK

### ΛCDM (k = 2)

| Parámetro | Mediana | −/+ 1σ |
|---|---|---|
| H0 | 67.8830 | −0.6499 / +0.6744 |
| Omega_m | 0.3270 | −0.0162 / +0.0159 |

Aceptación 0.72; τ: 34, 32; convergencia OK

## χ² por dataset (en el máximo a posteriori de cada modelo)

| Dataset | χ² MCMC | χ² ΛCDM |
|---|---|---|
| Hz | 14.42 | 14.43 |
| SNe | 1210.05 | 1210.04 |
| BAO | 1.56 | 1.56 |
| total | 1226.03 | 1226.03 |

## Criterios de información

Fórmulas: AIC = 2k − 2·ln L_max; BIC = k·ln(n) − 2·ln L_max. k_MCMC = 4 (H0, Ωm, ε, z_trans); k_ΛCDM = 2 (H0, Ωm); n = 1408.

| Modelo | k | ln L_max | AIC | BIC |
|---|---|---|---|---|
| MCMC | 4 | -613.02 | 1234.03 | 1255.03 |
| ΛCDM | 2 | -613.01 | 1230.03 | 1240.53 |

**ΔAIC (MCMC − ΛCDM) = +4.00; ΔBIC (MCMC − ΛCDM) = +14.50** (negativo favorece al MCMC; positivo, a ΛCDM). El resultado se publica sea cual sea.

Cadenas: `chains_mcmc.npz`, `chains_lcdm.npz` (en output/).
